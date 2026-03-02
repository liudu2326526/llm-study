# OpenClaw Agent 路由与工具选择机制

## 1. 总览

OpenClaw 支持多个 Agent、多个消息渠道（Telegram、Discord、Slack、Signal 等）、多种工具。整个从消息入站到工具执行的过程是一套**多层级、多维度的决策管道**：

```
消息入站 → 渠道处理 → Agent 路由 → 配置解析 → 消息分发 → 工具组装 → 策略过滤 → AI 决策 → 工具执行
```

相关代码：
- Agent 路由: `src/routing/resolve-route.ts`
- Session Key: `src/routing/session-key.ts`
- Agent 配置: `src/agents/agent-scope.ts`
- 消息分发: `src/auto-reply/dispatch.ts`, `src/auto-reply/reply/dispatch-from-config.ts`
- 回复生成: `src/auto-reply/reply/get-reply.ts`
- 工具组装: `src/agents/pi-tools.ts`
- 策略管道: `src/agents/tool-policy-pipeline.ts`
- 中间件: `src/agents/pi-tools.before-tool-call.ts`

---

## 2. 消息入站 — 渠道处理器

每个消息渠道有自己的 monitor（监控器），负责：
- 接收原始消息
- 构建统一的 `MsgContext` 上下文对象
- 提取渠道特定信息（`channel`、`accountId`、`peer`、`guildId`、`teamId`、`memberRoleIds` 等）

渠道代码位置：
- Telegram: `src/telegram/monitor.ts`
- Discord: `src/discord/monitor/message-handler.process.ts`
- Slack: `src/slack/monitor/message-handler/dispatch.ts`
- Signal: `src/signal/monitor/event-handler.ts`
- iMessage: `src/imessage/monitor/monitor-provider.ts`
- Web (WhatsApp): `src/web/auto-reply/monitor/on-message.ts`

---

## 3. Agent 路由 — `resolveAgentRoute()`

**核心代码**: `src/routing/resolve-route.ts:291`

### 3.1 八层优先级匹配

路由函数接收消息上下文，通过 8 层优先级匹配确定该消息由哪个 Agent 处理：

| 优先级 | 匹配方式 | 说明 |
|---|---|---|
| 1 | `binding.peer` | 精确匹配对话对象（如特定用户 ID、群 ID） |
| 2 | `binding.peer.parent` | 线程继承：匹配父会话的对话对象 |
| 3 | `binding.guild+roles` | Guild + 角色组合匹配（Discord 服务器 + 用户角色） |
| 4 | `binding.guild` | 仅 Guild 匹配 |
| 5 | `binding.team` | Team 匹配（如 Slack workspace） |
| 6 | `binding.account` | 账号匹配（特定的 bot 账号，accountId 非通配） |
| 7 | `binding.channel` | 渠道通配匹配（accountId 为 `"*"`） |
| 8 | `default` | 使用默认 Agent |

匹配到即停止，不继续向下匹配。

### 3.2 Binding 配置

在 `openclaw.json` 中配置 bindings 数组：

```json
{
  "bindings": [
    {
      "agentId": "customer-support",
      "match": {
        "channel": "telegram",
        "accountId": "bot123",
        "peer": { "kind": "group", "id": "support-group-456" }
      }
    },
    {
      "agentId": "dev-assistant",
      "match": {
        "channel": "discord",
        "guildId": "789",
        "roles": ["developer"]
      }
    }
  ]
}
```

相关代码: `src/routing/bindings.ts`（`listBindings()`）

### 3.3 Binding 匹配过程

```typescript
// src/routing/resolve-route.ts:370-440
const tiers = [
  { matchedBy: "binding.peer",        enabled: Boolean(peer),      predicate: peer匹配 },
  { matchedBy: "binding.peer.parent", enabled: Boolean(parentPeer), predicate: 父peer匹配 },
  { matchedBy: "binding.guild+roles", enabled: guildId && roles,    predicate: guild+roles匹配 },
  { matchedBy: "binding.guild",       enabled: Boolean(guildId),    predicate: 仅guild匹配 },
  { matchedBy: "binding.team",        enabled: Boolean(teamId),     predicate: team匹配 },
  { matchedBy: "binding.account",     enabled: true,               predicate: 非通配account },
  { matchedBy: "binding.channel",     enabled: true,               predicate: 通配account("*") },
];

for (const tier of tiers) {
  if (!tier.enabled) continue;
  const matched = bindings.find(candidate =>
    tier.predicate(candidate) && matchesBindingScope(candidate.match, scope)
  );
  if (matched) return choose(matched.binding.agentId, tier.matchedBy);
}
return choose(resolveDefaultAgentId(cfg), "default");
```

### 3.4 Session Key — 会话标识

路由确定 Agent 后，生成 Session Key 用于标识唯一会话：

```
格式: agent:<agentId>:<channel>:<scope>:<peerId>
示例: agent:main:telegram:direct:12345
      agent:support:discord:group:general-chat
```

**DM Scope 策略**（`session.dmScope`）：

| 策略 | Session Key 格式 | 效果 |
|---|---|---|
| `main` | `agent:<agentId>:main` | 所有 DM 共享一个会话 |
| `per-peer` | `agent:<agentId>:direct:<peerId>` | 按用户隔离 |
| `per-channel-peer` | `agent:<agentId>:<channel>:direct:<peerId>` | 按渠道+用户隔离 |
| `per-account-channel-peer` | `agent:<agentId>:<channel>:<accountId>:direct:<peerId>` | 完全隔离 |

**Identity Links**: 跨渠道身份关联，让同一用户在不同渠道共享会话。通过 `session.identityLinks` 配置。

相关代码: `src/routing/session-key.ts`（`buildAgentPeerSessionKey()`，115 行）

---

## 4. Agent 配置解析

**核心代码**: `src/agents/agent-scope.ts:117`

### 4.1 resolveAgentConfig()

确定 `agentId` 后，从 `openclaw.json` 的 `agents.list` 中查找对应配置：

```typescript
type ResolvedAgentConfig = {
  name?: string;           // Agent 显示名
  workspace?: string;      // 工作区路径
  agentDir?: string;       // Agent 内部状态目录
  model?: {                // AI 模型配置
    primary: string;       //   主模型
    fallbacks: string[];   //   回退模型链
  };
  skills?: string[];       // 技能过滤列表
  memorySearch?: ...;      // 记忆搜索配置
  identity?: ...;          // 人格/身份配置
  groupChat?: ...;         // 群聊行为
  subagents?: ...;         // 子 Agent 配置
  sandbox?: ...;           // 沙箱配置
  tools?: ...;             // 工具策略
};
```

### 4.2 默认 Agent 选择逻辑

```
resolveDefaultAgentId(cfg):
  1. 配置中标记 default: true 的 Agent → 优先（多个取第一个并警告）
  2. agents.list 数组第一个 Agent
  3. 列表为空 → 使用默认 ID "main"
```

相关代码: `src/agents/agent-scope.ts:71`（`resolveDefaultAgentId()`）

---

## 5. 消息分发管道

```
Channel Monitor 接收消息
  │
  ▼
dispatchInboundMessage()               ← src/auto-reply/dispatch.ts:35
  │
  ├─ finalizeInboundContext()           ← 统一消息上下文格式
  │
  ▼
dispatchReplyFromConfig()              ← src/auto-reply/reply/dispatch-from-config.ts:94
  │
  ├─ 去重检查 (shouldSkipDuplicateInbound)
  ├─ 解析 Session Store (resolveSessionStoreEntry)
  ├─ 触发插件钩子 (hookRunner.runMessageReceived)
  ├─ 触发内部钩子 (triggerInternalHook)
  ├─ 跨渠道路由判断 (shouldRouteToOriginating)
  ├─ 快速中止检查 (tryFastAbortFromMessage)
  │
  ▼
getReplyFromConfig()                   ← src/auto-reply/reply/get-reply.ts:55
  │
  ├─ resolveSessionAgentId()  → 从 SessionKey 解析 AgentId
  ├─ resolveDefaultModel()    → 解析模型（主模型 + 回退链）
  ├─ ensureAgentWorkspace()   → 确保工作区目录存在
  ├─ resolveReplyDirectives() → 解析指令（模型切换、思考模式等）
  ├─ handleInlineActions()    → 处理内联动作（命令）
  ├─ applyMediaUnderstanding() → 媒体理解（图片、音频转文本）
  │
  ▼
runEmbeddedAttempt()                   ← src/agents/pi-embedded-runner/run/attempt.ts
  │
  ├─ resolveSandboxContext()  → 是否启用 Docker 沙箱
  ├─ createOpenClawCodingTools()  → 组装工具集（见下节）
  ├─ createAgentSession()     → 创建 AI 会话
  │
  ▼
session.prompt()                       ← 将消息发送给 AI 模型，模型决定调用哪些工具
```

---

## 6. 工具组装 — `createOpenClawCodingTools()`

**核心代码**: `src/agents/pi-tools.ts:182`

这是工具选择的核心函数。它根据当前上下文动态组装工具集，整个过程包含四个阶段。

### 6.1 第一阶段：基础工具决策

根据沙箱模式决定使用哪种工具实现：

```
工具决策树:
├─ read     → 沙箱模式? → createSandboxedReadTool (Docker exec cat)
│             → 非沙箱?  → createHostWorkspaceReadTool (node:fs)
│             → workspaceOnly? → 额外包裹路径守卫
│
├─ write    → 沙箱模式? → createSandboxedWriteTool
│             → 非沙箱?  → createHostWorkspaceWriteTool
│             → workspace只读? → 不创建（沙箱模式）
│
├─ edit     → 沙箱模式? → createSandboxedEditTool
│             → 非沙箱?  → createHostWorkspaceEditTool
│
├─ exec     → 沙箱模式? → 在 Docker 容器内执行
│             → 非沙箱?  → 在宿主机执行
│             → 配置: host, security, scopeKey, timeout 等
│
├─ apply_patch → 仅在以下条件全满足时启用:
│               ├─ applyPatch.enabled = true
│               ├─ modelProvider 是 OpenAI
│               └─ modelId 在允许列表中
│
├─ process  → 进程管理（查看/终止后台进程）
├─ message  → 消息发送工具（如果未禁用）
├─ memory_search / memory_get → 记忆搜索/读取（只读）
├─ image    → 图片生成
└─ 渠道 Agent 工具 → listChannelAgentTools() 动态收集
```

### 6.2 第二阶段：工具策略过滤管道

**核心代码**: `src/agents/tool-policy-pipeline.ts`

组装完所有工具后，通过 **9 层策略过滤** 逐步缩小可用工具集：

```
全部工具
  │
  ├─ Step 1: tools.profile              ← 配置文件预设（如 "minimal", "full"）
  ├─ Step 2: tools.byProvider.profile   ← 按 AI 提供商的预设
  ├─ Step 3: tools.allow                ← 全局允许/拒绝列表
  ├─ Step 4: tools.byProvider.allow     ← 按提供商的允许列表
  ├─ Step 5: agents.X.tools.allow       ← 当前 Agent 专属的工具策略
  ├─ Step 6: agents.X.tools.byProvider.allow ← Agent + 提供商组合策略
  ├─ Step 7: group tools.allow          ← 群组级别工具策略
  ├─ Step 8: sandbox tools.allow        ← 沙箱模式额外限制
  ├─ Step 9: subagent tools.allow       ← 子 Agent 自动限制（深度越大越严）
  │
  ▼
最终可用工具集
```

每一层都可以配置 `allow`（白名单）和 `deny`（黑名单），采用**最严格者胜出**的叠加策略。

附加过滤:
- `applyMessageProviderToolPolicy()` — 按消息渠道过滤工具
- `applyOwnerOnlyToolPolicy()` — owner-only 工具仅 owner 可用

### 6.3 第三阶段：工具归一化

```typescript
// src/agents/pi-tools.ts:526-528
const normalized = subagentFiltered.map(tool =>
  normalizeToolParameters(tool, { modelProvider: options?.modelProvider })
);
```

针对不同 AI 提供商的 JSON Schema 兼容性问题做归一化（如 Gemini 需要去掉某些约束关键字）。

### 6.4 第四阶段：中间件包装（洋葱模型）

```
工具调用
  │
  ├─ AbortSignal 中间件    → 支持取消正在执行的工具
  │    └─ wrapToolWithAbortSignal()
  │
  ├─ beforeToolCall Hook   → 循环检测 + 插件钩子
  │    └─ wrapToolWithBeforeToolCallHook()
  │    └─ 循环检测: 短时间内连续调用同一工具相同参数超过阈值 → 自动阻断
  │
  └─ 实际 tool.execute()   → 执行工具逻辑
```

相关代码: `src/agents/pi-tools.before-tool-call.ts:175`（`wrapToolWithBeforeToolCallHook()`）

---

## 7. AI 模型如何选择工具

工具注册到 AI 会话后，**具体使用哪个工具由 AI 模型决定**（不是 OpenClaw 代码决定的）。OpenClaw 的角色是：

| 职责 | OpenClaw 框架 | AI 模型 |
|---|---|---|
| 路由到正确的 Agent | ✅ 框架决定 | |
| 组装可用工具集 | ✅ 框架决定 | |
| 施加安全约束 | ✅ 框架决定 | |
| 选择调用哪个工具 | | ✅ 模型决定 |
| 传递什么参数 | | ✅ 模型决定 |
| 后验安全检查 | ✅ 框架决定 | |

模型选择工具的依据：
1. **工具声明**: 每个工具的 `name`、`description`、`parameters`（JSON Schema）
2. **系统提示词引导**: 如 `buildMemorySection()` 指示模型在回答先前工作相关问题前必须先调用 `memory_search`
3. **对话上下文**: 用户的具体请求和当前对话历史

---

## 8. 子 Agent 生成 (Subagent Spawning)

当主 Agent 需要并行处理复杂任务时，可以生成子 Agent：

```
主 Agent (depth=0)
  │
  ├─ 调用 subagent 工具
  │
  ▼
子 Agent (depth=1)
  ├─ Session Key: agent:<agentId>:subagent:<parentKey>
  ├─ 自动工具限制: resolveSubagentToolPolicy()
  │   └─ 深度越大，可用工具越少
  ├─ 最大深度限制: 默认 3 层
  ├─ 最小系统提示模式: 减少 token 消耗
  │
  ├─ 可再生成子 Agent (depth=2)
  │   └─ 工具进一步受限
  │
  └─ 返回结果给父 Agent
```

深度通过 Session Key 中的 `subagent` 前缀计算：

```typescript
// src/sessions/session-key-utils.ts
export function getSubagentDepth(sessionKey: string): number {
  // 计算 Session Key 中 "subagent:" 出现的次数
}
```

相关代码: `src/auto-reply/reply/commands-subagents.ts`

---

## 9. 数据流全景

```
用户发消息（Telegram/Discord/Slack/...）
  │
  ▼ ① 渠道 Monitor 接收
  │
  ▼ ② resolveAgentRoute() — 8 层优先级匹配，找到 AgentId
  │
  ▼ ③ buildAgentSessionKey() — 生成唯一会话标识
  │
  ▼ ④ dispatchInboundMessage() — 消息入站分发
  │
  ▼ ⑤ getReplyFromConfig() — 解析 Agent 配置、模型、工作区
  │
  ▼ ⑥ createOpenClawCodingTools() — 组装工具集
  │     ├─ 基础工具（read/write/edit/exec/...）
  │     ├─ 沙箱 vs 宿主机实现选择
  │     ├─ 9 层策略过滤
  │     ├─ JSON Schema 归一化
  │     └─ 中间件包装（AbortSignal + beforeToolCall）
  │
  ▼ ⑦ session.prompt() — 发送给 AI 模型
  │
  ▼ ⑧ 模型返回工具调用决策
  │
  ▼ ⑨ beforeToolCall 中间件检查（循环检测 + 插件钩子）
  │
  ▼ ⑩ tool.execute() — 实际执行工具
  │
  ▼ ⑪ 工具结果返回模型 → 模型继续推理或生成最终回复
  │
  ▼ ⑫ 回复通过 ReplyDispatcher 发送回渠道
```

**设计哲学**: OpenClaw 的核心哲学是**"框架管路由和权限，AI 管决策"**——框架负责精确地把消息路由到正确的 Agent、组装恰当的工具集、施加安全约束，而具体选择哪个工具则交给 AI 模型基于对话上下文自主决定。
