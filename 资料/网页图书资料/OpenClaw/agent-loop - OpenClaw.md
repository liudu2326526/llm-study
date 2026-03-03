# 智能体循环 - OpenClaw

原文链接: [https://docs.openclaw.ai/zh-CN/concepts/agent-loop#%E5%8E%8B%E7%BC%A9-+-%E9%87%8D%E8%AF%95](https://docs.openclaw.ai/zh-CN/concepts/agent-loop#%E5%8E%8B%E7%BC%A9-+-%E9%87%8D%E8%AF%95)

智能体循环是智能体的完整”真实”运行：接收 → 上下文组装 → 模型推理 → 工具执行 → 流式回复 → 持久化。这是将消息转化为操作和最终回复的权威路径，同时保持会话状态的一致性。

在 OpenClaw 中，循环是每个会话的单次序列化运行，在模型思考、调用工具和流式输出时发出生命周期和流事件。本文档解释了这个真实循环是如何端到端连接的。

## 入口点

- Gateway 网关 RPC：`agent` 和 `agent.wait`。
- CLI：`agent` 命令。

## 工作原理（高层次）

1. `agent` RPC 验证参数，解析会话（`sessionKey`/`sessionId`），持久化会话元数据，立即返回 `{ runId, acceptedAt }`。
2. `agentCommand` 运行智能体：
   - 解析模型 + 思考/详细模式默认值
   - 加载 Skills 快照
   - 调用 `runEmbeddedPiAgent`（`pi-agent-core` 运行时）
   - 如果嵌入式循环未发出生命周期结束/错误事件，则发出该事件

3. `runEmbeddedPiAgent`：
   - 通过每会话 + 全局队列序列化运行
   - 解析模型 + 认证配置文件并构建 `pi` 会话
   - 订阅 `pi` 事件并流式传输助手/工具增量
   - 强制执行超时 -> 超时则中止运行
   - 返回有效负载 + 使用元数据

4. `subscribeEmbeddedPiSession` 将 `pi-agent-core` 事件桥接到 OpenClaw `agent` 流：
   - 工具事件 => `stream: "tool"`
   - 助手增量 => `stream: "assistant"`
   - 生命周期事件 => `stream: "lifecycle"`（`phase: "start" | "end" | "error"`）

5. `agent.wait` 使用 `waitForAgentJob`：
   - 等待 `runId` 的生命周期结束/错误
   - 返回 `{ status: ok|error|timeout, startedAt, endedAt, error? }`

## 队列 + 并发

- 运行按会话键（会话通道）序列化，可选择通过全局通道。
- 这可以防止工具/会话竞争并保持会话历史的一致性。
- 消息渠道可以选择队列模式（`collect`/`steer`/`followup`）来馈送此通道系统。参见命令队列。

## 会话 + 工作区准备

- 解析并创建工作区；沙箱隔离运行可能会重定向到沙箱工作区根目录。
- 加载 Skills（或从快照中复用）并注入到环境和提示中。
- 解析引导/上下文文件并注入到系统提示报告中。
- 获取会话写锁；在流式传输之前打开并准备 `SessionManager`。

## 提示组装 + 系统提示

- 系统提示由 OpenClaw 的基础提示、Skills 提示、引导上下文和每次运行的覆盖构建。
- 强制执行模型特定的限制和压缩保留令牌。
- 参见系统提示了解模型看到的内容。

## 钩子点（可以拦截的位置）

OpenClaw 有两个钩子系统：
1. **内部钩子（Gateway 网关钩子）**：用于命令和生命周期事件的事件驱动脚本。
2. **插件钩子**：智能体/工具生命周期和 Gateway 网关管道中的扩展点。

### 内部钩子（Gateway 网关钩子）

- `agent:bootstrap`：在系统提示最终确定之前构建引导文件时运行。用于添加/删除引导上下文文件。
- 命令钩子：`/new`、`/reset`、`/stop` 和其他命令事件（参见钩子文档）。

### 插件钩子（智能体 + Gateway 网关生命周期）

这些在智能体循环或 Gateway 网关管道内运行：
- `before_agent_start`：在运行开始前注入上下文或覆盖系统提示。
- `agent_end`：在完成后检查最终消息列表和运行元数据。
- `before_compaction` / `after_compaction`：观察或注释压缩周期。
- `before_tool_call` / `after_tool_call`：拦截工具参数/结果。
- `tool_result_persist`：在工具结果写入会话记录之前同步转换它们。
- `message_received` / `message_sending` / `message_sent`：入站 + 出站消息钩子。
- `session_start` / `session_end`：会话生命周期边界。
- `gateway_start` / `gateway_stop`：Gateway 网关生命周期事件。

## 流式传输 + 部分回复

- 助手增量从 `pi-agent-core` 流式传输并作为 `assistant` 事件发出。
- 分块流式传输可以在 `text_end` 或 `message_end` 时发出部分回复。
- 推理流式传输可以作为单独的流或作为块回复发出。
- 参见流式传输了解分块和块回复行为。

## 工具执行 + 消息工具

- 工具开始/更新/结束事件在 `tool` 流上发出。
- 工具结果在记录/发出之前会对大小和图像有效负载进行清理。
- 消息工具发送会被跟踪以抑制重复的助手确认。

## 回复整形 + 抑制

最终有效负载由以下内容组装：
- 助手文本（和可选的推理）
- 内联工具摘要（当详细模式 + 允许时）
- 模型出错时的助手错误文本

- `NO_REPLY` 被视为静默令牌，从出站有效负载中过滤。
- 消息工具重复项从最终有效负载列表中移除。
- 如果没有剩余可渲染的有效负载且工具出错，则发出回退工具错误回复（除非消息工具已经发送了用户可见的回复）。

## 压缩 + 重试

- 自动压缩发出 `compaction` 流事件，可以触发重试。
- 重试时，内存缓冲区和工具摘要会重置以避免重复输出。
- 参见压缩了解压缩管道。
