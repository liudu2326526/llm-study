# 多智能体路由 - OpenClaw

原文链接: [https://docs.openclaw.ai/zh-CN/concepts/multi-agent#%E5%A4%9A%E4%B8%AA%E6%99%BA%E8%83%BD%E4%BD%93-%3D-%E5%A4%9A%E4%B8%AA%E4%BA%BA%E3%80%81%E5%A4%9A%E7%A7%8D%E4%BA%BA%E6%A0%BC](https://docs.openclaw.ai/zh-CN/concepts/multi-agent#%E5%A4%9A%E4%B8%AA%E6%99%BA%E8%83%BD%E4%BD%93-%3D-%E5%A4%9A%E4%B8%AA%E4%BA%BA%E3%80%81%E5%A4%9A%E7%A7%8D%E4%BA%BA%E6%A0%BC)

## 多智能体路由

**目标**：多个隔离的智能体（独立的工作区 + agentDir + 会话），加上多个渠道账户（例如两个 WhatsApp）在一个运行的 Gateway 网关中。入站消息通过绑定路由到智能体。

### 什么是”一个智能体”？

一个智能体是一个完全独立作用域的大脑，拥有自己的：
- **工作区**（文件、AGENTS.md/SOUL.md/USER.md、本地笔记、人设规则）。
- **状态目录（agentDir）** 用于认证配置文件、模型注册表和每智能体配置。
- **会话存储**（聊天历史 + 路由状态）位于 `~/.openclaw/agents/<agentId>/sessions` 下。

> **注意**：认证配置文件是每智能体独立的。切勿在智能体之间重用 `agentDir`（这会导致认证/会话冲突）。

### 路径（快速映射）

- **配置**：`~/.openclaw/openclaw.json`（或 `OPENCLAW_CONFIG_PATH`）
- **状态目录**：`~/.openclaw`（或 `OPENCLAW_STATE_DIR`）
- **工作区**：`~/.openclaw/workspace`（或 `~/.openclaw/workspace-<agentId>`）
- **智能体目录**：`~/.openclaw/agents/<agentId>/agent`
- **会话**：`~/.openclaw/agents/<agentId>/sessions`

### 单智能体模式（默认）

如果你什么都不做，OpenClaw 运行单个智能体：
- `agentId` 默认为 `main`。
- 会话键为 `agent:main:<mainKey>`。
- 工作区默认为 `~/.openclaw/workspace`。

### 智能体助手

使用智能体向导添加新的隔离智能体：
```bash
openclaw agents add work
```
然后添加 bindings 来路由入站消息。

### 多个智能体 = 多个人、多种人格

使用多个智能体，每个 `agentId` 成为一个完全隔离的人格：
- 不同的电话号码/账户（每渠道 `accountId`）。
- 不同的人格（每智能体工作区文件如 `AGENTS.md` 和 `SOUL.md`）。
- 独立的认证 + 会话。

### 路由规则（消息如何选择智能体）

绑定是确定性的，最具体的优先：
1. `peer` 匹配（精确私信/群组/频道 id）
2. `guildId`（Discord）
3. `teamId`（Slack）
4. 渠道的 `accountId` 匹配
5. 渠道级匹配（`accountId: "*"`）
6. 回退到默认智能体

### 核心概念

- **agentId**：一个”大脑”（工作区、每智能体认证、每智能体会话存储）。
- **accountId**：一个渠道账户实例（例如 WhatsApp 账户 "personal" vs "biz"）。
- **binding**：通过 (channel, accountId, peer) 以及可选的 guild/team id 将入站消息路由到 `agentId`。
