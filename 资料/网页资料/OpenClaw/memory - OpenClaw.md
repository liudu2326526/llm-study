# 记忆 - OpenClaw

原文链接: [https://docs.openclaw.ai/zh-CN/concepts/memory#sqlite-%E5%90%91%E9%87%8F%E5%8A%A0%E9%80%9F%EF%BC%88sqlite-vec%EF%BC%89](https://docs.openclaw.ai/zh-CN/concepts/memory#sqlite-%E5%90%91%E9%87%8F%E5%8A%A0%E9%80%9F%EF%BC%88sqlite-vec%EF%BC%89)

OpenClaw 记忆是智能体工作空间中的纯 Markdown 文件。这些文件是唯一的事实来源；模型只”记住”写入磁盘的内容。

记忆搜索工具由活动的记忆插件提供（默认：`memory-core`）。使用 `plugins.slots.memory = "none"` 禁用记忆插件。

## 记忆文件（Markdown）

默认工作空间布局使用两个记忆层：

1. **`memory/YYYY-MM-DD.md`**
   - 每日日志（仅追加）。
   - 在会话开始时读取今天和昨天的内容。

2. **`MEMORY.md`（可选）**
   - 精心整理的长期记忆。
   - 仅在主要的私人会话中加载（绝不在群组上下文中加载）。

这些文件位于工作空间下（`agents.defaults.workspace`，默认 `~/.openclaw/workspace`）。

## 何时写入记忆

- 决策、偏好和持久性事实写入 `MEMORY.md`。
- 日常笔记和运行上下文写入 `memory/YYYY-MM-DD.md`。
- 如果有人说”记住这个”，就写下来（不要只保存在内存中）。
- 提醒模型存储记忆会有帮助；它会知道该怎么做。
- 如果你想让某些内容持久保存，请要求机器人将其写入记忆。

## 自动记忆刷新（压缩前触发）

当会话接近自动压缩时，OpenClaw 会触发一个静默的智能体回合，提醒模型在上下文被压缩之前写入持久记忆。默认提示明确说明模型可以回复，但通常 `NO_REPLY` 是正确的响应，因此用户永远不会看到这个回合。

这由 `agents.defaults.compaction.memoryFlush` 控制。

## 向量记忆搜索

OpenClaw 可以在 `MEMORY.md` 和 `memory/*.md`（以及你选择加入的任何额外目录或文件）上构建小型向量索引，以便语义查询可以找到相关笔记。

- **默认启用**。
- **监视记忆文件的更改**（去抖动）。
- **嵌入提供商**：OpenClaw 自动选择（OpenAI, Gemini 或本地）。
- **本地模式**：使用 `node-llama-cpp`。
- **SQLite 加速**：使用 `sqlite-vec`（如果可用）加速向量搜索。

## 额外记忆路径

你可以索引默认工作空间布局之外的 Markdown 文件：

```json
agents: {
  defaults: {
    memorySearch: {
      extraPaths: ["../team-docs", "/srv/shared-notes/overview.md"]
    }
  }
}
```

- 目录会递归扫描 `.md` 文件。
- 仅索引 Markdown 文件。

## Gemini 嵌入（原生）

将提供商设置为 `gemini` 以直接使用 Gemini 嵌入 API：

```json
agents: {
  defaults: {
    memorySearch: {
      provider: "gemini",
      model: "gemini-embedding-001",
      remote: {
        apiKey: "YOUR_GEMINI_API_KEY"
      }
    }
  }
}
```

## 自定义 OpenAI 兼容端点

可以使用 `remote` 配置与 OpenAI 提供商：

```json
agents: {
  defaults: {
    memorySearch: {
      provider: "openai",
      model: "text-embedding-3-small",
      remote: {
        baseUrl: "https://api.example.com/v1/",
        apiKey: "YOUR_OPENAI_COMPAT_API_KEY",
        headers: { "X-Custom-Header": "value" }
      }
    }
  }
}
```
