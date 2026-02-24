# AI Agent 核心架构与工程落地学习指南

本指南基于当前市场对 AI Agent 的高阶需求（任务编排、MCP、多智能体协同等）整理，旨在帮助开发者从“会调用 API”进阶到“构建生产级智能体系统”。

## 1. Agent 核心逻辑构建 (Core Logic Construction)

Agent 不仅仅是 LLM，而是 **LLM + Planning + Memory + Tools**。

### 1.1 任务规划与编排 (Planning & Orchestration)
核心目标：让 LLM 学会“三思而后行”，处理非线性复杂任务。
*   **思维链 (CoT) 与 ReAct**:
    *   **概念**: Reasoning + Acting。让模型在执行动作前先输出思考过程。
    *   **学习重点**: 如何通过 Prompt 引导模型进行自我修正 (Self-Correction)。
*   **工作流编排 (Workflow Orchestration)**:
    *   **LangGraph**: *（重点推荐）*
        *   学习基于图（Graph）的状态机编排。
        *   掌握 `State`（状态）、`Node`（节点/动作）和 `Edge`（条件跳转）的定义。
        *   解决传统 Chain 难以处理的**循环（Loop）**和**分支（Branching）**问题。
    *   **SOP (Standard Operating Procedure)**: 将业务流程（如销售SOP、客服SOP）硬编码为 Agent 的执行约束。

### 1.2 工具调用与 MCP (Tool Use & MCP)
核心目标：扩展 Agent 的边界，使其能操作真实世界。
*   **Function Calling**:
    *   深入理解 OpenAI/Anthropic 的 Tool Use 格式。
    *   实战：定义一个复杂的 JSON Schema，让 Agent 准确提取参数。
*   **MCP (Model Context Protocol)**: *（前沿热点）*
    *   **概念**: 统一 Agent 连接数据的标准协议，类似 USB 接口。
    *   **实战**:
        *   构建一个 **MCP Server**：连接本地 SQLite 数据库或文件系统。
        *   构建一个 **MCP Client**：让 Claude Desktop 或自定义 Agent 能够读取上述数据。
    *   **价值**: 解决 Agent 与数据源（Slack, Github, Local Files）连接的碎片化问题。
*   **Skill 封装**:
    *   将原子工具（如 `search_google`）组合成高阶能力（如 `research_competitor`）。

## 2. 记忆与上下文工程 (Memory & Context)

核心目标：解决“聊着聊着就忘了”和“上下文窗口爆炸”的问题。

### 2.1 长短期记忆架构
*   **短期记忆 (Short-term)**:
    *   **Sliding Window**: 滑动窗口机制，仅保留最近 N 轮对话。
    *   **Token Management**: 动态计算 Token 消耗，避免溢出。
*   **长期记忆 (Long-term)**:
    *   **向量存储 (Vector Store)**: 使用 Chroma/Milvus/Pinecone 存储历史交互。
    *   **记忆检索**: 在每一轮对话中，先根据用户 Query 去向量库检索相关的历史记忆，Inject 到 System Prompt 中。
    *   **实体记忆**: 提取对话中的实体（如用户姓名、偏好、提及的项目），存入 Graph DB 或结构化 SQL。

### 2.2 上下文工程
*   **信息压缩**: 使用 LLM 对过长的历史记录进行摘要（Summary），而非全量保留。
*   **注意力管理**: 识别当前任务的关键信息，过滤无关噪音，提升模型指令遵循能力。

## 3. 高级 RAG 系统 (Advanced RAG)

核心目标：让 Agent “外挂大脑”，回答准确且有据可查。

### 3.1 全链路优化
*   **文本处理 (Indexing)**:
    *   **Chunking**: 语义分块（Semantic Chunking）优于固定字符分块。
    *   **元数据提取**: 为文档块增加 `Title`, `Date`, `Author` 等元数据，用于后续过滤。
*   **检索与召回 (Retrieval)**:
    *   **混合检索 (Hybrid Search)**: 关键词检索 (BM25) + 向量检索 (Dense Retrieval)。
    *   **假设性文档嵌入 (HyDE)**: 先让 LLM 生成一个假设性答案，再用该答案去检索。
*   **重排序 (Reranking)**:
    *   使用 Cross-Encoder 模型（如 BGE-Reranker）对召回结果进行精排，大幅提升准确率。

## 4. 多智能体协同 (Multi-Agent Collaboration)

核心目标：三个臭皮匠，顶个诸葛亮。专人做专事。

### 4.1 协作模式
*   **Router / Supervisor 模式**:
    *   一个主 Agent（大脑）负责理解需求，分发任务给子 Agent（手脚）。
    *   例如：主 Agent 接收“写代码”请求 -> 分发给 `Coder Agent` -> 分发给 `Reviewer Agent` 审查。
*   **Team / Hierarchical 模式**:
    *   Agent 之间有层级关系，模拟公司组织架构（经理 -> 组长 -> 员工）。
*   **框架学习**:
    *   **AutoGen**: 微软推出的多 Agent 对话框架。
    *   **CrewAI**: 强调角色扮演（Role-Playing）和任务委派的框架。

## 5. 业务落地与生产级要求 (Production Ready)

### 5.1 评估与监控 (Eval & Monitor)
*   **Golden Dataset**: 建立“金标准”测试集（问题 + 标准答案），用于自动化回归测试。
*   **指标体系**:
    *   **幻觉率 (Hallucination)**: 答案是否包含原文未提及的信息。
    *   **准确率 (Accuracy)**: 意图识别是否正确，工具调用参数是否准确。
*   **Tracing**: 使用 LangSmith 或 Langfuse 追踪 Agent 的每一次思考、Token 消耗和延迟。

### 5.2 部署与交互
*   **API 化**: 将 Agent 封装为 RESTful API 或 WebSocket 服务。
*   **人机交互 (HCI)**: 在 Slack/Teams 集成 Agent，支持 Human-in-the-loop（人在回路），关键步骤需人工确认。

---

## 推荐学习路径

1.  **入门**: 跑通 LangChain Quickstart，理解 Prompt Template 和 Chain。
2.  **进阶**: 学习 **LangGraph**，尝试实现一个带有循环（如：代码写错了自动重写）的 Agent。
3.  **实战**: 实现一个 **RAG Agent**，能够读取本地 PDF 并回答问题。
4.  **高阶**: 研究 **MCP 协议**，尝试写一个简单的 MCP Server。
5.  **架构**: 搭建多 Agent 系统（如：一个负责搜索，一个负责写文章），并使用 LangSmith 进行评测。
