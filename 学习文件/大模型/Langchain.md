# 大模型微调实践
## 目录
1. [LangChain 定义与框架](#一、langchain-定义与框架)
2. [LangChain 核心价值](#二、langchain-核心价值)
3. [LangChain 核心组件](#三、langchain-核心组件)
4. [基于本地大模型的 Agent 搭建](#四、基于本地大模型的-agent-搭建)

---

## 一、LangChain 定义与框架
### 1. LangChain 定义
LangChain 是基于语言模型的应用程序开发框架，核心功能包括：
- 上下文感知（Context-aware）：连接语言模型与上下文信息（提示指令、少样本示例、内容），并基于这些信息回复。
- 推理（Reason）：依赖语言模型根据上下文明确要采取的行动并回复。

### 2. LangChain 框架组成
- **LangChain Libraries**：Python 和 JavaScript 库，包含组件接口、集成方案，以及开箱即用的 Chain 和 Agent。
- **LangChain Templates**：可快速部署的参考架构集合，适用于各类任务。
- **LangServe**：将 LangChain 链部署为 REST API 的库。
- **LangSmith**：开发者平台，支持调试、测试、评估和监控 Chain，与 LangChain 无缝集成。

## 二、LangChain 核心价值
### 1. 组件化
- 功能模块（数据预处理、文本生成、分类、命名实体识别等）被组件化，每个组件有特定接口和方法。
- 支持按需挑选组合，可独立使用，开发灵活、易扩展，加速部署。

### 2. 开箱即用的 Chains
- 预先设计组装的组件序列，针对特定高级任务优化。
- 无需从零搭建，可直接使用或定制调整，缩短开发周期。

## 三、LangChain 核心组件
### 1. Model I/O（模型输入输出）
核心元素包括三部分，实现“格式化-预测-解析”的完整流程：
- **Prompts**：模板化、动态选择和管理模型输入，支持预设模板与用户输入组合。
  - Prompt Template：参数化生成 Prompt，包含指令、少样本示例、问题等。
  - Example Selector：从示例集合中动态选择少样本（Few-Shot）示例。
- **Model**：通过通用接口调用语言模型，支持两种类型：
  - LLM：接收文本字符串输入，返回文本字符串（如 OpenAI 的 GPT-3）。
  - Chat Models：接收聊天消息列表，返回聊天消息（如 GPT-4、Claude）。
- **Output Parsers**：将模型输出转换为结构化信息（List/DateTime/Enum/Json 等），核心方法：
  - Get format instructions：返回输出格式化指令。
  - Parse：解析模型输出为指定结构。
  - Parse with prompt（可选）：结合提示信息修正输出。

### 2. Data Connection（数据连接）
为 LLM 应用提供用户特定数据的加载、转换、存储和查询能力：
- **核心功能**：
  - Document loaders：从多来源加载文档。
  - Document transformers：文档拆分、去重等操作。
  - Text embedding models：将非结构化文本转为向量。
  - Vector stores：存储和搜索嵌入数据。
  - Retrievers：查询嵌入数据。
- **关键工具：Faiss**
  - 全称 Facebook AI Similarity Search，开源高维向量相似性搜索库，支持十亿级向量检索。
  - 核心优势：高效搜索、多索引结构、多相似度度量（L2 距离、内积等）、多语言接口、易集成、高可定制。
  - 使用流程：准备数据集 → 创建索引结构 → 加载向量数据 → 训练索引 → 保存索引。

### 3. Callback（回调）
- 提供回调系统，可 Hook 到 LLM 应用的各个阶段，支持日志记录、监控、流式处理等。
- 通过 Callbacks 参数订阅事件，关键方法包括 on_chat_model_start、on_chain_end、on_tool_error 等。

### 4. Memory（记忆）
- 解决 Chains 和 Agent 的无状态问题，支持在交互中保持状态（适用于聊天机器人等场景）。
- 核心形式：
  - 模块化辅助工具：管理和操作先前的聊天消息。
  - 便捷集成方案：将记忆工具纳入 Chains。
- 关键组件：
  - ChatMessageHistory：保存人类/AI 消息的核心类。
  - ConversationBufferMemory：封装 ChatMessageHistory，提取消息变量。
  - 支持 MongoDB/Cassandra/Redis 等数据库存储历史信息。

### 5. Chains（链）
- 复杂应用需将 LLM 与其他组件链式调用，Chain 是组件调用序列（可嵌套其他 Chains）。
- 核心类型：
  - LLMChain：由 PromptTemplate、模型（LLM/ChatModel）、OutputParser（可选）组成。
  - SequentialChain：多个链组合构建的复杂链。
- 常见应用链：AnalyzeDocumentChain、Math chain、Graph DB QA chain、Summarization checker chain 等。

### 6. Agent（智能体）
- 模型封装器，通过用户输入理解意图，返回“action”和“action input”，调用工具满足需求。
- 核心特性：
  - 访问工具集，可动态选择工具、多工具联动（前一工具输出作为后一工具输入）。
  - 两种类型：Action agent（逐步骤决策，适用于小任务）、Plan-and-execute agent（预定义动作序列，适用于复杂任务）。
  - 相关概念：Tools（Agent 可执行的动作）、Toolkits（工具集合，适配特定用例）。
- 工作流程：
  1. 接收用户输入。
  2. 决策是否使用工具及工具输入。
  3. 调用工具并记录输出（观测）。
  4. 基于历史信息决定下一步操作。
  5. 重复步骤 3-4，直至直接回复用户。
- 核心组件（Agent Executor 封装）：
  - Prompt Template：构造发给语言模型的提示。
  - LM：接收提示并决策下一步动作。
  - Output Parser：解析模型输出为动作或最终答案。
- 示例代码片段：
  ```python
  tools = load_tools(["serpapi", "lm-math"], llm=llm)
  # 提示格式与问题示例（北京近期最高温及平方计算）
  ```

## 四、基于本地大模型的 Agent 搭建
### 1. 核心架构
- 工具：Qwen-14B Tool、Qwen2-7B-Tool
- 关键组件：FastChat、vLLM、LangChain（Server/Local）

### 2. FastChat
- 开放平台，用于训练、服务和评估聊天机器人。
- 架构组成：Server（接收请求分发）、Controller（管理 Worker 信息与健康状态）、Worker（模型计算）。
- 核心功能：分布式多模型部署、WebUI、OpenAI API 兼容（可复用 API 访问本地模型）。

### 3. vLLM
- 伯克利大学 LMSYS 开源的 LLM 高速推理框架，核心优化技术：
  - **PagedAttention**：借鉴虚拟内存分页技术，将 KV Cache 划分为块，减少显存碎片，显存利用率接近最优（浪费<4%）。
  - **Memory Sharing**：多序列共享物理 KV 块，结合 Copy-on-Write 机制，降低显存需求（最高降 55%），提升吞吐量 2.2 倍。
- 核心特性：连续批处理、张量并行推理、流式输出、HuggingFace 模型兼容、OpenAI 接口兼容。