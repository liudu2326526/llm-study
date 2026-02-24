# [AI Agent 记忆系统：从短期到长期的技术架构与实践](https://developer.aliyun.com/article/1710635)

## 简介
本文系统阐述 AI Agent 记忆系统的核心技术：短期记忆（会话级上下文管理）与长期记忆（跨会话知识沉淀）。涵盖上下文工程策略（压缩、卸载、隔离）、Record/Retrieve 架构、主流框架（ADK/LangChain/AgentScope）实现差异，及 Mem0 等开源方案集成，并探讨 MaaS、多模态记忆等前沿趋势。

## 前言
随着 AI Agent 应用的快速发展，智能体需要处理越来越复杂的任务和更长的对话历史。然而，LLM 的上下文窗口限制、不断增长的 token 成本，以及如何让 AI“记住”用户偏好和历史交互，都成为了构建实用 AI Agent 系统面临的核心挑战。

记忆系统（Memory System）正是为了解决这些问题而诞生的关键技术。记忆系统使 AI Agent 能够像人类一样，在单次对话中保持上下文连贯性（短期记忆），同时能够跨会话记住用户偏好、历史交互和领域知识（长期记忆）。这不仅提升了用户体验的连续性和个性化程度，也为构建更智能、更实用的 AI 应用奠定了基础。

---

## 一、Memory 基础概念

### 1.1 记忆的定义与分类
对于 AI Agent 而言，记忆至关重要，因为它使它们能够记住之前的互动、从反馈中学习，并适应用户的偏好。对“记忆”的定义有两个层面：
- **会话级记忆**：用户和智能体 Agent 在一个会话中的多轮交互（user-query & response）。
- **跨会话记忆**：从用户和智能体 Agent 的多个会话中抽取的通用信息，可以跨会话辅助 Agent 推理。

### 1.2 各 Agent 框架的定义差异
各个 Agent 框架对记忆的概念命名各有不同，但共同的是都遵循上述两个不同层面的划分：会话级和跨会话级。

| 框架 | 说明 |
| :--- | :--- |
| **Google ADK** | Session 表示单次持续交互；Memory 是长期知识库，可包含来自多次对话的信息。 |
| **LangChain** | Short-term memory 用于单线程或对话中记住之前的交互；Long-term memory 不属于基础核心组件，而是高阶的“个人知识库”外挂。 |
| **AgentScope** | 虽然官方文档强调需求驱动，但 API 层面仍然是两个组件（memory 和 long_term_memory），功能层面有明确区分。 |

习惯上，可以将会话级别的历史消息称为短期记忆，把可以跨会话共享的信息称为长期记忆。长期记忆的信息从短期记忆中抽取提炼而来，根据短期记忆中的信息实时地更新迭代，而其信息又会参与到短期记忆中辅助模型进行个性化推理。

---

## 二、Agent 框架集成记忆系统的架构

### 2.1 Agent 框架集成记忆的通用模式
各 Agent 框架集成记忆系统通常遵循以下通用模式：
1. **Step1：推理前加载** - 根据当前 user-query 从长期记忆中加载相关信息。
2. **Step2：上下文注入** - 从长期记忆中检索的信息加入当前短期记忆中辅助模型推理。
3. **Step3：记忆更新** - 短期记忆在推理完成后加入到长期记忆中。
4. **Step4：信息处理** - 长期记忆模块中结合 LLM+向量化模型进行信息提取和检索。

### 2.2 短期记忆（Session 会话）
短期记忆存储会话中产生的各类消息，包括用户输入、模型回复、工具调用及其结果等。这些消息直接参与模型推理，实时更新，并受模型的 maxToken 限制。

**核心特点**：
- 存储会话中的所有交互消息（用户输入、模型回复、工具调用等）。
- 直接参与模型推理，作为 LLM 的输入上下文。
- 实时更新，每次交互都会新增消息。
- 受模型 maxToken 限制，需要上下文工程策略进行优化。

### 2.3 长期记忆（跨会话）
长期记忆与短期记忆形成双向交互：一方面，长期记忆从短期记忆中提取“事实”、“偏好”、“经验”等有效信息进行存储（Record）；另一方面，长期记忆中的信息会被检索并注入到短期记忆中，辅助模型进行个性化推理（Retrieve）。

**与短期记忆的交互**：
- **Record（写入）**：从短期记忆的会话消息中提取有效信息，通过 LLM 进行语义理解和抽取，存储到长期记忆中。
- **Retrieve（检索）**：根据当前用户查询，从长期记忆中检索相关信息，注入到短期记忆中作为上下文，辅助模型推理。

**信息组织维度**：
- **用户维度（个人记忆）**：面向用户维度组织的实时更新的个人知识库。
- **业务领域维度**：沉淀的经验（包括领域经验和工具使用经验）。

---

## 三、短期记忆的上下文工程策略

### 3.1 核心策略
针对短期记忆的上下文处理，主要有以下几种策略：

1. **上下文缩减（Context Reduction）**
   - 保留预览内容：对于大块内容，只保留前 N 个字符或关键片段。
   - 总结摘要：使用 LLM 对整段内容进行总结摘要，保留关键信息。

2. **上下文卸载（Context Offloading）**
   - 当内容被缩减后，原始完整内容被卸载到外部存储（文件系统、数据库等），消息中只保留引用（路径、UUID）。

3. **上下文隔离（Context Isolation）**
   - 通过多智能体架构，将上下文拆分到不同的子智能体中。

### 3.2 各框架的实现方式

#### Google ADK
通过 `EventsCompactionConfig` 设置上下文处理策略。
```python
from google.adk.apps.app import App, EventsCompactionConfig
app = App(
    name='my-agent',
    root_agent=root_agent,
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=3,  # 每3次新调用触发压缩
        overlap_size=1          # 包含前一个窗口的最后一次调用
    ),
)
```

#### LangChain
通过 `SummarizationMiddleware` 设置上下文处理参数。
```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            max_tokens_before_summary=4000,  # 4000 tokens时触发摘要
            messages_to_keep=20,  # 摘要后保留最后20条消息
        ),
    ],
)
```

#### AgentScope
通过 `AutoContextMemory` 提供智能化的上下文工程解决方案，支持 6 种渐进式压缩策略。
```java
AutoContextMemory memory = new AutoContextMemory(
    AutoContextConfig.builder()
        .msgThreshold(100)
        .maxToken(128 * 1024)
        .tokenRatio(0.75)
        .build(),
    model
);
ReActAgent agent = ReActAgent.builder()
    .name("Assistant")
    .model(model)
    .memory(memory)
    .build();
```

---

## 四、长期记忆技术架构及 Agent 框架集成

### 4.1 核心组件
1. **LLM 大模型**：提取短期记忆中的有效信息。
2. **Embedder 向量化**：将文本转换为语义向量。
3. **VectorStore 向量数据库**：持久化存储记忆向量和元数据。
4. **GraphStore 图数据库**：存储实体-关系知识图谱。
5. **Reranker（重排序器）**：对初步检索结果进行重新排序。
6. **SQLite**：记录所有记忆操作的审计日志。

### 4.2 Record & Retrieve 流程
- **Record（记录）**：LLM 事实提取 → 信息向量化 → 向量存储 →（复杂关系存储）→ SQLite 操作日志。
- **Retrieve（检索）**：User query 向量化 → 向量数据库语义检索 → 图数据库关系补充 →（Reranker-LLM）→ 结果返回。

### 4.3 长期记忆与 RAG 的区别
虽然技术架构相似（向量化存储、相似性检索、注入上下文），但长期记忆更侧重于**动态更新的用户偏好和经验沉淀**，而 RAG 通常侧重于**静态知识库的检索**。

### 4.4 关键问题与挑战
1. **准确性**：依赖记忆管理机制（巩固、更新、遗忘）和检索相关度。
2. **安全和隐私**：防止数据中毒，保障用户隐私。
3. **多模态记忆**：支持文本、视觉、语音的统一记忆空间。

### 4.5 Agent 框架集成示例

#### 集成 Mem0
```java
// 初始化Mem0长期记忆
Mem0LongTermMemory mem0Memory = new Mem0LongTermMemory(
    Mem0Config.builder()
        .apiKey("your-mem0-api-key")
        .build()
);
// 创建Agent并集成长期记忆
ReActAgent agent = ReActAgent.builder()
    .name("Assistant")
    .model(model)
    .memory(memory)  // 短期记忆
    .longTermMemory(mem0Memory)  // 长期记忆
    .build();
```

---

## 五、行业趋势与产品对比

### 5.1 AI 记忆系统发展趋势
- **记忆即服务（MaaS）**：记忆系统将成为 AI 应用的基础设施。
- **精细化记忆管理**：借鉴人脑机制，构建分层动态的记忆架构。
- **多模态记忆系统**：向跨模态关联与毫秒级响应演进。
- **参数化记忆**：在模型层集成记忆单元（如 Memory Adapter）。

### 5.2 相关开源产品对比
从实际情况看，**Mem0** 仍然是长期记忆产品的领头地位，占据开源社区活跃度的领先位置。

---

## 结语
记忆系统作为 AI Agent 的核心基础设施，其发展直接影响着智能体的能力和用户体验。虽然目前的上下文工程策略已能解决大部分通用场景，但对于特定行业或场景（如医疗、法律等），仍需更深度的记忆建模与管理。
