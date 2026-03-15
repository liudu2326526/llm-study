# 阿里国际 Agent
## 清单

### 1. 模型训练与对齐

---

#### Q: 常见的后训练方法有哪些？它们的优缺点和区别是什么？

常见的后训练（Post-Training）方法主要包括：

| 方法 | 核心思路 | 优点 | 缺点 |
| :--- | :--- | :--- | :--- |
| **SFT（监督微调）** | 用"指令-回答"对直接微调模型 | 简单高效，效果直接 | 依赖高质量标注数据，容易过拟合 |
| **RLHF（基于人类反馈的强化学习）** | 训练奖励模型 + PPO 优化策略模型 | 能对齐人类偏好，处理主观任务 | 训练流程复杂，奖励模型容易被 hack |
| **DPO（直接偏好优化）** | 直接用偏好对数据优化策略模型，无需奖励模型 | 简化 RLHF 流程，训练更稳定 | 对数据质量敏感，优化上限可能低于 RLHF |
| **GRPO（Group Relative Policy Optimization）** | 组内相对排序优化，无需 Critic 模型 | 节省显存，适合推理任务 | 需要可验证的奖励信号 |
| **KTO（Kahneman-Tversky Optimization）** | 只需"好/坏"二元标签，不需要偏好对 | 数据收集更简单 | 理论上不如偏好对精确 |

**核心区别**：
- SFT 是"模仿学习"——告诉模型正确答案是什么
- RLHF/DPO 是"偏好学习"——告诉模型哪个更好
- SFT 通常作为第一阶段，RLHF/DPO 作为第二阶段进一步对齐

**典型流程**：Pretrain → SFT → RLHF/DPO（如 Qwen3 采用 SFT + GRPO 两阶段后训练）

---

#### Q: 介绍 SFT 的流程，以及如何构建高质量、多样化的数据集？

**SFT 流程**：

```
1. 数据准备 → 2. 数据预处理 → 3. 模型选择 → 4. 训练配置 → 5. 训练 → 6. 评估迭代
```

1. **数据准备**：收集"指令-输入-输出"三元组格式的数据
2. **数据预处理**：清洗、去重、格式化为 Chat Template（如 `[{"role": "user", ...}, {"role": "assistant", ...}]`）
3. **模型选择**：选择合适的预训练基座模型
4. **训练配置**：设定学习率（通常 1e-5 ~ 5e-5）、Epoch（1-3 轮）、Batch Size 等
5. **训练**：仅对 assistant 的输出计算 Loss（Masked Loss），忽略 user 部分
6. **评估迭代**：通过人工评估和自动指标检验效果，迭代数据和超参

**构建高质量数据集的关键**：

| 维度 | 方法 |
| :--- | :--- |
| **质量** | 人工审核 + LLM 打分过滤；移除低质量、有毒、错误的样本 |
| **多样性** | 覆盖多种任务类型（QA、摘要、创作、推理、代码）；覆盖不同领域和难度 |
| **数据合成** | 用强模型（如 GPT-4）生成种子数据，再人工校验（Self-Instruct / Evol-Instruct） |
| **去污染** | 移除与评测集重叠的数据，防止 Benchmark 泄漏 |
| **平衡** | 控制各类任务的比例，避免某类任务主导导致能力退化 |

实践建议：**数据质量 > 数据数量**。几千条高质量数据的效果往往优于几万条低质量数据。

---

#### Q: 在什么业务场景下，必须引入 RLHF 或 DPO 这种偏好对齐技术？

以下场景中，单纯 SFT 不够，需要偏好对齐：

1. **主观质量判断**：如聊天助手的回答风格、语气、详略程度——没有唯一"正确答案"，但有"好坏之分"
2. **安全与有害内容控制**：需要让模型学会拒绝危险请求，SFT 容易过度拒绝或拒绝不足，偏好对齐能更精细地调节边界
3. **减少幻觉**：通过偏好数据教模型"诚实说不知道"优于"编造看似合理的答案"
4. **多轮对话质量**：Agent 场景中的对话连贯性、任务完成度等难以用单一 Loss 衡量
5. **指令遵循的精细化**：当模型已基本能完成任务但输出格式、长度、风格不够理想时
6. **竞争性排序场景**：如搜索排序、推荐系统中，需要模型输出相对排序而非绝对答案

**核心判断标准**：当任务的"好"是一个相对概念（A 比 B 好），而非绝对正确答案时，就适合引入偏好对齐。

---

#### Q: KL 散度的数学意义是什么？在模型对齐（如 PPO/DPO）中起什么作用？

**数学定义**：

KL 散度（Kullback-Leibler Divergence）衡量两个概率分布 $P$ 和 $Q$ 之间的差异：

$$D_{KL}(P \| Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}$$

**数学性质**：
- $D_{KL} \geq 0$，当且仅当 $P = Q$ 时等于 0
- **不对称**：$D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$
- 直观理解：用 $Q$ 近似 $P$ 时，信息损失了多少（额外所需的信息量）

**在模型对齐中的作用**：

在 PPO/DPO 中，KL 散度用于 **约束策略模型不要偏离参考模型（SFT 模型）太远**：

$$\text{Objective} = \mathbb{E}[r(x, y)] - \beta \cdot D_{KL}(\pi_\theta \| \pi_{\text{ref}})$$

- $r(x, y)$：奖励信号（鼓励模型生成更好的回答）
- $\beta \cdot D_{KL}$：KL 惩罚项（约束模型不要跑偏）

**为什么需要 KL 约束？**

| 问题 | KL 约束的作用 |
| :--- | :--- |
| **奖励模型 Hacking** | 模型可能找到奖励模型的漏洞，生成高奖励但实际质量差的回答。KL 约束防止模型过度偏离 |
| **语言能力退化** | 过度优化偏好可能导致模型"忘记"预训练学到的语言能力 |
| **分布崩塌（Mode Collapse）** | 模型可能退化为只生成少数几种"安全"回答，KL 约束保持输出多样性 |

**在 DPO 中**：KL 约束被隐式地编码进了损失函数中，无需显式计算：

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

其中 $y_w$ 是偏好回答，$y_l$ 是较差回答。$\beta$ 控制 KL 约束的强度。

---

### 2. Agent 架构与工具

---

#### Q: 除了 ReAct，目前主流的 Agent 架构还有哪些？

| 架构 | 核心思想 | 特点 | 代表 |
| :--- | :--- | :--- | :--- |
| **ReAct** | Reasoning + Acting 交替进行 | 思考→行动→观察 循环 | LangChain Agent |
| **Plan-and-Execute** | 先制定完整计划，再逐步执行 | 分离规划与执行，适合复杂任务 | LangGraph Plan-and-Execute |
| **Reflexion** | 执行后自我反思，迭代改进 | 引入"反思"环节提升任务完成率 | Reflexion 论文 |
| **LATS（Language Agent Tree Search）** | 树搜索 + LLM 评估 | 探索多条路径，选择最优 | 复杂推理任务 |
| **Multi-Agent（多智能体）** | 多个专业 Agent 协作 | Router/Supervisor 分发，专人专事 | AutoGen、CrewAI |
| **Tool-augmented LLM** | 直接 Function Calling，无显式推理链 | 简单直接，适合确定性任务 | OpenAI Function Calling |
| **Cognitive Architecture** | 模拟人类认知（感知→思考→行动→记忆） | 包含完整的记忆、规划、学习模块 | CoALA 框架 |
| **Flow Engineering** | 用工作流图（DAG/状态机）编排 Agent 步骤 | 可控性强，适合生产环境 | LangGraph、Dify |

**选型建议**：
- 简单工具调用 → Function Calling
- 需要推理过程 → ReAct
- 复杂多步任务 → Plan-and-Execute
- 需要高可靠性 → Flow Engineering (LangGraph)
- 需要多角色协作 → Multi-Agent

---

#### Q: Agent 系统中的 LangGraph 是如何搭建的？其 Memory 组件的工作机制是怎样的？

**LangGraph 搭建核心三要素**：

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage
from typing import TypedDict

# 1. 定义状态（State）—— Agent 在执行过程中维护的全部信息
class AgentState(TypedDict):
    messages: list[BaseMessage]       # 对话历史
    current_step: str                 # 当前步骤
    tool_results: dict                # 工具调用结果

# 2. 定义节点（Node）—— 每个节点是一个处理函数
def call_llm(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def call_tool(state: AgentState) -> AgentState:
    # 执行工具调用
    ...

# 3. 定义边（Edge）—— 节点间的跳转逻辑
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("tool", call_tool)
graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", should_use_tool, {"yes": "tool", "no": END})
graph.add_edge("tool", "llm")  # 工具执行后回到 LLM

app = graph.compile(checkpointer=MemorySaver())
```

> 注：以上为教学示例，省略了 `llm` 与 `should_use_tool` 的具体实现。

**Memory 组件的工作机制**：

LangGraph 的 Memory 分为两层：

| 层次 | 机制 | 作用 |
| :--- | :--- | :--- |
| **短期记忆（State）** | Graph 的 State 对象在单次对话中持续传递 | 维护当前对话的上下文、中间结果 |
| **持久化记忆（Checkpointer）** | 通过 `MemorySaver` 或数据库 Checkpointer 存储状态快照 | 支持多轮对话、断点恢复 |

**Checkpointer 工作流程**：
1. 每个节点执行完毕后，Checkpointer 自动保存当前 State 的快照
2. 每个快照关联一个 `thread_id`（对话线程）和 `checkpoint_id`
3. 新一轮对话到来时，通过 `thread_id` 加载最新快照，恢复完整上下文
4. 支持回溯到任意历史节点（Time Travel），重新执行

```python
# 使用 Memory 的调用方式
config = {"configurable": {"thread_id": "user_123"}}
result = app.invoke({"messages": [HumanMessage("你好")]}, config)
# 后续调用同一 thread_id 会自动恢复上下文
result = app.invoke({"messages": [HumanMessage("继续上个话题")]}, config)
```

---

#### Q: 相比于 LangChain，LangGraph 在处理循环任务和状态管理上有哪些优势？

| 维度 | LangChain (Chain/Agent) | LangGraph |
| :--- | :--- | :--- |
| **执行模型** | 线性链（DAG），数据从前到后流动 | 有向图（支持环），节点可循环执行 |
| **循环支持** | 不原生支持循环，需要手动 hack | 原生支持环（Cycle），如"失败重试"、"迭代优化" |
| **状态管理** | 状态通过 Chain 参数传递，不够灵活 | 显式的 State 对象，所有节点共享，可自定义 |
| **条件分支** | 通过 Router Chain 实现，配置复杂 | `add_conditional_edges` 一行搞定 |
| **可观测性** | 依赖 Callback 机制 | 每步 State 可追踪、可回放 |
| **错误恢复** | 整个 Chain 从头重试 | 从失败节点的上一个快照恢复，断点续跑 |
| **人机交互** | 需要额外封装 | 原生 `interrupt_before`/`interrupt_after`，内置 Human-in-the-loop |

**LangGraph 解决的核心痛点**：

1. **循环任务**：Agent 调用工具 → 检查结果 → 不满意 → 重新调用 → 再检查... 这种"循环直到满意"的模式在 LangChain 中很难优雅实现
2. **复杂分支**：不同条件走不同路径，LangGraph 用条件边（conditional edges）自然表达
3. **状态一致性**：多步骤任务中状态的一致性维护，LangGraph 通过全局 State 保证

---

#### Q: LangGraph 的状态快照机制是如何实现任务回溯和持久化的？

**状态快照（Checkpoint）机制**：

```
Node A 执行 → 保存 Checkpoint 1 → Node B 执行 → 保存 Checkpoint 2 → ...
```

**核心实现**：

1. **快照时机**：每个节点执行完毕后自动触发
2. **快照内容**：完整的 State 对象（包含消息历史、中间变量、工具结果等）
3. **存储结构**：
   - `thread_id`：标识一个对话线程
   - `checkpoint_id`：标识该线程中的具体时间点
   - `parent_checkpoint_id`：指向上一个快照，形成链式结构

**任务回溯（Time Travel）**：

```python
# 获取历史快照列表
checkpoints = list(app.get_state_history(config))

# 回溯到特定快照
target_config = checkpoints[2].config
result = app.invoke(None, target_config)  # 从该点重新执行
```

**持久化方案**：

| Checkpointer | 存储介质 | 适用场景 |
| :--- | :--- | :--- |
| `MemorySaver` | 内存 | 开发调试 |
| `SqliteSaver` | SQLite | 单机部署 |
| `PostgresSaver` | PostgreSQL | 生产环境 |

**实际应用**：
- **断点续跑**：Agent 执行到一半崩溃，重启后从最后一个 Checkpoint 恢复
- **分支探索**：从某个决策点回溯，尝试不同的路径
- **审计追踪**：回放 Agent 的每一步决策过程，便于调试和合规审查

---

### 3. Transformer 架构深度理解

---

#### Q: 描述 Transformer Decoder 的完整解码流程。

**完整解码流程（以 Decoder-Only 为例）**：

```
输入 Token → Embedding + 位置编码 → [Decoder Block × N] → Linear → Softmax → 输出 Token
```

**逐步拆解**：

**1. Embedding + 位置编码**
- 输入 Token 经过 Token Embedding 映射为 $d_{\text{model}}$ 维向量
- 加上位置编码（RoPE/ALiBi）注入位置信息

**2. Decoder Block（重复 N 层）**

每层包含两个核心子层（Attention + FFN）：

```
输入 x
  → RMSNorm → Masked Self-Attention → + x（残差）   # 子层1
  → RMSNorm → Feed Forward (SwiGLU)  → + x（残差）   # 子层2
```

- **Masked Self-Attention**：
  - 生成 Q、K、V：$Q = xW^Q, K = xW^K, V = xW^V$
  - 计算注意力：$\text{Attention} = \text{softmax}(\frac{QK^T}{\sqrt{d_k}} + \text{Mask})V$
  - Mask 为上三角矩阵（$-\infty$），确保位置 $i$ 只能看到 $\leq i$ 的 Token
  - 若为 Encoder-Decoder 架构，还有 Cross-Attention 层（Q 来自 Decoder，K/V 来自 Encoder）

- **FFN（前馈网络）**：对每个位置独立做非线性变换

**3. 输出层**
- 最后一层的隐藏状态经 Linear 投影到词表大小维度，得到 Logits
- Softmax 将 Logits 转为概率分布
- 根据解码策略（Greedy/Sampling/Beam Search）选择下一个 Token

**4. 自回归循环**
- 将生成的 Token 追加到输入序列，重复上述过程
- 利用 KV Cache 缓存历史 Token 的 K/V，避免重复计算
- 直到生成 EOS Token 或达到最大长度

---

#### Q: Transformer 中 Attention 的本质是什么？请从数学角度解释。

**本质：Attention 是一种加权聚合机制——根据"相关性"动态地从所有位置收集信息。**

**数学角度**：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

逐步理解：

1. **$QK^T$（相似度矩阵）**：Q 和 K 的点积衡量两个向量的方向一致性。$QK^T$ 的第 $(i,j)$ 个元素表示位置 $i$ 对位置 $j$ 的"关注程度"

2. **$\frac{1}{\sqrt{d_k}}$（缩放）**：标准化点积的方差，防止值过大

3. **$\text{softmax}$（归一化）**：将相似度转为概率分布（注意力权重），使权重和为 1

4. **$\times V$（加权求和）**：按注意力权重对 Value 加权求和，得到聚合后的表示

**从核函数角度**：Attention 本质上是一个**非参数化的核平滑（Kernel Smoothing）**。$QK^T$ 充当核函数，衡量 Token 间的"距离"，然后用这个距离对 V 做加权平均。

**从数据库角度**：
- Q = 查询条件（"我想找什么"）
- K = 索引键（"我有什么"）
- V = 实际数据（"我能给你什么"）
- Attention = 软检索（soft lookup），所有记录按相关性加权返回

---

#### Q: 为什么在计算 Attention 时需要进行 Scaling？

**问题根源**：当 $d_k$（Key 的维度）较大时，$QK^T$ 的值会变得很大。

**数学解释**：

假设 Q 和 K 的元素服从独立的 $\mathcal{N}(0, 1)$ 分布，则：
- 点积 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$
- 每项 $q_i k_i$ 的期望为 0，方差为 1
- 求和后 $q \cdot k$ 的方差为 $d_k$
- 即 $\text{Var}(q \cdot k) = d_k$

当 $d_k = 64$ 时，点积的标准差为 $\sqrt{64} = 8$，值可能在 $[-20, 20]$ 范围波动。

**Softmax 在大值下的问题**：
- Softmax 输出趋近于 one-hot（某个位置接近 1，其余接近 0）
- 梯度趋近于 0（**梯度消失**），模型难以学习
- 注意力权重几乎全部集中在一个 Token 上，失去了"加权聚合"的意义

**Scaling 的效果**：

$$\frac{QK^T}{\sqrt{d_k}}$$

除以 $\sqrt{d_k}$ 后，点积的方差被归一化为 1，Softmax 输出不会过于极端，梯度保持健康。

---

#### Q: Self-Attention 和 Cross-Attention 在作用和输入来源上有什么区别？

| 维度 | Self-Attention | Cross-Attention |
| :--- | :--- | :--- |
| **Q/K/V 来源** | Q、K、V 全部来自**同一序列** | Q 来自 **Decoder**，K/V 来自 **Encoder** |
| **作用** | 序列内部的信息交互（每个 Token 关注其他 Token） | 跨序列的信息桥梁（Decoder 从 Encoder 获取源信息） |
| **出现位置** | Encoder 和 Decoder 都有 | 仅在 Encoder-Decoder 架构的 Decoder 中 |
| **典型应用** | 理解上下文语义、建模依赖关系 | 机器翻译中关注源语言、VQA 中关注图像特征 |
| **注意力矩阵形状** | $L \times L$（序列长度 × 序列长度） | $L_{\text{dec}} \times L_{\text{enc}}$（目标序列 × 源序列） |

**Self-Attention 示例**：翻译句子"我爱你"时，Encoder 中"我"需要关注"爱"和"你"来理解完整语义。

**Cross-Attention 示例**：Decoder 生成 "I" 时，通过 Cross-Attention 重点关注源语言中的"我"，建立翻译映射。

**在 Decoder-Only 架构中**：只有 Masked Self-Attention（因果自注意力），没有 Cross-Attention。

---

#### Q: 面对极长序列 Attention 的 $O(L^2)$ 复杂度问题，目前有哪些主流解决方案？

| 方案 | 原理 | 复杂度 | 代表 |
| :--- | :--- | :--- | :--- |
| **Sliding Window Attention** | 每个 Token 只关注前后固定窗口 $W$ 内的 Token | $O(L \times W)$ | Mistral、Longformer |
| **Global + Local** | 滑窗 + 少量 Token 拥有全局视野 | $O(L \times (W+G))$ | Longformer（[CLS] Token 全局） |
| **稀疏注意力（BigBird）** | 随机 + 滑窗 + 全局的组合 | $O(L)$ | BigBird |
| **FlashAttention** | IO 感知的分块计算，不存储完整注意力矩阵 | 仍 $O(L^2)$ 但极快 | 大模型训练标配 |
| **线性注意力（Linear Attention）** | 用核函数近似 Softmax，避免显式计算 $L \times L$ 矩阵 | $O(L)$ | Performer、Linear Transformer |
| **长序列替代架构（非标准 Softmax Attention）** | 用状态空间/递推机制建模长依赖 | 通常近似 $O(L)$ | Mamba、RWKV、RetNet |
| **Ring Attention** | 将长序列分片到多设备，环形传递 KV 块 | $O(L^2/N)$（N 为设备数） | 超长上下文训练 |
| **MLA（Multi-head Latent Attention）** | 低秩压缩 KV，减少每个 Token 的存储开销 | $O(L^2)$ 但显存大幅降低 | DeepSeek-V3 |
| **位置编码外推** | RoPE + NTK 插值 / YaRN，使模型处理超训练长度的序列 | 不改变复杂度 | LLaMA 扩展上下文 |

**工程实践中的组合方案**：
- 训练时用 FlashAttention（精确 + 快速）
- 推理时用 GQA/MLA 减少 KV Cache + Paged KV Cache 管理显存
- 超长文本用 Sliding Window + 少量全局 Token

---

#### Q: 在 Agent 多轮对话任务中，Attention 机制的局限性体现在哪些方面？

1. **上下文窗口限制**：多轮对话的累积 Token 容易超出模型的最大上下文长度，早期对话被截断丢失

2. **注意力稀释（Attention Dilution）**：随着上下文变长，每个 Token 分到的注意力权重减小，模型难以精准定位关键信息——尤其是"Lost in the Middle"问题（中间位置的信息容易被忽略）

3. **KV Cache 显存膨胀**：每轮对话都追加 Token，KV Cache 持续增长，显存压力巨大

4. **无法显式遗忘**：Attention 没有"遗忘门"机制（不像 LSTM），已进入上下文的无关信息也会消耗注意力资源

5. **跨轮次引用困难**：用户说"用你之前推荐的方案"，模型需要在大量历史 Token 中精确定位，Attention 可能无法准确建立这种远距离关联

6. **工具调用结果的噪音**：Agent 调用工具返回的大量中间结果（如长 JSON）会占据大量上下文空间，稀释有用信息

**缓解方案**：
- 历史消息摘要压缩（Summary Memory）
- 向量数据库存储长期记忆，按需检索注入
- Sliding Window + 关键信息锚定
- LangGraph 的显式 State 管理替代纯上下文传递

---

#### Q: 为什么模型在长上下文对话中容易出现"信息遗忘"？有哪些缓解机制？

**"遗忘"的根本原因**：

1. **Softmax 的归一化效应**：Attention 权重必须归一化为和为 1。上下文越长，每个早期 Token 被分到的权重越小，等效于"被遗忘"

2. **位置编码衰减**：RoPE 等位置编码使得距离远的 Token 的注意力分数天然更低，远距离信息更难被关注

3. **Lost in the Middle**：研究表明模型对上下文开头和结尾的信息利用率最高，中间部分容易被忽略（U 形注意力分布）

4. **信息竞争**：新信息不断进入上下文，与旧信息争夺有限的注意力资源

**缓解机制**：

| 机制 | 方法 | 效果 |
| :--- | :--- | :--- |
| **上下文压缩** | 用 LLM 对历史对话做摘要，用精简摘要替代完整历史 | 减少 Token 数，保留核心信息 |
| **检索增强记忆** | 将历史对话存入向量数据库，每轮按相关性检索注入 | 精准召回关键历史信息 |
| **Sliding Window** | 只保留最近 N 轮对话 | 简单有效，但会丢失早期信息 |
| **结构化记忆** | 提取实体/关系存入图数据库或 KV 存储 | 不依赖上下文窗口 |
| **分层记忆架构** | 短期（State）+ 中期（摘要）+ 长期（向量库）| 模拟人类记忆系统 |
| **Landmark Attention** | 在上下文中插入"标记点"，强制模型关注关键位置 | 缓解 Lost in the Middle |
| **模型架构改进** | 增大上下文窗口（128K+）、使用线性注意力 | 从根本上扩展容量 |

---

### 4. 工程实践与算法优化

---

#### Q: MoE 架构的具体实现原理是什么？路由（Router）是如何工作的？

**MoE 架构原理**：

MoE（Mixture of Experts）将 Transformer 中每层的 FFN 替换为多个并行的"专家"FFN，通过路由网络动态选择激活哪些专家。

```
标准 Transformer:  x → Attention → FFN → output
MoE Transformer:   x → Attention → Router → 选中的专家 FFN → output
```

**核心组件**：

1. **专家网络（Experts）**：N 个结构相同但参数独立的 FFN（如 DeepSeek-V3 有 256 个专家）
2. **路由网络（Router/Gating Network）**：一个可学习的线性层，决定每个 Token 分配给哪些专家

**Router 工作流程**：

```python
# 简化实现
def router(x, W_gate, top_k=2, num_experts=8):
    # 1. 计算每个专家的门控分数
    gate_logits = x @ W_gate  # [batch, num_experts]

    # 2. 选择 Top-K 个专家
    top_k_values, top_k_indices = torch.topk(gate_logits, k=top_k)

    # 3. 对选中专家的分数做 Softmax，得到权重
    weights = F.softmax(top_k_values, dim=-1)

    # 4. 加权组合选中专家的输出
    output = sum(weights[i] * experts[top_k_indices[i]](x) for i in range(top_k))
    return output
```

> 注：以上为伪代码，实际实现需处理 batch 维度、token 维度对齐、capacity 限制与专家并行调度。

**路由策略**：

| 策略 | 说明 |
| :--- | :--- |
| **Top-K Routing** | 每个 Token 选择得分最高的 K 个专家（如 K=2） |
| **Expert Choice** | 反向选择——每个专家选择与自己最匹配的 Token |
| **Shared Expert** | 部分专家始终激活（DeepSeek-V3 使用 1 个共享专家 + Top-8 路由专家） |

**负载均衡（Load Balancing）**：

Router 容易出现"赢者通吃"——少数专家被大量选中，其他专家闲置。解决方案：

- **Auxiliary Loss**：在训练 Loss 中添加负载均衡惩罚项，鼓励各专家被均匀选中
- **Token Drop**：超出容量的 Token 被丢弃或路由到其他专家
- **Noise Injection**：在门控分数中添加噪声，增加探索性

---

#### Q: 面对模型在生成过程中出现循环、重复回答的问题，有哪些解决办法？

| 层次 | 方法 | 说明 |
| :--- | :--- | :--- |
| **解码策略** | **Repetition Penalty** | 对已生成的 Token 的 Logit 施加惩罚（除以 > 1 的系数），降低其再次被选中的概率 |
| | **N-gram 惩罚** | `no_repeat_ngram_size=N`，禁止重复出现的 N-gram |
| | **Temperature 调节** | 适度提高 Temperature 增加多样性 |
| | **Top-p/Top-K 采样** | 引入随机性，避免贪心搜索的确定性循环 |
| | **Frequency Penalty** | 按 Token 出现频率线性增加惩罚 |
| | **Presence Penalty** | Token 只要出现过就施加固定惩罚 |
| **Prompt 工程** | **明确指令** | 在 System Prompt 中明确要求"不要重复" |
| | **Few-shot 示例** | 提供简洁、不重复的示例引导模型 |
| **模型层面** | **SFT 数据质量** | 确保训练数据中无重复模式 |
| | **RLHF/DPO** | 在偏好数据中标注"重复"为负样本 |
| **工程层面** | **后处理截断** | 检测输出中的重复模式，提前终止生成 |
| | **重试机制** | 检测到重复时自动重试，使用更高的 Temperature |

**重复的根本原因**：模型陷入了概率分布的局部循环——某些 Token 序列形成闭环，每个 Token 都高概率指向下一个，最终回到起点。

---

#### Q: 如果单次生成的任务量远大于模型的 Max Tokens 限制，如何实现断点继续生成？

**方案一：Chunked Generation（分块生成）**

```python
full_output = ""
context = initial_prompt

while not is_complete(full_output):
    response = llm.generate(context, max_tokens=4096)
    full_output += response

    # 构建续写 Prompt
    context = initial_prompt + "\n[前文]\n" + full_output[-N:] + "\n[请继续]"
```

**方案二：摘要 + 续写**

```
Round 1: Prompt → 生成 Part 1（达到 max_tokens 停止）
Round 2: "以下是前文摘要：{summary(Part1)}，请继续..." → 生成 Part 2
Round 3: "以下是前文摘要：{summary(Part1+Part2)}，请继续..." → 生成 Part 3
```

**方案三：结构化分段**

将大任务拆分为小任务，每段独立生成后拼接：

```
大纲生成 → 第1章 → 第2章 → ... → 第N章 → 合并
```

**方案四：Agent 自动编排**

使用 LangGraph 等框架将"生成大文本"编排为多步工作流，Agent 自动判断是否完成、是否需要续写。

**关键注意点**：
- 续写时要提供足够的上下文（前文末尾 + 任务指令），避免语义断裂
- 使用摘要而非全文拼接，避免超出上下文窗口
- 设置明确的完成标志（如特定标记），让模型知道何时结束

---

#### Q: 当大模型产生错误回答或幻觉时，在工程和算法层面有哪些规避手段？

**工程层面**：

| 方法 | 说明 |
| :--- | :--- |
| **RAG（检索增强生成）** | 让模型基于检索到的真实文档回答，而非凭"记忆"生成 |
| **Grounding 验证** | 生成后检查答案是否有检索文档支撑，无支撑则标记为不可信 |
| **Self-Consistency** | 多次采样同一问题，取多数一致的答案（投票法） |
| **Chain-of-Verification** | 让模型先生成答案，再自己提问验证，不一致则修正 |
| **外部知识库校验** | 将模型输出与结构化知识库（如知识图谱）交叉验证 |
| **置信度阈值** | 监控生成 Token 的概率，低于阈值时触发人工审核或拒绝回答 |
| **受限解码** | 用 FSM/CFG 约束输出格式，防止格式层面的幻觉 |
| **Human-in-the-Loop** | 关键输出由人工确认 |

**算法层面**：

| 方法 | 说明 |
| :--- | :--- |
| **SFT 数据清洗** | 移除训练数据中的错误信息，提高数据质量 |
| **RLHF/DPO 对齐** | 在偏好数据中标注幻觉回答为负样本，教模型"不知道就说不知道" |
| **Factual Probing** | 训练时加入事实验证任务 |
| **Decoding 策略** | 降低 Temperature、使用 Top-p 限制采样范围，减少"胡说"概率 |
| **知识蒸馏** | 用更强模型的输出作为训练目标 |

**实践优先级**：RAG > Self-Consistency > 受限解码 > RLHF 对齐

---

### 5. RAG 与多模态

---

#### Q: BM25 算法的数学原理是什么？它相比于简单的 TF-IDF 有哪些改进？

**TF-IDF 回顾**：

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

- $\text{TF}(t,d)$：词频，词 $t$ 在文档 $d$ 中出现的次数
- $\text{IDF}(t) = \log\frac{N}{n_t}$：逆文档频率，$N$ 为总文档数，$n_t$ 为包含 $t$ 的文档数

**BM25 公式**：

$$\text{BM25}(t, d) = \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})}$$

其中：
- $f(t, d)$：词 $t$ 在文档 $d$ 中的词频
- $|d|$：文档长度
- $\text{avgdl}$：平均文档长度
- $k_1$（通常 1.2-2.0）：控制词频饱和度
- $b$（通常 0.75）：控制文档长度归一化的程度

**BM25 相比 TF-IDF 的三大改进**：

| 改进点 | TF-IDF | BM25 |
| :--- | :--- | :--- |
| **词频饱和** | TF 线性增长——出现 10 次的权重是 1 次的 10 倍 | TF 有上界——词频增加到一定程度后增益趋于饱和（由 $k_1$ 控制） |
| **文档长度归一化** | 不考虑文档长度，长文档天然占优 | 通过 $b$ 参数惩罚长文档，短文档中出现同一关键词得分更高 |
| **IDF 优化** | 简单的 $\log(N/n_t)$，常见实现不够平滑 | 常用平滑写法：$\log\left(\frac{N - n_t + 0.5}{n_t + 0.5} + 1\right)$，更稳定且非负 |

**直观理解**：
- TF-IDF：一个词出现越多分数越高（无上限）
- BM25：一个词出现几次很重要，但出现 100 次和出现 50 次差别不大（饱和效应）

---

#### Q: MinerU 在解析复杂的工业文档（如图文混排）时，具体的处理逻辑是怎样的？

**MinerU** 是一个开源的文档解析工具，主要处理 PDF 等复杂格式文档。其核心处理流程：

```
PDF 输入 → 版面分析 → 元素分类 → 分类处理 → 结构化输出
```

**详细步骤**：

**1. 版面分析（Layout Detection）**
- 使用基于深度学习的目标检测模型（如 YOLO、LayoutLMv3）识别页面中的区域
- 将页面分割为：标题、段落、表格、图片、页眉页脚、公式等区域
- 输出每个区域的坐标和类型

**2. 元素分类处理**

| 元素类型 | 处理方式 |
| :--- | :--- |
| **文本块** | OCR 识别（如有需要）→ 按阅读顺序排列 → 提取纯文本 |
| **表格** | 表格结构识别 → 行列检测 → 转为 HTML/Markdown 表格 |
| **图片** | 提取图片 → 可选：用多模态模型生成图片描述（Caption） |
| **公式** | LaTeX 识别（如 Nougat 模型）→ 转为 LaTeX 代码 |
| **页眉/页脚** | 识别并过滤掉（通常不参与正文内容） |

**3. 阅读顺序重建**
- 多栏排版中确定正确的阅读顺序（左栏→右栏 vs 跨栏）
- 处理跨页的段落续接

**4. 结构化输出**
- 输出 Markdown 格式或 JSON 格式的结构化文档
- 保留标题层级关系、表格结构、图片引用

**在 RAG 中的价值**：高质量的文档解析是 RAG 的基础。解析质量直接影响分块和检索效果。

---

#### Q: 在多模态检索中，文本和图片是如何映射到同一个统一向量空间的？

**核心方法：对比学习（Contrastive Learning）**

以 **CLIP（Contrastive Language-Image Pre-training）** 为代表：

```
            文本: "一只猫在沙发上"          图片: 🐱🛋️
                    ↓                          ↓
            Text Encoder               Image Encoder
            (Transformer)               (ViT/ResNet)
                    ↓                          ↓
            文本向量 [0.2, 0.8, ...]    图片向量 [0.3, 0.7, ...]
                    ↓                          ↓
                    └──── 同一向量空间 ────┘
```

**训练过程**：

1. **数据**：大量（图片, 文本描述）配对数据（如 LAION-5B，50 亿对）
2. **双塔编码**：Text Encoder 和 Image Encoder 分别编码文本和图片
3. **对比损失**：

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \left[\log \frac{\exp(\text{sim}(t_i, v_i)/\tau)}{\sum_{j=1}^{N}\exp(\text{sim}(t_i, v_j)/\tau)}\right]$$

- 配对的文本-图片向量拉近（正样本）
- 不配对的拉远（负样本）
- $\tau$ 是温度参数

**检索时的工作方式**：

```
用户查询 "红色跑车" → Text Encoder → 查询向量
                                        ↓
                              在统一向量空间中检索
                                        ↓
                 返回距离最近的文本向量 AND/OR 图片向量
```

**主流多模态 Embedding 模型**：

| 模型 | 特点 |
| :--- | :--- |
| **CLIP** | 开创性工作，文本-图片对齐 |
| **SigLIP** | CLIP 改进版，使用 Sigmoid 损失替代 Softmax |
| **BGE-M3** | 支持文本多语言 + 多粒度检索 |
| **Jina CLIP v2** | 支持文本和图片的混合检索 |
| **Nomic Embed Vision** | 开源的多模态 Embedding |

---

#### Q: Ragas 评测框架中的 Faithfulness 和 Answer Relevance 指标的具体计算逻辑是什么？

**Ragas** 是一个专门用于评估 RAG 系统质量的框架。

**Faithfulness（忠实度）**：

衡量生成的回答是否忠实于检索到的上下文（有没有编造信息）。

**计算步骤**：

```
1. 将回答拆分为独立的陈述（Statements）
   回答: "巴黎是法国的首都，建于公元前3世纪"
   → ["巴黎是法国的首都", "巴黎建于公元前3世纪"]

2. 对每条陈述，判断是否能从 Context 中推导出来
   "巴黎是法国的首都" → Context 中有提到 → ✓ 支持
   "巴黎建于公元前3世纪" → Context 中未提到 → ✗ 不支持

3. 计算 Faithfulness 分数
   Faithfulness = 被支持的陈述数 / 总陈述数 = 1/2 = 0.5
```

$$\text{Faithfulness} = \frac{|\text{Supported Statements}|}{|\text{Total Statements}|}$$

**Answer Relevance（答案相关性）**：

衡量生成的回答与原始问题的相关程度（有没有答非所问）。

**计算步骤**：

```
1. 根据回答反向生成 N 个问题（Reverse Engineering）
   回答: "巴黎是法国的首都"
   → 生成问题: ["法国的首都是哪里?", "巴黎是哪个国家的首都?", "什么城市是法国的首都?"]

2. 计算每个生成问题与原始问题的语义相似度（Embedding Cosine Similarity）
   原始问题: "法国的首都是什么?"
   sim("法国的首都是哪里?", 原始问题) = 0.95
   sim("巴黎是哪个国家的首都?", 原始问题) = 0.82
   sim("什么城市是法国的首都?", 原始问题) = 0.91

3. 取平均值
   Answer Relevance = (0.95 + 0.82 + 0.91) / 3 = 0.89
```

$$\text{Answer Relevance} = \frac{1}{N}\sum_{i=1}^{N} \text{sim}(q_{\text{generated}}^i, q_{\text{original}})$$

**Ragas 核心指标总览**：

| 指标 | 衡量什么 | 输入 |
| :--- | :--- | :--- |
| **Faithfulness** | 回答是否基于 Context（无幻觉） | Answer + Context |
| **Answer Relevance** | 回答是否切题 | Answer + Question |
| **Context Precision** | 检索到的 Context 中有用信息排序是否靠前 | Context + Ground Truth |
| **Context Recall** | Ground Truth 中的信息是否都被 Context 覆盖 | Context + Ground Truth |

---

### 6. 评估指标

---

#### Q: 在大模型评估中，如何平衡 Precision 与召回率 Recall？哪一个指标在 Agent 任务中更重要？

**基本定义**：

$$\text{Precision} = \frac{TP}{TP + FP} \quad \text{（预测为正的样本中，真正为正的比例）}$$

$$\text{Recall} = \frac{TP}{TP + FN} \quad \text{（所有正样本中，被正确预测的比例）}$$

**平衡方法**：

| 方法 | 说明 |
| :--- | :--- |
| **F1 Score** | $F_1 = 2 \cdot \frac{P \times R}{P + R}$，Precision 和 Recall 的调和平均 |
| **$F_\beta$ Score** | $F_\beta = (1+\beta^2) \cdot \frac{P \times R}{\beta^2 P + R}$，$\beta > 1$ 侧重 Recall，$\beta < 1$ 侧重 Precision |
| **阈值调整** | 调节分类/检索的阈值，在 P-R 曲线上选择最优点 |
| **业务导向** | 根据错误的代价决定侧重——漏报代价高则重 Recall，误报代价高则重 Precision |

**在 Agent 任务中，哪个更重要？取决于具体场景**：

| Agent 场景 | 更重要的指标 | 原因 |
| :--- | :--- | :--- |
| **工具调用（Tool Calling）** | **Precision** | 调用错误的工具可能产生不可逆后果（如错误下单、删除数据），宁可不调用也不要调错 |
| **意图识别** | **Recall** | 漏识别用户意图会导致任务失败，用户体验差 |
| **RAG 检索** | **Recall** | 检索阶段宁可多召回，后续可通过 Reranker 精排。漏召回的信息无法恢复 |
| **安全审核** | **Recall** | 漏放有害内容的代价远高于误拦截 |
| **信息提取** | 平衡 / 看业务 | 提取错误信息和遗漏关键信息都有害 |

**Agent 整体视角**：

在 Agent 的**规划和执行链路**中，**Precision 通常更关键**。原因：
1. Agent 的行动是串联的，一步错误会级联放大
2. 错误的工具调用可能造成不可逆的副作用
3. "不做"通常比"做错"安全，Agent 可以通过重试弥补遗漏，但难以撤销错误操作

在 Agent 的**信息获取环节**（如 RAG 检索），**Recall 更关键**，因为后续步骤可以过滤多余信息，但无法补回遗漏信息。
