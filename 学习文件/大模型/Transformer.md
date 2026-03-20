# Transformer 深度解析

> Transformer 结构是当前大模型的核心架构，自 2017 年论文 *"Attention Is All You Need"* 发表以来，已成为自然语言处理乃至多模态 AI 领域最重要的基础架构。

---

## 目录

- [一、背景与动机](#一背景与动机)
- [二、整体架构概览](#二整体架构概览)
- [三、核心组件详解](#三核心组件详解)
  - [1. Embedding（嵌入层）](#1-embedding嵌入层)
  - [2. Positional Encoding（位置编码）](#2-positional-encoding位置编码)
  - [3. Attention Mechanism（注意力机制）](#3-attention-mechanism注意力机制)
  - [4. Add & Norm（残差连接与层归一化）](#4-add--norm残差连接与层归一化)
  - [5. Feed Forward Network（前馈神经网络）](#5-feed-forward-network前馈神经网络)
  - [6. Linear & Softmax（输出层）](#6-linear--softmax输出层)
- [四、三大架构范式](#四三大架构范式)
- [五、训练机制](#五训练机制)
- [六、解码策略与生成参数](#六解码策略与生成参数)
- [七、受限解码深度解析](#七受限解码深度解析)
- [八、实际业务应用示例](#八实际业务应用示例)

---

## 一、背景与动机

### 1.1 传统序列模型的困境

在 Transformer 出现之前，序列建模主要依赖 **RNN** 及其变体（LSTM、GRU）：

| 模型 | 核心问题 |
| :--- | :--- |
| **RNN** | 梯度消失/爆炸，难以捕捉长距离依赖 |
| **LSTM** | 通过门控机制缓解了梯度问题，但时间步必须串行计算，**训练效率极低** |
| **GRU** | 简化版 LSTM，问题类似 |
| **CNN (序列建模)** | 可以并行，但感受野有限，需要多层堆叠才能捕捉远距离信息 |

核心瓶颈总结：
- **串行计算**：RNN 系列必须按时间步逐个处理，无法充分利用 GPU 并行能力。
- **长程依赖**：即使是 LSTM，在序列超过数百个 Token 后，早期信息仍会显著衰减。

### 1.2 Transformer 的突破

Transformer 完全抛弃了时间步迭代结构，采用 **全并行的自注意力机制（Self-Attention）**，核心突破在于：

1. **全局视野**：任意两个 Token 之间可以直接交互，无需像 RNN 那样逐步传递信息。
2. **完全并行**：所有位置的计算可以同时进行，训练速度大幅提升。
3. **模块化设计**：编码器-解码器的堆叠式结构具有极强的可扩展性，便于构建从亿级到万亿级参数的模型。

这些特性使 Transformer 成为后续 BERT、GPT、T5、LLaMA 等主流模型的统一基础架构。

---

## 二、整体架构概览

![Transformer Architecture](../resource/transformers_architecture.png)

Transformer 采用经典的 **编码器-解码器（Encoder-Decoder）** 架构，原始论文中编码器和解码器各堆叠 6 层（$N=6$）。

### 2.1 数据流概览

```text
输入序列 → [Token Embedding + Positional Encoding]
                        ↓
              ┌─────────────────────┐
              │   Encoder × N 层    │
              │  ┌───────────────┐  │
              │  │ Self-Attention │  │
              │  │   Add & Norm  │  │
              │  │  Feed Forward │  │
              │  │   Add & Norm  │  │
              │  └───────────────┘  │
              └────────┬────────────┘
                       │ 编码器输出（K, V）
                       ↓
              ┌─────────────────────┐
              │   Decoder × N 层    │
              │  ┌───────────────┐  │
              │  │Masked Self-Att│  │
              │  │   Add & Norm  │  │
              │  │Cross-Attention│ ←── 编码器的 K, V
              │  │   Add & Norm  │  │
              │  │  Feed Forward │  │
              │  │   Add & Norm  │  │
              │  └───────────────┘  │
              └────────┬────────────┘
                       ↓
              [Linear → Softmax → 输出概率]
```

### 2.2 编码器 vs 解码器的关键区别

| 特性 | 编码器 (Encoder) | 解码器 (Decoder) |
| :--- | :--- | :--- |
| **注意力类型** | 双向自注意力（可看全文） | 因果自注意力（只看左侧已生成的内容） |
| **遮蔽机制** | 无遮蔽 | **Masked Self-Attention**：防止"偷看"未来 Token |
| **交叉注意力** | 无 | **Cross-Attention**：从编码器获取源序列信息 |
| **典型用途** | 理解、分类、编码表示 | 生成、翻译、自回归输出 |

### 2.3 三种注意力机制一览

Transformer 中实际包含 **三种不同的注意力**：

1. **Encoder Self-Attention**：编码器中的双向自注意力，Q/K/V 均来自编码器输入。每个 Token 可以关注输入序列中的所有位置。
2. **Masked Self-Attention（因果注意力）**：解码器中的自注意力，Q/K/V 均来自解码器输入。通过遮蔽矩阵（上三角为 $-\infty$），确保位置 $i$ 只能关注位置 $\leq i$ 的 Token，防止信息泄露。
3. **Cross-Attention（交叉注意力）**：解码器中的第二个注意力层。**Q 来自解码器，K 和 V 来自编码器的输出**。这是编码器信息流向解码器的唯一通道。

---

## 三、核心组件详解

### 1. Embedding（嵌入层）

- **Token Embedding**：将输入的离散 Token（如单词或子词）映射为高维连续向量。通过学习到的权重矩阵，将每个 Token 转换为固定维度（如 $`d_{\text{model}}=512`$）的特征表示。
- **Output Embedding**：在解码阶段，将目标 Token 也转换为向量表示，以便与编码器的输出进行交互。

#### 1.1 分词算法（Tokenization）

| 算法 | 核心原理 | 代表模型 |
| :--- | :--- | :--- |
| **BPE** | 从字符级开始，反复合并最高频的相邻子词对 | GPT 系列、LLaMA、Baichuan |
| **WordPiece** | 类似 BPE，但基于似然度（而非频率）选择合并项 | BERT |
| **Unigram** | 基于概率语言模型，从大词表逐步剪枝至目标大小 | T5、XLNet |

**BPE 详细步骤**：
1. 准备语料，将单词拆分为字符序列，并添加结束符 `</w>`。
2. 统计所有相邻子词对的频率。
3. 找到频率最高的子词对，将其合并为一个新的 Token。
4. 重复上述步骤，直到词表达到设定大小或无法再合并。

**具体示例**：
- 假设语料中有 `{"hug": 10, "pug": 5, "pun": 12, "bun": 4}`。
- 第一步合并频率最高的 `u` 和 `n`（共 $12+4=16$ 次），得到 `un`。
- 接着合并 `h` 和 `ug` 等。

**优点**：能够有效处理未登录词（OOV），平衡了字符级（太碎）和词级（词表太大）的缺点。

#### 1.2 词表与权重变体

- **Tie Embedding**：共享输入与输出层的 Embedding 权重，减少模型参数量。
  - 应用：GPT-2、PaLM。
- **Scaling**：在 Embedding 后乘以 $`\sqrt{d_{\text{model}}}`$，防止 Embedding 值过小被位置编码淹没。
  - 应用：原始 Transformer。

#### 1.3 中文向量化策略

针对中文特有的语言特性，常见的向量化（分词）策略包括：

| 策略 | 特点 | 代表模型 |
| :--- | :--- | :--- |
| **字级别** | 词表小（几千字），无 OOV，但无法捕捉词组语义 | BERT-base-chinese |
| **词级别** | 语义明确，但词表庞大、OOV 严重、分词歧义 | （需 Jieba/HanLP 预分词） |
| **BBPE（字节级 BPE）** | 在 UTF-8 字节流上运行 BPE，彻底解决 OOV | GPT-3/4、LLaMA、DeepSeek |
| **SentencePiece** | 集成化框架，直接处理原始文本，中英混合支持极佳 | LLaMA-chinese、Baichuan、ChatGLM |

部分模型还采用 **字词兼顾策略**：在词表中显式加入高频中文词汇（如"人工智能"），提升中文处理效率。

#### 1.4 Token 数计算逻辑

在实际应用中，准确估算 Token 数对于成本控制和上下文管理至关重要。

- **中英文差异估算**：
  - **英文**：约 1000 Tokens ≈ 750 个英文单词（1 word ≈ 1.33 tokens）。
  - **中文**：通常 1 个汉字 ≈ 1.2 ~ 2 个 Tokens（取决于分词器）。
- **影响因素**：
  - **词表大小**：词表越大，单个 Token 承载信息越多，总 Token 数越少。
  - **特殊字符**：空格、换行符、Emoji、代码缩进等都会占用独立 Tokens。
- **计算公式**：

$$\text{Total Tokens} = \sum (\text{分词后的子词数量}) + \text{特殊占位符数量 (如 [CLS], [SEP])}$$

- **实用工具**：
  - **Tiktoken**：OpenAI 提供的 BPE 分词器。
  - **HuggingFace Tokenizers**：支持多种主流开源模型的分词计算。

---

### 2. Positional Encoding（位置编码）

#### 2.1 为什么需要位置编码？

Transformer 的注意力机制对序列顺序天然不敏感（Permutation Invariant）。如果不加位置编码，模型会将 `"I love you"` 和 `"you love I"` 视为完全相同的输入。位置编码的作用就是为每个 Token 注入位置信息。

#### 2.2 绝对位置编码（Absolute PE）

**Sinusoidal（正余弦编码）**：原始 Transformer 采用的方法，使用不同频率的正余弦函数生成固定编码：

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{\text{model}}})$$

$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{\text{model}}})$$

- 优点：无需学习参数，理论上具有一定的长度外推性。
- 直觉：不同维度使用不同频率的波，低频维度编码粗粒度位置，高频维度编码细粒度位置。

**Learned（可学习编码）**：为每个位置分配一个可学习的向量（如 BERT 最大 512、GPT-2 最大 1024）。缺点是无法处理超过训练时最大长度的序列。

#### 2.3 旋转位置编码（RoPE）

- **核心思想**：通过旋转矩阵将相对位置信息注入到 Query 和 Key 中。它在保持绝对位置信息的同时，通过旋转角度的差值自然地体现相对位置关系。
- **关键优势**：
  - 良好的 **长度外推性**（配合 NTK-aware 插值等技术，模型可以处理比训练时更长的序列）。
  - 随着距离增加，注意力分数的衰减更自然。
- **代表模型**：LLaMA、PaLM、GLM、DeepSeek。

#### 2.4 线性偏置注意力（ALiBi）

- **核心思想**：不在 Embedding 上加位置信息，而是直接在计算 Attention Score（$QK^T$）时，根据 Token 间距离添加一个线性惩罚项。距离越远，惩罚越大。
- **优势**：极强的外推性，即使在极长序列下也能保持稳定。
- **代表模型**：BLOOM、Falcon、Baichuan2-13B。

#### 2.5 相对位置编码（Relative PE）

- **核心思想**：不考虑绝对坐标，只考虑词与词之间的相对距离（如 $i-j$）。
- **代表模型**：T5（使用可学习的相对位置偏置）。

#### 2.6 位置编码方案对比

| 方案 | 类型 | 外推性 | 代表模型 |
| :--- | :--- | :--- | :--- |
| Sinusoidal | 绝对 | 一般 | 原始 Transformer |
| Learned | 绝对 | 无 | BERT、GPT-2 |
| RoPE | 绝对+相对 | 强（配合插值） | LLaMA、DeepSeek |
| ALiBi | 相对偏置 | 极强 | BLOOM、Falcon |
| T5 Bias | 相对 | 中等 | T5 |

---

### 3. Attention Mechanism（注意力机制）

#### 3.1 自注意力的计算过程

自注意力是 Transformer 的核心。给定输入矩阵 $X$（每行是一个 Token 的向量），计算过程如下：

**Step 1：生成 Q、K、V**

$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

其中 $W^Q, W^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$，$W^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$ 是可学习的投影矩阵。

**Step 2：计算注意力分数**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- $QK^T$：计算每对 Token 之间的相似度（点积），得到 $L \times L$ 的注意力矩阵。
- $\sqrt{d_k}$：**缩放因子**。当 $d_k$ 较大时，点积的方差也会增大，导致 Softmax 输出趋近于 one-hot（梯度接近于 0）。除以 $\sqrt{d_k}$ 可以稳定梯度。
- Softmax：将分数归一化为概率分布，使每个 Token 对其他 Token 的注意力权重之和为 1。
- 乘以 $V$：按注意力权重对 Value 向量加权求和，得到每个位置的输出。

**直观理解**：
- **Q（查询）**：「我在找什么信息？」
- **K（键）**：「我有什么信息可以提供？」
- **V（值）**：「我实际能提供的内容是什么？」

Q 和 K 的点积衡量"需求"与"供给"的匹配度，匹配度越高的 V 被赋予越大的权重。

#### 3.2 多头注意力（Multi-Head Attention）

单个注意力头只能捕捉一种类型的关联模式。多头注意力通过并行运行多组注意力头，让模型同时关注不同子空间的信息：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$

$$\text{where head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

- 每个头使用独立的投影矩阵，在不同的子空间中学习不同的注意力模式。
- 例如：某些头关注语法结构，某些头关注语义关系，某些头关注位置邻近信息。
- 原始 Transformer：$h=8$，$d_k = d_v = d_{\text{model}} / h = 64$。

#### 3.3 多头注意力变体

为了平衡计算效率（尤其是推理时的 KV Cache 占用）与模型性能，演化出了以下变体：

| 变体 | 机制 | 优缺点 | 代表模型 |
| :--- | :--- | :--- | :--- |
| **MHA** | 每个 Q 头对应独立的 K/V 头 | 表达能力最强，但 KV Cache 占用大 | 原始 Transformer、BERT |
| **MQA** | 所有 Q 头共享同一组 K/V | 极大压缩 KV Cache，精度略降 | PaLM、Falcon |
| **GQA** | Q 头分组，每组共享一组 K/V | MHA 与 MQA 的最佳折中 | LLaMA-2/3 |
| **MLA** | 通过低秩投影压缩/解压 KV | 显存占用极低，性能接近 MHA | DeepSeek-V3 |

#### 3.4 稀疏注意力（Sparse Attention）

标准自注意力的复杂度为 $O(L^2)$，限制了序列长度。稀疏注意力通过限制注意力范围来降低复杂度：

| 方法 | 机制 | 复杂度 | 代表模型 |
| :--- | :--- | :--- | :--- |
| **Sliding Window** | 每个 Token 只关注前后固定窗口 | $O(L \times W)$ | Mistral、Longformer |
| **Global + Sliding** | 滑窗 + 部分 Token 具有全局视野 | $O(L \times (W+G))$ | Longformer |
| **BigBird** | 随机 + 滑窗 + 全局的组合 | $O(L)$ | BigBird |
| **FlashAttention** | IO 感知的显存优化（非稀疏算法） | $O(L^2)$ 但速度极快 | 大模型训练标配 |

> **FlashAttention** 虽然不改变算法复杂度，但通过精心设计的 GPU 显存访问模式（tiling + recomputation），避免了对 $L \times L$ 注意力矩阵的显式存储，在实践中将注意力计算速度提升 2-4 倍。

#### 3.5 KV Cache（键值缓存）

**一句话理解**：KV Cache 将历史 Token 的 Key 和 Value 缓存起来，避免每次生成新 Token 时重复计算，是 LLM 推理最重要的加速技术。

**为什么需要 KV Cache？**

LLM 推理本质是 **自回归生成（Next-token Prediction）**。例如生成 `"I love machine learning"`：
1. `I` → `love`
2. `I love` → `machine`
3. `I love machine` → `learning`

每生成一个新 Token，都要执行一次 Forward。在计算 Attention 时，若无缓存，生成第 $n$ 个 Token 需要重新计算前 $n-1$ 个 Token 的 K 和 V，计算量呈 $O(n^2)$ 增长。

**核心思想：缓存不变项**

关键观察：**历史 Token 的 K 和 V 在生成后续 Token 时不会改变**。因此可以缓存：
- $\text{K-cache} = [K_1, K_2, \dots, K_{n-1}]$
- $\text{V-cache} = [V_1, V_2, \dots, V_{n-1}]$

生成新 Token 时，只需计算当前 Token 的 $`Q_{\text{new}}`$，然后与缓存进行计算：

$$\text{Attention}(Q_{\text{new}}, K_{\text{cache}}, V_{\text{cache}})$$

**性能提升：$O(n^2) \to O(n)$**

| 模式 | 每步计算量 | 总复杂度 |
| :--- | :--- | :--- |
| 无 KV Cache | 重算全部 Token | $O(n^2)$ |
| 有 KV Cache | 只算当前 Token | $O(n)$ |

**显存挑战（The Memory Wall）**

KV Cache 极度消耗显存，计算公式：

$$\text{Size} \approx 2 \times \text{layers} \times \text{tokens} \times \text{hidden\_dim} \times \text{precision\_bytes}$$

例如 7B 模型（4096 上下文），KV Cache 可能占用 3-5 GB 显存。

**工程优化方案**：
| 方案 | 原理 |
| :--- | :--- |
| **Paged KV Cache (vLLM)** | 解决显存碎片，类似操作系统的虚拟内存分页 |
| **KV Cache 量化** | 使用 FP8/INT8 存储，减少一半显存 |
| **Prefix Cache** | 共享公共前缀缓存（如系统提示词） |

**直观类比**：
- 无 KV Cache = 做数学题没有草稿纸，每算下一步都要重新推导前面所有步骤。
- 有 KV Cache = 有草稿纸，只需在已有结果上继续往下算。

---

### 4. Add & Norm（残差连接与层归一化）

- **Add（残差连接）**：借鉴 ResNet 思想，将子层的输入直接与其输出相加（$x + \text{Sublayer}(x)$）。缓解深层网络中的梯度消失问题，使训练更稳定。
- **Norm（层归一化）**：将激活值归一化到均值为 0、方差为 1 的分布，加速模型收敛。

#### 4.1 归一化方法对比

| 方法 | 原理 | 代表模型 |
| :--- | :--- | :--- |
| **LayerNorm (LN)** | 对每层所有神经元的输出进行均值-方差规范化 | 原始 Transformer、BERT |
| **RMSNorm** | 只做缩放不做平移（去掉均值中心化），计算开销更小 | LLaMA、DeepSeek、Gemma |
| **DeepNorm** | 特殊的初始化+归一化策略，支持训练极深（1000+层）网络 | GLM-130B |
| **BatchNorm (BN)** | 在 batch 维度上规范化，对序列数据不太适用 | CV 领域为主 |
| **GroupNorm (GN)** | 介于 BN 和 LN 之间，分组归一化 | CV 领域为主 |

![Different Normalization](../resource/different_Normalization.webp)

#### 4.2 归一化位置

| 位置 | 特点 | 代表模型 |
| :--- | :--- | :--- |
| **Post-LN** | Norm 在残差连接之后；可能训练不稳定（梯度爆炸），但性能可能略优 | 原始 Transformer、BERT |
| **Pre-LN** | Norm 在子层之前；训练极其稳定，**大模型标准配置** | GPT-2、LLaMA、Baichuan |
| **Sandwich-LN** | Pre-LN 基础上，子层输出后再加一层 LN，防止值溢出 | CogView |

#### 4.3 Attention Residuals（注意力残差）

传统 Transformer 的残差连接默认是“等权相加”：

$$h_l = h_{l-1} + f_{l-1}(h_{l-1})$$

把它展开后，可以理解成当前层拿到的是“所有历史层输出的统一累加和”。这种设计的优点是简单、稳定、梯度路径短，但也有一个越来越受关注的问题：

- **所有历史层贡献的权重都固定为 1**，模型无法主动决定“当前层更该看哪几层”。
- 在 **Pre-LN** 结构里，隐藏状态会随着深度不断累加，容易出现“**越深越稀释**”的问题。
- 早期层的信息被混进总和后，后续层很难再有选择地精确取回。

这正是 **Attention Residuals（AttnRes）** 想解决的问题。它把“固定相加”改成“沿深度方向做一次注意力加权”：

$$h_l = \sum_{i=0}^{l-1} \alpha_{i \to l} \, v_i$$

其中：
- $v_0 = h_1$ 表示初始 embedding。
- $v_i = f_i(h_i)$ 表示第 $i$ 层的输出。
- $\alpha_{i \to l}$ 是当前层对历史各层分配的权重，满足和为 1。

也就是说，**普通残差是在深度维度做固定加和，AttnRes 则是在深度维度做“可学习的 softmax 注意力”**。这可以把它理解为：

- 普通 Self-Attention：在 **Token 维度** 选择该看哪些位置。
- Attention Residuals：在 **Layer 维度** 选择该看哪些历史层。

##### 4.3.1 核心思想

AttnRes 的一个关键观察是：RNN 在时间维度上曾经依赖递归状态压缩历史，后来被 Attention 替代；而 Transformer 的残差流在深度维度上，其实也在做类似的“递归累加”。因此可以把同样的思路迁移到深度维度。

一个简化理解是：
- **标准残差**：每层默认平均重视所有历史层。
- **AttnRes**：每层根据输入内容，动态决定“更依赖近层、远层，还是 embedding 本身”。

##### 4.3.2 Full AttnRes 与 Block AttnRes

AttnRes 有两个主要版本：

| 版本 | 核心做法 | 优点 | 代价 |
| :--- | :--- | :--- | :--- |
| **Full AttnRes** | 当前层直接对所有历史层输出做注意力 | 表达能力最强 | 需要保存全部层输出，显存/通信开销更高 |
| **Block AttnRes** | 先把多层压成 block 表示，再对 block 做注意力 | 更适合大规模训练与推理 | 比 Full 版本略有信息压缩 |

`Block AttnRes` 的直觉很像：
- 不是逐层回看全部历史笔记；
- 而是先把若干层压缩成“章节摘要”，再决定重点看哪些摘要。

论文里的结论是：**用大约 8 个 block，通常就能保留大部分收益，同时把代价压下来**。

##### 4.3.3 它为什么重要

Attention Residuals 对 Transformer 的启发主要有三点：

1. **残差连接不只是“训练技巧”**，它本质上也是一种信息路由机制。
2. **注意力不仅可以发生在 Token 之间，也可以发生在层之间**。
3. 随着模型越来越深，残差流本身可能成为新的瓶颈，因此“深度维度的信息选择”会越来越重要。

从工程角度看，它特别适合放在我们理解 `Pre-LN` 的语境里：

- `Pre-LN` 解决了深层训练稳定性问题；
- 但也带来了残差持续累加、层贡献被稀释的问题；
- `AttnRes` 则是在不放弃残差主干的前提下，让模型学会“**挑着加**”。

##### 4.3.4 与普通 Transformer 的关系

对现有 Transformer 可以这样理解：

| 机制 | 本质 |
| :--- | :--- |
| **Self-Attention** | 在序列维度做内容选择 |
| **FFN** | 对每个位置做非线性变换 |
| **Residual** | 在深度维度累积历史表示 |
| **Attention Residuals** | 在深度维度做可学习的信息检索与加权 |

所以 AttnRes 并不是推翻 Transformer，而是把 Transformer 中原本“固定写死”的残差累积，升级成一种更灵活的深度路由方式。

> 延伸阅读：详细整理见 [Attention_Residuals.md](../论文/Kimi/Attention_Residuals.md)

---

### 5. Feed Forward Network（前馈神经网络）

- **结构**：每个位置独立经过一个两层的全连接网络（MLP），中间包含非线性激活函数。
- **作用**：提供非线性映射能力，对注意力机制提取的特征进行进一步的变换和抽象。
- **公式**：

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

> **关键理解**：注意力层负责「信息聚合」（决定看哪些 Token），FFN 负责「信息变换」（决定怎么处理看到的信息）。一个类比：注意力是"开会讨论"，FFN 是"独立思考"。

#### 5.1 激活函数变体

| 激活函数 | 特点 | 代表模型 |
| :--- | :--- | :--- |
| **ReLU** | 原始 Transformer 使用，简单高效 | 原始 Transformer |
| **GELU** | 平滑版 ReLU，零点附近具有非零梯度 | BERT、GPT 系列 |
| **SwiGLU** | GLU 的变体 + Swish 激活，显著提升表现 | LLaMA、PaLM、Baichuan |

#### 5.2 MoE（混合专家模型）

MoE 将 FFN 替换为多个专家网络，通过路由（Router）选择部分专家激活，实现 **参数量大但计算量小** 的效果。

##### 核心组件

- **稀疏 MoE 层**：包含若干"专家"（如 8 个），每个专家是一个独立的 FFN。
- **门控网络 / 路由（Router）**：决定哪些 Token 被分发到哪个专家。路由器由可学习参数组成，与网络其他部分同步预训练。

![MoE Router](../resource/MOE_Architecture.png)

##### MoE 的优缺点

| 维度 | 说明 |
| :--- | :--- |
| **优点：降低推理耗时** | FFN 权重维度大（$`d_{\text{model}} \times d_{\text{ff}}`$），MoE 推理时仅激活少数专家，实际 FLOPs 远小于同等规模稠密模型 |
| **优点：参数扩展性** | 可以在不增加计算成本的前提下，通过增加专家数量极大提升知识容量 |
| **缺点：显存占用大** | 需要提前加载所有专家参数 |
| **缺点：训练困难** | 容易过拟合，路由器的负载均衡（Load Balancing）难以优化 |
| **缺点：微调不稳定** | 在 Fine-tuning 阶段的稳定性不如稠密模型 |

**代表模型**：GPT-4（据传）、Mixtral、DeepSeek-V3。

---

### 6. Linear & Softmax（输出层）

- **Linear 层**：在解码器顶层，通过一个全连接层将隐藏状态映射到词表大小（Vocabulary Size）的维度。每个维度的数值代表对应词的未归一化得分（**Logits**）。
- **Softmax 层**：将 Logits 转换为概率分布。通过指数运算使得分高的词概率更大，且所有词的概率之和为 1，最终输出下一个 Token 的预测概率。

---

## 四、三大架构范式

基于 Transformer 的模型根据使用的组件不同，形成了三大架构范式：

| 范式 | 结构 | 预训练目标 | 典型任务 | 代表模型 |
| :--- | :--- | :--- | :--- | :--- |
| **Encoder-Only** | 仅编码器 | MLM（掩码语言模型）：随机遮蔽部分 Token，预测被遮蔽的 Token | 文本分类、NER、语义理解 | BERT、RoBERTa、ALBERT |
| **Decoder-Only** | 仅解码器 | CLM（因果语言模型）：给定前文，预测下一个 Token | 文本生成、对话、代码生成 | GPT 系列、LLaMA、DeepSeek |
| **Encoder-Decoder** | 完整结构 | Seq2Seq（如 Span Corruption）：编码输入，解码输出 | 翻译、摘要、问答 | T5、BART、mBART |

> **当前趋势**：Decoder-Only 架构已成为大模型的绝对主流。GPT-4、Claude、LLaMA、DeepSeek 等均采用此架构。原因在于其自回归生成范式天然适合对话和通用生成任务，且 Scaling Law 表现最佳。

---

## 五、训练机制

### 5.1 损失函数

**交叉熵损失（Cross-Entropy Loss）** 是 Transformer 最常用的训练目标：

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(w_t | w_{<t})$$

即最大化模型对每个正确 Token 的预测概率。

### 5.2 优化器

| 优化器 | 特点 | 使用场景 |
| :--- | :--- | :--- |
| **Adam** | 自适应学习率，结合动量和 RMSProp | 原始 Transformer |
| **AdamW** | Adam + 权重衰减（Weight Decay），正则化效果更好 | BERT、GPT 系列 |
| **Adafactor** | 节省内存的 Adam 变体 | T5、PaLM |

### 5.3 学习率调度

原始 Transformer 提出了经典的 **Warmup + Decay** 策略：

$$lr = d_{\text{model}}^{-0.5} \cdot \min(\text{step}^{-0.5}, \text{step} \cdot \text{warmup\_steps}^{-1.5})$$

- **Warmup 阶段**：学习率从 0 线性增长到峰值，避免训练初期梯度不稳定。
- **Decay 阶段**：学习率逐步衰减。
- 现代大模型常用 **Cosine Decay**（余弦退火），更平滑。

### 5.4 正则化技术

| 技术 | 原理 | 位置 |
| :--- | :--- | :--- |
| **Dropout** | 训练时随机丢弃部分神经元，防止过拟合 | Attention 权重、FFN 输出、Embedding |
| **Label Smoothing** | 将 one-hot 标签软化（如 0.9/0.1），防止模型过度自信 | 损失函数 |
| **Weight Decay** | 对权重施加 L2 正则化 | 优化器（AdamW） |

---

## 六、解码策略与生成参数

LLM 在生成文本时，主要采用以下解码策略：

### 6.1 贪心搜索（Greedy Search）

每个时间步选择概率最高的 Token：

$$w_{t} = \text{argmax}_{w} P(w | w_{1:t-1})$$

![Greedy search](../resource/greedy_search.png)

从 **The** 开始，算法贪心地选择条件概率最高的词 **nice**，然后 **woman**。最终序列 `(The, nice, woman)` 联合概率为 $0.5 \times 0.4 = 0.2$。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

input_ids = tokenizer.encode('I enjoy walking with my cute dog', return_tensors='pt')
greedy_output = model.generate(input_ids, max_length=50)

print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))
```

**典型输出**（注意重复问题）：
```text
I enjoy walking with my cute dog, but I'm not sure if I'll ever be able to
walk with my dog. I'm not sure if I'll ever be able to walk with my dog.
```

**缺点**：容易陷入局部最优，错过隐藏在低概率词后面的高概率序列。例如 `(The, dog, has)` 的联合概率为 $0.3 \times 0.9 = 0.27 > 0.2$。此外，极易导致文本循环重复。

### 6.2 波束搜索（Beam Search）

通过在每个时间步保留最可能的 `num_beams` 个候选序列，降低丢失高概率序列的风险。

| 初始词 | 分支 1 (概率) | 分支 2 (概率) | 最终选择 |
| :--- | :--- | :--- | :--- |
| **The** | nice (0.5) → woman (0.4) = 0.20 | dog (0.3) → has (0.9) = **0.27** | **(The, dog, has)** |

```python
beam_output = model.generate(
    input_ids, max_length=50, num_beams=5, early_stopping=True
)
```

**优化：n-gram 惩罚**

通过 `no_repeat_ngram_size` 将已出现的 n-gram 候选词概率设为 0：

```python
beam_output = model.generate(
    input_ids, max_length=50, num_beams=5,
    no_repeat_ngram_size=2, early_stopping=True
)
```

> **注意**：n-gram 惩罚需谨慎。例如生成关于"New York"的文章时，`no_repeat_ngram_size=2` 会导致该地名只能出现一次。

**波束搜索的局限性**：
1. **惊喜度低**：倾向于生成中规中矩的文本，缺乏"人性"。
2. **长度偏差**：难以平衡不同长度的序列。
3. **重复风险**：即便有惩罚，仍难以完全消除循环生成。

人类文本的"惊喜度"（概率波动）远高于波束搜索：
```text
P (Probability)
1.0 ─┐
0.8 ─┤          /--\      Human (High Variance)
0.6 ─┤   /--\--/    \--
0.4 ─┤  ----------------  BeamSearch (Flat/Predictable)
0.2 ─┤
0.0 ─┴──────────────────────────────────────
      0  20  40  60  80  100 (Timestep)
```

### 6.3 随机采样（Sampling）

根据当前条件概率分布随机选择下一个 Token：

$$w_{t} \sim P(w | w_{1:t-1})$$

```python
import torch
torch.manual_seed(0)

sample_output = model.generate(
    input_ids, do_sample=True, max_length=50, top_k=0
)
```

**问题**：直接采样可能导致文本不连贯（低概率词被意外选中），也是大模型输出 JSON 等结构化数据格式不稳定的根源。

### 6.4 温度调节（Temperature）

通过 Temperature $T$ 缩放 Softmax 分布，调节生成的"创造力"：

$$P_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

| 温度 | 效果 |
| :--- | :--- |
| $T < 1$（低温） | 分布更"尖锐"，高概率词占主导，结果保守连贯 |
| $T > 1$（高温） | 分布更"平坦"，低概率词机会增加，结果多样但可能乱码 |
| $T \to 0$ | 退化为贪心搜索 |

### 6.5 Top-K 采样

将采样池限制在概率最大的 $K$ 个词中，过滤长尾噪声。

**局限性**：固定 $K$ 无法适应不同概率分布——分布陡峭时 $K=50$ 引入太多干扰，分布平坦时 $K=50$ 又限制发挥。

### 6.6 Top-p 采样（核采样 / Nucleus Sampling）

根据累积概率动态选择采样池，解决固定 $K$ 的问题：
1. 将词按概率降序排列。
2. 选取最小的集合 $`V_{\text{top-p}}`$，使其累积概率 $\geq p$。

```text
     Flat Distribution (p=0.92)        Sharp Distribution (p=0.92)
P    ┌────────────────              P    ┌────────────────
     │   ________                        │   __
     │  |        | → Pool: 50+          │  |  | → Pool: 2-3
     └─┴──────────┴──────              └─┴──┴────────
```

```python
sample_output = model.generate(
    input_ids, do_sample=True, max_length=50, top_p=0.92, top_k=0
)
```

### 6.7 综合建议：Top-K + Top-p + Temperature

实际应用中通常将三者结合使用：

```python
from transformers import GenerationConfig

gen_config = GenerationConfig(
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.8,
    repetition_penalty=1.2,
    max_new_tokens=100,
    pad_token_id=tokenizer.eos_token_id
)

output = model.generate(input_ids, generation_config=gen_config)
```

### 6.8 解码策略对比总结

| 策略 | 核心思想 | 优点 | 缺点 | 适用场景 |
| :--- | :--- | :--- | :--- | :--- |
| **贪心搜索** | 每步取最高概率 | 速度快，确定性强 | 易陷入局部最优，易重复 | 确定性任务（公式推导） |
| **波束搜索** | 保留多条路径 | 序列联合概率高 | 缺乏多样性，计算量大 | 翻译、摘要、QA |
| **Top-K 采样** | 限制前 K 个词 | 过滤长尾噪声 | K 值固定，适应性差 | 创意写作、故事生成 |
| **Top-p 采样** | 动态累积概率 | 适应性强，文本连贯 | 采样池大小不确定 | 通用对话、开放式生成 |
| **受限解码** | FSM/CFG 约束 | **100% 格式正确** | 首次编译延迟 | **JSON 提取、结构化输出** |

---

## 七、受限解码深度解析

受限解码的工程实现主要基于两种计算模型：**有限状态机（FSM）** 和 **上下文无关文法（CFG）**。

### 7.1 FSM 与 CFG 核心对比

| 特性 | **FSM（有限状态机）** | **CFG（上下文无关文法）** |
| :--- | :--- | :--- |
| **特点** | 无栈结构，状态数固定 | 支持递归，带栈自动机 |
| **表达能力** | 正则语言 | 上下文无关语言 |
| **适合场景** | 正则表达式、固定格式、简单 JSON | 复杂 JSON、SQL、编程语言 |
| **优点** | 推理速度快，实现简单 | 表达能力强，支持无限嵌套 |
| **缺点** | 无法表达无限嵌套结构 | 推理开销大，实现复杂 |

### 7.2 LLM 结构化输出流程

```text
Schema / Grammar → CFG 或 FSM 编译 → 生成 Token Mask → Logit Masking → 受限解码输出
```

在生成每个 Token 时，动态限制模型只能输出符合语法规则的 Token，保证 JSON 100% 合法、Tool Call 参数结构正确、SQL 语法无误。

### 7.3 业界方案分布

| 方案类别 | 厂商 / 框架 | 应用场景 |
| :--- | :--- | :--- |
| **CFG** | OpenAI | Structured Outputs, JSON Schema, Tool Calling |
| | Anthropic | Claude Tool Use, Structured Output |
| | Microsoft | Guidance, Semantic Kernel |
| **FSM / Regex** | vLLM | Guided Decoding (FSM/Regex/JSON) |
| | Hugging Face | Constrained Decoding, Outlines |
| | Mistral AI | Function Calling, Structured Outputs |

### 7.4 工程实践建议

- **简单 JSON / Regex / 固定格式** → 优先选择 **FSM**
- **复杂 JSON Schema / SQL / DSL** → 必须使用 **CFG**
- **混合模式**：现代系统（如 OpenAI）采用 **CFG → 编译 → FSM** 链路，兼顾表达能力与推理速度

> **一句话总结**：FSM 快但有限，适合简单约束；CFG 强但略慢，适合复杂嵌套。现代 LLM 的结构化输出本质上就是 **语言模型 + 语法约束解码（FSM/CFG）**。

---

## 八、实际业务应用示例

### 场景 1：追求精确性的提取任务

从文本中提取关键信息，需要模型尽量保守：

```python
output = model.generate(
    input_ids=input_ids,
    num_beams=5,
    do_sample=False,
    temperature=0.1,
    early_stopping=True
)
```

### 场景 2：追求多样性的创意生成

写诗、写故事，需要模型有更多发挥空间：

```python
output = model.generate(
    input_ids=input_ids,
    do_sample=True,
    top_k=50,
    top_p=0.9,
    temperature=0.9,
    repetition_penalty=1.1
)
```

### 场景 3：结构化输出

需要模型稳定返回 JSON 格式时，传统的贪心或采样策略往往不够可靠，建议使用受限解码或框架原生的 Structured Output 功能。
