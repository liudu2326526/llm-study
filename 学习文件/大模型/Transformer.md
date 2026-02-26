

# 背景
Transformer结构是当前大模型的核心架构，自2017年被提出以来，已成为自然语言处理领域最重要的基础技术之一。其设计初衷是为了解决传统循环神经网络在序列建模中存在的长期依赖问题与训练效率瓶颈。相比于RNN与LSTM，Transformer完全抛弃了时间步迭代结构，采用**全并行的自注意力机制**，使得模型在序列建模中**兼顾了建模能力与计算效率**，成为后续BERT、GPT、T5等主流模型的基础结构。
Transformer的核心在于其堆叠式的编码器−解码器架构与全局注意力机制，能够实现对输入序列中任意位置信息的建模，是支持语言理解与生成任务的关键机制，其架构的模块化、层级化特点也极大地增强了系统的可扩展性，便于与其他智能体组件协同构建复杂任务流程。

# 架构
![Transformer Architecture](../resource/transformers_architecture.png)

## 核心组件介绍

### 1. Embedding (嵌入层)
- **Token Embedding**: 将输入的离散 Token（如单词或字符）映射为高维连续向量空间。通过学习到的权重矩阵，将每个 Token 转换为固定维度（如 $d_{model}=512$）的特征表示。
- **Output Embedding**: 在解码阶段，将目标 Token 也转换为向量表示，以便与编码器的输出进行交互。

#### 1.1 分词算法变体 (Tokenization)
- **BPE (Byte Pair Encoding)**: 通过合并高频字节对实现子词切分。
  - **应用**: GPT 系列、LLaMA、Baichuan。
- **WordPiece**: 类似于 BPE，但基于似然度选择合并项。
  - **应用**: BERT。
- **Unigram**: 基于概率语言模型。
  - **应用**: T5、XLNet。

#### 1.2 词表与权重变体
- **Tie Embedding**: 共享输入与输出层的 Embedding 权重，减少模型参数量。
  - **应用**: GPT-2、PaLM。
- **Scaling**: 在 Embedding 后乘以 $\sqrt{d_{model}}$。
  - **应用**: 原始 Transformer。

### 2. Positional Encoding (位置编码)
- **核心作用**: 由于 Transformer 的自注意力机制（Self-Attention）是位置无关的（置换不变性），它无法感知序列中词汇的先后顺序。位置编码通过在 Embedding 中注入位置信息，使模型能够区分不同位置的词。

#### 2.1 绝对位置编码 (Absolute PE)
- **Sinusoidal (正余弦编码)**: 原始 Transformer 采用的方法，使用不同频率的正余弦函数生成固定编码。优点是无需学习，且理论上具有一定的长度外推性。
- **公式**:
$$
PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})
$$
$$
PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})
$$
- **Learned (可学习编码)**: 为每个位置（如 0-511）分配一个可学习的向量（如 BERT、GPT-2）。缺点是无法处理超过训练时最大长度的序列。

#### 2.2 旋转位置编码 (RoPE - Rotary Positional Embedding)
- **核心思想**: 通过旋转矩阵将相对位置信息注入到 Query 和 Key 中。它在保持绝对位置信息的同时，通过旋转角度的差值体现相对位置。
- **优点**: 
  - 具有良好的**长度外推性**（模型可以处理比训练时更长的序列）。
  - 随着距离增加，注意力分数的衰减更自然。
- **代表模型**: LLaMA、PaLM、GLM、DeepSeek。

#### 2.3 相对位置偏差 (ALiBi - Attention with Linear Biases)
- **核心思想**: 不在 Embedding 上加位置信息，而是直接在计算 Attention Score ($QK^T$) 时，根据距离添加一个线性惩罚项。
- **优点**: 极强的外推性，即使在极长序列下也能保持稳定。
- **代表模型**: BLOOM、Falcon、Baichuan2-13B。

#### 2.4 相对位置编码 (Relative PE)
- **核心思想**: 不考虑绝对坐标，只考虑词与词之间的相对距离（如 $i-j$）。
- **代表模型**: T5 (使用可学习的相对位置偏置)。

### 3. Attention Mechanism (注意力机制)
- **自注意力 (Self-Attention)**: 通过计算查询（Query）、键（Key）和值（Value）之间的关联度，让模型在处理当前词时，能够“关注”到序列中其他相关的词。
- **公式**:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

#### 3.1 多头注意力变体 (Multi-Head Variants)
为了平衡计算效率（尤其是推理时的 KV Cache 占用）与模型性能，演化出了以下变体：

- **MHA (Multi-Head Attention)**: 每个 Query 头都有对应的 Key 和 Value 头。参数量大，但表达能力最强。
- **MQA (Multi-Query Attention)**: 所有的 Query 头共享同一组 Key 和 Value 头。极大压缩了 KV Cache，提升了推理吞吐，但模型精度略有下降。
  - **代表模型**: PaLM、Falcon、GPT-NeoX。
- **GQA (Grouped-Query Attention)**: MHA 与 MQA 的折中方案。将 Query 头分组，每组共享一组 Key 和 Value 头。在保持接近 MHA 性能的同时，获得接近 MQA 的速度。
  - **代表模型**: LLaMA-2 (70B)、LLaMA-3。
- **MLA (Multi-head Latent Attention)**: 通过低秩投影对 KV 进行压缩和解压。显著降低了 KV Cache 的显存占用，同时保持了高性能。
  - **代表模型**: DeepSeek-V3。

#### 3.2 稀疏注意力 (Sparse Attention)
旨在解决自注意力机制中 $O(L^2)$ 的复杂度问题，使其能处理更长的序列：

- **Sliding Window Attention (滑窗注意力)**: 每个 Token 只关注其前后固定范围内的 Token。将复杂度降为 $O(L \times W)$。
  - **代表模型**: Mistral、Longformer。
- **Global + Sliding Window**: 在滑窗基础上，指定部分 Token（如 `[CLS]` 或特定关键词）具有全局视野，关注所有位置。
- **BigBird / Longformer**: 结合了随机注意力、滑窗注意力和全局注意力，实现对超长序列的高效建模。
- **FlashAttention**: 虽然不是算法层面的稀疏，但通过 IO 感知的显存优化，极大地提升了标准注意力的计算速度，是大模型训练的标配。

### 4. Add & Norm (残差连接与层归一化)
- **Add (残差连接)**: 借鉴 ResNet 思想，将子层（Attention 或 Feed Forward）的输入直接与其输出相加（$x + \text{Sublayer}(x)$）。这有助于缓解深层网络中的梯度消失问题，使训练更稳定。
- **Norm (层归一化)**: 在每一层之后进行 Layer Normalization，将神经元的激活值归一化到均值为 0、方差为 1 的分布，加速模型收敛。

#### 4.1 归一化变体 (Normalization Variants)
- **LayerNorm (LN)**: 标准的层归一化。
- **RMSNorm (Root Mean Square Layer Normalization)**: 只做缩放不做平移，计算开销更小。
  - **应用**: LLaMA 系列、DeepSeek、Gemma。
- **DeepNorm**: 一种特殊的初始化和归一化策略，允许训练极深（如 1000 层）的 Transformer。
  - **应用**: GLM-130B。

#### 4.2 归一化位置 (Normalization Position)
- **Post-LN**: Norm 放在残差连接之后。容易导致梯度爆炸，训练不稳定，但性能可能略优。
  - **应用**: 原始 Transformer、BERT。
- **Pre-LN**: Norm 放在子层之前。训练极其稳定，是大模型的标准配置。
  - **应用**: GPT-2、LLaMA、Baichuan。
- **Sandwich-LN**: 在 Pre-LN 基础上，在子层输出后再加一层 LN，防止值溢出。
  - **应用**: CogView。

### 5. Feed Forward (前馈神经网络)
- **结构**: 每个位置独立经过一个两层的全连接网络（MLP），中间包含一个非线性激活函数。
- **作用**: 提供非线性映射能力，对注意力机制提取的特征进行进一步的变换和抽象。
- **公式**:

$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

#### 5.1 激活函数变体 (Activation Variants)
- **ReLU**: 原始 Transformer 使用。
- **GELU (Gaussian Error Linear Unit)**: 平滑版的 ReLU，在零点附近具有非零梯度。
  - **应用**: BERT、GPT 系列。
- **SwiGLU**: GLU (Gated Linear Unit) 的变体，使用 Swish 激活函数。显著提升模型表现。
  - **应用**: LLaMA、PaLM、Baichuan。

#### 5.2 架构变体 (Architecture Variants)
- **Standard FFN**: 两层全连接。
- **MoE (Mixture of Experts)**: 将 FFN 替换为多个专家网络，通过路由（Router）选择部分专家激活。
  - **优点**: 极大增加模型参数量的同时，保持较低的推理计算量。
  - **代表模型**: GPT-4 (据传)、Mixtral、DeepSeek-V3。

### 6. Linear (线性层)
- **作用**: 在解码器的顶层，通过一个全连接层将隐藏状态映射到词表大小（Vocabulary Size）的维度。每个维度的数值代表对应词的未归一化得分（Logits）。

### 7. SoftMax (归一化层)
- **作用**: 将 Linear 层的输出 Logits 转换为概率分布。通过指数运算使得分高的词概率更大，且所有词的概率之和为 1，从而选出概率最高的词作为预测结果。


