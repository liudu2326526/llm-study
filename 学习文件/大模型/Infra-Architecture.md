# LLM 推理架构优化 (Infra Architecture)

LLM 推理吞吐量的关键不在模型，而在 Infra 架构。为了应对 KV Cache 带来的显存压力和自回归生成的串行瓶颈，业界演进出了以下核心优化技术：

## 1. 核心优化技术

### 1️⃣ Continuous Batching (连续批处理)
*   **痛点**：传统 Batching 需要等待 Batch 中最长的序列生成结束才能开始下一轮，导致 GPU 存在严重的空闲（Bubble）。
*   **原理**：在 Token 级别进行调度，一旦某个请求生成结束（遇到 EOS），立即插入新请求，无需等待整个 Batch 完成。
*   **效果**：极大提升了 GPU 利用率和系统吞吐量。

### 2️⃣ Paged KV Cache (分页 KV 缓存)
*   **痛点**：KV Cache 随序列增长，预先分配显存会导致严重的碎片化（External Fragmentation），且无法有效处理变长序列。
*   **原理**：借鉴操作系统虚拟内存管理，将 KV Cache 划分为固定大小的“页（Blocks）”，按需动态分配，不要求物理连续。
*   **代表作**：vLLM。
*   **效果**：显存碎片率降至近 0%，允许更大的 Batch Size。

### 3️⃣ Prefix Cache (前缀缓存)
*   **场景**：在多轮对话或固定 Prompt（如 System Message）场景下，相同的前缀会重复计算。
*   **原理**：将公共前缀的 KV Cache 缓存并在多个请求间共享。
*   **效果**：显著降低 Prefill 阶段的计算量和首字延迟（TTFT）。

### 4️⃣ Prefill / Decode 分离
*   **痛点**：Prefill（计算密集型）和 Decode（访存密集型）在资源需求上存在冲突，混合在一起会互相干扰。
*   **原理**：将请求的预填充阶段和解码阶段分配到不同的算力单元或集群上独立调度。
*   **效果**：优化了资源配比，降低了长文本生成的排队等待时间。

### 5️⃣ Speculative Decoding (投机采样)
*   **原理**：使用一个小模型（Draft Model）快速预测多个后续 Token，再由大模型（Oracle Model）一次性并行校验。
*   **效果**：在不改变输出质量的前提下，通过增加并行度来打破自回归的串行限制，加速生成速度。

## 2. 性能指标参考 (vLLM vs 原始 Transformers)

| 指标 | Transformers | vLLM (Optimized Infra) |
| :--- | :--- | :--- |
| **吞吐量 (Requests/s)** | 低 (1x) | 高 (2x - 4x) |
| **显存利用率** | 碎片化严重 | 极高 (Paged) |
| **首字延迟 (TTFT)** | 随 Batch Size 增加剧增 | 相对稳定 |

---
> **相关文档**：
> - 了解解码原理：[Transformer.md](./Transformer.md)
> - 了解受限解码：[Transformer.md#8-受限解码深度解析-fsm-vs-cfg](./Transformer.md#L439)