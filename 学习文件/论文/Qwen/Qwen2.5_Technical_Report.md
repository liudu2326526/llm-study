# Qwen2.5 Technical Report
**Qwen Team**
**2025-01-06**
**Links**:
- [Hugging Face](https://huggingface.co/Qwen)
- [ModelScope](https://modelscope.cn/organization/qwen)
- [GitHub](https://github.com/QwenLM/Qwen2.5)

## Abstract
This report introduces Qwen2.5, a comprehensive series of large language models (LLMs) with significant improvements in pre-training and post-training. Pre-training data is scaled from 7 trillion to **18 trillion tokens**, laying a solid foundation for common sense, expert knowledge, and reasoning. Post-training includes supervised finetuning with over 1 million samples and multistage reinforcement learning (offline DPO + online GRPO), enhancing human preference alignment, long text generation, structural data analysis, and instruction following.

Qwen2.5 offers rich configurations: open-weight base/instruction-tuned models in 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B parameter sizes (with quantized versions), and over 100 models available on Hugging Face, ModelScope, and Kaggle. Proprietary MoE variants (**Qwen2.5Turbo** and **Qwen2.5-Plus**) are available on Alibaba Cloud Model Studio.

Qwen2.5 achieves top-tier performance on diverse benchmarks:
- Qwen2.5-72B-Instruct (open-weight flagship) rivals Llama-3-405B-Instruct (5x larger) and outperforms many open/proprietary models.
- Qwen2.5-Turbo and Qwen2.5-Plus match GPT-4o-mini and GPT-4o respectively with superior cost-effectiveness.
- Serves as the foundation for specialized models (Qwen2.5-Math, Qwen2.5-Coder, QwQ) and multimodal models.

*Figure 1*: Data scaling (3T→7T→18T) in the Qwen series drives capability improvements, with Qwen2.5 (18T) leading in domain expertise.

## 1 Introduction
The rapid development of LLMs has brought AGI closer, with advancements in model/data scaling, pre-training + SFT + RLHF paradigms, and inference-time reasoning (e.g., OpenAI o1). In the past two years, open-weight LLMs (Llama, Mistral, Qwen series) have democratized access, fostering research and application innovation.

This report details Qwen2.5, with key features:
### Better in Size
- Restores 3B, 14B, 32B models (cost-effective for resource-limited scenarios) alongside 0.5B, 1.5B, 7B, 72B.
- MoE models (Qwen2.5Turbo/Plus) balance accuracy, latency, and cost.

### Better in Data
- Pre-training: 18T tokens (up from 7T), focused on knowledge, coding, mathematics, with staged training and optimized data mixture.
- Post-training: 1M examples across SFT, DPO, GRPO.

### Better in Use
- Eliminates Qwen2 limitations: generation length extended from 2K to **8K tokens**, improved structured input/output (tables, JSON), easier tool use.
- Qwen2.5-Turbo supports **1 million token context length**.

## 2 Architecture & Tokenizer
Qwen2.5 includes **dense open-source models** (0.5B/1.5B/3B/7B/14B/32B/72B) and **MoE API models** (Qwen2.5-Turbo/Plus).

### Dense Model Architecture
Based on Transformer decoder (consistent with Qwen2), with key components:
- Grouped Query Attention (GQA) for efficient KV cache
- SwiGLU activation function
- Rotary Positional Embeddings (RoPE) for position encoding
- QKV bias in attention mechanism
- RMSNorm with pre-normalization for stable training

### MoE Model Architecture
Extends dense models by replacing FFN layers with MoE layers (multiple FFN experts + top-K routing). Implements:
- Fine-grained expert segmentation
- Shared experts routing
(Following Qwen1.5-MoE, boosts downstream task performance)

### Tokenizer
- Uses Qwen's byte-level BPE (BBPE) with **151,643 regular tokens**.
- Expands control tokens from 3 to **22** (adds 2 for tool functionality, others for model capabilities).
- Unified vocabulary across all Qwen2.5 models for consistency and compatibility.

### Table 1: Model architecture and license of Qwen2.5 open-weight models
| Models | Layers | Heads (Q / KV) | Tie Embedding | Context / Generation Length | License |
|--------|--------|---------------|---------------|----------------------------|---------|
| 0.5B   | 24     | 14 / 2        | Yes           | 32K / 8K                   | Apache 2.0 |
| 1.5B   | 36     | 64 / 8        | No            | 128K / 8K                  | Qwen Research |
| 3B     | 80     | 28 / 4        | No            | 128K / 8K                  | Apache 2.0 |
| 7B     | 28     | 40 / 8        | Yes           | 128K / 8K                  | Apache 2.0 |
| 14B    | 64     | 12 / 2        | No            | 128K / 8K                  | Apache 2.0 |
| 32B    | 48     | 16 / 2        | Yes           | 32K / 8K                   | Apache 2.0 |
| 72B    | 64     | 40 / 8        | No            | 32K / 8K                   | Qwen |

## 3 Pre-training
Pre-training consists of **high-quality data curation**, **hyperparameter optimization**, and **long-context pre-training**.

### 3.1 Pre-training Data
Qwen2.5 improves data quality vs. Qwen2 in four key aspects:
1. **Better data filtering**: Uses Qwen2-Instruct as a multi-dimensional quality filter, retaining high-quality samples and filtering low-quality ones across languages (leveraging Qwen2's multilingual pre-training).
2. **Better math and code data**: Integrates Qwen2.5-Math and Qwen2.5-Coder datasets, enabling SOTA math reasoning and code generation.
3. **Better synthetic data**: Generates math/code/knowledge data with Qwen2-72B-Instruct and Qwen2Math-72B-Instruct, filtered by proprietary general reward model and Qwen2-Math-RM-72B.
4. **Better data mixture**: Uses Qwen2-Instruct to classify/balance domain content—down-samples overrepresented domains (e-commerce, social media) and up-samples high-value domains (technology, science, academia).

**Result**: 18 trillion token pre-training dataset (up from 7T in Qwen2).

### 3.2 Scaling Law for Hyper-parameters
Develops scaling laws for dense/MoE models based on Qwen2.5's pre-training data, to:
- Determine optimal batch size (B) and learning rate (μ) for different model sizes (N) and data sizes (D).
- Experiment on dense models (44M→14B) and MoE models (44M→1B activated parameters) with 0.8B→600B tokens.
- Predict MoE model performance vs. dense counterparts, guide hyperparameter tuning for performance parity (e.g., with Qwen2.5-72B/14B).

### 3.3 Long-context Pre-training
#### Two-phase pre-training (all except Qwen2.5-Turbo)
1. Initial phase: 4,096-token context length.
2. Extension phase: 4,096→**32,768 tokens** (final pre-training stage).
3. RoPE base frequency increased from 10,000 to 1,000,000 (ABF technique).

#### Progressive expansion (Qwen2.5-Turbo)
Four stages: 32K→65K→131K→**262,144 tokens** (RoPE base frequency = 10,000,000).
- 40% of training data at current max length, 60% shorter sequences for smooth adaptation.

#### Inference-time long-sequence optimization
Implements **YARN** and **Dual Chunk Attention (DCA)** for 4x sequence length capacity:
- Qwen2.5-Turbo: up to **1 million tokens**.
- Other models: up to **131,072 tokens**.
- Reduces perplexity for long sequences, maintains performance on short sequences.

## 4 Post-training
Qwen2.5's post-training has two core advancements vs. Qwen2: **1M+ SFT examples** (expanded coverage) and **two-stage RL** (offline + online).

### 4.1 Supervised Fine-tuning (SFT)
Constructs a **1 million+ sample SFT dataset** with enhancements in 9 key areas, fine-tuned for 2 epochs (32,768-token sequence length), learning rate from \(7×10^{-6}\) to \(7×10^{-7}\), weight decay 0.1, gradient norm clipping 1.0.

Key enhancements:
1. **Long-sequence Generation**: 8K token output (up from <2K), uses back-translation to generate long-text queries, filters low-quality data with Qwen2.
2. **Mathematics**: Integrates Qwen2.5-Math chain-of-thought data (public/K-12/synthetic problems), uses rejection sampling + reward modeling for step-by-step reasoning.
3. **Coding**: Incorporates Qwen2.5-Coder instruction data (40+ programming languages), synthesizes from code Q&A/GitHub, validates with multilingual sandbox (static check + unit testing).
4. **Instruction-following**: Rigorous code-based validation (LLMs generate instructions + verification code + unit tests), rejection sampling via execution feedback for high-quality data.
5. **Structured Data Understanding**: Comprehensive dataset (tabular QA, fact verification, complex structured/semi-structured tasks), adds reasoning chains to boost inference ability.
6. **Logical Reasoning**: 70,000 new diverse queries (multiple-choice/true-false/open-ended), trains deductive/inductive/analogical/causal/statistical reasoning, filters flawed data.
7. **Cross-Lingual Transfer**: Translates high-resource instructions to low-resource languages, evaluates semantic alignment to preserve logic/style.
8. **Robust System Instruction**: Hundreds of general system prompts, reduces performance variance across different prompts.
9. **Response Filtering**: Multiple automatic annotation methods (critic model + multi-agent scoring), retains only flawless responses.

### 4.2 Offline Reinforcement Learning (Offline RL)
Targets tasks with standard answers but hard to evaluate via reward models (math, coding, instruction following, logical reasoning).
- Reuses SFT quality pipeline (execution feedback + answer matching).
- SFT model resamples responses for new queries: **positive examples** (pass quality checks), **negative examples** (fail).
- Uses **Direct Preference Optimization (DPO)** with human + automated review for reliable signals.
- **150,000 training pairs**, 1 epoch training with Online Merging Optimizer, learning rate \(7×10^{-7}\).

### 4.3 Online Reinforcement Learning (Online RL)
Uses **Group Relative Policy Optimization (GRPO)** to align with human preferences (truthfulness, helpfulness, conciseness, relevance, harmlessness, debiasing).

#### Reward model labeling criteria
1. **Truthfulness**: Factual accuracy, no unsupported information.
2. **Helpfulness**: Addresses queries, positive/educational/relevant content.
3. **Conciseness**: Succinct, no unnecessary verbosity.
4. **Relevance**: Aligned with query/dialogue/context.
5. **Harmlessness**: No illegal/immoral/harmful content.
6. **Debiasing**: Free from gender/race/nationality/political bias.

#### Training details
- Queries: public open-source data + proprietary high-complexity data.
- Responses: sampled from Qwen checkpoints (SFT/DPO/RL) at different temperatures.
- Preference pairs: human + automated labeling (integrates DPO data).
- Training: same query set as reward model, prioritizes high-variance queries, 8 responses per query, 2048 global batch size, 2048 samples per episode.

### 4.4 Long Context Fine-tuning (Qwen2.5-Turbo)
#### SFT phase (two-stage)
1. Stage 1: Only short instructions (<32K tokens) – same data/steps as other Qwen2.5 models (strong short-task performance).
2. Stage 2: Mix short (<32K) and long (<262K) instructions – boosts long-context instruction following.

#### RL phase
- Only short instructions (computationally expensive for long tasks; lack of long-context reward models).
- RL on short instructions still enhances long-context human preference alignment.

## 5 Evaluation
Evaluates pre-trained base models and instruction-tuned models with a **comprehensive automatic evaluation suite** (open benchmarks + in-house datasets), with minimal human interaction.

### Test Data Leakage Prevention
Removes training sequences \(s_t\) if:
- Longest common subsequence (LCS) with test sequence \(s_e\): \(|LCS(s_t,s_e)| ≥13\) **and** \(|LCS(s_t,s_e)| ≥0.6×min(|s_t|,|s_e|)\).

### 5.1 Base Models
Evaluates on **natural language understanding, QA, coding, math, scientific knowledge, reasoning, multilingual capabilities** with datasets including MMLU, BBH, GPQA, HumanEval, Multi-Exam, etc. Compares with Qwen2 and leading open-weight models (Llama3, Mixtral, Gemma2, Yi).

#### Key Results
1. **Qwen2.5-72B & Qwen2.5-Plus**
   - Qwen2.5-72B outperforms peers (Llama3-70B, Mixtral-8x22B) across tasks, matches Llama3-405B (1/5 the parameters), and surpasses Qwen2-72B in nearly all benchmarks (MATH 62.1, GSM8K 91.5).
   - Qwen2.5-Plus (lower cost) outperforms baselines on Hellaswag, TheoremQA, MATH (64.4), GSM8K (93.0), and scores 64.0 on MMLU-Pro (5.9 points higher than Qwen2.5-72B).

2. **Qwen2.5-14B/32B & Qwen2.5-Turbo**
   - Qwen2.5-14B outperforms larger competitors (e.g., MMLU 79.7, BBH 78.2).
   - Qwen2.5-32B surpasses Qwen1.5-32B (MATH 57.7, MBPP 84.5) and outperforms similar-sized models.
   - Qwen2.5-Turbo (lower cost than 14B) achieves comparable results, with better MMLU-Pro than 32B.

3. **Qwen2.5-7B**
   - Surpasses Mistral-7B, Llama3-8B, Gemma2-9B (fewer non-embedding parameters: 6.5B vs. 8.2B for Gemma2-9B) – MMLU 74.2, MATH 49.8, HumanEval 57.9.

4. **Qwen2.5-0.5B/1.5B/3B (edge-side)**
   - Maintains strong performance across benchmarks; Qwen2.5-0.5B outperforms Gemma2-2.6B on math/coding tasks.

*Tables 2-5*: Detailed performance of base models across scales (70B+, 14B-30B+, 7B+, smaller models).

### 5.2 Instruction-tuned Model
Evaluates **foundational skills, human preference, long-context capability, multilingualism** with open benchmarks + in-house evaluations (English/Chinese/multilingual).

#### 5.2.1 Open Benchmark Evaluation
Assesses with MMLU-Pro, LiveBench, MATH, HumanEval, IFEval, MT-Bench, Arena-Hard (basic capabilities + alignment). Key results by model scale:

1. **Qwen2.5-72B-Instruct & Qwen2.5-Plus**
   - Qwen2.5-72B-Instruct surpasses Llama3.1-405B-Instruct on MMLU-redux, MATH (83.1), MBPP, LiveCodeBench, Arena-Hard (81.2), MTBench (9.35).
   - Qwen2.5-Plus outperforms Qwen2.5-72B-Instruct on 9/13 benchmarks.

2. **Qwen2.5-14B/32B-Instruct & Qwen2.5-Turbo**
   - Qwen2.5-32B-Instruct is superior across most tasks for similar-sized models.
   - Qwen2.5-14B-Instruct rivals GPT-4o-mini.
   - Qwen2.5-Turbo (lower cost) outperforms 14B-Instruct on 8/10 benchmarks.

3. **Other Instruction-tuned Models**
   - Qwen2.5-7B-Instruct outperforms Gemma2-9B-IT/Llama3.1-8B-Instruct (MATH 75.5, HumanEval 84.8).
   - Qwen2.5-3B-Instruct (fewer parameters) surpasses Phi3.5-Mini/MiniCPM3-4B on math/coding.
   - Qwen2.5-0.5B/1.5B-Instruct show substantial improvements over Qwen2 versions (ideal for edge-side).

*Tables 6-10*: Detailed performance of instruction-tuned models across scales.

#### 5.2.2 In-house Automatic Evaluation
Develops in-house datasets for **English/Chinese/multilingual evaluation** (knowledge, generation, coding, reasoning). Key results:

1. **English & Chinese**
   - Small Qwen2.5 models (0.5B) match/surpass Qwen2 larger models (1.5B); 3B matches Qwen2-7B; 32B outperforms Qwen2-72B.
   - Qwen2.5-72B narrows the gap with GPT-4/Claude3.5-sonnet, matches/exceeds Llama3.1-405B (except instruction following).
   - Qwen2.5-Plus improves Chinese instruction following and enhances other capabilities.

2. **Multilingual Evaluation**
   - Extends benchmarks (IFEval-multilingual, AMMLU/JMMLU, MGSM8K-extended, BLEnD for cultural nuances).
   - Qwen2.5 excels in instruction following, multilingual knowledge, math reasoning; improved cultural nuance understanding vs. Qwen2 (still room for refinement).

*Tables 11-14*: English, Chinese, and multilingual performance of instruction-tuned models.

#### 5.2.3 Reward Model
Evaluates Qwen2.5-RM-72B on Reward Bench, RMB, PPE, and in-house Chinese human preference benchmark (compares with Nemotron-4-340B-Reward, Llama3.1-Nemotron-70B-Reward, Athene-RM70B). Key results:
- Qwen2.5-RM-72B **leads in PPE and Human-Preference-Chinese**, ranks 2nd on RMB, and matches Nemotron-4-340B-Reward on Reward Bench.
- **Key Insight**: Over-optimization on a single RM benchmark triggers Goodhart’s law (degraded performance on others); current RM benchmarks **do not predict RL model performance** (higher RM scores ≠ better RL models).

*Table 15*: Detailed reward model performance.

#### 5.2.4 Long Context Capabilities
Evaluates with **RULER, LV-Eval (keyword recall), Longbench-Chat**; uses YARN+DCA for length extrapolation. Key results:
1. Qwen2.5 models show strong long-context processing – **Qwen2.5-72B-Instruct** outperforms open-weight and proprietary models (GPT-4o-mini, GPT-4) across all context lengths.
2. Qwen2.5-Turbo achieves **100% accuracy** on 1M-token passkey retrieval.
3. Sparse attention mechanism (based on Minference) boosts inference speed:
   - Reduces attention compute load by **12.5x** for 1M-token sequences.
   - **3.2-4.3x TTFT (Time To First Token) speedup** across hardware (H20, A100).

*Tables 16-17, Figures 2-3*: Detailed long-context performance and inference speed.

## 6 Conclusion
Qwen2.5 is a significant advancement in LLMs, with:
- **18T-token pre-training** and sophisticated post-training (1M+ SFT samples, two-stage RL) boosting human preference alignment, long text generation, and structural data analysis.
- **Diverse configurations**: 0.5B-72B open-weight models (base/instruction-tuned/quantized) and proprietary MoE models (Qwen2.5Turbo/Plus) with superior cost-effectiveness.
- **SOTA performance**: Qwen2.5-72B-Instruct matches Llama-3-405B-Instruct (1/6 the size); serves as a foundation for specialized/multimodal models.

Qwen2.5 is valuable for **academic research and industrial applications**; future work focuses on:
1. Refining base/instruction-tuned models with broader/higher-quality data.
2. Developing multimodal models (unified textual/visual/auditory framework).
3. Enhancing reasoning capabilities via inference compute scaling.

## 7 Authors
### Core Contributors
An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, Zihan Qiu

### Contributors
Biao Sun, Bin Luo, Bin Zhang, Binghai Wang, Chaojie Yang, Chang Si, Cheng Chen, Chengpeng Li, Chujie Zheng, Fan Hong, Guanting Dong, Guobin Zhao, Hangrui Hu, Hanyu Zhao, Hao Lin, Hao Xiang, Haoyan Huang, Humen Zhong, Jialin Wang, Jialong Tang, Jiandong Jiang, Jianqiang Wan, Jianxin Ma, Jianyuan Zeng, Jie Zhang, Jin Xu, Jinkai Wang, Jinzheng He, Jun Tang, Ke Yi, Keqin Chen, Langshi Chen, Le Jiang, Lei Zhang, Liang Chen, Man Yuan, Mingkun Yang, Minmin Sun, Na Ni, Nuo Chen, Peng Wang, Peng Zhu, Pengcheng Zhang, Pengfei Wang, Qiaoyu Tang, Qing Fu, Rong Zhang, Ru Peng, Ruize Gao, Shanghaoran Quan, Shen Huang, Shuai Bai, Shuang Luo, Sibo Song, Song Chen, Tao He, Ting He, Wei Ding, Wei Liao, Weijia Xu, Wenbin Ge, Wenbiao Yin, Wenyuan Yu, Xianyan Jia, Xianzhong Shi, Xiaodong Deng, Xiaoming Huang, Ximing Zhou, Xinyu Wang, Xipin Wei, Xuejing Liu, Yang Liu, Yang Yao, Yang Zhang, Yibo Miao, Yidan Zhang, Yikai Zhu, Yinger Zhang, Yong Jiang, Yong Li, Yongan Yue, Yuanzhi Zhu, Yunfei Chu, Zekun Wang, Zhaohai Li, Zheren Fu, Zhi Li, Zhibo Yang, Zhifang Guo, Zhipeng Zhang, Zhiying Xu, Zile Qiao, Ziye Meng

## References
[Full reference list as in the original report (26 pages of citations)]