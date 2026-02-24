# PyTorch深度学习框架

## 1. PyTorch 基础与核心组件
### 1.1 张量 (Tensor) 与 自动求导 (Autograd)
- **Tensor**: PyTorch 的核心数据结构，类似于 NumPy 的 ndarray，但支持 GPU 加速。
    - 创建与操作：`torch.tensor`, `torch.zeros`, `torch.randn`。
    - 维度变换：`view`, `reshape`, `permute`, `transpose`。
    - 设备管理：`.to('cuda')`, `.cuda()`, `.cpu()`。
- **Autograd**: 自动微分引擎，是神经网络训练的基石。
    - 计算图（Computational Graph）：动态图机制（Dynamic Computational Graph），运行时构建。
    - 梯度计算：`.requires_grad=True` 追踪操作，`.backward()` 反向传播计算梯度。
    - 梯度控制：`torch.no_grad()` 上下文管理器（用于推理/评估，减少显存占用）。

### 1.2 优化器 (Optimizer)
- **优化器**: 用于更新模型参数的算法。
    - SGD, Adam, RMSprop 等。
    - `optimizer.step()`: 执行一次参数更新。
    - `optimizer.zero_grad()`: 手动将梯度置零（每次迭代前调用）。

#### 代码示例：张量操作与自动求导
```python
import torch

# 1. 张量创建与设备移动
x = torch.randn(2, 3)  # 创建 2x3 的随机张量
if torch.cuda.is_available():
    x = x.to('cuda')  # 移动到 GPU

# 2. 自动求导
w = torch.tensor([1.0], requires_grad=True)  # 定义权重，需要求导
b = torch.tensor([2.0], requires_grad=True)  # 定义偏置
x = torch.tensor([3.0])
y = w * x + b  # 前向计算图构建

y.backward()  # 反向传播

print(f"dy/dw: {w.grad}")  # 输出 3.0 (因为 y = wx+b, dy/dw = x = 3)
print(f"dy/db: {b.grad}")  # 输出 1.0 (因为 y = wx+b, dy/db = 1)
```

### 1.2 神经网络构建 (nn.Module)
- **nn.Module**: 所有神经网络模块的基类。
    - `__init__`: 定义网络层（如 `nn.Linear`, `nn.Conv2d`）。
    - `forward`: 定义前向传播逻辑（必须实现）。
- **常用层**:
    - 全连接层：`nn.Linear`
    - 卷积层：`nn.Conv2d`
    - 循环层：`nn.LSTM`, `nn.GRU`
    - 归一化层：`nn.BatchNorm2d`, `nn.LayerNorm`
    - 激活函数：`nn.ReLU`, `nn.GELU`, `nn.Sigmoid`
- **容器**:
    - `nn.Sequential`: 顺序容器，快速构建简单模型。
    - `nn.ModuleList`: 模块列表，用于管理一组子模块。

#### 代码示例：构建一个简单的分类网络
```python
import torch.nn as nn

class SimpleClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

# 实例化模型
model = SimpleClassifier(input_size=784, num_classes=10)
print(model)
```

### 1.3 数据处理 (Dataset & DataLoader)
- **Dataset**: 定义数据源和样本获取逻辑。
    - `Map-style dataset`: 实现 `__getitem__` 和 `__len__`。
    - `Iterable-style dataset`: 实现 `__iter__`（适用于流式数据）。
- **DataLoader**: 封装 Dataset，提供批量加载、多进程读取、打乱数据等功能。
    - 关键参数：`batch_size`, `shuffle`, `num_workers`, `collate_fn`（自定义 batch 组装逻辑，常用于 padding）。

#### 代码示例：自定义 Dataset
```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 模拟 Tokenizer 处理
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 模拟数据与调用
# dataset = CustomDataset(train_texts, train_labels, tokenizer)
# loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## 2. 训练与评估流程
### 2.1 完整训练循环 (Training Loop)
一个标准的 PyTorch 训练步骤通常包含：
1.  **前向传播**：`outputs = model(inputs)`
2.  **计算损失**：`loss = criterion(outputs, targets)`
3.  **梯度清零**：`optimizer.zero_grad()`（防止梯度累积）
4.  **反向传播**：`loss.backward()`
5.  **参数更新**：`optimizer.step()`

#### 代码示例：标准训练循环
```python
import torch.optim as optim

# 假设 model, dataloader 已定义
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()  # 切换到训练模式
    running_loss = 0.0
    
    for batch in dataloader:
        inputs, labels = batch['input_ids'], batch['labels']
        
        # 1. 梯度清零
        optimizer.zero_grad()
        
        # 2. 前向传播
        outputs = model(inputs)
        
        # 3. 计算损失
        loss = criterion(outputs, labels)
        
        # 4. 反向传播
        loss.backward()
        
        # 5. 参数更新
        optimizer.step()
        
        running_loss += loss.item()
        
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader)}")
```

### 2.2 模型保存与加载
- **保存权重 (推荐)**: `torch.save(model.state_dict(), 'model.pth')`
- **加载权重**: `model.load_state_dict(torch.load('model.pth'))`
- **保存完整模型**: `torch.save(model, 'model_full.pth')`（依赖代码结构，不推荐）。
- **Checkpoint**: 保存 optimizer 状态、epoch 数，用于断点续训。

### 2.3 可视化与监控
- **TensorBoard**: `torch.utils.tensorboard`，记录 Loss 曲线、参数分布、图像等。
- **Weights & Biases (WandB)**: 第三方实验追踪工具，功能更强大，支持多人协作。

## 3. 大模型微调 (Fine-tuning) 实战
### 3.1 预训练模型集成 (Hugging Face Transformers)
- **Transformers 库**: 事实上的 NLP 标准库，与 PyTorch 无缝集成。
- **核心组件**:
    - `AutoModelForCausalLM`: 加载预训练大模型（如 Llama, Qwen）。
    - `AutoTokenizer`: 加载对应的分词器。
    - `Trainer` API: 封装了训练循环的高级接口，支持多卡、混合精度。
- **集成流程**: 加载预训练权重 -> 冻结部分层（可选） -> 定义微调任务头 -> 训练。

#### 代码示例：加载 Llama 模型
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto",  # 自动分配到 GPU
    torch_dtype=torch.float16 # 使用半精度加载
)

input_text = "Hello, how are you?"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 3.2 参数高效微调 (PEFT)
针对大模型全量微调成本高的问题，采用 PEFT (Parameter-Efficient Fine-Tuning) 技术。
- **LoRA (Low-Rank Adaptation)**:
    - 原理：在冻结的预训练权重旁增加低秩矩阵分支，$W' = W + BA$。
    - 优势：训练参数量减少 99%，显存占用大幅降低，推理无延迟（权重合并）。
    - 实现：`peft` 库集成，`LoraConfig` 配置秩 (r)、缩放系数 (alpha)。
- **其他方法**: Prefix Tuning, P-Tuning, Prompt Tuning。

#### 代码示例：使用 LoRA 微调
```python
from peft import LoraConfig, get_peft_model, TaskType

# 1. 定义 LoRA 配置
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=8,            # 秩大小
    lora_alpha=32,  # 缩放系数
    lora_dropout=0.1
)

# 2. 将模型转换为 PEFT 模型
model = get_peft_model(model, peft_config)
model.print_trainable_parameters() 
# 输出示例: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.062
```

### 3.3 分布式训练基础
- **DataParallel (DP)**: 单机多卡，简单的参数服务器模式，主卡负载重（已不推荐）。
- **DistributedDataParallel (DDP)**:
    - 多进程模式，每张卡独立计算梯度，通过 Ring-AllReduce 同步。
    - 速度快，均衡性好，是工业界标准。
- **FSDP (Fully Sharded Data Parallel)**:
    - 针对超大模型，将模型参数、梯度、优化器状态切分到不同 GPU。
    - 类似 DeepSpeed ZeRO 系列优化，极大降低单卡显存需求。

## 4. 推理优化与部署
### 4.1 模型量化 (Quantization)
- **精度类型**: FP32 -> FP16 / BF16 -> INT8 / INT4。
- **量化策略**:
    - **PTQ (Post-Training Quantization)**: 训练后量化，无需重新训练，速度快。
    - **QAT (Quantization-Aware Training)**: 训练感知量化，精度损失小。
- **工具**: `torch.quantization`, `bitsandbytes` (常用 4-bit/8-bit 加载)。

#### 代码示例：4-bit 量化加载 (bitsandbytes)
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 配置 4-bit 量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
```

### 4.2 编译优化 (torch.compile)
- **PyTorch 2.0+ 核心特性**:
    - `model = torch.compile(model)`: 一行代码加速。
    - 原理：JIT (Just-In-Time) 编译，将 PyTorch 操作融合为 Triton 内核，减少 Python 开销和 GPU 显存访问。
    - 模式：`default`, `reduce-overhead`, `max-autotune`。

#### 代码示例：torch.compile 加速
```python
import torch

# 定义简单模型
model = SimpleClassifier(784, 10).cuda()

# 一行代码优化
opt_model = torch.compile(model)

# 正常使用
inputs = torch.randn(32, 784).cuda()
output = opt_model(inputs)
```

### 4.3 部署框架
- **TorchServe**: PyTorch 官方生产级服务框架。
- **ONNX Runtime**: 将 PyTorch 模型导出为 ONNX 通用格式 (`torch.onnx.export`)，跨平台加速。
- **vLLM / TGI**: 针对大语言模型的高性能推理服务框架（支持 PagedAttention）。

## 5. 实战项目建议
1.  **基础入门**: 使用 MLP/CNN 在 MNIST/CIFAR-10 上实现手写数字/图像分类。
2.  **进阶 NLP**: 使用 LSTM/Transformer 实现情感分析或机器翻译。
3.  **大模型微调**: 使用 LoRA 微调 Llama 3 / Qwen 2.5 模型，使其具备特定领域的问答能力（如医疗、法律）。
4.  **RAG 系统**: 结合 LangChain + PyTorch (Embedding 模型) 构建本地知识库问答助手。
