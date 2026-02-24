# AutoGen 编码实战指南

使用 AutoGen 进行编码的核心在于利用 **AssistantAgent (思考者)** 和 **UserProxyAgent (执行者)** 的协作。

## 1. 核心流程

AutoGen 编码遵循以下闭环逻辑：
1.  **用户发起任务**：通过 `user_proxy.initiate_chat` 发送需求。
2.  **Assistant 生成代码**：LLM 根据需求编写 Python 代码块。
3.  **UserProxy 提取并执行**：
    - 自动识别消息中的 ` ```python ` 代码块。
    - 将代码保存到本地临时文件。
    - 运行代码并捕获标准输出（stdout）和错误（stderr）。
4.  **反馈与 Debug**：
    - 如果代码成功运行，输出结果发回 Assistant。
    - 如果运行报错，报错信息发回 Assistant。
    - Assistant 根据反馈修复代码，直到任务完成。

## 2. 关键配置详解：`code_execution_config`

在定义 `UserProxyAgent` 时，必须配置此项才能开启编码执行能力：

```python
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    code_execution_config={
        "work_dir": "coding",       # 代码运行所在的文件夹（自动创建）
        "use_docker": False,        # 是否在 Docker 容器中运行（推荐生产环境开启）
        "last_n_messages": 3,       # 每次执行参考的历史消息轮数
        "timeout": 60,              # 运行超时时间（秒）
    }
)
```

## 3. 编写高效的 Coding 提示词

为了让 Assistant 更好地编码，建议在 `system_message` 中包含以下内容：
- **明确格式**：要求输出完整的、可运行的代码块。
- **终止条件**：告诉它任务完成后回复 "TERMINATE"。
- **环境说明**：告知当前环境可用的库或限制。

## 4. 自动 Debug 演示

假设 LLM 写出了如下有错误的代码：
```python
# 错误：忘记导入 math 库
print(math.sqrt(16))
```

**执行过程：**
1. `user_proxy` 运行代码，报错：`NameError: name 'math' is not defined`。
2. `user_proxy` 自动回复该报错给 `assistant`。
3. `assistant` 收到报错，意识到漏了导入，生成修正后的代码：
   ```python
   import math
   print(math.sqrt(16))
   ```
4. 任务最终成功。

## 5. 最佳实践

-   **Temperature 设为 0**：在 `llm_config` 中设置 `temperature: 0`，可以提高代码生成的确定性。
-   **工作目录隔离**：始终指定 `work_dir`，避免代码执行产生的文件污染项目根目录。
-   **Docker 沙箱**：处理不信任的任务或复杂依赖时，将 `use_docker` 设为 `True`（需本地安装 Docker）。

---

**参考示例：**
- [simple_chat.py](file:///Users/macbook/Documents/trae_projects/llm-study/代码测试/autogen_demo/simple_chat.py) - 基础编码与斐波那契计算演示。
