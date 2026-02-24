# AutoGen 多 Agent 协作 Demo

本项目包含两个基于 Microsoft [AutoGen](https://github.com/microsoft/autogen) 框架的示例，展示了 AI Agent 如何协作完成任务。

## 目录结构
```
autogen_demo/
├── requirements.txt    # 依赖包
├── simple_chat.py      # 双 Agent 模式：代码编写与执行 (Assistant + UserProxy)
└── group_chat.py       # 多 Agent 模式：群聊讨论 (PM + Engineer + Admin)
```

## 环境准备

1.  **安装依赖**：
    ```bash
    pip install -r requirements.txt
    ```

2.  **配置 API Key**：
    你需要一个 OpenAI API Key (或者兼容 OpenAI 格式的其他 LLM Key)。
    在运行代码前，请设置环境变量：
    ```bash
    export OPENAI_API_KEY="sk-你的密钥"
    ```
    *或者直接修改代码中的 `api_key` 变量。*

## 示例说明

### 1. Simple Chat (代码编写与执行)
这个示例展示了最经典的 AutoGen 用法：一个负责写代码，一个负责执行代码。

*   **AssistantAgent**: 接收任务，编写计算斐波那契数列的 Python 代码。
*   **UserProxyAgent**: 接收代码，自动在本地执行（在 `coding` 目录下），并将执行结果（如报错或输出）反馈给 Assistant。
*   **运行**:
    ```bash
    python simple_chat.py
    ```

### 2. Group Chat (多角色群聊)
这个示例展示了多个 Agent 在一个群组中协作讨论。

*   **Product Manager**: 提出“程序员相亲 App”的功能点。
*   **Engineer**: 根据功能点设计技术栈。
*   **Admin**: 管理对话流程。
*   **运行**:
    ```bash
    python group_chat.py
    ```

## 注意事项
*   `UserProxyAgent` 在 `simple_chat.py` 中被配置为 `human_input_mode="NEVER"`，这意味着它会自动执行代码且不询问你。请确保生成的代码是安全的，或者将其改为 `"ALWAYS"` 以便在执行前进行人工确认。
*   代码执行默认在 `coding` 子目录下进行。
