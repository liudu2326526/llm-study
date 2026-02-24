import autogen
import os

# 配置 API Key
# 在实际使用中，建议将 API Key 放在环境变量中或独立的配置文件中
# 这里为了演示方便，使用一个占位符配置
# 请在使用前设置环境变量 OPENAI_API_KEY，或者直接在这里填入你的 Key
api_key = os.environ.get("OPENAI_API_KEY", "e07ae987-1258-4bb5-94bb-277bdc9fc310")

config_list = [
    {
        "model": "doubao-seed-1-6-flash-250828",  # 或者 gpt-3.5-turbo
        "api_key": api_key,
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
    }
]

llm_config = {
    "config_list": config_list,
    "temperature": 0,  # 代码生成任务建议设为 0
    "seed": 42,        # 缓存种子，相同输入产生相同输出
}

# 1. 定义助手 Agent (AssistantAgent)
# 这个 Agent 负责接收任务，思考并生成代码或方案
assistant = autogen.AssistantAgent(
    name="coding_assistant",
    llm_config=llm_config,
    system_message="""你是一个擅长 Python 编程的 AI 助手。
    如果任务需要写代码，请输出完整的 Python 代码块。
    如果代码需要执行，请确保代码是完整的、可运行的。
    当任务完成时，请回复 'TERMINATE'。
    """
)

# 2. 定义用户代理 Agent (UserProxyAgent)
# 这个 Agent 代表用户，负责执行代码并反馈结果给 Assistant
# human_input_mode="NEVER": 自动执行，不询问人类 (适合全自动 Demo)
# code_execution_config: 配置代码执行环境 (使用 Docker 还是本地进程)
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",  # 可选: ALWAYS, TERMINATE, NEVER
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "coding",     # 代码执行的工作目录
        "use_docker": False,      # 演示方便，使用本地环境 (生产环境建议用 Docker)
    },
    llm_config=False,  # UserProxy 通常不需要 LLM，它只是负责执行和反馈
)

# 3. 定义任务
task = """
编写一个 Python 脚本，计算斐波那契数列的前 20 个数字，并将结果打印出来。
然后，计算这 20 个数字的平均值。
"""

print(f"开始执行任务: {task}")
print("-" * 50)

# 4. 发起对话
# 由 user_proxy 向 assistant 发起聊天
user_proxy.initiate_chat(
    assistant,
    message=task,
)
