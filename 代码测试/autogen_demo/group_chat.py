import autogen
import os

# 同样的配置
api_key = os.environ.get("OPENAI_API_KEY", "e07ae987-1258-4bb5-94bb-277bdc9fc310")
config_list = [
    {
        "model": "doubao-seed-1-6-flash-250828",
        "api_key": api_key,
        "base_url": "https://ark.cn-beijing.volces.com/api/v3"
    }
]
llm_config = {"config_list": config_list, "temperature": 0.7}

# 1. 定义产品经理 (Product Manager)
pm = autogen.AssistantAgent(
    name="Product_Manager",
    system_message="你是一个有创意的产品经理。你的工作是针对用户的主题提出具体的应用功能点。",
    llm_config=llm_config,
)

# 2. 定义工程师 (Engineer)
engineer = autogen.AssistantAgent(
    name="Engineer",
    system_message="你是一个资深后端工程师。你的工作是根据产品经理的功能点，设计技术栈和简单的 API 接口定义。",
    llm_config=llm_config,
)

# 3. 定义群聊管理器 (User Proxy / Admin)
user_proxy = autogen.UserProxyAgent(
    name="Admin",
    system_message="你是项目管理员。负责启动讨论，并在讨论结束时总结。",
    human_input_mode="TERMINATE", # 在需要时请求人类输入
    code_execution_config=False,
    llm_config=llm_config, # 赋予 Admin 思考能力，防止其自动回复空消息
)

# 4. 创建群聊 (GroupChat)
groupchat = autogen.GroupChat(
    agents=[user_proxy, pm, engineer], 
    messages=[], 
    max_round=6 # 限制对话轮数
)

# 5. 创建群聊管理器
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# 6. 开始群聊
task = "我们要开发一个‘针对程序员的相亲 App’，请讨论出核心功能和技术架构。"
user_proxy.initiate_chat(
    manager,
    message=task,
)
