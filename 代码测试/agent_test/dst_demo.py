"""
对话状态追踪(DST) Demo —— 酒店预订场景

本 Demo 演示智能体在多轮对话中如何通过状态追踪机制维持连贯性：
1. Memory 模块：回合内容注入 + 上下文变量管理
2. StateTracker 模块：槽位提取、状态快照、回退纠错
3. Agent 决策引擎：基于状态驱动 Function Call

运行方式：python dst_demo.py

特殊命令：
  /status  - 查看当前对话状态、槽位值、快照历史
  /history - 查看完整对话历史
  /debug   - 查看详细调试信息（Memory + StateTracker + Phase）
  /reset   - 重置对话
  /help    - 显示帮助信息
  /quit    - 退出
"""

from agent import Agent


WELCOME = """
╔══════════════════════════════════════════════════════════════╗
║           🏨  酒店预订助手 —— 对话状态追踪 Demo             ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  本 Demo 演示对话状态追踪(DST)的核心机制：                   ║
║    · Memory 模块：对话历史记录 & 上下文变量管理              ║
║    · 槽位追踪：自动从对话中提取关键信息                      ║
║    · 状态快照：支持回退与纠错                                ║
║    · Function Call：根据状态自动决定调用工具                 ║
║                                                              ║
║  特殊命令: /status /history /debug /reset /help /quit        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
📖 使用帮助：

  直接输入自然语言与预订助手对话，例如：
    · "我想订一间房"
    · "我姓张"
    · "3月10号入住，3月12号退房"
    · "大床房，2位客人"

  修改信息：
    · "修改日期" / "改一下房型" / "换个名字"

  查看状态：
    · "当前状态" 或 /status

  特殊命令：
    /status  - 查看槽位状态和快照历史
    /history - 查看完整对话历史
    /debug   - 查看详细调试信息
    /reset   - 重置对话，重新开始
    /help    - 显示本帮助
    /quit    - 退出程序
"""


def handle_command(cmd: str, agent: Agent) -> str | None:
    """处理特殊命令，返回输出文本；非命令返回 None。"""
    cmd = cmd.strip().lower()

    if cmd == "/help":
        return HELP_TEXT

    elif cmd == "/status":
        return agent._handle_show_status()

    elif cmd == "/history":
        history = agent.memory.format_history()
        return f"📜 对话历史：\n{history}" if history else "📜 暂无对话记录"

    elif cmd == "/debug":
        return (
            f"{'=' * 50}\n"
            f"🔍 DEBUG 信息\n"
            f"{'=' * 50}\n\n"
            f"📊 Agent 阶段: {agent.phase}\n\n"
            f"📋 槽位状态:\n{agent.tracker.format_slots()}\n\n"
            f"📸 快照历史:\n{agent.tracker.format_snapshots()}\n\n"
            f"🔑 上下文变量:\n  {agent.memory.get_all_vars()}\n\n"
            f"📜 对话历史 (最近5轮):\n{agent.memory.format_history(last_n=5)}\n"
            f"{'=' * 50}"
        )

    elif cmd == "/reset":
        return "__RESET__"

    elif cmd == "/quit":
        return "__QUIT__"

    return None


def main():
    print(WELCOME)
    agent = Agent()

    # 初始问候
    greeting = "欢迎光临！我是酒店预订助手，请问有什么可以帮您？"
    agent.memory.add_turn("assistant", greeting)
    print(f"🤖 {greeting}\n")

    while True:
        try:
            user_input = input("👤 ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 再见！")
            break

        if not user_input:
            continue

        # 处理特殊命令
        if user_input.startswith("/"):
            result = handle_command(user_input, agent)
            if result == "__QUIT__":
                print("\n👋 感谢使用，再见！")
                break
            elif result == "__RESET__":
                agent = Agent()
                print("\n🔄 已重置对话。\n")
                greeting = "欢迎光临！我是酒店预订助手，请问有什么可以帮您？"
                agent.memory.add_turn("assistant", greeting)
                print(f"🤖 {greeting}\n")
                continue
            elif result:
                print(f"\n{result}\n")
                continue

        # 正常对话处理
        response = agent.process_input(user_input)
        print(f"\n🤖 {response}\n")


if __name__ == "__main__":
    main()
