"""
Memory 模块 —— 对话状态追踪的基础设施

职责：
1. 回合内容注入：将每一轮用户输入与系统输出以结构化格式写入记忆队列
2. 上下文变量管理：维护动态键值对，记录任务进度、分支条件等元信息
"""

from datetime import datetime


class Memory:
    """智能体记忆管理模块，负责对话历史存储与上下文变量维护。"""

    def __init__(self):
        self._history: list[dict] = []
        self._context_variables: dict = {}
        self._turn_counter: int = 0

    # ── 对话历史管理 ──────────────────────────────────────

    def add_turn(self, role: str, content: str, metadata: dict | None = None) -> int:
        """记录一轮对话，返回本轮 turn_id。"""
        self._turn_counter += 1
        turn = {
            "turn_id": self._turn_counter,
            "role": role,  # "user" | "assistant" | "system" | "tool"
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        self._history.append(turn)
        return self._turn_counter

    def get_history(self, last_n: int | None = None) -> list[dict]:
        """获取对话历史，可选只返回最近 n 轮。"""
        if last_n is None:
            return list(self._history)
        return list(self._history[-last_n:])

    def get_turn(self, turn_id: int) -> dict | None:
        """按 turn_id 获取指定一轮。"""
        for turn in self._history:
            if turn["turn_id"] == turn_id:
                return turn
        return None

    # ── 上下文变量管理 ────────────────────────────────────

    def set_var(self, key: str, value) -> None:
        """设置上下文变量。"""
        self._context_variables[key] = value

    def get_var(self, key: str, default=None):
        """获取上下文变量。"""
        return self._context_variables.get(key, default)

    def get_all_vars(self) -> dict:
        """获取全部上下文变量的快照。"""
        return dict(self._context_variables)

    def delete_var(self, key: str) -> None:
        """删除指定上下文变量。"""
        self._context_variables.pop(key, None)

    # ── 辅助 ──────────────────────────────────────────────

    @property
    def turn_count(self) -> int:
        return self._turn_counter

    def format_history(self, last_n: int | None = None) -> str:
        """格式化输出对话历史，方便调试查看。"""
        lines = []
        for turn in self.get_history(last_n):
            role_tag = {"user": "👤", "assistant": "🤖", "system": "⚙️", "tool": "🔧"}.get(
                turn["role"], "❓"
            )
            lines.append(f"  [{turn['turn_id']}] {role_tag} {turn['role']}: {turn['content']}")
            if turn["metadata"]:
                lines.append(f"       metadata: {turn['metadata']}")
        return "\n".join(lines)
