"""
Agent 决策引擎 —— 基于对话状态驱动 Function Call

对应文档中的「结合 Function Call 的状态驱动决策」：
- 条件性工具调用：槽位齐备时才触发查询/预订
- 子任务流程跳转：进入确认子流程
- 回退与纠错：检测到修改意图时触发 rollback
"""

import re
from memory import Memory
from state_tracker import DialogueStateTracker, SLOT_DEFINITIONS
from tools import execute_tool


# ── 任务阶段定义 ──────────────────────────────────────────

class Phase:
    COLLECTING = "collecting"   # 信息采集阶段
    SEARCHING = "searching"     # 查询房间阶段
    CONFIRMING = "confirming"   # 确认预订阶段
    COMPLETED = "completed"     # 预订完成
    CANCELLED = "cancelled"     # 已取消


class Agent:
    """
    酒店预订智能体：根据对话状态驱动决策。
    模拟真实 Agent 的 感知 → 决策 → 行动 循环。
    """

    def __init__(self):
        self.memory = Memory()
        self.tracker = DialogueStateTracker()
        self.phase = Phase.COLLECTING
        self._last_search_result: dict | None = None
        self._booking_result: dict | None = None

        # 初始化上下文变量
        self.memory.set_var("phase", self.phase)
        self.memory.set_var("interaction_count", 0)

    def process_input(self, user_input: str) -> str:
        """
        处理用户输入，返回智能体回复。
        这是智能体的主循环入口。
        """
        # 1. 感知：记录用户输入到 Memory
        self.memory.add_turn("user", user_input)
        interaction_count = self.memory.get_var("interaction_count", 0) + 1
        self.memory.set_var("interaction_count", interaction_count)

        # 2. 决策：根据当前阶段和输入内容决定动作
        action = self._decide_action(user_input)

        # 3. 行动：执行动作并生成回复
        response = self._execute_action(action, user_input)

        # 4. 记录回复到 Memory
        self.memory.add_turn("assistant", response)
        self.memory.set_var("phase", self.phase)

        return response

    def _decide_action(self, user_input: str) -> dict:
        """
        决策引擎：分析用户意图，决定下一步动作。
        返回 {action, tool?, params?, message?} 结构，模拟 Function Call 决策。
        """
        text = user_input.strip()

        # ── 全局意图检测（任何阶段都可触发）──

        # 检测取消意图
        if re.search(r"取消|不要了|算了|不订了", text):
            return {"action": "cancel"}

        # 检测修改/回退意图
        modify_match = re.search(r"(?:修改|改|更改|换)(?:一下)?(.{0,6})", text)
        if modify_match:
            target = modify_match.group(1).strip()
            return {"action": "rollback", "target": target}

        # 检测查看状态意图
        if re.search(r"(?:当前|目前|现在).{0,4}(?:状态|信息|进度)", text):
            return {"action": "show_status"}

        # ── 阶段性决策 ──

        if self.phase == Phase.COLLECTING:
            return {"action": "collect_info"}

        elif self.phase == Phase.SEARCHING:
            return {"action": "search_rooms", "tool": "search_rooms"}

        elif self.phase == Phase.CONFIRMING:
            if re.search(r"确认|好的|可以|没问题|预订|订吧|确定", text):
                return {"action": "book_room", "tool": "book_room"}
            elif re.search(r"不|换|其他|再看看", text):
                return {"action": "back_to_collect"}
            else:
                return {"action": "ask_confirm"}

        elif self.phase == Phase.COMPLETED:
            if re.search(r"再订|新的|另外", text):
                return {"action": "restart"}
            return {"action": "completed_reply"}

        return {"action": "collect_info"}

    def _execute_action(self, action: dict, user_input: str) -> str:
        """执行决策动作，返回回复文本。"""
        action_type = action["action"]

        if action_type == "collect_info":
            return self._handle_collect(user_input)

        elif action_type == "search_rooms":
            return self._handle_search()

        elif action_type == "book_room":
            return self._handle_book()

        elif action_type == "ask_confirm":
            return "请问您确认预订吗？回复「确认」完成预订，或告诉我需要修改什么。"

        elif action_type == "rollback":
            return self._handle_rollback(action.get("target", ""))

        elif action_type == "cancel":
            return self._handle_cancel()

        elif action_type == "show_status":
            return self._handle_show_status()

        elif action_type == "back_to_collect":
            self.phase = Phase.COLLECTING
            return "好的，请告诉我您想修改哪些信息？"

        elif action_type == "restart":
            self.__init__()
            return "好的，让我们重新开始！请告诉我您的预订需求。"

        elif action_type == "completed_reply":
            return f"您的预订已完成！订单号: {self._booking_result['booking_id']}。如需再订一间，请告诉我。"

        return "抱歉，我没有理解您的意思。请告诉我您的预订需求。"

    # ── 具体动作处理 ──────────────────────────────────────

    def _handle_collect(self, user_input: str) -> str:
        """信息采集：提取槽位 → 保存快照 → 询问下一个缺失项或进入查询。"""
        # 提取槽位
        extracted = self.tracker.extract_slots(user_input)
        if extracted:
            updated = self.tracker.update_slots(extracted)
            # 保存快照（状态变更时）
            self.tracker.save_snapshot(trigger=f"filled: {', '.join(updated)}")
            self.memory.set_var("last_updated_slots", updated)

        # 检查是否所有必填项齐备
        if self.tracker.is_complete():
            # 条件性工具调用：所有槽位齐备 → 自动触发查询
            self.phase = Phase.SEARCHING
            return self._handle_search()

        # 询问下一个缺失槽位
        next_slot = self.tracker.get_next_slot_to_ask()
        prompt = SLOT_DEFINITIONS[next_slot]["prompt"]

        # 构建友好回复
        if extracted:
            filled_info = "、".join(
                f"{SLOT_DEFINITIONS[k]['label']}({v})" for k, v in extracted.items()
            )
            name = self.tracker.slots.get("name")
            greeting = f"{name}{'先生' if name and len(name) <= 2 else ''}" if name else "您"
            return f"收到，{greeting}！已记录{filled_info}。{prompt}"
        else:
            return prompt

    def _handle_search(self) -> str:
        """查询房间：调用 search_rooms 工具。"""
        params = {
            "checkin": self.tracker.slots["checkin"],
            "checkout": self.tracker.slots["checkout"],
            "room_type": self.tracker.slots["room_type"],
        }

        # 模拟 Function Call
        self.memory.add_turn(
            "tool",
            f"Function Call: search_rooms({params})",
            metadata={"tool": "search_rooms", "params": params},
        )

        result = execute_tool("search_rooms", params)
        self._last_search_result = result

        if result["success"]:
            self.phase = Phase.CONFIRMING
            summary = self._format_booking_summary()
            return (
                f"🔍 [Function Call: search_rooms]\n"
                f"   {result['message']}\n\n"
                f"📋 预订信息确认：\n{summary}\n\n"
                f"请确认是否预订？"
            )
        else:
            self.phase = Phase.COLLECTING
            return f"抱歉，{result['message']}。请重新选择房型。"

    def _handle_book(self) -> str:
        """执行预订：调用 book_room 工具。"""
        params = {
            "name": self.tracker.slots["name"],
            "checkin": self.tracker.slots["checkin"],
            "checkout": self.tracker.slots["checkout"],
            "room_type": self.tracker.slots["room_type"],
            "guests": self.tracker.slots["guests"],
        }

        self.memory.add_turn(
            "tool",
            f"Function Call: book_room({params})",
            metadata={"tool": "book_room", "params": params},
        )

        result = execute_tool("book_room", params)
        self._booking_result = result

        if result["success"]:
            self.phase = Phase.COMPLETED
            self.tracker.save_snapshot(trigger="booking_completed")
            return (
                f"✅ [Function Call: book_room]\n"
                f"   {result['message']}\n\n"
                f"📋 订单详情：\n{self._format_booking_summary()}\n\n"
                f"感谢您的预订！如需其他帮助请随时告诉我。"
            )
        return f"预订失败: {result['message']}，请稍后重试。"

    def _handle_rollback(self, target: str) -> str:
        """回退处理：根据用户指定的修改目标回退到对应状态。"""
        # 识别要修改的槽位
        slot_map = {
            "名字": "name", "姓名": "name", "名": "name",
            "入住": "checkin", "入住日期": "checkin", "日期": "checkin",
            "退房": "checkout", "退房日期": "checkout",
            "房型": "room_type", "房间": "room_type", "房": "room_type",
            "人数": "guests", "几位": "guests", "人": "guests",
        }

        target_slot = None
        for keyword, slot_name in slot_map.items():
            if keyword in target:
                target_slot = slot_name
                break

        if target_slot:
            self.tracker.rollback_slot(target_slot)
            self.phase = Phase.COLLECTING
            label = SLOT_DEFINITIONS[target_slot]["label"]
            prompt = SLOT_DEFINITIONS[target_slot]["prompt"]
            self.memory.add_turn(
                "system",
                f"Rollback: 回退槽位 {target_slot}",
                metadata={"action": "rollback", "slot": target_slot},
            )
            return f"🔄 [Rollback] 回退到{label}填写步骤\n{prompt}"
        else:
            # 无法识别具体目标，回退一步
            if self.tracker.rollback(steps=1):
                self.phase = Phase.COLLECTING
                self.memory.add_turn("system", "Rollback: 回退一步")
                return "🔄 [Rollback] 已回退到上一步。\n请告诉我您要修改什么？"
            return "已经是最初状态了，请直接告诉我您的预订需求。"

    def _handle_cancel(self) -> str:
        """取消预订。"""
        if self._booking_result and self._booking_result.get("booking_id"):
            result = execute_tool("cancel_booking", {"booking_id": self._booking_result["booking_id"]})
            self.memory.add_turn("tool", f"Function Call: cancel_booking", metadata={"tool": "cancel_booking"})
            self.phase = Phase.CANCELLED
            return f"🚫 [Function Call: cancel_booking]\n   {result['message']}"
        self.phase = Phase.CANCELLED
        return "好的，已取消本次预订流程。如需重新预订请告诉我。"

    def _handle_show_status(self) -> str:
        """展示当前状态（调试/演示用）。"""
        return (
            f"📊 当前状态：\n"
            f"  阶段: {self.phase}\n"
            f"  对话轮次: {self.memory.turn_count}\n\n"
            f"📋 槽位状态：\n{self.tracker.format_slots()}\n\n"
            f"📸 快照历史：\n{self.tracker.format_snapshots()}"
        )

    # ── 辅助方法 ──────────────────────────────────────────

    def _format_booking_summary(self) -> str:
        """格式化预订摘要。"""
        s = self.tracker.slots
        price = self._last_search_result.get("price_per_night", "N/A") if self._last_search_result else "N/A"
        return (
            f"  姓名: {s['name']}\n"
            f"  入住日期: {s['checkin']}\n"
            f"  退房日期: {s['checkout']}\n"
            f"  房型: {s['room_type']}\n"
            f"  入住人数: {s['guests']}\n"
            f"  参考价格: ¥{price}/晚"
        )
