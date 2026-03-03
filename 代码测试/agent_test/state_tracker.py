"""
DialogueStateTracker —— 对话状态追踪核心模块

职责：
1. 关键标记提取：从用户输入中识别槽位值(Slot Values)
2. 状态快照：在每次状态变更时保存快照，支持回退
3. 回退与纠错：当用户想修改已填信息时，回退到对应状态
"""

import copy
import re


# ── 槽位定义 ──────────────────────────────────────────────

SLOT_DEFINITIONS = {
    "name": {
        "label": "姓名",
        "required": True,
        "prompt": "请问您贵姓？",
        "order": 1,
    },
    "checkin": {
        "label": "入住日期",
        "required": True,
        "prompt": "请问入住日期是哪天？（如：3月10号）",
        "order": 2,
    },
    "checkout": {
        "label": "退房日期",
        "required": True,
        "prompt": "退房日期是哪天？",
        "order": 3,
    },
    "room_type": {
        "label": "房型",
        "required": True,
        "prompt": "请问需要什么房型？（标准间 / 大床房 / 套房）",
        "order": 4,
    },
    "guests": {
        "label": "入住人数",
        "required": True,
        "prompt": "入住几位客人？",
        "order": 5,
    },
}


class DialogueStateTracker:
    """对话状态追踪器：管理槽位提取、快照与回退。"""

    def __init__(self):
        self.slots: dict[str, str | None] = {key: None for key in SLOT_DEFINITIONS}
        self._snapshots: list[dict] = []
        self._current_asking: str | None = None  # 当前正在询问的槽位

    # ── 槽位提取 ──────────────────────────────────────────

    def extract_slots(self, user_input: str) -> dict[str, str]:
        """
        从用户输入中提取槽位值。
        使用规则 + 正则的方式模拟 NLU 槽位提取。
        返回本次提取到的 {slot_name: value} 字典。
        """
        extracted = {}
        text = user_input.strip()

        # 姓名提取：「姓X」「我叫XX」「X先生/女士」
        name_patterns = [
            r"(?:我姓|免贵姓|姓)(\S{1,2})",
            r"(?:我叫|我是|名字是?)(\S{2,4})",
            r"(\S{1,4})(?:先生|女士|小姐)",
        ]
        for pattern in name_patterns:
            m = re.search(pattern, text)
            if m:
                extracted["name"] = m.group(1)
                break

        # 日期提取：「X月X号/日」
        date_pattern = r"(\d{1,2})\s*月\s*(\d{1,2})\s*[号日]?"
        dates_found = re.findall(date_pattern, text)
        if dates_found:
            # 如果当前正在问入住日期，或者还没填入住日期
            if self._current_asking == "checkin" or self.slots["checkin"] is None:
                month, day = dates_found[0]
                extracted["checkin"] = f"{int(month)}月{int(day)}日"
                if len(dates_found) > 1:
                    month2, day2 = dates_found[1]
                    extracted["checkout"] = f"{int(month2)}月{int(day2)}日"
            elif self._current_asking == "checkout" or self.slots["checkout"] is None:
                month, day = dates_found[0]
                extracted["checkout"] = f"{int(month)}月{int(day)}日"

        # 房型提取
        for room_type in ["标准间", "大床房", "套房"]:
            if room_type in text:
                extracted["room_type"] = room_type
                break

        # 人数提取：「X位」「X个人」「X人」
        guest_pattern = r"(\d+)\s*(?:位|个人?|人)"
        m = re.search(guest_pattern, text)
        if m:
            extracted["guests"] = f"{m.group(1)}位"

        # 如果正在询问某个特定槽位，且用户输入较短（直接回答），做兜底处理
        if self._current_asking and self._current_asking not in extracted:
            if len(text) <= 10 and not any(kw in text for kw in ["修改", "改", "回退", "取消"]):
                if self._current_asking == "name" and not extracted:
                    extracted["name"] = text
                elif self._current_asking == "guests" and re.search(r"\d+", text):
                    num = re.search(r"\d+", text).group()
                    extracted["guests"] = f"{num}位"

        return extracted

    def update_slots(self, extracted: dict[str, str]) -> list[str]:
        """用提取到的值更新槽位，返回被更新的槽位名列表。"""
        updated = []
        for key, value in extracted.items():
            if key in self.slots:
                old_value = self.slots[key]
                self.slots[key] = value
                updated.append(key)
                if old_value is not None and old_value != value:
                    # 值被修改，记录到快照元信息
                    pass
        return updated

    # ── 快照管理 ──────────────────────────────────────────

    def save_snapshot(self, trigger: str = "") -> int:
        """保存当前状态快照，返回快照序号。"""
        snapshot = {
            "index": len(self._snapshots),
            "slots": copy.deepcopy(self.slots),
            "current_asking": self._current_asking,
            "trigger": trigger,
        }
        self._snapshots.append(snapshot)
        return snapshot["index"]

    def rollback(self, steps: int = 1) -> bool:
        """回退指定步数的状态。"""
        target_index = len(self._snapshots) - 1 - steps
        if target_index < 0:
            return False
        snapshot = self._snapshots[target_index]
        self.slots = copy.deepcopy(snapshot["slots"])
        self._current_asking = snapshot["current_asking"]
        # 截断后续快照
        self._snapshots = self._snapshots[: target_index + 1]
        return True

    def rollback_slot(self, slot_name: str) -> bool:
        """回退到指定槽位被填充之前的状态。"""
        for i in range(len(self._snapshots) - 1, -1, -1):
            if self._snapshots[i]["slots"].get(slot_name) is None:
                self.slots = copy.deepcopy(self._snapshots[i]["slots"])
                self._current_asking = slot_name
                self._snapshots = self._snapshots[: i + 1]
                return True
        # 没找到就直接清空该槽位
        self.slots[slot_name] = None
        self._current_asking = slot_name
        return True

    # ── 状态查询 ──────────────────────────────────────────

    def get_missing_slots(self) -> list[str]:
        """返回尚未填充的必填槽位列表（按 order 排序）。"""
        missing = []
        for key, definition in sorted(SLOT_DEFINITIONS.items(), key=lambda x: x[1]["order"]):
            if definition["required"] and self.slots[key] is None:
                missing.append(key)
        return missing

    def get_next_slot_to_ask(self) -> str | None:
        """获取下一个应该询问的槽位。"""
        missing = self.get_missing_slots()
        if missing:
            self._current_asking = missing[0]
            return missing[0]
        self._current_asking = None
        return None

    def is_complete(self) -> bool:
        """判断所有必填槽位是否已齐备。"""
        return len(self.get_missing_slots()) == 0

    def set_current_asking(self, slot_name: str | None):
        self._current_asking = slot_name

    # ── 格式化输出 ────────────────────────────────────────

    def format_slots(self) -> str:
        """格式化当前槽位状态，方便调试。"""
        lines = []
        for key, definition in sorted(SLOT_DEFINITIONS.items(), key=lambda x: x[1]["order"]):
            value = self.slots[key]
            status = f"✅ {value}" if value else "⬜ 未填写"
            required_tag = "*" if definition["required"] else " "
            lines.append(f"  {required_tag} {definition['label']}: {status}")
        return "\n".join(lines)

    def format_snapshots(self) -> str:
        """格式化快照历史。"""
        if not self._snapshots:
            return "  (无快照)"
        lines = []
        for s in self._snapshots:
            filled = [f"{k}={v}" for k, v in s["slots"].items() if v]
            lines.append(f"  [{s['index']}] {s['trigger']} | {', '.join(filled) or '空'}")
        return "\n".join(lines)
