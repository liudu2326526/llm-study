"""
Tools 模块 —— 模拟酒店预订相关的工具函数

对应文档中的「函数调用(Function Call)机制」，
智能体根据上下文状态决定是否调用某个工具。
"""

import random
from datetime import datetime


# ── 模拟数据 ──────────────────────────────────────────────

_ROOM_INVENTORY = {
    "标准间": {"price": 288, "total": 10},
    "大床房": {"price": 388, "total": 8},
    "套房": {"price": 688, "total": 4},
}

_bookings: dict[str, dict] = {}


# ── 工具函数定义 ──────────────────────────────────────────

# 每个工具函数都有 name / description / parameters 描述，
# 模拟真实 Function Call 的声明格式。

TOOL_DEFINITIONS = [
    {
        "name": "search_rooms",
        "description": "查询指定日期和房型的可用房间",
        "parameters": {
            "checkin": "入住日期",
            "checkout": "退房日期",
            "room_type": "房型(标准间/大床房/套房)",
        },
    },
    {
        "name": "book_room",
        "description": "预订房间",
        "parameters": {
            "name": "客人姓名",
            "checkin": "入住日期",
            "checkout": "退房日期",
            "room_type": "房型",
            "guests": "入住人数",
        },
    },
    {
        "name": "cancel_booking",
        "description": "取消预订",
        "parameters": {"booking_id": "订单号"},
    },
]


def search_rooms(checkin: str, checkout: str, room_type: str) -> dict:
    """查询可用房间。"""
    room_info = _ROOM_INVENTORY.get(room_type)
    if not room_info:
        return {"success": False, "message": f"未知房型: {room_type}"}

    # 模拟随机可用数量
    available = random.randint(1, room_info["total"])
    return {
        "success": True,
        "room_type": room_type,
        "checkin": checkin,
        "checkout": checkout,
        "available_count": available,
        "price_per_night": room_info["price"],
        "message": f"找到 {available} 间可用{room_type}，价格 ¥{room_info['price']}/晚",
    }


def book_room(name: str, checkin: str, checkout: str, room_type: str, guests: str) -> dict:
    """执行预订。"""
    booking_id = f"HTL{datetime.now().strftime('%Y%m%d')}{random.randint(100, 999)}"
    booking = {
        "booking_id": booking_id,
        "name": name,
        "checkin": checkin,
        "checkout": checkout,
        "room_type": room_type,
        "guests": guests,
    }
    _bookings[booking_id] = booking
    return {
        "success": True,
        "booking_id": booking_id,
        "message": f"预订成功！订单号: {booking_id}",
        "details": booking,
    }


def cancel_booking(booking_id: str) -> dict:
    """取消预订。"""
    if booking_id in _bookings:
        del _bookings[booking_id]
        return {"success": True, "message": f"订单 {booking_id} 已取消"}
    return {"success": False, "message": f"未找到订单: {booking_id}"}


def execute_tool(tool_name: str, params: dict) -> dict:
    """统一的工具执行入口，模拟 Function Call 的调用过程。"""
    tool_map = {
        "search_rooms": search_rooms,
        "book_room": book_room,
        "cancel_booking": cancel_booking,
    }
    func = tool_map.get(tool_name)
    if not func:
        return {"success": False, "message": f"未知工具: {tool_name}"}
    return func(**params)
