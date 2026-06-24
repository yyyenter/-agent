#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""模拟飞书发消息, 触发 bot 完整后端响应 (不依赖真实飞书 App)

调用链: do_p2_im_message_receive_v1 -> _process_message_async -> _call_agent
        -> TravelWorkflow.run_for_user -> 6 状态机 -> final_report -> _send_reply

_send_reply 在 mock 模式下被替换, 不真发飞书, 但走完整 Agent 流程。
"""
import sys
import json
from unittest.mock import MagicMock

ROOT = "E:/Python/agent_test0"
sys.path.insert(0, f"{ROOT}/src")

from agent_test0.connectors.feishu import bot


def mock_send_reply(open_id, text):
    print(f"\n[mock_send_reply] open_id={open_id}")
    print(f"[mock_send_reply] text 长度={len(text)}")
    print("-" * 60)
    print(text)
    print("-" * 60)


def mock_send_reply_immediate(open_id, text):
    print(f"[mock_send_reply_immediate] {text}")


bot._send_reply = mock_send_reply
bot._send_reply_immediate = mock_send_reply_immediate


USER_TEXT = "想去北京玩 3 天，预算 5000，两个人"
SENDER_OPEN_ID = "mock_user_open_id_001"
CHAT_ID = "mock_chat_id_001"
MESSAGE_ID = "mock_msg_001"

mock_message = MagicMock()
mock_message.message_id = MESSAGE_ID
mock_message.content = json.dumps({"text": USER_TEXT})
mock_message.chat_id = CHAT_ID

mock_sender = MagicMock()
mock_sender.sender_id.open_id = SENDER_OPEN_ID

mock_event = MagicMock()
mock_event.message = mock_message
mock_event.sender = mock_sender

mock_data = MagicMock()
mock_data.event = mock_event

print("=" * 70)
print("【飞书 Mock 测试】")
print(f"  用户消息: {USER_TEXT}")
print(f"  发送者:   {SENDER_OPEN_ID}")
print(f"  消息 ID:  {MESSAGE_ID}")
print("=" * 70)

bot.do_p2_im_message_receive_v1(mock_data)

print("\n[主线程] 等待后台 Agent 线程...")
bot._agent_executor.shutdown(wait=True, cancel_futures=False)
print("[主线程] 后台 Agent 线程已结束")
print("=" * 70)
print("【测试完成】")
