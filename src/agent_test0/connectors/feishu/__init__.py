# agent_test0/connectors/feishu/__init__.py
"""
飞书长连接机器人 — 按官方示例实现

架构（单向数据流）:
  飞书用户消息 → WebSocket → do_p2_im_message_receive_v1()
  → _call_agent() → TravelWorkflow.run_for_user()
  → _send_reply() → 飞书用户收到回复

启动方式:
  uv run python -m agent_test0.connectors.feishu.bot

参考文档:
  https://open.feishu.cn/document/server-side-sdk/python--sdk/invoke-server-api
  https://open.feishu.cn/document/server-side-sdk/python--sdk/handle-events
"""

from agent_test0.connectors.feishu.bot import (
    start,
    do_p2_im_message_receive_v1,
    _call_agent,
    _send_reply,
)

__all__ = ["start", "do_p2_im_message_receive_v1", "_call_agent", "_send_reply"]
