# agent_test0/feishu/__init__.py —— 兼容入口（薄壳）
"""
向后兼容 shim。

飞书长连接 bot 已搬到 agent_test0/connectors/feishu/，本文件保留以兼容：
    from agent_test0.feishu import start
    uv run python -m agent_test0.feishu.long_conn_bot

新位置：
    from agent_test0.connectors.feishu import start
    uv run python -m agent_test0.connectors.feishu.bot
"""

from agent_test0.connectors.feishu import (
    start,
    do_p2_im_message_receive_v1,
    _call_agent,
    _send_reply,
)

__all__ = ["start", "do_p2_im_message_receive_v1", "_call_agent", "_send_reply"]
