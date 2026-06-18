# agent_test0/feishu/long_conn_bot.py —— 兼容入口（薄壳）
"""
向后兼容 shim。

原 long_conn_bot.py 已搬到 agent_test0/connectors/feishu/bot.py，
本文件保留以兼容旧命令：

    uv run python -m agent_test0.feishu.long_conn_bot

新启动方式：
    uv run python -m agent_test0.connectors.feishu.bot
"""

from agent_test0.connectors.feishu.bot import *  # noqa: F401, F403
from agent_test0.connectors.feishu.bot import start


if __name__ == "__main__":
    start()
