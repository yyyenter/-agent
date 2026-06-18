# agent_test0/debug_main.py —— 兼容入口（薄壳）
"""
向后兼容 shim。

原 debug_main.py 已搬到 agent_test0/api/debug_cli.py，本文件保留以兼容：
    uv run python src/agent_test0/debug_main.py

新位置启动方式：
    uv run python src/agent_test0/api/debug_cli.py
    uv run python -m agent_test0.api.debug_cli
"""

from agent_test0.api.debug_cli import run_standalone_debug


if __name__ == "__main__":
    run_standalone_debug()
