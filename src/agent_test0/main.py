# agent_test0/main.py —— 兼容入口（薄壳）
"""
向后兼容 shim。

原 main.py 已搬到 agent_test0/api/server.py，本文件保留以兼容：
    uv run python src/agent_test0/main.py

新位置启动方式：
    uv run python src/agent_test0/api/server.py
    uv run python -m agent_test0.api.server
"""

# import 触发 server.py 模块加载（创建 FastAPI app 等）
from agent_test0.api.server import app  # noqa: F401
from agent_test0.api.server import run as _run_server


if __name__ == "__main__":
    _run_server()
