# agent_test0/api/__init__.py
"""
HTTP API 入口包。

模块：
    server.py    —— FastAPI 服务（/api/chat_stream SSE 端点等）
    debug_cli.py —— 同步本地调试入口（不启 HTTP，命令行交互）

启动方式：
    uv run python -m agent_test0.api.server      # FastAPI 服务（端口 8000）
    uv run python -m agent_test0.api.debug_cli   # 命令行交互调试
"""
