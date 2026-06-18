# agent_test0/memory/__init__.py
"""
记忆系统包。

主要 export：
    from agent_test0.memory import MemoryManager, get_redis_or_fallback

内部模块：
    manager.py —— MemoryManager 主类（原 harness.py）
"""

from agent_test0.memory.manager import (
    MemoryManager,
    get_redis_or_fallback,
)

__all__ = [
    "MemoryManager",
    "get_redis_or_fallback",
]
