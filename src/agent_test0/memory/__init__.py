# agent_test0/memory/__init__.py
"""
记忆系统包。

主要 export：
    from agent_test0.memory import MemoryManager, get_redis_or_fallback
    from agent_test0.memory import InMemoryFallback, ToolCacheManager

内部模块：
    manager.py —— MemoryManager / ToolCacheManager / InMemoryFallback 等
"""

from agent_test0.memory.manager import (
    MemoryManager,
    get_redis_or_fallback,
    InMemoryFallback,
    ToolCacheManager,
)

__all__ = [
    "MemoryManager",
    "get_redis_or_fallback",
    "InMemoryFallback",
    "ToolCacheManager",
]
