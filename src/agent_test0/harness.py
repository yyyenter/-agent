# agent_test0/harness.py —— 兼容入口（薄壳）
"""
向后兼容 shim。

原 harness.py 已搬到 agent_test0/memory/manager.py，本文件仅 re-export，
保证以下旧代码 import 路径继续工作：

    from agent_test0.harness import MemoryManager
    from agent_test0.harness import get_redis_or_fallback

新代码请改用：
    from agent_test0.memory import MemoryManager, get_redis_or_fallback
"""

from agent_test0.memory.manager import *  # noqa: F401, F403
from agent_test0.memory.manager import (  # noqa: F401
    MemoryManager,
    get_redis_or_fallback,
)
