# agent_test0/workflow/trace.py
"""
轻量计时 trace：定位单轮 Flow 里哪个环节耗时最多、哪个环节被反复调用。

用法：
    from agent_test0.workflow.trace import timed, reset, report
    with timed("Planner"):
        ...
    report()  # 打印按总耗时排序的汇总表

为什么是模块级 dict 而不是挂在 flow 上：
    记忆蒸馏（manager.convert_*）在 run_for_user 里调用，不经过 flow；
    统一用模块级收集，一处 report 即可看到全貌。单轮测试场景是单线程顺序执行，
    无并发问题。飞书 bot 多线程场景下若需要可改 thread-local，当前够用。
"""

import time
from collections import defaultdict

# label -> {"count", "total", "samples"}
_calls = defaultdict(lambda: {"count": 0, "total": 0.0, "samples": []})


class timed:
    """上下文管理器：with timed("Planner"): ..."""
    def __init__(self, label: str):
        self.label = label
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        dt = time.perf_counter() - self.t0
        v = _calls[self.label]
        v["count"] += 1
        v["total"] += dt
        v["samples"].append(dt)
        return False  # 不吞异常


def reset():
    _calls.clear()


def report():
    """打印按总耗时降序的汇总表。"""
    if not _calls:
        print("\n[Trace] 无计时记录")
        return
    print("\n" + "=" * 78)
    print("[Trace] 耗时汇总（按总耗时降序）")
    print("=" * 78)
    rows = sorted(_calls.items(), key=lambda x: -x[1]["total"])
    total = sum(v["total"] for v in _calls.values())
    print(f"{'环节':<42} {'次数':>6} {'总耗时(s)':>10} {'占比':>8}")
    print("-" * 78)
    for label, v in rows:
        pct = v["total"] / total * 100 if total else 0
        print(f"{label:<42} {v['count']:>6} {v['total']:>10.2f} {pct:>7.1f}%")
    print("-" * 78)
    print(f"{'合计':<42} {'':>6} {total:>10.2f}")
    print("=" * 78)
