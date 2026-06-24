#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Span-based trace 测试 (Trace A + B 验证, 不调 LLM)

覆盖:
  1. Span 基本: span() 上下文 + 自动 start/end/duration/status
  2. Span 父子层级: with span() 嵌套 → parent_id 正确
  3. Span 数据属性: set_output / set_status
  4. 错误捕获: span 内抛异常 → status=error
  5. 树形打印: tree() 输出格式
  6. JSON dump: dump_json 写盘 + 字段完整
  7. 兼容旧 timed(): with timed("label") 仍能跑
  8. 静默 CrewAI: quiet_crewai() 不抛
  9. 线程隔离: 不同线程的 span 互不干扰
  10. 复杂 attr: dict / list / 长字符串 不被打印撑爆
"""

import sys
import json
import time
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_test0.workflow.trace import (
    Span, Tracer, span, timed, reset, report, dump_json, get_all_spans, quiet_crewai,
)


# ============================================================
# 1. Span 基本
# ============================================================

def test_span_basic():
    reset()
    with span("foo"):
        time.sleep(0.01)
    spans = get_all_spans()
    assert len(spans) == 1
    s = spans[0]
    assert s.name == "foo"
    assert s.start > 0
    assert s.end is not None
    assert s.duration >= 0.01
    assert s.status == "ok"
    assert s.parent_id is None
    print("[OK] span_basic")


# ============================================================
# 2. Span 父子层级
# ============================================================

def test_span_parent_child():
    reset()
    with span("outer") as outer:
        with span("inner1") as inner1:
            pass
        with span("inner2") as inner2:
            pass
    spans = get_all_spans()
    assert len(spans) == 3
    assert outer.children == [inner1, inner2]
    assert inner1.parent_id == outer.span_id
    assert inner2.parent_id == outer.span_id
    assert outer.parent_id is None
    print("[OK] span_parent_child")


# ============================================================
# 3. Span 数据属性
# ============================================================

def test_span_attrs_input_output():
    reset()
    with span("search", query="北京天气", top_k=5) as s:
        s.set_output(results=["故宫", "颐和园"], count=2)
    attrs = get_all_spans()[0].attrs
    assert attrs["query"] == "北京天气"
    assert attrs["top_k"] == 5
    assert attrs["out.results"] == ["故宫", "颐和园"]
    assert attrs["out.count"] == 2
    print("[OK] span_attrs_input_output")


# ============================================================
# 4. 错误捕获
# ============================================================

def test_span_error_captured():
    reset()
    try:
        with span("do_thing"):
            raise ValueError("something went wrong")
    except ValueError:
        pass
    s = get_all_spans()[0]
    assert s.status == "error"
    assert "something went wrong" in s.attrs.get("error", "")
    assert s.attrs.get("error_type") == "ValueError"
    print("[OK] span_error_captured")


# ============================================================
# 5. 树形打印
# ============================================================

def test_tree_print():
    reset()
    with span("root", user="alice") as r:
        with span("step1", idx=0) as s1:
            s1.set_output(result="ok")
        with span("step2", idx=1) as s2:
            with span("sub-step"):
                pass
    tree = get_all_spans()[0].to_dict()  # 先验证 dump
    # 然后验证 tree 文本
    from agent_test0.workflow.trace import _get_tracer
    tree_text = _get_tracer().tree()
    assert "root" in tree_text
    assert "step1" in tree_text
    assert "step2" in tree_text
    assert "sub-step" in tree_text
    assert "in: user=" in tree_text
    assert "out: result=" in tree_text
    assert "总耗时" in tree_text
    print("[OK] tree_print")


# ============================================================
# 6. JSON dump
# ============================================================

def test_dump_json():
    reset()
    with span("op", x=1) as s:
        s.set_output(y=2)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        path = f.name
    try:
        dump_json(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "root_spans" in data
        assert data["span_count"] == 1
        assert data["root_spans"][0]["name"] == "op"
        assert data["root_spans"][0]["attrs"]["x"] == 1
        assert data["root_spans"][0]["attrs"]["out.y"] == 2
        assert data["root_spans"][0]["status"] == "ok"
    finally:
        Path(path).unlink(missing_ok=True)
    print("[OK] dump_json")


# ============================================================
# 7. 兼容旧 timed
# ============================================================

def test_backward_compat_timed():
    reset()
    with timed("LegacyLabel") as s:
        time.sleep(0.005)
    spans = get_all_spans()
    assert len(spans) == 1
    assert spans[0].name == "LegacyLabel"
    assert spans[0].status == "ok"
    print("[OK] backward_compat_timed")


# ============================================================
# 8. 静默 CrewAI
# ============================================================

def test_quiet_crewai_does_not_throw():
    quiet_crewai()  # 不传参数, 不应抛
    # 再调一次幂等
    quiet_crewai()
    import os
    assert os.environ.get("CREWAI_TRACING_ENABLED") == "false"
    import logging
    assert logging.getLogger("crewai").level >= logging.WARNING
    print("[OK] quiet_crewai_does_not_throw")


# ============================================================
# 9. 线程隔离
# ============================================================

def test_thread_isolation():
    reset()
    results = {}

    def worker(name, delay):
        with span(f"worker_{name}") as s:
            time.sleep(delay)
            s.set_output(thread_name=name)
        results[name] = len(get_all_spans())

    t1 = threading.Thread(target=worker, args=("A", 0.01))
    t2 = threading.Thread(target=worker, args=("B", 0.02))
    t1.start(); t2.start()
    t1.join(); t2.join()

    # 每个线程独立 tracer, 各自只看到自己的 span
    assert results["A"] == 1
    assert results["B"] == 1

    # 主线程仍然是空的 (reset 过)
    assert len(get_all_spans()) == 0
    print("[OK] thread_isolation")


# ============================================================
# 10. 复杂 attr 截断
# ============================================================

def test_long_value_truncated():
    reset()
    long_str = "x" * 200
    long_list = list(range(100))
    big_dict = {f"k{i}": i for i in range(50)}
    with span("big") as s:
        s.set_output(text=long_str, items=long_list, mapping=big_dict)
    tree_text = get_all_spans()[0].to_dict()
    # attrs 完整保留
    assert len(tree_text["attrs"]["out.text"]) == 200
    # tree 打印会截断 (但我们这里不直接读 tree text, 验证 _format_value 函数)
    from agent_test0.workflow.trace import _format_value
    assert "..." in _format_value(long_str, max_len=20)
    assert "[100 items]" in _format_value(long_list, max_len=20)
    assert "{50 keys}" in _format_value(big_dict, max_len=20)
    print("[OK] long_value_truncated")


# ============================================================
# 11. report() 打印无内容时
# ============================================================

def test_report_empty():
    reset()
    # 强制重定向 stdout 抓 report() 输出
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report()
    out = buf.getvalue()
    assert "无 span 记录" in out
    print("[OK] report_empty")


# ============================================================
# 12. reset 清空
# ============================================================

def test_reset_clears():
    reset()
    with span("a"):
        pass
    with span("b"):
        pass
    assert len(get_all_spans()) == 2
    reset()
    assert len(get_all_spans()) == 0
    print("[OK] reset_clears")


if __name__ == "__main__":
    test_span_basic()
    test_span_parent_child()
    test_span_attrs_input_output()
    test_span_error_captured()
    test_tree_print()
    test_dump_json()
    test_backward_compat_timed()
    test_quiet_crewai_does_not_throw()
    test_thread_isolation()
    test_long_value_truncated()
    test_report_empty()
    test_reset_clears()
    print("\nALL PASSED")
