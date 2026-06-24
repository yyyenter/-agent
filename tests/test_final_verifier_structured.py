#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FinalVerifier 结构化 + 确定性规则 测试 (P1.2, 不调用 LLM)

覆盖:
  1. _check_deterministic_rules R1: 没 completed step → fail
  2. _check_deterministic_rules R2: 有 failed step → fail
  3. _check_deterministic_rules R3: completed 但没 data → fail
  4. _check_deterministic_rules pass: 正常 plan
  5. run_final_verifier 写 structured_plan 到 state
  6. run_final_verifier 规则不通过直接 replan, 不调 LLM
  7. generate_final_report 优先 structured_plan 输入
  8. generate_final_report fallback 路径 (无 structured_plan)
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_test0.workflow.state import StepPlan, TravelState
from agent_test0.workflow.nodes import (
    _check_deterministic_rules, final_verifier_node, generate_final_report,
)
from agent_test0.workflow.structured import StructuredCallError


# ============================================================
# 1. R1: 没 completed step
# ============================================================

def test_rule_r1_no_completed():
    plan = {
        "steps": [
            {"index": 0, "status": "pending", "data": None},
        ],
        "failed_steps": [],
        "data_sources": [],
        "warnings": [],
    }
    failures = _check_deterministic_rules(plan)
    assert any("R1" in f for f in failures)
    print("[OK] rule_r1_no_completed")


# ============================================================
# 2. R2: 有 failed step
# ============================================================

def test_rule_r2_has_failed():
    plan = {
        "steps": [
            {"index": 0, "status": "completed", "data": {"x": 1}},
            {"index": 1, "status": "failed", "data": None},
        ],
        "failed_steps": [1],
        "data_sources": [],
        "warnings": [],
    }
    failures = _check_deterministic_rules(plan)
    assert any("R2" in f for f in failures)
    print("[OK] rule_r2_has_failed")


# ============================================================
# 3. R3: completed 但没 data
# ============================================================

def test_rule_r3_no_data():
    plan = {
        "steps": [
            {"index": 0, "status": "completed", "data": None},
            {"index": 1, "status": "completed", "data": ""},
        ],
        "failed_steps": [],
        "data_sources": [],
        "warnings": [],
    }
    failures = _check_deterministic_rules(plan)
    assert any("R3" in f for f in failures)
    print("[OK] rule_r3_no_data")


# ============================================================
# 4. pass: 正常 plan
# ============================================================

def test_rule_pass():
    plan = {
        "steps": [
            {"index": 0, "status": "completed", "data": {"温度": "23"}},
            {"index": 1, "status": "completed", "data": [{"name": "故宫"}]},
        ],
        "failed_steps": [],
        "data_sources": ["weather_tool", "Tavily Search"],
        "warnings": [],
    }
    failures = _check_deterministic_rules(plan)
    assert failures == [], f"应有空 failures, 实际: {failures}"
    print("[OK] rule_pass")


# ============================================================
# 5. run_final_verifier 写 structured_plan
# ============================================================

def test_run_final_verifier_writes_structured_plan():
    state = TravelState()
    state.location = "北京"
    state.message = "test"
    state.assumptions = []
    state.failed_steps_indices = []
    state.final_verifier_done = False
    s = StepPlan(index=0, description="weather", status="completed",
                 tools=["weather_tool"])
    s.result = {"温度": "23"}
    s.result_text = "[weather_tool] 23"
    state.steps = [s]
    state.current_step_index = 1  # 等于 len(steps), 触发 FinalVerifier

    flow = MagicMock()
    flow.state = state
    flow._check_ask_user_hook.return_value = False
    flow.notify = lambda x: None
    flow._set_ask_user_question = lambda x: None
    flow.max_replan_attempts = 3
    flow.step_retry_counts = {}
    flow.max_step_retries = 3

    # 让 LLM 调用的 load_task_prompt 抛错, 走默认 pass
    from agent_test0.workflow import nodes
    original_load = nodes.load_task_prompt
    def mock_load(*a, **kw):
        raise StructuredCallError("mock: no LLM in this test")
    nodes.load_task_prompt = mock_load
    # pass 分支会内部调 generate_final_report → zhipu_llm.call, mock 掉避免真实 LLM
    original_call = nodes.zhipu_llm.call
    nodes.zhipu_llm.call = MagicMock(side_effect=Exception("mock: no LLM"))
    try:
        verdict = final_verifier_node(flow.state, {})
    finally:
        nodes.load_task_prompt = original_load
        nodes.zhipu_llm.call = original_call

    # 规则通过 (R1/R2/R3 都 pass) → LLM 失败但默认 pass → generate_final_report 调用
    # (LLM 也失败 → fallback 到 structured_plan summary)
    assert state.structured_plan != {}, "structured_plan 应已写入"
    assert state.structured_plan["destination"] == "北京"
    assert len(state.structured_plan["steps"]) == 1
    assert state.structured_plan["steps"][0]["data"] == {"温度": "23"}
    print("[OK] run_final_verifier_writes_structured_plan")


# ============================================================
# 6. 规则不通过直接 replan, 不调 LLM
# ============================================================

def test_rule_fail_triggers_replan_no_llm():
    state = TravelState()
    state.location = "北京"
    state.message = "test"
    state.final_verifier_done = False
    state.failed_steps_indices = []
    # R3 触发: completed 但没 data
    s = StepPlan(index=0, description="weather", status="completed", tools=["weather_tool"])
    s.result = None
    s.result_text = ""
    state.steps = [s]
    state.current_step_index = 1

    flow = MagicMock()
    flow.state = state
    flow._check_ask_user_hook.return_value = False
    flow.notify = lambda x: None
    flow.max_replan_attempts = 3
    flow.step_retry_counts = {}
    flow.max_step_retries = 3
    flow.replan_count = 0  # 关键: replan 计数起点

    # load_task_prompt 不能被调用 (规则失败应直接走 replan)
    from agent_test0.workflow import nodes
    def fail_load(*a, **kw):
        raise AssertionError("规则失败时不应调 LLM prompt")
    original = nodes.load_task_prompt
    nodes.load_task_prompt = fail_load
    try:
        verdict = final_verifier_node(flow.state, {})
    finally:
        nodes.load_task_prompt = original

    # 新签名返回 state 增量 dict (不再返回 verdict 字符串); 规则失败由 replan_count 体现
    assert isinstance(verdict, dict), f"应返回 state 增量 dict, 实际: {type(verdict)}"
    assert state.structured_plan != {}, "即使规则失败, structured_plan 也应已组装 (用于 replan 输入)"
    # replan_count 应被增加
    assert state.replan_count >= 1
    print("[OK] rule_fail_triggers_replan_no_llm")


# ============================================================
# 7. generate_final_report 优先 structured_plan
# ============================================================

def test_generate_final_report_prefers_structured_plan():
    state = TravelState()
    state.location = "北京"
    state.message = "想去北京 3 天"
    state.assumptions = []
    state.structured_plan = {
        "destination": "北京",
        "user_query": "想去北京 3 天",
        "assumptions": [],
        "steps": [{"index": 0, "description": "weather", "tools_used": ["weather_tool"],
                    "data": {"温度": "23"}, "status": "completed"}],
        "data_sources": ["weather_tool"],
        "warnings": [],
        "failed_steps": [],
    }
    s = StepPlan(index=0, description="weather", status="completed", tools=["weather_tool"])
    s.result = {"温度": "23"}
    s.result_text = "should not be used"
    state.steps = [s]

    flow = MagicMock()
    flow.state = state

    # LLM 失败, 应 fallback 到 structured_plan summary
    from agent_test0.workflow import nodes
    original_call = nodes.zhipu_llm.call
    nodes.zhipu_llm.call = MagicMock(side_effect=Exception("mock: no LLM"))
    try:
        report = generate_final_report(flow.state)
    finally:
        nodes.zhipu_llm.call = original_call

    # fallback 应包含 destination / data_sources
    assert "北京" in report
    assert "weather_tool" in report
    print("[OK] generate_final_report_prefers_structured_plan")


# ============================================================
# 8. fallback 路径 (无 structured_plan)
# ============================================================

def test_generate_final_report_fallback():
    state = TravelState()
    state.location = "北京"
    state.message = "test"
    state.assumptions = []
    state.structured_plan = {}  # 空的
    s = StepPlan(index=0, description="weather", status="completed", tools=["weather_tool"])
    s.result = "legacy text result"
    s.result_text = "[weather_tool] legacy data"
    state.steps = [s]

    flow = MagicMock()
    flow.state = state

    from agent_test0.workflow import nodes
    original_call = nodes.zhipu_llm.call
    nodes.zhipu_llm.call = MagicMock(side_effect=Exception("mock: no LLM"))
    try:
        report = generate_final_report(flow.state)
    finally:
        nodes.zhipu_llm.call = original_call

    # fallback 应包含 result_text
    assert "legacy data" in report
    print("[OK] generate_final_report_fallback")


if __name__ == "__main__":
    test_rule_r1_no_completed()
    test_rule_r2_has_failed()
    test_rule_r3_no_data()
    test_rule_pass()
    test_run_final_verifier_writes_structured_plan()
    test_rule_fail_triggers_replan_no_llm()
    test_generate_final_report_prefers_structured_plan()
    test_generate_final_report_fallback()
    print("\nALL PASSED")
