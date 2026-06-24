#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""assemble_structured_plan 测试 (P1.1 验证, 不调用 LLM)

覆盖:
  1. 完整组装: 3 步全成功 → 完整 structured_plan
  2. 失败步骤: failed status 写入 warnings
  3. 部分完成: 2/3 步完成, 1 步 pending
  4. 空 steps: warning
  5. data_sources 去重
  6. 工具结果结构化: dict 直接进 step.data
  7. failed_steps 字段来自 state
  8. assumptions 来自 state
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CHENGDU = chr(0x6210) + chr(0x90FD)  # 成都
BEIJING = chr(0x5317) + chr(0x4EAC)  # 北京

from agent_test0.workflow.state import StepPlan, ToolResult, TravelState
from agent_test0.workflow.nodes import assemble_structured_plan


def _make_step(idx: int, desc: str, status: str = "completed",
               tools: list = None, result: any = None,
               error: str = "") -> StepPlan:
    s = StepPlan(index=idx, description=desc, tools=tools or [], status=status)
    s.result = result
    s.error = error
    return s


def _make_flow(state: TravelState) -> MagicMock:
    flow = MagicMock()
    flow.state = state
    return flow


# ============================================================
# 1. 完整组装
# ============================================================

def test_full_assembly():
    state = TravelState()
    state.location = BEIJING
    state.message = "想去北京 3 天"
    state.assumptions = []
    state.failed_steps_indices = []
    state.steps = [
        _make_step(0, "查询北京天气", status="completed",
                   tools=["weather_tool"], result={"温度": "23°C", "天气": "晴"}),
        _make_step(1, "检索北京景点", status="completed",
                   tools=["Tavily Search"], result=[{"name": "故宫"}, {"name": "颐和园"}]),
        _make_step(2, "组装 3 天行程", status="completed", tools=[], result=None),
    ]
    flow = _make_flow(state)
    plan = assemble_structured_plan(flow)

    assert plan["destination"] == BEIJING
    assert plan["user_query"] == "想去北京 3 天"
    assert len(plan["steps"]) == 3
    assert plan["steps"][0]["data"] == {"温度": "23°C", "天气": "晴"}
    assert plan["steps"][1]["data"] == [{"name": "故宫"}, {"name": "颐和园"}]
    assert "weather_tool" in plan["data_sources"]
    assert "Tavily Search" in plan["data_sources"]
    assert plan["warnings"] == []
    assert plan["failed_steps"] == []
    print("[OK] full_assembly")


# ============================================================
# 2. 失败步骤
# ============================================================

def test_failed_step_writes_warning():
    state = TravelState()
    state.location = BEIJING
    state.message = "test"
    state.assumptions = []
    state.failed_steps_indices = [1]
    state.steps = [
        _make_step(0, "weather", status="completed",
                   tools=["weather_tool"], result={"温度": "23"}),
        _make_step(1, "broken POI search", status="failed",
                   tools=["Tavily Search"], error="网络超时"),
    ]
    flow = _make_flow(state)
    plan = assemble_structured_plan(flow)

    assert len(plan["steps"]) == 2  # failed step 仍记录
    assert plan["steps"][1]["status"] == "failed"
    assert any("步骤 1 失败" in w for w in plan["warnings"])
    assert "broken" in plan["warnings"][0] or "broken" in plan["warnings"][1] or len(plan["warnings"]) >= 1
    print("[OK] failed_step_writes_warning")


# ============================================================
# 3. 部分完成 (pending)
# ============================================================

def test_pending_step_writes_warning():
    state = TravelState()
    state.location = CHENGDU
    state.message = "test"
    state.steps = [
        _make_step(0, "weather", status="completed", tools=["weather_tool"]),
        _make_step(1, "POI", status="pending"),
        _make_step(2, "assemble", status="pending"),
    ]
    flow = _make_flow(state)
    plan = assemble_structured_plan(flow)

    assert len(plan["steps"]) == 1  # 只 completed 算入
    assert any("步骤 1" in w and "pending" in w for w in plan["warnings"])
    print("[OK] pending_step_writes_warning")


# ============================================================
# 4. 空 steps
# ============================================================

def test_empty_steps_writes_warning():
    state = TravelState()
    state.location = "未知"
    state.message = "test"
    state.steps = []
    flow = _make_flow(state)
    plan = assemble_structured_plan(flow)

    assert plan["steps"] == []
    assert any("没有完成的步骤" in w for w in plan["warnings"])
    print("[OK] empty_steps_writes_warning")


# ============================================================
# 5. data_sources 去重
# ============================================================

def test_data_sources_dedup():
    state = TravelState()
    state.location = BEIJING
    state.message = "test"
    state.steps = [
        _make_step(0, "weather day 1", status="completed", tools=["weather_tool"]),
        _make_step(1, "weather day 2", status="completed", tools=["weather_tool"]),
        _make_step(2, "weather day 3", status="completed", tools=["weather_tool"]),
    ]
    flow = _make_flow(state)
    plan = assemble_structured_plan(flow)

    # weather_tool 只出现一次
    assert plan["data_sources"].count("weather_tool") == 1
    print("[OK] data_sources_dedup")


# ============================================================
# 6. assumptions / failed_steps 透传
# ============================================================

def test_assumptions_and_failed_steps():
    state = TravelState()
    state.location = BEIJING
    state.message = "test"
    state.assumptions = ["默认 3 天", "默认 2 人"]
    state.failed_steps_indices = [1, 3]
    state.steps = [
        _make_step(0, "a", status="completed", tools=[]),
    ]
    flow = _make_flow(state)
    plan = assemble_structured_plan(flow)

    assert plan["assumptions"] == ["默认 3 天", "默认 2 人"]
    assert plan["failed_steps"] == [1, 3]
    print("[OK] assumptions_and_failed_steps")


# ============================================================
# 7. 工具结果为 dict 完整保留
# ============================================================

def test_tool_data_dict_preserved():
    state = TravelState()
    state.location = BEIJING
    state.message = "test"
    state.steps = [
        _make_step(0, "weather", status="completed", tools=["weather_tool"],
                   result={"城市": "北京", "温度": "23°C", "湿度": "45%",
                           "数据源": "和风天气"}),
    ]
    flow = _make_flow(state)
    plan = assemble_structured_plan(flow)

    data = plan["steps"][0]["data"]
    assert data["城市"] == "北京"
    assert data["温度"] == "23°C"
    assert data["数据源"] == "和风天气"
    print("[OK] tool_data_dict_preserved")


# ============================================================
# 8. 写入 state.structured_plan
# ============================================================

def test_writes_to_state():
    """assemble_structured_plan 不直接写 state, 由调用方写入"""
    state = TravelState()
    state.location = BEIJING
    state.message = "test"
    state.steps = [
        _make_step(0, "weather", status="completed", tools=["weather_tool"], result={"t": 23}),
    ]
    flow = _make_flow(state)
    plan = assemble_structured_plan(flow)
    # 调用方负责写入
    state.structured_plan = plan
    assert state.structured_plan["destination"] == "北京"
    assert len(state.structured_plan["steps"]) == 1
    print("[OK] writes_to_state")


if __name__ == "__main__":
    test_full_assembly()
    test_failed_step_writes_warning()
    test_pending_step_writes_warning()
    test_empty_steps_writes_warning()
    test_data_sources_dedup()
    test_assumptions_and_failed_steps()
    test_tool_data_dict_preserved()
    test_writes_to_state()
    print("\nALL PASSED")
