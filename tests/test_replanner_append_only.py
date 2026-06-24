#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PartialReplanner append-only 测试 (P2.1 验证, 不调用 LLM)

覆盖:
  1. 已完成 steps 全部保留 (不被丢弃)
  2. 失败 steps 保留为 status="failed" 历史
  3. 新 steps 追加到末尾, current_step_index 指向第一个新 step
  4. LLM 输出 index 与已有冲突时, 自动修正
  5. 兼容旧 prompt 输出 (new_coarse_steps) 仍能 append
  6. 空 new_steps_raw 不抛异常
  7. failed_steps_indices 被清空 (重置)
  8. replan_count 正确累加
  9. dependencies 引用已失败步骤也合法 (不要求只引用 completed)
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_test0.workflow.state import StepPlan, TravelState, ReplanOutput
from agent_test0.workflow.structured import StructuredCallError
from agent_test0.workflow import nodes


def _make_step(idx: int, desc: str, status: str = "completed",
               deps: list = None, result_text: str = "",
               error: str = "") -> StepPlan:
    s = StepPlan(
        index=idx, description=desc, dependencies=deps or [],
        status=status,
    )
    s.result_text = result_text
    s.error = error
    return s


def _make_flow(state: TravelState) -> MagicMock:
    flow = MagicMock()
    flow.state = state
    flow._check_ask_user_hook.return_value = False
    flow.notify = lambda x: None
    flow.max_replan_attempts = 3
    return flow


# ============================================================
# 1. 保留全部已完成 steps
# ============================================================

def test_keeps_all_completed_steps():
    state = TravelState()
    state.steps = [
        _make_step(0, "weather", status="completed", result_text="[weather] 23"),
        _make_step(1, "poi search", status="completed", result_text="[search] 故宫"),
    ]
    state.failed_steps_indices = [2]
    state.steps.append(_make_step(2, "broken", status="failed", error="timeout"))
    state.current_step_index = 0
    state.replan_count = 0

    flow = _make_flow(state)

    # 模拟 LLM 返回 1 个新追加 step
    def mock_call(name, prompt, schema):
        return ReplanOutput(
            reason="补救",
            preserved_steps=[0, 1],
            original_remaining_steps=[2],
            new_appended_steps=[
                StepPlan(index=3, description="重新搜索景点",
                         dependencies=[0, 2], tools=["Tavily Search"]),
            ],
            replan_retry_count=1,
        )

    original_call = nodes.call_structured
    nodes.call_structured = mock_call
    try:
        nodes.run_partial_replanner(flow, {"reason": "test fail"})
    finally:
        nodes.call_structured = original_call

    # 关键: 原有 3 个 step 全部保留
    assert len(flow.state.steps) == 4
    assert flow.state.steps[0].status == "completed"
    assert flow.state.steps[1].status == "completed"
    assert flow.state.steps[2].status == "failed"  # 失败步骤保留
    # 新追加的 step
    assert flow.state.steps[3].description == "重新搜索景点"
    assert flow.state.steps[3].dependencies == [0, 2]  # 含失败依赖
    # current_step_index 指向第一个新 step
    assert flow.state.current_step_index == 3
    print("[OK] keeps_all_completed_steps")


# ============================================================
# 2. 失败 steps 保留为历史
# ============================================================

def test_failed_steps_kept_as_history():
    state = TravelState()
    state.steps = [
        _make_step(0, "ok", status="completed"),
        _make_step(1, "ok", status="completed"),
        _make_step(2, "fail", status="failed", error="network"),
        _make_step(3, "fail2", status="failed", error="timeout"),
    ]
    state.failed_steps_indices = [2, 3]
    state.current_step_index = 0
    state.replan_count = 0

    flow = _make_flow(state)

    def mock_call(name, prompt, schema):
        return ReplanOutput(
            reason="补救",
            new_appended_steps=[
                StepPlan(index=4, description="fresh start",
                         dependencies=[0, 1], tools=[]),
            ],
        )

    original_call = nodes.call_structured
    nodes.call_structured = mock_call
    try:
        nodes.run_partial_replanner(flow, {"reason": "x"})
    finally:
        nodes.call_structured = original_call

    # 4 个原有 steps 全部保留
    assert len(flow.state.steps) == 5
    # 失败 step 仍是 failed
    assert flow.state.steps[2].status == "failed"
    assert flow.state.steps[3].status == "failed"
    # failed_steps_indices 被清空 (供下次重规划重新登记)
    assert flow.state.failed_steps_indices == []
    print("[OK] failed_steps_kept_as_history")


# ============================================================
# 3. LLM 输出 index 冲突, 自动修正
# ============================================================

def test_index_collision_auto_fixed():
    state = TravelState()
    state.steps = [
        _make_step(0, "a", status="completed"),
        _make_step(1, "b", status="completed"),
    ]
    state.failed_steps_indices = [2]
    state.steps.append(_make_step(2, "fail", status="failed"))
    state.current_step_index = 0
    state.replan_count = 0

    flow = _make_flow(state)

    def mock_call(name, prompt, schema):
        # LLM 给出重复 index (试图用 0/1/2)
        return ReplanOutput(
            new_appended_steps=[
                StepPlan(index=0, description="x", dependencies=[]),  # 冲突
                StepPlan(index=1, description="y", dependencies=[]),  # 冲突
                StepPlan(index=3, description="z", dependencies=[]),  # 合法
            ],
        )

    original_call = nodes.call_structured
    nodes.call_structured = mock_call
    try:
        nodes.run_partial_replanner(flow, {"reason": "x"})
    finally:
        nodes.call_structured = original_call

    # 修正后: 3 个新 step 的 index 应该是 3, 4, 5
    assert flow.state.steps[3].index == 3
    assert flow.state.steps[4].index == 4
    assert flow.state.steps[5].index == 5
    assert flow.state.current_step_index == 3
    print("[OK] index_collision_auto_fixed")


# ============================================================
# 4. 兼容旧 prompt (new_coarse_steps)
# ============================================================

def test_legacy_new_coarse_steps_still_works():
    state = TravelState()
    state.steps = [
        _make_step(0, "a", status="completed"),
        _make_step(1, "b", status="failed", error="x"),
    ]
    state.failed_steps_indices = [1]
    state.current_step_index = 0
    state.replan_count = 0

    flow = _make_flow(state)

    # 旧 prompt 输出 (用 new_coarse_steps, index 从失败点起)
    def mock_call(name, prompt, schema):
        return ReplanOutput(
            new_appended_steps=[],  # 空
            new_coarse_steps=[      # 旧字段
                StepPlan(index=2, description="legacy 补救", dependencies=[0], tools=[]),
            ],
        )

    original_call = nodes.call_structured
    nodes.call_structured = mock_call
    try:
        nodes.run_partial_replanner(flow, {"reason": "x"})
    finally:
        nodes.call_structured = original_call

    # 兼容路径也能 append
    assert len(flow.state.steps) == 3
    assert flow.state.steps[2].description == "legacy 补救"
    print("[OK] legacy_new_coarse_steps_still_works")


# ============================================================
# 5. LLM 失败时不抛异常
# ============================================================

def test_llm_failure_does_not_crash():
    state = TravelState()
    state.steps = [
        _make_step(0, "a", status="completed"),
        _make_step(1, "b", status="failed", error="x"),
    ]
    state.failed_steps_indices = [1]
    state.current_step_index = 0
    state.replan_count = 0

    flow = _make_flow(state)

    def mock_call(name, prompt, schema):
        raise StructuredCallError("mock: LLM down")

    original_call = nodes.call_structured
    nodes.call_structured = mock_call
    try:
        nodes.run_partial_replanner(flow, {"reason": "x"})
    finally:
        nodes.call_structured = original_call

    # 状态保留 (没有追加任何新 step)
    assert len(flow.state.steps) == 2
    assert flow.state.failed_steps_indices == []
    print("[OK] llm_failure_does_not_crash")


# ============================================================
# 6. replan_count 累加
# ============================================================

def test_replan_count_increments():
    state = TravelState()
    state.steps = [
        _make_step(0, "a", status="failed", error="x"),
    ]
    state.failed_steps_indices = [0]
    state.current_step_index = 0
    state.replan_count = 0

    flow = _make_flow(state)

    def mock_call(name, prompt, schema):
        raise StructuredCallError("mock")

    original_call = nodes.call_structured
    nodes.call_structured = mock_call
    try:
        nodes.run_partial_replanner(flow, {"reason": "x"})
    finally:
        nodes.call_structured = original_call

    assert state.replan_count == 1
    print("[OK] replan_count_increments")


# ============================================================
# 7. failed_steps_indices 被清空
# ============================================================

def test_failed_steps_indices_reset():
    state = TravelState()
    state.steps = [
        _make_step(0, "a", status="completed"),
        _make_step(1, "b", status="failed"),
    ]
    state.failed_steps_indices = [1]
    state.current_step_index = 0
    state.replan_count = 0

    flow = _make_flow(state)

    def mock_call(name, prompt, schema):
        return ReplanOutput(
            new_appended_steps=[
                StepPlan(index=2, description="补救", dependencies=[0]),
            ],
        )

    original_call = nodes.call_structured
    nodes.call_structured = mock_call
    try:
        nodes.run_partial_replanner(flow, {"reason": "x"})
    finally:
        nodes.call_structured = original_call

    # replan 处理完应清空 failed_steps_indices
    assert state.failed_steps_indices == []
    print("[OK] failed_steps_indices_reset")


# ============================================================
# 8. 多次 replan 累加
# ============================================================

def test_multiple_replans_accumulate():
    state = TravelState()
    state.steps = [_make_step(0, "a", status="completed")]
    state.failed_steps_indices = []
    state.current_step_index = 0
    state.replan_count = 0

    flow = _make_flow(state)

    def mock_call(name, prompt, schema):
        return ReplanOutput(
            new_appended_steps=[
                StepPlan(index=len(flow.state.steps), description="appended",
                         dependencies=[0]),
            ],
        )

    original_call = nodes.call_structured
    nodes.call_structured = mock_call
    try:
        # 第一次 replan
        state.failed_steps_indices = [0]
        state.steps[0].status = "failed"
        nodes.run_partial_replanner(flow, {"reason": "first fail"})
        # 第二次 replan (假设新追加的 step 又失败)
        last = state.steps[-1]
        last.status = "failed"
        state.failed_steps_indices = [last.index]
        nodes.run_partial_replanner(flow, {"reason": "second fail"})
    finally:
        nodes.call_structured = original_call

    # 2 次 replan, 每次都追加 1 个 step
    assert state.replan_count == 2
    # 1 (original) + 1 (first append) + 1 (second append) = 3
    assert len(state.steps) == 3
    print("[OK] multiple_replans_accumulate")


if __name__ == "__main__":
    test_keeps_all_completed_steps()
    test_failed_steps_kept_as_history()
    test_index_collision_auto_fixed()
    test_legacy_new_coarse_steps_still_works()
    test_llm_failure_does_not_crash()
    test_replan_count_increments()
    test_failed_steps_indices_reset()
    test_multiple_replans_accumulate()
    print("\nALL PASSED")
