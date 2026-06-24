#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TravelState 4 类字段 + MemoryManager 跨轮字段测试 (不调用 LLM)。

覆盖:
  1. TravelState 默认值: 所有 4 类字段都有默认值, 可直接 TravelState() 构造。
  2. TravelState 旧 session 兼容: 缺字段 dict 能 model_validate。
  3. TravelState 字段分组完整性: Process / Working / Business / Output 关键字段都在。
  4. MemoryManager.new_task_id 唯一性: 同一 session 多次调用, task_id 互不相同。
  5. MemoryManager.bind_to_state: memory 跨轮字段同步到 TravelState, 不覆盖已有非空值。
  6. MemoryManager.sync_from_state: state 字段回写到 memory, 为下一轮 run_for_user 准备。
  7. MemoryManager.mark_current_task_completed: 完成后索引中该 task 标记 is_completed_task=True。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CHENGDU = chr(0x6210) + chr(0x90FD)
CHONGQING = chr(0x91CD) + chr(0x5E86)

from agent_test0.workflow.state import TravelState
from agent_test0.memory.manager import InMemoryFallback, MemoryManager


# ============================================================
# 1. TravelState 默认值
# ============================================================

def test_travelstate_defaults():
    s = TravelState()
    # Process / Control
    assert s.current_node == ""
    assert s.loop_count == 0
    assert s.total_steps_counted == 0
    assert s.retry_count == 0
    assert s.replan_count == 0
    assert s.final_verifier_done is False
    assert s.is_interrupted is False
    assert s.is_done is False
    # Working Data
    assert s.steps == []
    assert s.current_step_index == 0
    assert s.step_results == []
    assert s.failed_steps_indices == []
    # Business Context
    assert s.message == ""
    assert s.current_task_id is None
    assert s.current_destination is None
    assert s.current_topic == "general"
    assert s.is_invalid_reply is False
    assert s.is_delegation is False
    # Final Output
    assert s.needs_user_input is False
    assert s.user_question == ""
    assert s.user_choice == ""
    assert s.final_report == ""
    assert s.assumptions == []
    assert s.warnings == []
    assert s.asked_fields == []
    print("[OK] travelstate_defaults")


# ============================================================
# 2. TravelState 旧 session 兼容
# ============================================================

def test_travelstate_legacy_compat():
    # 旧 session 持久化的 dict 只有部分字段, 不应因缺字段炸掉
    old = {"message": "hi", "user_id": "u1", "session_id": "s1"}
    s = TravelState.model_validate(old)
    assert s.message == "hi"
    assert s.user_id == "u1"
    assert s.session_id == "s1"
    # 新字段都有默认值
    assert s.loop_count == 0
    assert s.current_destination is None
    assert s.current_topic == "general"
    assert s.warnings == []
    print("[OK] travelstate_legacy_compat")


# ============================================================
# 3. 字段分组完整性
# ============================================================

def test_field_groups_complete():
    s = TravelState()
    process_fields = ["current_node", "loop_count", "total_steps_counted",
                      "retry_count", "replan_count", "final_verifier_done",
                      "is_interrupted", "is_done"]
    working_fields = ["steps", "current_step_index", "step_results",
                      "failed_steps_indices", "last_validation"]
    business_fields = ["message", "user_id", "session_id", "focus",
                       "is_complex", "simple_answer", "location",
                       "current_task_id", "current_destination",
                       "current_topic", "is_invalid_reply", "is_delegation"]
    output_fields = ["needs_user_input", "user_question", "user_choice",
                     "skip_remaining_steps", "final_report", "assumptions",
                     "warnings", "asked_fields"]

    for f in process_fields:
        assert hasattr(s, f), f"missing Process field: {f}"
    for f in working_fields:
        assert hasattr(s, f), f"missing Working field: {f}"
    for f in business_fields:
        assert hasattr(s, f), f"missing Business field: {f}"
    for f in output_fields:
        assert hasattr(s, f), f"missing Output field: {f}"
    print("[OK] field_groups_complete")


# ============================================================
# 4. MemoryManager.new_task_id 唯一性
# ============================================================

def test_new_task_id_unique():
    m = MemoryManager("sess_001", "u1", InMemoryFallback(), True)
    ids = [m.new_task_id() for _ in range(5)]
    assert len(set(ids)) == 5, f"task_id 不唯一: {ids}"
    for i, tid in enumerate(ids, 1):
        assert tid.startswith("t_"), tid
        assert tid.endswith(f"_{i}"), tid
    print("[OK] new_task_id_unique")


# ============================================================
# 5. bind_to_state: 跨轮字段同步, 不覆盖非空值
# ============================================================

def test_bind_to_state_does_not_overwrite():
    m = MemoryManager("sess_002", "u1", InMemoryFallback(), True)
    m.current_task_id = "t_old_1"
    m.current_destination = CHONGQING
    m.current_topic = "trip_planning"

    s = TravelState()
    m.bind_to_state(s)
    assert s.current_task_id == "t_old_1"
    assert s.current_destination == CHONGQING
    assert s.current_topic == "trip_planning"

    # state 已有非空值时, 不应被 memory 覆盖
    s2 = TravelState()
    s2.current_destination = CHENGDU
    m.bind_to_state(s2)
    assert s2.current_destination == CHENGDU, "bind 不应覆盖 state 已有非空值"
    print("[OK] bind_to_state_does_not_overwrite")


# ============================================================
# 6. sync_from_state: state 字段回写到 memory
# ============================================================

def test_sync_from_state():
    m = MemoryManager("sess_003", "u1", InMemoryFallback(), True)
    assert m.current_destination is None
    assert m.current_task_id is None

    s = TravelState()
    s.current_task_id = "t_new_1"
    s.current_destination = CHENGDU
    s.current_topic = "trip_planning"

    m.sync_from_state(s)
    assert m.current_task_id == "t_new_1"
    assert m.current_destination == CHENGDU
    assert m.current_topic == "trip_planning"
    print("[OK] sync_from_state")


# ============================================================
# 7. mark_current_task_completed: 索引标记 + 清空指针
# ============================================================

def test_mark_current_task_completed():
    m = MemoryManager("sess_004", "u1", InMemoryFallback(), True)
    m.current_task_id = "t_a_1"
    m.add_message("user", "想去" + CHONGQING,
                  task_id="t_a_1", topic="trip_planning",
                  destination=CHONGQING, has_slots=True)
    m.add_message("assistant", "请补充天数、预算、人数",
                  task_id="t_a_1", topic="ask_user")
    m.add_message("user", "3天 预算3000 两人",
                  task_id="t_a_1", topic="slot_answer",
                  destination=CHONGQING, has_slots=True,
                  extracted_slots={"duration_days": 3, "budget": 3000,
                                    "companions": "2人"})

    m.mark_current_task_completed()
    index = m.get_short_term_index()
    user_entries = [e for e in index if e.get("role") == "user"
                    and e.get("task_id") == "t_a_1"]
    assert user_entries, "t_a_1 的 user 条目不见了"
    assert user_entries[0]["is_completed_task"] is True
    assert m.current_task_id is None
    print("[OK] mark_current_task_completed")


# ============================================================
# 8. 完整跨轮链路: bind → 节点读 → sync → mark_completed
# ============================================================

def test_full_multi_turn_lifecycle():
    m = MemoryManager("sess_005", "u1", InMemoryFallback(), True)
    # 第一轮: 重庆
    s1 = TravelState()
    m.bind_to_state(s1)
    assert s1.current_destination is None  # 首次

    s1.current_task_id = "t_5_1"
    s1.current_destination = CHONGQING
    s1.current_topic = "trip_planning"

    m.add_message("user", "想去" + CHONGQING,
                  task_id=s1.current_task_id,
                  destination=s1.current_destination,
                  topic="trip_planning", has_slots=True)

    m.sync_from_state(s1)
    assert m.current_destination == CHONGQING

    m.mark_current_task_completed()
    assert m.current_task_id is None

    # 第二轮: 成都
    s2 = TravelState()
    m.bind_to_state(s2)
    assert s2.current_destination is None, \
        "新轮 state 不应继承上一轮已完成的 destination"

    s2.current_task_id = m.new_task_id()
    s2.current_destination = CHENGDU
    m.add_message("user", "想去" + CHENGDU,
                  task_id=s2.current_task_id,
                  destination=s2.current_destination,
                  topic="trip_planning", has_slots=False)

    ctx = m.retrieve_short_term_context("想去" + CHENGDU)
    assert ctx["current_destination"] == CHENGDU
    # new_task_id() 用 session 后缀生成, 这里只需要验证是 t_ 开头 + 包含 sess_005
    assert ctx["current_task_id"].startswith("t_"), \
        f"current_task_id 格式错: {ctx['current_task_id']}"
    assert "sess_005" in ctx["current_task_id"], \
        f"current_task_id 应含 session 后缀: {ctx['current_task_id']}"
    assert any(CHONGQING in line for line in ctx["excluded_history"]), \
        "已完成的重庆应进入 excluded_history"
    print("[OK] full_multi_turn_lifecycle")


if __name__ == "__main__":
    test_travelstate_defaults()
    test_travelstate_legacy_compat()
    test_field_groups_complete()
    test_new_task_id_unique()
    test_bind_to_state_does_not_overwrite()
    test_sync_from_state()
    test_mark_current_task_completed()
    test_full_multi_turn_lifecycle()
    print("\nALL PASSED")
