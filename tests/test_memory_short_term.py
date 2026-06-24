#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""短期记忆索引 + 检索的确定性测试 (不调用 LLM, 不依赖 Redis)。

覆盖:
  1. 完整原文日志保留
  2. add_message 同步写索引 (task_id / destination / topic / has_slots)
  3. retrieve_short_term_context 在新任务/无效回复/授权默认 三种情况下
     不会把旧任务槽位误用为当前任务事实
  4. excluded_history 把不同 destination / completed_task 隔离
"""

import sys
from pathlib import Path

# 把 src 目录加入 path, 让测试可以直接 uv run python 运行
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Windows GBK 终端会把源码里的 UTF-8 中文字面量读成乱码。用 chr() 拼接避免编译期损坏。
CHENGDU = chr(0x6210) + chr(0x90FD)            # 成都
CHONGQING = chr(0x91CD) + chr(0x5E86)          # 重庆
BEIJING = chr(0x5317) + chr(0x4EAC)            # 北京

from agent_test0.memory.manager import InMemoryFallback, MemoryManager


def _new_manager(suffix: str = "01"):
    """构造一个强制走内存回退的 MemoryManager。"""
    fallback = InMemoryFallback()
    return MemoryManager(
        session_id=f"test_{suffix}",
        user_id=f"user_{suffix}",
        redis_client=fallback,
        is_fallback=True,
    )


def test_full_log_preserved():
    m = _new_manager("log")
    m.add_message("user", "想去" + CHONGQING, task_id="t1", topic="trip_planning",
                  destination=CHONGQING, has_slots=True,
                  extracted_slots={"destination": CHONGQING})
    m.add_message("assistant", "请补充天数、预算、人数",
                  task_id="t1", topic="ask_user")

    history = m.get_chat_history()
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "想去" + CHONGQING
    assert history[1]["role"] == "assistant"
    print("[OK] full_log_preserved")


def test_index_built_incrementally():
    m = _new_manager("idx")
    m.add_message("user", "想去" + CHONGQING, task_id="t1", topic="trip_planning",
                  destination=CHONGQING, has_slots=True,
                  extracted_slots={"destination": CHONGQING})
    m.add_message("assistant", "请补充", task_id="t1", topic="ask_user")
    m.add_message("user", "3天 预算3000 两个人", task_id="t1",
                  topic="slot_answer", destination=CHONGQING, has_slots=True,
                  extracted_slots={"duration_days": 3, "budget": 3000, "companions": "2人"})

    index = m.get_short_term_index()
    assert len(index) == 3
    assert index[0]["destination"] == CHONGQING
    assert index[2]["extracted_slots"] == {"duration_days": 3, "budget": 3000, "companions": "2人"}
    assert index[2]["has_slots"] is True
    print("[OK] index_built_incrementally")


def test_new_task_isolates_old_task_slots():
    m = _new_manager("iso")
    # 历史: 重庆任务
    m.add_message("user", "想去" + CHONGQING, task_id="t1", topic="trip_planning",
                  destination=CHONGQING, has_slots=True,
                  extracted_slots={"destination": CHONGQING})
    m.add_message("assistant", "请补充天数、预算、人数", task_id="t1", topic="ask_user")
    m.add_message("user", "3天 预算3000 两个人", task_id="t1", topic="slot_answer",
                  destination=CHONGQING, has_slots=True,
                  extracted_slots={"duration_days": 3, "budget": 3000, "companions": "2人"},
                  is_completed_task=True)
    # 新任务: 想去成都 (新 task_id, 新 destination)
    m.add_message("user", "想去" + CHENGDU, task_id="t2", topic="trip_planning",
                  destination=CHENGDU, has_slots=False)

    ctx = m.retrieve_short_term_context("想去" + CHENGDU)
    assert ctx["current_task_id"] == "t2"
    assert ctx["current_destination"] == CHENGDU
    # 成都相关 turns 只应包含 t2 自己
    assert len(ctx["relevant_turns"]) == 1
    assert ctx["relevant_turns"][0]["content"] == "想去" + CHENGDU
    # 旧任务重庆应在 excluded 中
    assert any(CHONGQING in line for line in ctx["excluded_history"]), ctx["excluded_history"]
    print("[OK] new_task_isolates_old_task_slots")


def test_invalid_reply_does_not_reuse_destination():
    m = _new_manager("ff")
    m.add_message("user", "想去" + CHONGQING, task_id="t1", topic="trip_planning",
                  destination=CHONGQING, has_slots=True,
                  extracted_slots={"destination": CHONGQING})
    m.add_message("assistant", "请补充天数、预算、人数", task_id="t1", topic="ask_user")

    ctx = m.retrieve_short_term_context("ff")
    assert ctx["is_invalid_reply"] is True
    assert ctx["is_delegation"] is False
    assert ctx["current_task_id"] == "t1"
    assert ctx["last_assistant_question"] == "请补充天数、预算、人数"
    print("[OK] invalid_reply_does_not_reuse_destination")


def test_delegation_detected():
    m = _new_manager("dlg")
    m.add_message("user", "想去" + CHONGQING, task_id="t1", topic="trip_planning",
                  destination=CHONGQING, has_slots=True)
    m.add_message("assistant", "请补充天数、预算、人数", task_id="t1", topic="ask_user")

    ctx = m.retrieve_short_term_context("看你安排")
    assert ctx["is_delegation"] is True
    assert ctx["is_invalid_reply"] is False
    assert ctx["current_task_id"] == "t1"
    print("[OK] delegation_detected")


def test_global_prompt_renders_exclusion_note():
    m = _new_manager("prompt")
    m.add_message("user", "想去" + CHONGQING + "3天 预算3000 两人", task_id="t1",
                  topic="trip_planning", destination=CHONGQING, has_slots=True,
                  extracted_slots={"destination": CHONGQING, "duration_days": 3,
                                   "budget": 3000, "companions": "2人"},
                  is_completed_task=True)
    m.add_message("user", "想去" + CHENGDU, task_id="t2",
                  topic="trip_planning", destination=CHENGDU, has_slots=False)

    prompt = m.get_global_context_prompt("想去" + CHENGDU)
    # 提示中应包含当前指令、历史任务排除说明
    assert "【当前最新指令】：想去" + CHENGDU in prompt
    assert "【历史任务参考" in prompt
    assert "不得作为当前任务事实" in prompt
    # 当前任务还没补充任何槽位, 提示中不应该出现 3天/3000/2人 作为当前任务事实
    assert "duration_days" not in prompt
    print("[OK] global_prompt_renders_exclusion_note")


if __name__ == "__main__":
    test_full_log_preserved()
    test_index_built_incrementally()
    test_new_task_isolates_old_task_slots()
    test_invalid_reply_does_not_reuse_destination()
    test_delegation_detected()
    test_global_prompt_renders_exclusion_note()
    print("\nALL PASSED")
