#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Planner 信息不足护栏的确定性测试（不调用 LLM）。"""

from agent_test0.workflow.flow import TravelWorkflow
from agent_test0.workflow import nodes
from agent_test0.workflow.nodes import _enforce_first_turn_question


def test_first_turn_must_ask():
    nodes._llm_followup_question = lambda state, plan_data, reason, missing=None: "LLM追问：请补充计划玩几天、预算和几位出行。"
    flow = TravelWorkflow()
    flow.state.message = "想去重庆"
    flow.state.focus = "【近期对话上下文（原文）】：\nuser: 想去重庆\n【当前最新指令】：想去重庆"
    plan_data = {
        "location": "重庆",
        "focus": "重庆行程规划",
        "assumptions": ["用户未指定天数，模型试图默认", "未指定预算，模型试图默认", "未指定人数，模型试图默认"],
        "steps": [{"index": 0, "description": "查询重庆天气", "dependencies": []}],
        "needs_user_input": False,
    }

    # 方案A1: _enforce_first_turn_question(state, plan_data) 接 state 而非 flow
    assert _enforce_first_turn_question(flow.state, plan_data) is True
    assert flow.state.needs_user_input is True
    assert "LLM追问" in flow.state.user_question
    assert "计划玩几天" in flow.state.user_question
    assert "预算" in flow.state.user_question
    assert "几位出行" in flow.state.user_question
    print("✅ first_turn_must_ask")


def test_weather_query_not_blocked():
    flow = TravelWorkflow()
    flow.state.message = "重庆天气"
    flow.state.focus = "【近期对话上下文】：\nuser: 重庆天气\n【当前最新指令】：重庆天气"
    plan_data = {
        "location": "重庆",
        "focus": "查询重庆天气",
        "assumptions": [],
        "steps": [{"index": 0, "description": "查询重庆天气", "dependencies": []}],
        "needs_user_input": False,
    }

    assert _enforce_first_turn_question(flow.state, plan_data) is False
    assert flow.state.needs_user_input is False
    print("✅ weather_query_not_blocked")


def test_previous_user_constraints_do_not_fill_new_turn():
    nodes._llm_followup_question = lambda state, plan_data, reason, missing=None: "LLM追问：成都这次计划玩几天、预算多少、几位出行？"
    flow = TravelWorkflow()
    flow.state.message = "想去成都"
    flow.state.focus = """【近期对话上下文（原文）】：
user: 想去重庆3天，预算3000，两个人
assistant: 已为您生成重庆行程。
user: 想去成都
【当前最新指令】：想去成都"""
    plan_data = {
        "location": "成都",
        "focus": "成都行程规划",
        "assumptions": ["模型试图沿用历史3天/3000/2人"],
        "steps": [{"index": 0, "description": "查询成都天气", "dependencies": []}],
        "needs_user_input": False,
    }

    assert _enforce_first_turn_question(flow.state, plan_data) is True
    assert flow.state.needs_user_input is True
    assert "LLM追问" in flow.state.user_question
    assert "成都" in flow.state.user_question
    print("✅ previous_user_constraints_do_not_fill_new_turn")


def test_invalid_reply_to_previous_question_does_not_reuse_location():
    nodes._llm_followup_question = lambda state, plan_data, reason, missing=None: "LLM追问：我没理解 ff，请补充有效的天数、预算和人数。"
    flow = TravelWorkflow()
    flow.state.message = "ff"
    flow.state.focus = """【近期对话上下文】：
user: 想去重庆
assistant: 好的，重庆行程没问题！为了规划得更贴合您的需求，请先补充：计划玩几天？大概预算是多少？几位出行？
user: ff
【当前最新指令】：ff"""
    plan_data = {
        "location": "重庆",
        "focus": "重庆行程规划",
        "assumptions": [],
        "steps": [{"index": 0, "description": "查询重庆天气", "dependencies": []}],
        "needs_user_input": False,
    }

    assert _enforce_first_turn_question(flow.state, plan_data) is True
    assert flow.state.needs_user_input is True
    assert "LLM追问" in flow.state.user_question
    assert "ff" in flow.state.user_question
    assert "重庆行程没问题" not in flow.state.user_question
    print("✅ invalid_reply_to_previous_question_does_not_reuse_location")


def test_second_turn_delegated_can_assume():
    flow = TravelWorkflow()
    flow.state.message = "看你安排"
    flow.state.focus = """【近期对话上下文】：
user: 想去重庆
assistant: 好的，重庆行程没问题！为了规划得更贴合您的需求，请先补充：计划玩几天？大概预算是多少？几位出行？
user: 看你安排
【当前最新指令】：看你安排"""
    plan_data = {
        "location": "重庆",
        "focus": "重庆行程规划",
        "assumptions": ["用户未指定天数，默认按 3 天规划", "未指定预算，按中等档位", "未指定人数，按 2 人"],
        "steps": [{"index": 0, "description": "查询重庆天气", "dependencies": []}],
        "needs_user_input": False,
    }

    assert _enforce_first_turn_question(flow.state, plan_data) is False
    assert flow.state.needs_user_input is False
    print("✅ second_turn_delegated_can_assume")


if __name__ == "__main__":
    test_first_turn_must_ask()
    test_weather_query_not_blocked()
    test_previous_user_constraints_do_not_fill_new_turn()
    test_invalid_reply_to_previous_question_does_not_reuse_location()
    test_second_turn_delegated_can_assume()
