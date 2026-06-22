#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Planner 信息不足护栏的确定性测试（不调用 LLM）。"""

from agent_test0.workflow.flow import TravelWorkflow
from agent_test0.workflow.nodes import _enforce_first_turn_question


def test_first_turn_must_ask():
    flow = TravelWorkflow()
    flow.state.message = "想去杭州"
    flow.state.focus = "【近期对话上下文】：\nuser: 想去杭州\n【当前最新指令】：想去杭州"
    plan_data = {
        "location": "杭州",
        "focus": "杭州行程规划",
        "assumptions": ["用户未指定天数，默认按 3 天规划", "未指定预算，按中等档位", "未指定人数，按 2 人"],
        "steps": [{"index": 0, "description": "查询杭州天气", "dependencies": []}],
        "needs_user_input": False,
    }

    assert _enforce_first_turn_question(flow, plan_data) is True
    assert flow.state.needs_user_input is True
    assert "计划玩几天" in flow.state.user_question
    assert "大概预算" in flow.state.user_question
    assert "几位出行" in flow.state.user_question
    print("✅ first_turn_must_ask")


def test_weather_query_not_blocked():
    flow = TravelWorkflow()
    flow.state.message = "杭州天气"
    flow.state.focus = "【近期对话上下文】：\nuser: 杭州天气\n【当前最新指令】：杭州天气"
    plan_data = {
        "location": "杭州",
        "focus": "查询杭州天气",
        "assumptions": [],
        "steps": [{"index": 0, "description": "查询杭州天气", "dependencies": []}],
        "needs_user_input": False,
    }

    assert _enforce_first_turn_question(flow, plan_data) is False
    assert flow.state.needs_user_input is False
    print("✅ weather_query_not_blocked")


def test_second_turn_delegated_can_assume():
    flow = TravelWorkflow()
    flow.state.message = "看你安排"
    flow.state.focus = """【近期对话上下文】：
user: 想去杭州
assistant: 好的，杭州行程没问题！为了规划得更贴合您的需求，请先补充：计划玩几天？大概预算是多少？几位出行？
user: 看你安排
【当前最新指令】：看你安排"""
    plan_data = {
        "location": "杭州",
        "focus": "杭州行程规划",
        "assumptions": ["用户未指定天数，默认按 3 天规划", "未指定预算，按中等档位", "未指定人数，按 2 人"],
        "steps": [{"index": 0, "description": "查询杭州天气", "dependencies": []}],
        "needs_user_input": False,
    }

    assert _enforce_first_turn_question(flow, plan_data) is False
    assert flow.state.needs_user_input is False
    print("✅ second_turn_delegated_can_assume")


if __name__ == "__main__":
    test_first_turn_must_ask()
    test_weather_query_not_blocked()
    test_second_turn_delegated_can_assume()
