#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Python 工具执行器测试（不调用 LLM / 外部网络）。"""

from unittest.mock import patch

from agent_test0.tools.registry import execute_tool_calls
from agent_test0.workflow.state import StepPlan, ToolCall
from agent_test0.workflow.flow import TravelWorkflow
from agent_test0.workflow.nodes import run_step_executor


def test_weather_tool_success():
    calls = [ToolCall(order=1, tool_name="weather_tool", parameters={"city": "重庆"})]
    with patch("agent_test0.tools.registry.WeatherTool._run", return_value='{"城市":"重庆","天气":"多云"}'):
        results = execute_tool_calls(calls)
    assert len(results) == 1
    assert results[0]["tool_name"] == "weather_tool"
    assert results[0]["input"] == {"city": "重庆"}
    assert "重庆" in results[0]["output"]
    assert results[0]["error"] == ""
    print("✅ weather_tool_success")


def test_unknown_tool_records_error():
    calls = [ToolCall(order=1, tool_name="mcp://weather/get", parameters={"city": "重庆"})]
    results = execute_tool_calls(calls)
    assert len(results) == 1
    assert "不支持的工具" in results[0]["error"]
    print("✅ unknown_tool_records_error")


def test_tool_exception_records_error_and_continues():
    calls = [
        ToolCall(order=2, tool_name="weather_tool", parameters={"city": "重庆"}),
        ToolCall(order=1, tool_name="bad_tool", parameters={}),
    ]
    with patch("agent_test0.tools.registry.WeatherTool._run", side_effect=RuntimeError("boom")):
        results = execute_tool_calls(calls)
    assert results[0]["tool_name"] == "bad_tool"
    assert "不支持的工具" in results[0]["error"]
    assert results[1]["tool_name"] == "weather_tool"
    assert "boom" in results[1]["error"]
    print("✅ tool_exception_records_error_and_continues")


def test_run_step_executor_no_tool_step_completed():
    flow = TravelWorkflow()
    flow.state.steps = [StepPlan(index=0, description="整合生成报告", prepared=True, tool_calls=[])]
    flow.state.current_step_index = 0
    run_step_executor(flow)
    step = flow.state.steps[0]
    assert step.status == "completed"
    assert step.result
    assert len(flow.state.step_results) == 1
    assert flow.state.step_results[0].passed is True
    print("✅ run_step_executor_no_tool_step_completed")


def test_run_step_executor_python_weather():
    flow = TravelWorkflow()
    flow.state.steps = [StepPlan(
        index=0,
        description="查询重庆天气",
        prepared=True,
        tools=["weather_tool"],
        tool_calls=[ToolCall(order=1, tool_name="weather_tool", parameters={"city": "重庆"})],
    )]
    flow.state.current_step_index = 0
    with patch("agent_test0.tools.registry.WeatherTool._run", return_value='{"城市":"重庆","天气":"多云"}'):
        run_step_executor(flow)
    step = flow.state.steps[0]
    assert step.status == "completed"
    assert len(step.tool_results) == 1
    assert "重庆" in step.result
    assert len(flow.state.step_results) == 1
    print("✅ run_step_executor_python_weather")


if __name__ == "__main__":
    test_weather_tool_success()
    test_unknown_tool_records_error()
    test_tool_exception_records_error_and_continues()
    test_run_step_executor_no_tool_step_completed()
    test_run_step_executor_python_weather()
