#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""StepPlan.result 拆分测试 (P0.2 修复验证, 不调用 LLM)

覆盖:
  1. ToolResult.output 是 Any, 解析后存 dict; 解析失败存 str
  2. ToolResult.output_text 是格式化字符串
  3. StepPlan.result 聚合 tool_results[].output (单一 dict)
  4. StepPlan.result 多工具时包成 {"items": [...]}
  5. StepResult.result / result_text 同步
  6. 旧 session 兼容: 老 dict 缺 output_text 也能 model_validate
  7. format_tool_results 优先 output_text, 兼容旧 output
  8. End-to-end: WeatherTool JSON 字符串 → registry 解析为 dict → 写入 StepPlan.result
"""

import sys
import json
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_test0.workflow.state import (
    StepPlan, StepResult, ToolResult, TravelState,
)
from agent_test0.tools.registry import (
    _try_parse_structured, _format_text_output, format_tool_results,
)


# ============================================================
# 1. _try_parse_structured
# ============================================================

def test_try_parse_json_dict():
    s = '{"城市": "北京", "温度": "23°C", "天气": "晴"}'
    parsed = _try_parse_structured(s)
    assert isinstance(parsed, dict)
    assert parsed["城市"] == "北京"
    print("[OK] try_parse_json_dict")


def test_try_parse_cache_hit_prefix():
    s = '[Cache Hit] {"城市": "上海", "温度": "28°C"}'
    parsed = _try_parse_structured(s)
    assert isinstance(parsed, dict)
    assert parsed["城市"] == "上海"
    print("[OK] try_parse_cache_hit_prefix")


def test_try_parse_compressed_fallback_to_str():
    """压缩后的摘要无法解析, 应回退原字符串"""
    s = "城市: 北京, 温度: 23°C [完整数据已缓存]"
    parsed = _try_parse_structured(s)
    assert parsed == s
    print("[OK] try_parse_compressed_fallback_to_str")


def test_try_parse_empty():
    assert _try_parse_structured("") == ""
    assert _try_parse_structured("普通文本") == "普通文本"
    print("[OK] try_parse_empty_and_plain_text")


# ============================================================
# 2. _format_text_output
# ============================================================

def test_format_text_output_success():
    s = _format_text_output("weather_tool", {"city": "北京"}, {"温度": "23"}, "")
    assert "[weather_tool]" in s
    assert "北京" in s
    assert "23" in s
    assert "错误" not in s
    print("[OK] format_text_output_success")


def test_format_text_output_with_error():
    s = _format_text_output("weather_tool", {"city": "北京"}, "", "网络超时")
    assert "网络超时" in s
    print("[OK] format_text_output_with_error")


# ============================================================
# 3. StepPlan.result 聚合
# ============================================================

def test_step_plan_result_aggregates_single_tool():
    """单一工具 → StepPlan.result 直接是该工具的 output"""
    s = StepPlan(index=0, description="查询天气")
    s.tool_results = [
        ToolResult(tool_name="weather_tool", input={"city": "北京"},
                   output={"城市": "北京", "温度": "23°C"}, output_text="[weather_tool] ..."),
    ]
    # 模拟 StepExecutor 聚合逻辑 (与 nodes.py:run_step_executor 一致)
    structured = [r.output for r in s.tool_results if r.output is not None]
    if len(structured) == 1:
        s.result = structured[0]
    elif len(structured) > 1:
        s.result = {"items": structured}
    s.result_text = format_tool_results(s.tool_results)
    assert isinstance(s.result, dict)
    assert s.result["城市"] == "北京"
    assert s.result_text == "[weather_tool] ..."
    print("[OK] step_plan_result_aggregates_single_tool")


def test_step_plan_result_aggregates_multiple_tools():
    """多工具 → StepPlan.result 包成 {"items": [...]}"""
    s = StepPlan(index=0, description="多步查询")
    s.tool_results = [
        ToolResult(tool_name="weather_tool", output={"温度": "23"}, output_text="weather text"),
        ToolResult(tool_name="Tavily Search", output=[{"title": "景点 A"}], output_text="search text"),
    ]
    structured = [r.output for r in s.tool_results if r.output is not None]
    s.result = {"items": structured}
    s.result_text = format_tool_results(s.tool_results)
    assert isinstance(s.result, dict)
    assert "items" in s.result
    assert len(s.result["items"]) == 2
    assert s.result_text  # 应有格式化文本
    print("[OK] step_plan_result_aggregates_multiple_tools")


# ============================================================
# 4. 旧 session 兼容
# ============================================================

def test_legacy_dict_without_output_text():
    """旧数据缺 output_text 字段, Pydantic 默认值兜底"""
    old = {"tool_name": "weather_tool", "input": {"city": "北京"}, "output": "legacy str"}
    r = ToolResult.model_validate(old)
    assert r.output == "legacy str"
    assert r.output_text == ""  # 默认空
    print("[OK] legacy_dict_without_output_text")


def test_legacy_step_plan_result_str():
    """旧数据 result 是 str, 不应崩"""
    old = {"index": 0, "description": "老步骤", "result": "老的 result 字符串"}
    s = StepPlan.model_validate(old)
    assert s.result == "老的 result 字符串"
    assert s.result_text == ""
    print("[OK] legacy_step_plan_result_str")


# ============================================================
# 5. format_tool_results 兼容
# ============================================================

def test_format_tool_results_uses_output_text():
    results = [
        ToolResult(tool_name="weather_tool", input={"city": "北京"},
                   output={"温度": "23"}, output_text="[weather_tool] 23"),
    ]
    s = format_tool_results(results)
    assert s == "[weather_tool] 23"
    print("[OK] format_tool_results_uses_output_text")


def test_format_tool_results_fallback_to_legacy_output():
    """老 ToolResult (无 output_text) 仍能 format"""
    # 模拟旧 ToolResult: 用 __dict__ 注入, Pydantic v2 允许 extra=ignore
    tr = ToolResult(tool_name="weather_tool", input={"city": "北京"}, output="legacy output")
    # 模拟无 output_text 的情况
    s = format_tool_results([tr])
    assert "weather_tool" in s
    assert "legacy output" in s
    print("[OK] format_tool_results_fallback_to_legacy_output")


def test_format_tool_results_empty():
    assert "无需外部工具" in format_tool_results([])
    print("[OK] format_tool_results_empty")


# ============================================================
# 6. TravelState.steps 装填新格式
# ============================================================

def test_travel_state_with_structured_results():
    s = TravelState()
    s.steps = [
        StepPlan(index=0, description="查询天气", result={"城市": "北京"}),
        StepPlan(index=1, description="检索景点", result={"items": [{"name": "故宫"}]}),
    ]
    assert isinstance(s.steps[0].result, dict)
    assert s.steps[1].result["items"][0]["name"] == "故宫"
    print("[OK] travel_state_with_structured_results")


# ============================================================
# 7. End-to-end: registry 解析为 dict
# ============================================================

def test_registry_parses_weather_json_output(monkeypatch=None):
    """模拟 registry 调用 _run_weather, 验证 output 是 dict"""
    from agent_test0.tools import registry

    # 模拟 _run_weather 返回 JSON 字符串
    def fake_run_weather(params):
        return json.dumps({"城市": "北京", "温度": "23°C", "天气": "晴"}, ensure_ascii=False)

    # 构造 ToolCall-like
    tc = MagicMock()
    tc.tool_name = "weather_tool"
    tc.parameters = {"city": "北京"}
    tc.order = 1

    # 替换 registry 里的 _run_weather
    original = registry.TOOL_REGISTRY["weather_tool"]
    registry.TOOL_REGISTRY["weather_tool"] = fake_run_weather
    try:
        result = registry.execute_tool_call(tc)
    finally:
        registry.TOOL_REGISTRY["weather_tool"] = original

    assert result["tool_name"] == "weather_tool"
    assert isinstance(result["output"], dict), f"output 应是 dict, 实际: {type(result['output'])}"
    assert result["output"]["城市"] == "北京"
    assert "[weather_tool]" in result["output_text"]
    print("[OK] registry_parses_weather_json_output")


if __name__ == "__main__":
    test_try_parse_json_dict()
    test_try_parse_cache_hit_prefix()
    test_try_parse_compressed_fallback_to_str()
    test_try_parse_empty()
    test_format_text_output_success()
    test_format_text_output_with_error()
    test_step_plan_result_aggregates_single_tool()
    test_step_plan_result_aggregates_multiple_tools()
    test_legacy_dict_without_output_text()
    test_legacy_step_plan_result_str()
    test_format_tool_results_uses_output_text()
    test_format_tool_results_fallback_to_legacy_output()
    test_format_tool_results_empty()
    test_travel_state_with_structured_results()
    test_registry_parses_weather_json_output()
    print("\nALL PASSED")
