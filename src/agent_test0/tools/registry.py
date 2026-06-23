# agent_test0/tools/registry.py
"""
确定性工具执行器。

StepPreparer 负责生成细粒度 tool_calls（工具名 + 参数），本模块负责用 Python
按顺序执行工具，避免 StepExecutor 再让 LLM 走 ReAct 工具调用导致解析失败。
"""

from __future__ import annotations

import time
from typing import Any, Callable

from agent_test0.tools.custom_tool import WeatherTool


ToolRunner = Callable[[dict], str]


def _normalize_tool_name(name: str) -> str:
    """统一工具名，兼容 LLM 输出的大小写/空格/下划线差异。"""
    raw = (name or "").strip()
    lowered = raw.lower().replace(" ", "_")
    aliases = {
        "weather": "weather_tool",
        "getweathertool": "weather_tool",
        "get_weather_tool": "weather_tool",
        "weather_tool": "weather_tool",
        "tavily": "tavily_search",
        "tavily_search": "tavily_search",
        "tavily_search_tool": "tavily_search",
    }
    return aliases.get(lowered, raw)


def _run_weather(params: dict) -> str:
    city = params.get("city") or params.get("location")
    if not city:
        raise ValueError("weather_tool 缺少参数 city")
    return WeatherTool()._run(city=str(city))


def _run_tavily(params: dict) -> str:
    query = params.get("query") or params.get("search_query")
    if not query:
        raise ValueError("tavily_search 缺少参数 query")

    # 延迟导入，避免 tools.registry -> workflow.llm -> workflow.__init__ -> nodes 的循环导入。
    from agent_test0.workflow.llm import search_tool

    # crewai_tools 的 BaseTool 版本之间 run/_run 签名略有差异，按最稳定方式兜底。
    try:
        return str(search_tool.run(query=str(query)))
    except TypeError:
        return str(search_tool._run(query=str(query)))


TOOL_REGISTRY: dict[str, ToolRunner] = {
    "weather_tool": _run_weather,
    "Tavily Search": _run_tavily,
    "tavily_search": _run_tavily,
}


_ERROR_MARKERS = (
    "工具调用失败",
    "天气查询失败",
    "天气查询出错",
    "未配置",
    "无法解析",
    "error",
    "Error",
)


def execute_tool_call(tool_call: Any) -> dict:
    """执行一个 ToolCall-like 对象，永不向上抛异常，错误写入 dict['error']。"""
    tool_name = getattr(tool_call, "tool_name", "")
    params = getattr(tool_call, "parameters", {}) or {}
    normalized_name = _normalize_tool_name(tool_name)
    started = time.perf_counter()

    runner = TOOL_REGISTRY.get(normalized_name) or TOOL_REGISTRY.get(tool_name)
    if runner is None:
        return {
            "tool_name": tool_name,
            "input": params,
            "output": "",
            "error": f"不支持的工具: {tool_name}",
            "duration_ms": 0,
        }

    try:
        output = runner(params)
        duration_ms = int((time.perf_counter() - started) * 1000)
        error = ""
        if any(marker in output for marker in _ERROR_MARKERS):
            error = output[:300]
        return {
            "tool_name": normalized_name,
            "input": params,
            "output": output,
            "error": error,
            "duration_ms": duration_ms,
        }
    except Exception as exc:
        duration_ms = int((time.perf_counter() - started) * 1000)
        return {
            "tool_name": normalized_name,
            "input": params,
            "output": "",
            "error": str(exc),
            "duration_ms": duration_ms,
        }


def execute_tool_calls(tool_calls: list[Any]) -> list[dict]:
    """按 order 升序执行工具调用。"""
    ordered = sorted(tool_calls, key=lambda c: getattr(c, "order", 0))
    return [execute_tool_call(call) for call in ordered]


def format_tool_results(results: list[Any]) -> str:
    """把结构化工具结果转成旧 result 字符串，兼容 StepVerifier / FinalReporter。"""
    if not results:
        return "（本步骤无需外部工具，作为整合/撰写步骤处理）"

    chunks = []
    for result in results:
        if result.error:
            chunks.append(
                f"[{result.tool_name}] 输入: {result.input}\n错误: {result.error}\n输出: {result.output[:800]}"
            )
        else:
            chunks.append(
                f"[{result.tool_name}] 输入: {result.input}\n输出: {result.output[:1000]}"
            )
    return "\n\n".join(chunks)
