# agent_test0/tools/registry.py
"""
确定性工具执行器。

StepPreparer 负责生成细粒度 tool_calls（工具名 + 参数），本模块负责用 Python
按顺序执行工具，避免 StepExecutor 再让 LLM 走 ReAct 工具调用导致解析失败。
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from agent_test0.tools.custom_tool import WeatherTool


ToolRunner = Callable[[dict], str]


def _try_parse_structured(s: str) -> Any:
    """P0.2: 尝试把工具返回的字符串解析为结构化数据 (dict/list)。

    解析失败返回原字符串。常见场景:
      - WeatherTool 返回 '[Cache Hit] {"城市": "北京", "温度": "23°C", ...}' → 解析为 dict
      - WeatherTool 返回压缩摘要 '城市: 北京, 温度: 23°C [完整数据已缓存]' → 解析失败, 回退 str
      - Tavily 搜索返回纯文本段落 → 解析失败, 回退 str
    """
    if not s:
        return s
    # 去掉 [Cache Hit] 前缀
    raw = s
    if raw.startswith("[Cache Hit]"):
        raw = raw[len("[Cache Hit]"):].strip()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return s


def _format_text_output(tool_name: str, input_params: dict, output: Any, error: str) -> str:
    """P0.2: 把 tool_result 格式化为 LLM 可读的文本 (output_text 字段)。"""
    if error:
        out_preview = (output if isinstance(output, str) else json.dumps(output, ensure_ascii=False))[:800]
        return f"[{tool_name}] 输入: {input_params}\n错误: {error}\n输出: {out_preview}"
    if isinstance(output, str):
        out_preview = output[:1000]
    else:
        out_preview = json.dumps(output, ensure_ascii=False)[:1000]
    return f"[{tool_name}] 输入: {input_params}\n输出: {out_preview}"


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
    """执行一个 ToolCall-like 对象，永不向上抛异常，错误写入 dict['error']。

    P0.2: 返回 dict 同时含结构化 (output) + 文本 (output_text):
      - output: 解析后的 dict/list; 解析失败回退 str
      - output_text: 格式化后给 LLM 看的字符串
    """
    tool_name = getattr(tool_call, "tool_name", "")
    params = getattr(tool_call, "parameters", {}) or {}
    normalized_name = _normalize_tool_name(tool_name)
    started = time.perf_counter()

    runner = TOOL_REGISTRY.get(normalized_name) or TOOL_REGISTRY.get(tool_name)
    if runner is None:
        return {
            "tool_name": tool_name,
            "input": params,
            "output": None,
            "output_text": "",
            "error": f"不支持的工具: {tool_name}",
            "duration_ms": 0,
        }

    try:
        output_str = runner(params)
        duration_ms = int((time.perf_counter() - started) * 1000)
        error = ""
        if any(marker in output_str for marker in _ERROR_MARKERS):
            error = output_str[:300]
        # P0.2: 尝试解析为结构化
        structured = _try_parse_structured(output_str)
        formatted = _format_text_output(normalized_name, params, structured, error)
        return {
            "tool_name": normalized_name,
            "input": params,
            "output": structured,
            "output_text": formatted,
            "error": error,
            "duration_ms": duration_ms,
        }
    except Exception as exc:
        duration_ms = int((time.perf_counter() - started) * 1000)
        return {
            "tool_name": normalized_name,
            "input": params,
            "output": None,
            "output_text": "",
            "error": str(exc),
            "duration_ms": duration_ms,
        }


def execute_tool_calls(tool_calls: list[Any]) -> list[dict]:
    """按 order 升序执行工具调用。"""
    ordered = sorted(tool_calls, key=lambda c: getattr(c, "order", 0))
    return [execute_tool_call(call) for call in ordered]


def format_tool_results(results: list[Any]) -> str:
    """P0.2: 优先使用 ToolResult.output_text, 兼容旧的 output 字段。"""
    if not results:
        return "（本步骤无需外部工具，作为整合/撰写步骤处理）"

    chunks = []
    for result in results:
        # P0.2: 优先 output_text
        text = getattr(result, "output_text", "") or ""
        if text:
            chunks.append(text)
            continue
        # 旧路径兼容: 旧的 output 字段
        out = getattr(result, "output", "") or ""
        err = getattr(result, "error", "") or ""
        if err:
            chunks.append(
                f"[{result.tool_name}] 输入: {result.input}\n错误: {err}\n输出: {out[:800]}"
            )
        else:
            chunks.append(
                f"[{result.tool_name}] 输入: {result.input}\n输出: {out[:1000]}"
            )
    return "\n\n".join(chunks)
