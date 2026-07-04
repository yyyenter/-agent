# tools/registry.py
"""
工具注册表 + 批量执行器。

【定位】
- 路由层: name → (Input类, Output类, runner) 的映射
- executor 节点从 state 拿到 ToolCall 列表, 交给这里执行
- 统一错误兜底: 未知工具 / 参数校验失败 / 网络异常 都包成 ToolResult(error=...), 不抛出
- 统一格式化: 提供 format_for_llm, 把结构化输出转成"给 LLM 看"的文本

【和 state.ToolCall / ToolResult 的对接】
  ToolCall.parameters (dict)
    ↓ spec.input_cls.model_validate(...)
  WeatherInput 实例
    ↓ spec.runner(inp)
  WeatherOutput 实例
    ↓ out.model_dump()
  ToolResult.output (dict, 结构化) + ToolResult.output_text (str, 给 LLM 看)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel

from agent_test0.workflow.state import ToolCall, ToolResult
from agent_practice.tools.impls import run_tavily_search, run_weather
from agent_practice.tools.schemas import (
    TavilySearchInput,
    TavilySearchOutput,
    WeatherInput,
    WeatherOutput,
)


# ============================================================
# 工具规格 & 注册表
# ============================================================

@dataclass
class ToolSpec:
    """一个工具的完整规格。"""
    input_cls: type[BaseModel]
    output_cls: type[BaseModel]
    runner: Callable[[BaseModel], BaseModel]


REGISTRY: dict[str, ToolSpec] = {
    "weather_tool": ToolSpec(
        input_cls=WeatherInput,
        output_cls=WeatherOutput,
        runner=run_weather,
    ),
    "Tavily Search": ToolSpec(
        input_cls=TavilySearchInput,
        output_cls=TavilySearchOutput,
        runner=run_tavily_search,
    ),
}


# ============================================================
# 批量执行入口 (executor 节点调用点)
# ============================================================

def execute_tool_calls(tool_calls: list[ToolCall]) -> list[ToolResult]:
    """按 order 顺序执行 ToolCall 列表, 每个调用包成 ToolResult。"""
    ordered = sorted(tool_calls, key=lambda c: c.order)
    return [_execute_one(call) for call in ordered]


def _execute_one(call: ToolCall) -> ToolResult:
    """执行单个 ToolCall, 一切异常都不外抛, 包成 error 字段。"""
    t0 = time.time()
    spec = REGISTRY.get(call.tool_name)

    # ① 未注册工具 → error
    if spec is None:
        return _err(call, f"未注册的工具: {call.tool_name!r}", t0)

    # ② 输入参数 Pydantic 校验
    try:
        inp = spec.input_cls.model_validate(call.parameters)
    except Exception as exc:
        return _err(call, f"输入参数校验失败: {exc}", t0)

    # ③ 真正调用
    try:
        out = spec.runner(inp)
    except Exception as exc:
        return _err(call, f"{type(exc).__name__}: {exc}", t0)

    # ④ 输出类型防御 (impls 层写错的最后一道保险)
    if not isinstance(out, spec.output_cls):
        return _err(call, f"工具输出类型不符, 期望 {spec.output_cls.__name__}", t0)

    # ⑤ 成功: 落成 ToolResult
    return ToolResult(
        tool_name=call.tool_name,
        input=inp.model_dump(),
        output=out.model_dump(),
        output_text=format_for_llm(call.tool_name, out),
        error="",
        duration_ms=int((time.time() - t0) * 1000),
    )


def _err(call: ToolCall, error: str, t0: float) -> ToolResult:
    return ToolResult(
        tool_name=call.tool_name,
        input=call.parameters,
        output=None,
        output_text="",
        error=error,
        duration_ms=int((time.time() - t0) * 1000),
    )


# ============================================================
# 给 LLM 看的字符串格式化
# ============================================================

def format_for_llm(tool_name: str, out: BaseModel) -> str:
    """把结构化输出转成一段给 LLM 看的文本, 控制长度, 防上下文膨胀。"""
    data = out.model_dump()

    if tool_name == "weather_tool":
        return f"[天气] {data['city']}: {data['temp_c']}°C, {data['condition']}"

    if tool_name == "Tavily Search":
        items = data.get("items") or []
        if items:
            head = f"[搜索] {data['query']} → {len(items)} 条结果:\n"
            lines = [
                f"  · {x['title']}: {x['snippet'][:100]}"
                for x in items[:5]
            ]
            return head + "\n".join(lines)
        raw = (data.get("raw_text") or "")[:500]
        return f"[搜索] {data['query']} → 原始返回:\n{raw}"

    # 兜底
    return str(data)[:500]


# ============================================================
# 简易内省接口 (给 StepPreparer prompt 用: 展示工具白名单)
# ============================================================

def list_tools_for_prompt() -> str:
    """把工具白名单渲染成 prompt 可读的文本。"""
    lines = []
    for name, spec in REGISTRY.items():
        schema = spec.input_cls.model_json_schema()
        props = schema.get("properties", {})
        params_desc = ", ".join(
            f'"{k}": {v.get("type", "any")}' for k, v in props.items()
        )
        lines.append(f"- {name}: 参数 {{{params_desc}}}")
    return "\n".join(lines)
