# tools/impls.py
"""
每个工具的真实调用实现。

【定位】
- 有副作用层: 这里才 import requests / crewai_tools / 外部 SDK
- 一个函数只负责一个工具, 签名统一: (Input) -> Output
- 出错直接抛异常, 由 registry 层统一兜底包成 ToolResult(error=...)

【新增工具的步骤】
  1. schemas.py 加一组 XxxInput / XxxOutput
  2. 这里加一个 run_xxx(inp) 函数
  3. registry.py 的 REGISTRY 加一行
"""
from __future__ import annotations

import os

import requests

from agent_practice.tools.schemas import (
    TavilySearchInput,
    TavilySearchItem,
    TavilySearchOutput,
    WeatherInput,
    WeatherOutput,
)


# ============================================================
# WeatherTool - 和风天气
# ============================================================

def run_weather(inp: WeatherInput) -> WeatherOutput:
    """和风天气 API: city → LocationID → 实时天气。"""
    api_key = os.getenv("QWEATHER_API_KEY", "")
    host = os.getenv("QWEATHER_API_HOST", "geoapi.qweather.com")
    if not api_key:
        raise RuntimeError("环境变量 QWEATHER_API_KEY 未设置")

    # ① 城市 → LocationID
    r = requests.get(
        f"https://{host}/v2/city/lookup",
        params={"location": inp.city, "key": api_key},
        timeout=10,
    )
    r.raise_for_status()
    payload = r.json()
    locations = payload.get("location") or []
    if not locations:
        raise RuntimeError(f"和风天气找不到城市: {inp.city}")
    location_id = locations[0]["id"]

    # ② LocationID → 实时天气
    r = requests.get(
        "https://devapi.qweather.com/v7/weather/now",
        params={"location": location_id, "key": api_key},
        timeout=10,
    )
    r.raise_for_status()
    data = r.json()
    now = data.get("now") or {}

    return WeatherOutput(
        city=inp.city,
        temp_c=float(now.get("temp", 0)),
        condition=str(now.get("text", "")),
        obs_time=str(data.get("updateTime", "")),
    )


# ============================================================
# Tavily Search - 复用 crewai_tools 已装好的实例
# ============================================================

# crewai_tools.TavilySearchTool 是有状态对象, 模块级单例即可
_tavily_tool = None


def _get_tavily_tool():
    """惰性初始化 Tavily Tool 单例, 避免 import 时联网。"""
    global _tavily_tool
    if _tavily_tool is None:
        from crewai_tools import TavilySearchTool
        _tavily_tool = TavilySearchTool()
    return _tavily_tool


def run_tavily_search(inp: TavilySearchInput) -> TavilySearchOutput:
    """调 crewai_tools 的 TavilySearchTool。

    crewai_tools 返回的是格式化字符串, 这里先塞进 raw_text 兜底,
    未来若要精细拆分 items, 可改为直接调 tavily-python 库。
    """
    tool = _get_tavily_tool()
    # BaseTool 的调用接口: 关键字参数直接透传给 _run
    try:
        raw = tool.run(query=inp.query)
    except TypeError:
        # 某些版本要求 dict
        raw = tool.run({"query": inp.query})

    return TavilySearchOutput(
        query=inp.query,
        items=[],
        raw_text=str(raw),
    )
