# tools/schemas.py
"""
每个工具的输入/输出 Pydantic 契约。

【定位】
- 纯 Pydantic, 无外部依赖 (不 import requests / redis / crewai_tools)
- 谁想看某个工具"要什么参数、给什么结果", 只 import 这里
- StepPreparer 生成参数 / Executor 执行前后校验 / LLM 看 schema, 三方都用这里

【和 state.ToolCall / ToolResult 的区别】
- ToolCall / ToolResult 是"通用信封" (跨工具通用), 字段是 tool_name / parameters / ...
- 这里的 WeatherInput / WeatherOutput 是"信里的具体内容", 每个工具都不一样
"""
from __future__ import annotations

from pydantic import BaseModel, Field


# ============================================================
# WeatherTool
# ============================================================

class WeatherInput(BaseModel):
    """weather_tool 的输入。"""
    city: str = Field(..., description="城市名称, 支持中英文, 如: 北京 / Beijing / 成都")


class WeatherOutput(BaseModel):
    """weather_tool 的输出。字段与和风天气 API 语义对齐, 但对外只暴露必要项。"""
    city: str = Field(..., description="回显输入的城市")
    temp_c: float = Field(..., description="摄氏温度")
    condition: str = Field(..., description="天气现象文本, 如: 晴 / 多云 / 小雨")
    obs_time: str = Field("", description="观测时间, ISO 8601 字符串; 未知时为空")


# ============================================================
# TavilySearchTool
# ============================================================

class TavilySearchInput(BaseModel):
    """Tavily Search 的输入。"""
    query: str = Field(..., description="搜索关键词, 支持中文")
    max_results: int = Field(5, ge=1, le=20, description="返回结果条数上限")


class TavilySearchItem(BaseModel):
    """搜索结果单条命中。"""
    title: str = ""
    url: str = ""
    snippet: str = Field("", description="摘要或页面片段")


class TavilySearchOutput(BaseModel):
    """Tavily Search 的输出。"""
    query: str
    items: list[TavilySearchItem] = Field(default_factory=list)
    raw_text: str = Field("", description="兜底: 当上游返回难以拆分时, 存原始文本")
