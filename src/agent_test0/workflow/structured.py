# agent_test0/workflow/structured.py
"""
直接 LLM + Pydantic 的结构化调用工具。

目的：替代 CrewAI Agent 的 ReAct 解析层。当前节点（Planner / StepPreparer /
Verifier / Replanner / FinalVerifier）都不需要工具调用，只需要“输入上下文 →
结构化对象”，因此不应让 CrewAI Agent 先按 Thought/Action/Final Answer 解析。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from agent_test0.workflow.llm import zhipu_llm
from agent_test0.workflow.parsing import extract_json_object
from agent_test0.workflow.trace import timed


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


class StructuredCallError(RuntimeError):
    """结构化 LLM 调用多次失败。"""


def _safe_render(template: str, values: dict[str, Any]) -> str:
    """
    只替换 {key} 这种输入变量，避免 YAML 里的 JSON 大括号被 str.format 误伤。
    """
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered


def load_task_prompt(yaml_name: str, task_key: str, values: dict[str, Any]) -> str:
    """读取 config/<yaml_name> 中 task 的 description + expected_output 并替换变量。"""
    path = CONFIG_DIR / yaml_name
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    task = data[task_key]
    parts = [task.get("description", ""), task.get("expected_output", "")]
    return _safe_render("\n\n".join(parts), values)


def call_structured(
    label: str,
    prompt: str,
    model_cls: type[BaseModel],
    *,
    max_retries: int = 2,
) -> BaseModel:
    """
    调 LLM 并用 Pydantic 校验结构化输出。

    第一版不依赖 provider 的 response_format；用 extract_json_object 兜底提取 JSON，
    校验失败时把错误和上一轮输出回灌给 LLM 重试。
    """
    messages = [{"role": "user", "content": prompt}]
    last_raw = ""
    last_error = ""

    for attempt in range(max_retries + 1):
        with timed(f"LLM:{label}"):
            raw = zhipu_llm.call(messages).strip()
        last_raw = raw

        data = extract_json_object(raw)
        if data is None:
            last_error = "未找到 JSON 对象"
        else:
            try:
                return model_cls.model_validate(data)
            except ValidationError as exc:
                last_error = str(exc)

        messages = [
            {"role": "user", "content": prompt},
            {
                "role": "user",
                "content": (
                    "上一次输出不符合结构化字段约束，请只重新输出一个合法 JSON 对象。\n"
                    f"校验错误：{last_error}\n"
                    f"上一次输出：{last_raw[:2000]}"
                ),
            },
        ]

    raise StructuredCallError(f"{label} 结构化输出失败: {last_error}; raw={last_raw[:500]}")
