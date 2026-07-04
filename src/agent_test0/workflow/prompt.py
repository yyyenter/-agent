# agent_test0/workflow/prompt.py
"""
YAML 任务模板加载 + slot 填充 (主项目版).

【与 agent_practice 版的差异】
主项目历史 YAML 命名不规整 (tasks.yaml 里叫 planning_task, replan_tasks.yaml 里叫 replan_task),
所以维护一份 TASK_REGISTRY 显式映射, 让 nodes.py 只关心 task_name, 不管文件名。

【职责】
- 私有函数各负责单一操作 (读文件 / 找 slot / 校验+填充)
- 公开 API load_task_prompt(task_name, slots) 组合这些操作, 供 nodes.py 使用
- 只对 description 做 slot 填充, expected_output 原样附加
  (expected_output 含 JSON 示例, {} 密度高, str.format 会把它们误当 slot)
"""
from __future__ import annotations

from pathlib import Path
from string import Formatter

import yaml

CONFIG_DIR = Path(__file__).parent.parent / "config"


# ─── task_name → 文件名 显式映射 (屏蔽历史命名不一致) ───
TASK_REGISTRY: dict[str, str] = {
    "planning_task":            "tasks.yaml",
    "step_preparer_task":       "step_preparer_tasks.yaml",
    "step_validator_task":      "step_validator_tasks.yaml",
    "replan_task":              "replan_tasks.yaml",
    "final_validator_task":     "final_validator_tasks.yaml",
    "domain_classifier_task":   "domain_classifier_task.yaml",
}


def _load_yaml(file_name: str) -> dict:
    path = CONFIG_DIR / file_name
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_task_prompt(task_name: str) -> tuple[str, str]:
    if task_name not in TASK_REGISTRY:
        raise KeyError(f"未注册的 task_name: {task_name!r}, 请先在 TASK_REGISTRY 登记")
    data = _load_yaml(TASK_REGISTRY[task_name])
    task = data[task_name]
    return task["description"], task["expected_output"]


def _find_slots(description: str) -> set[str]:
    """扫模板里所有 {xxx} 占位符, 返回 slot 名集合. 忽略 {{}} 转义与末尾字面段."""
    keys = set()
    for literal_text, field_name, format_spec, conversion in Formatter().parse(description):
        if field_name is None:
            continue
        keys.add(field_name)
    return keys


def _safe_render(description: str, slots: dict) -> str:
    """校验 slot 完整性, 缺失时一次列出所有缺失名再报错; 否则调 str.format 填充."""
    required = _find_slots(description)
    provided = set(slots.keys())

    missing = required - provided
    if missing:
        raise ValueError(f"模板缺少 slot: {sorted(missing)}")

    return description.format(**slots)


# ─── 公开 API ────────────────────────────────────────────────
def load_task_prompt(task_name: str, slots: dict) -> str:
    """加载 YAML 任务模板并用 slots 填充, 返回可直接喂给 LLM 的完整 prompt.

    参数:
        task_name: TASK_REGISTRY 里注册的键 (如 'planning_task')
        slots:     用于填充 description 的字典

    返回:
        单个 str, 结构为: [填充后的 description]\\n\\n【期望输出】\\n[expected_output 原样]

    异常:
        KeyError:   task_name 未注册
        ValueError: description 缺少 slot (一次列出所有缺失)
    """
    description, expected_output = _load_task_prompt(task_name)
    rendered = _safe_render(description, slots)
    return f"{rendered}\n\n【期望输出】\n{expected_output}"


class StructuredCallError(Exception):
    """结构化调用失败 (LLM 解析失败 / Pydantic 校验失败). 让 nodes.py 用 except 兜底."""


if __name__ == "__main__":
    fake_state = {
        "message": "去成都玩 3 天",
        "user_id": "u_001",
        "focus": "美食",
        "previous_plan": "无",
        "current_step": "无",
        "current_draft": "无",
    }
    p = load_task_prompt("planning_task", fake_state)
    print(f"planning_task rendered, len={len(p)}")
