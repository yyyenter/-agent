#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""StepPlan DAG 字段测试 (P0.1 修复验证, 不调用 LLM)

覆盖:
  1. StepPlan 默认值: dependencies 字段存在且默认为 []
  2. DAG 字段 round-trip: LLM 输出含 dependencies 时, 能正确解析到 StepPlan
  3. 旧 session 兼容: 旧 dict 缺 dependencies 也能 model_validate
  4. Planner prompt schema 期望与 StepPlan 一致 (DAG 不再静默丢失)
  5. 多步骤 DAG 示例: tasks.yaml 例子里的 3 步 DAG 能正确还原
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_test0.workflow.state import StepPlan, ReplanOutput, TravelState


# ============================================================
# 1. StepPlan 默认值
# ============================================================

def test_step_plan_has_dependencies_default():
    s = StepPlan(index=0, description="查询天气")
    assert s.dependencies == [], f"dependencies 默认值应为空列表, 实际: {s.dependencies}"
    print("[OK] step_plan_has_dependencies_default")


# ============================================================
# 2. DAG 字段 round-trip (P0.1 核心修复)
# ============================================================

def test_dag_field_round_trip_from_llm_output():
    """模拟 LLM 输出 tasks.yaml 示例 2 的 steps 列表, 验证 dependencies 不丢失。"""
    llm_steps = [
        {"index": 0, "description": "查询<目的地>天气和出行季节信息", "dependencies": []},
        {"index": 1, "description": "检索<目的地>热门景点和开放信息", "dependencies": []},
        {"index": 2, "description": "结合天气、景点和用户授权假设规划行程", "dependencies": [0, 1]},
    ]
    parsed = [StepPlan(**s) for s in llm_steps]
    assert len(parsed) == 3
    assert parsed[0].dependencies == []
    assert parsed[1].dependencies == []
    assert parsed[2].dependencies == [0, 1], \
        f"DAG 依赖丢失! 实际: {parsed[2].dependencies}, 期望: [0, 1]"
    print("[OK] dag_field_round_trip_from_llm_output")


# ============================================================
# 3. 旧 session 兼容 (Pydantic v2 默认值保证)
# ============================================================

def test_legacy_dict_without_dependencies():
    """旧持久化数据 (无 dependencies 字段) 仍能 model_validate, 默认 []"""
    old = {"index": 0, "description": "老 session 的步骤"}
    s = StepPlan.model_validate(old)
    assert s.dependencies == []
    assert s.description == "老 session 的步骤"
    print("[OK] legacy_dict_without_dependencies")


# ============================================================
# 4. ReplanOutput 也能透传 dependencies
# ============================================================

def test_replan_output_preserves_dependencies():
    """replan_tasks.yaml 示例里的 new_coarse_steps 含 dependencies 时不丢失"""
    replan = {
        "reason": "步骤2天气数据缺失",
        "preserved_steps": [0, 1],
        "original_remaining_steps": [2, 3, 4],
        "new_coarse_steps": [
            {"index": 2, "description": "重新查询天气", "dependencies": [0, 1], "tools": ["weather_tool"]},
            {"index": 3, "description": "查找周一开放景点", "dependencies": [2], "tools": ["Tavily Search"]},
            {"index": 4, "description": "重新规划行程", "dependencies": [2, 3], "tools": []},
        ],
        "replan_retry_count": 1,
    }
    out = ReplanOutput.model_validate(replan)
    assert out.new_coarse_steps[0].dependencies == [0, 1]
    assert out.new_coarse_steps[1].dependencies == [2]
    assert out.new_coarse_steps[2].dependencies == [2, 3]
    print("[OK] replan_output_preserves_dependencies")


# ============================================================
# 5. TravelState 仍然能装下 steps (含 dependencies)
# ============================================================

def test_travel_state_with_dag_steps():
    s = TravelState()
    s.steps = [
        StepPlan(index=0, description="weather", dependencies=[]),
        StepPlan(index=1, description="poi", dependencies=[]),
        StepPlan(index=2, description="assemble", dependencies=[0, 1]),
    ]
    assert s.steps[2].dependencies == [0, 1]
    print("[OK] travel_state_with_dag_steps")


# ============================================================
# 6. StepExecutor 依赖检查 (使用 mock 不调真实 LLM)
# ============================================================

def test_step_executor_dependency_check_warns_on_unmet():
    """StepExecutor 应在依赖未完成时打 warn, 但不阻塞 (P0.1 线性执行模式)"""
    from unittest.mock import MagicMock
    from agent_test0.workflow import nodes

    flow = MagicMock()
    flow.state.steps = [
        StepPlan(index=0, description="weather", dependencies=[], status="completed"),
        StepPlan(index=1, description="assemble", dependencies=[5]),  # 5 不存在
    ]
    flow.state.current_step_index = 1
    flow._check_ask_user_hook.return_value = False
    flow.notify = lambda x: None  # 静默 notify

    # run_step_executor 不应抛异常, 只 warn
    try:
        nodes.run_step_executor(flow)
    except Exception as e:
        raise AssertionError(f"StepExecutor 在依赖未满足时应仅 warn, 不应抛异常: {e}")
    # 步骤应仍进入执行 (status=executing), 不因依赖未满足被跳过
    assert flow.state.steps[1].status in ("executing", "completed", "failed"), \
        f"步骤应进入执行流, 实际 status: {flow.state.steps[1].status}"
    print("[OK] step_executor_dependency_check_warns_on_unmet")


if __name__ == "__main__":
    test_step_plan_has_dependencies_default()
    test_dag_field_round_trip_from_llm_output()
    test_legacy_dict_without_dependencies()
    test_replan_output_preserves_dependencies()
    test_travel_state_with_dag_steps()
    test_step_executor_dependency_check_warns_on_unmet()
    print("\nALL PASSED")
