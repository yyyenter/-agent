# agent_test0/workflow/__init__.py
"""
旅游规划状态机工作流包。

使用方式：
    from agent_test0.workflow import TravelWorkflow
    from agent_test0.workflow.state import TravelState, StepPlan

模块结构：
    state.py      —— Pydantic 状态模型（StepPlan/StepResult/ValidationFeedback/TravelState）
    llm.py        —— 共享 LLM 实例（zhipu_llm）+ 共享工具（search_tool）
    crews.py      —— 7 个 @CrewBase 类
    callbacks.py  —— Agent 步骤回调与日志（agent_step_logger / make_step_callback / run_crew_with_callback）
    parsing.py    —— 统一 JSON 解析（extract_json_object / parse_step_feedback）
    ask_user.py   —— AskUser 中断机制（AskUserInterrupt / ask_user_and_exit / has_already_asked / 旧 hook）
    nodes.py      —— 6 个状态节点的业务逻辑（run_planner / run_step_preparer / ...）
    flow.py       —— TravelWorkflow Flow 编排骨架 + run_for_user 外部入口
"""

from agent_test0.workflow.state import (
    StepPlan,
    StepResult,
    ValidationFeedback,
    TravelState,
)
from agent_test0.workflow.flow import (
    TravelWorkflow,
    _redis_client,
    _is_redis_fallback,
)

__all__ = [
    "TravelWorkflow",
    "TravelState",
    "StepPlan",
    "StepResult",
    "ValidationFeedback",
    "_redis_client",
    "_is_redis_fallback",
]
