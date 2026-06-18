# agent_test0/crew.py
"""
兼容入口（阶段 D 重构后）。

历史上这个文件有 1083 行，把状态机/Crew 定义/业务逻辑/外部入口全揉在一起。
现已全部搬到 `agent_test0.workflow` 包下，本文件仅 re-export 旧路径，
保证既有外部代码（main.py / long_conn_bot / test_*）import 路径不变。

新代码请改用：
    from agent_test0.workflow import TravelWorkflow
    from agent_test0.workflow.state import TravelState, StepPlan

模块导航（搬家对照表）：

    原 crew.py 的内容              →    新位置
    ─────────────────────────────────────────────────────────────
    StepPlan/StepResult/...        →    workflow/state.py
    TravelState                    →    workflow/state.py
    zhipu_llm / search_tool        →    workflow/llm.py
    7 个 @CrewBase 类              →    workflow/crews.py
    agent_step_logger              →    workflow/callbacks.py
    _make_step_callback            →    workflow/callbacks.py（make_step_callback）
    _run_crew_with_callback        →    workflow/callbacks.py（run_crew_with_callback）
    _parse_step_feedback           →    workflow/parsing.py（parse_step_feedback）
    _check_ask_user_hook           →    workflow/ask_user.py（check_ask_user_hook）
    _set_ask_user_question         →    workflow/ask_user.py（set_ask_user_question）
                                        + 新增 ask_user_and_exit / AskUserInterrupt / has_already_asked
    plan_steps 节点逻辑            →    workflow/nodes.py（run_planner）
    step_preparer 节点逻辑         →    workflow/nodes.py（run_step_preparer）
    step_executor 节点逻辑         →    workflow/nodes.py（run_step_executor）
    step_verifier 节点逻辑         →    workflow/nodes.py（run_step_verifier）
    partial_replanner 节点逻辑     →    workflow/nodes.py（run_partial_replanner）
    final_verifier 节点逻辑        →    workflow/nodes.py（run_final_verifier）
    _generate_final_report         →    workflow/nodes.py（generate_final_report）
    finalize                       →    workflow/nodes.py（run_finalize）
    TravelWorkflow 类              →    workflow/flow.py
    TravelWorkflow.run_for_user    →    workflow/flow.py
"""

# ─── 状态模型 ───
from agent_test0.workflow.state import (
    StepPlan,
    StepResult,
    ValidationFeedback,
    TravelState,
)

# ─── Flow 主类与全局 Redis ───
from agent_test0.workflow.flow import (
    TravelWorkflow,
    _redis_client,
    _is_redis_fallback,
)

# ─── LLM / 工具（旧代码偶尔直接 from .crew import zhipu_llm）───
from agent_test0.workflow.llm import zhipu_llm, search_tool

# ─── Crew 类（保留 re-export，万一旧代码 import）───
from agent_test0.workflow.crews import (
    PlannerCrew,
    ValidatorCrew,
    StepPreparerCrew,
    StepExecutorCrew,
    StepVerifierCrew,
    PartialReplannerCrew,
    FinalVerifierCrew,
)

# ─── 回调与日志（旧代码可能直接 import agent_step_logger）───
from agent_test0.workflow.callbacks import agent_step_logger


__all__ = [
    # 数据模型
    "StepPlan",
    "StepResult",
    "ValidationFeedback",
    "TravelState",
    # Flow
    "TravelWorkflow",
    "_redis_client",
    "_is_redis_fallback",
    # LLM / 工具
    "zhipu_llm",
    "search_tool",
    # Crews
    "PlannerCrew",
    "ValidatorCrew",
    "StepPreparerCrew",
    "StepExecutorCrew",
    "StepVerifierCrew",
    "PartialReplannerCrew",
    "FinalVerifierCrew",
    # 工具函数
    "agent_step_logger",
]
