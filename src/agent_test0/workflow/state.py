# agent_test0/workflow/state.py
"""
状态机数据模型。

本文件是整个 Flow 的"数据契约"——所有节点读写的状态都定义在 TravelState 中。
任何节点新增字段一律先改这里，不要在节点内部偷偷往 state 上挂属性。

字段分组：
  - 基础元数据：用户/会话标识
  - 状态机核心：steps / current_step_index / step_results
  - 流程控制：needs_user_input / asked_fields / skip_remaining_steps / 计数器
  - 业务字段：location / focus / assumptions / final_report
"""

from typing import Literal

from pydantic import BaseModel, model_validator


# ============================================
# 单步骤数据结构
# ============================================

class ToolCall(BaseModel):
    """StepPreparer 生成的细粒度工具调用计划"""
    order: int = 0
    tool_name: str
    parameters: dict = {}
    expected_output_schema: dict = {}


class ToolResult(BaseModel):
    """Python 工具执行器返回的单次工具执行结果"""
    tool_name: str
    input: dict = {}
    output: str = ""
    error: str = ""
    duration_ms: int = 0


class StepPlan(BaseModel):
    """单个执行步骤的计划与执行状态"""
    index: int
    description: str            # 步骤描述（不指定工具）
    tools: list[str] = []       # 需要的工具名摘要（兼容旧逻辑）
    tool_calls: list[ToolCall] = []      # StepPreparer 生成的完整小计划（含参数）
    tool_results: list[ToolResult] = []  # Python StepExecutor 写入的结构化执行结果
    prepared: bool = False      # True 表示 StepPreparer 已处理；tool_calls=[] 表示无需工具
    status: str = "pending"     # pending | executing | completed | failed
    result: str = ""
    error: str = ""
    validation_feedback: str = ""  # StepVerifier 写入的反馈


class StepResult(BaseModel):
    """步骤执行结果归档（写入 state.step_results 历史）"""
    step_index: int
    step_description: str
    result: str
    passed: bool
    validation_feedback: str


class ValidationFeedback(BaseModel):
    """统一验证反馈结构（StepVerifier / FinalVerifier 共用）"""
    is_valid: bool
    feedback_type: str  # pass | retry | partial_fail | full_replan
    failed_indices: list[int] = []          # 失败的步骤索引
    reason: str = ""                        # 为什么失败
    suggested_corrections: dict[int, str] = {}  # 每个失败步骤的修正建议


# ============================================
# LLM 结构化输出模型
# ============================================

class PlannerOutput(BaseModel):
    """Planner 的结构化输出：大计划 + 缺失信息判断"""
    is_complex: bool = True
    location: str = "未知"
    focus: str = ""
    assumptions: list[str] = []
    steps: list[StepPlan] = []
    needs_user_input: bool = False
    user_question: str = ""
    simple_answer: str = ""
    plan_summary: str = ""


class StepPreparerOutput(BaseModel):
    """StepPreparer 的结构化输出：单步骤对应工具小计划"""
    step_index: int = 0
    tools_to_call: list[ToolCall] = []


class StepVerifierOutput(BaseModel):
    """StepVerifier 的结构化输出：单步审核 verdict"""
    verdict: Literal["pass", "retry", "fail", "ask_user"] = "pass"
    passed: bool = True
    feedback_type: str = "pass"
    reason: str = ""
    suggested_corrections: dict[str, str] = {}
    question: str = ""

    @model_validator(mode="before")
    @classmethod
    def flatten_feedback(cls, data):
        # 兼容旧 prompt 输出: {"verdict": "pass", "feedback": {...}}
        if isinstance(data, dict) and isinstance(data.get("feedback"), dict):
            feedback = data.get("feedback") or {}
            merged = {**feedback, **{k: v for k, v in data.items() if k != "feedback"}}
            return merged
        return data


class ReplanOutput(BaseModel):
    """PartialReplanner 的结构化输出：失败处之后的新步骤"""
    reason: str = ""
    preserved_steps: list[int] = []
    original_remaining_steps: list[int] = []
    new_coarse_steps: list[StepPlan] = []
    replan_retry_count: int = 0


class FinalVerifierOutput(BaseModel):
    """FinalVerifier 的结构化输出：整体审核 verdict"""
    global_verdict: Literal["pass", "fail_with_patches"] = "pass"
    reason: str = ""
    failed_step_ids: list[int] = []
    suggested_corrections: dict[str, str] = {}

    @model_validator(mode="before")
    @classmethod
    def flatten_global_feedback(cls, data):
        # 兼容旧 prompt 输出: {"global_verdict": "...", "global_feedback": {...}}
        if isinstance(data, dict) and isinstance(data.get("global_feedback"), dict):
            feedback = data.get("global_feedback") or {}
            merged = {**feedback, **{k: v for k, v in data.items() if k != "global_feedback"}}
            return merged
        return data


# ============================================
# Flow 全局状态
# ============================================

class TravelState(BaseModel):
    """
    Flow 的全局状态容器。Pydantic 会自动持久化到 Redis（CrewAI Flow 内置）。
    所有字段必须有默认值，否则旧 session 反序列化时会因缺字段炸掉。
    """

    # === 基础元数据 ===
    message: str = ''
    user_id: str = 'default_user'
    session_id: str = 'default_sess'

    # === 状态机核心字段 ===
    steps: list[StepPlan] = []                  # 步骤列表
    current_step_index: int = 0                 # 当前执行步骤索引
    step_results: list[StepResult] = []         # 步骤执行结果历史
    failed_steps_indices: list[int] = []        # 失败步骤索引列表

    # === 流程控制（结构化字段，替代旧的文本前缀解析） ===
    needs_user_input: bool = False              # True 时主流程中断，向用户提问
    user_question: str = ""                     # needs_user_input=True 时填入问题
    skip_remaining_steps: bool = False          # 跳过剩余步骤直接输出结果

    # === Loop 安全（阶段 A 顺便修 bug：asked_fields 默认值之前是 ""，应为 []）===
    total_steps_counted: int = 0                # 已经进入步骤节点的总次数
    asked_fields: list[str] = []                # 本轮已经问过用户的字段名（防重复问）

    # === 显式驱动循环用的计数/标志 ===
    # 之前用 flow.state._replan_count / _final_verifier_executed 动态挂属性，
    # 但 Pydantic v2 默认 extra='ignore' 会丢弃 → 限流与重入保护失效。
    # 这里提升为正式字段。
    replan_count: int = 0                       # PartialReplanner 已触发次数（上限 max_replan_attempts）
    final_verifier_done: bool = False           # FinalVerifier 是否已执行（防重入）

    # === 业务字段 ===
    is_complex: bool = True
    simple_answer: str = ""
    location: str = "未知地点"
    focus: str = ""
    assumptions: list[str] = []                 # Planner 在信息不足时所做的合理假设
    final_report: str = ""
