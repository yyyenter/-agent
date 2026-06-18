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

from pydantic import BaseModel


# ============================================
# 单步骤数据结构
# ============================================

class StepPlan(BaseModel):
    """单个执行步骤的计划与执行状态"""
    index: int
    description: str            # 步骤描述（不指定工具）
    tools: list[str] = []       # 需要的工具列表（StepPreparer 填充）
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

    # === 业务字段 ===
    is_complex: bool = True
    simple_answer: str = ""
    location: str = "未知地点"
    focus: str = ""
    assumptions: list[str] = []                 # Planner 在信息不足时所做的合理假设
    final_report: str = ""
