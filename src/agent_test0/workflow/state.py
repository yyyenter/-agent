# agent_test0/workflow/state.py
"""
状态机数据模型。

本文件是整个 Flow 的"数据契约"——所有节点读写的状态都定义在 TravelState 中。
任何节点新增字段一律先改这里，不要在节点内部偷偷往 state 上挂属性。

设计参考 LangGraph 风格 (TypedDict + Annotated reducers):
LangGraph 在底层只有"一个全局状态 + 归约器", 不区分业务/控制状态;
本项目使用 Pydantic BaseModel (累加/合并需代码显式 += / append),
字段按设计者心智分为 4 组, 物理上仍都在同一个 State 里:

  ┌─────────────────────────────────────────────────────────────┐
  │  1. Process / Control   流程控制: 循环/重试/节点/中断        │
  │  2. Working Data        步骤数据: steps / tool_calls / 结果  │
  │  3. Business Context    业务上下文: 用户/目的地/任务/话题     │
  │  4. Final Output        输出: 报告/追问/假设/警告             │
  └─────────────────────────────────────────────────────────────┘

跨轮持久 (Multi-turn Persistence):
  - 跨轮字段 (current_task_id / current_destination / current_topic)
    同时存在 MemoryManager (Redis) 和 TravelState (单轮);
    MemoryManager 是权威源, TravelState 是单轮缓存。
  - 单轮字段 (loop_count / current_step_index / final_report) 只在 TravelState。

重要约束:
  - 所有字段必须有默认值, 否则旧 session 反序列化时会因缺字段炸掉 (Pydantic v2)。
  - 不要在节点内部偷偷往 state 上挂属性 (extra='ignore' 会丢)。
"""

from typing import Any, Literal

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
    """Python 工具执行器返回的单次工具执行结果

    P0.2: 拆分为 output (结构化) + output_text (格式化文本):
      - output: 原始结构化数据 (dict / list / 标量), 若工具返回纯文本则存 str
      - output_text: 格式化后给 LLM 看的字符串 (含输入/错误信息, 长度受控)
      - 同时保留两个字段, 让 assemble_structured_plan 能用 output 拼装结构化 plan
    """
    tool_name: str
    input: dict = {}
    output: Any = None          # P0.2: 结构化 (None=无; dict/list/str=数据)
    output_text: str = ""       # P0.2: 格式化文本 (给 LLM 看)
    error: str = ""
    duration_ms: int = 0


class StepPlan(BaseModel):
    """单个执行步骤的计划与执行状态

    字段:
      - description: 任务级描述 (如"查询<目的地>天气"), 不是 draft itinerary
      - dependencies: DAG 依赖, 当前步骤的前置步骤 index 列表
        (Planner prompt 要求输出, Pydantic 之前因缺字段静默丢失)
      - tools/tool_calls: StepPreparer 填充的细粒度工具调用
    """
    index: int
    description: str            # 步骤描述（任务动词, 不指定工具）
    dependencies: list[int] = []    # DAG 依赖: 前置步骤 index 列表; [] 表示无依赖
    tools: list[str] = []       # 需要的工具名摘要（兼容旧逻辑）
    tool_calls: list[ToolCall] = []      # StepPreparer 生成的完整小计划（含参数）
    tool_results: list[ToolResult] = []  # Python StepExecutor 写入的结构化执行结果
    prepared: bool = False      # True 表示 StepPreparer 已处理；tool_calls=[] 表示无需工具
    status: str = "pending"     # pending | executing | completed | failed
    result: Any = None          # P0.2: 聚合的结构化结果 (来自 tool_results[].output)
    result_text: str = ""       # P0.2: 格式化的文本 (给 LLM 看)
    error: str = ""
    validation_feedback: str = ""  # StepVerifier 写入的反馈


class StepResult(BaseModel):
    """步骤执行结果归档（写入 state.step_results 历史）"""
    step_index: int
    step_description: str
    result: Any = None          # P0.2: 结构化结果
    result_text: str = ""       # P0.2: 文本结果
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
    # 缺失字段的结构化 key. 用于跨轮精确匹配 (asked_fields 去重).
    # LLM 必须从 ask_user.ASKABLE_FIELDS 候选池中选; 落到其他值会被归一化为 "unknown".
    missing_field: str = ""
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
    """PartialReplanner 的结构化输出 (P2.1: 改为 append-only)

    旧语义: 失败处之后整体替换 (new_coarse_steps, index 从失败点起)
    新语义: 已完成/失败步骤全部保留, 新步骤追加到末尾 (new_appended_steps, index 从 len(steps) 起)

    保留 new_coarse_steps 字段作为 fallback 兼容路径, 但节点优先用 new_appended_steps。
    """
    reason: str = ""
    preserved_steps: list[int] = []
    original_remaining_steps: list[int] = []
    new_appended_steps: list[StepPlan] = []   # P2.1: append-only 主路径
    new_coarse_steps: list[StepPlan] = []     # 兼容旧 prompt 输出
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


class DomainClassifierOutput(BaseModel):
    """DomainClassifier 的结构化输出: 用户消息触发了哪些领域.

    只在关键词硬匹配漏检时才调 (混合检测第 ②层).
    active_domains 里的字符串必须是 DOMAIN_FIELDS 的 key
    (companion / pet / activity / medical / international / logistics / occasion).
    LLM 输出的字符串若不在这些值里, 归一化时会被过滤.
    """
    active_domains: list[str] = []   # LLM 判断被激活的领域名
    reasoning: str = ""              # 可选说明, 便于调试


# ============================================
# Flow 全局状态 (4 类分组, LangGraph 风格)
# ============================================

class TravelState(BaseModel):
    """
    Flow 的全局状态容器。Pydantic 会自动持久化到 Redis（CrewAI Flow 内置）。
    所有字段必须有默认值，否则旧 session 反序列化时会因缺字段炸掉。

    字段 4 类分组 (物理上都是字段, 分组只用于设计者心智):
      1. Process / Control   流程控制
      2. Working Data        步骤数据
      3. Business Context    业务上下文
      4. Final Output        输出
    """

    # === 1. Process / Control (流程控制) ===
    current_node: str = ""                    # 当前在哪个节点 (debug / 排查用)
    loop_count: int = 0                       # 主循环总次数 (防主流程死循环)
    total_steps_counted: int = 0              # 已经进入步骤节点的总次数 (防步骤节点死循环)
    retry_count: int = 0                      # 当前步骤重试次数 (StepVerifier retry 上限)
    replan_count: int = 0                     # PartialReplanner 已触发次数 (replan 上限)
    final_verifier_done: bool = False         # FinalVerifier 是否已执行 (防重入)
    is_interrupted: bool = False              # 流程是否已被 AskUser 中断
    is_done: bool = False                     # 流程是否已结束 (FinalReport 已生成)
    # 重试计数与上限 (方案A1: 从 TravelWorkflow 实例属性搬进 state, 进 checkpoint)
    # 之前是 flow.step_retry_counts, 跨轮不持久 (飞书多轮重试计数会丢); 现入 state 修复。
    step_retry_counts: dict[int, int] = {}    # 每个步骤 index → 已重试次数
    max_step_retries: int = 3                 # 单步骤最大重试次数 (原 DEFAULT_MAX_STEP_RETRIES)
    max_replan_attempts: int = 3              # 最大重规划次数 (原 DEFAULT_MAX_REPLAN_ATTEMPTS)
    max_asks: int = 3                         # 单会话追问总次数上限 (硬护栏, 防"审问用户")

    # === 2. Working Data (步骤数据) ===
    steps: list[StepPlan] = []                # 步骤列表
    current_step_index: int = 0               # 当前执行步骤索引
    step_results: list[StepResult] = []       # 步骤执行结果历史
    failed_steps_indices: list[int] = []      # 失败步骤索引列表
    last_validation: ValidationFeedback | None = None  # 最近一次验证反馈 (跨节点共享)

    # === 3. Business Context (业务上下文) ===
    # --- 3a. 用户与会话标识 ---
    message: str = ""                         # 当前用户消息原文
    user_id: str = "default_user"
    session_id: str = "default_sess"

    # --- 3b. 上下文拼装 ---
    focus: str = ""                           # 渲染后的 LLM 上下文 (来自 MemoryManager)

    # --- 3c. 当前轮推断结果 (Planner / 意图路由后填) ---
    is_complex: bool = True                   # True 走完整流程, False 走 simple_answer
    simple_answer: str = ""                   # 简单问题直接回答
    location: str = "未知地点"                # 兼容旧字段, 新代码用 current_destination

    # --- 3d. 跨轮业务字段 (从 MemoryManager 同步过来, 默认 None) ---
    current_task_id: str | None = None        # 当前任务 ID (跨轮持久)
    current_destination: str | None = None    # 当前目的地 (跨轮持久)
    current_topic: str = "general"            # 当前话题: trip_planning / ask_user / slot_answer / invalid_reply / delegation / chitchat
    is_invalid_reply: bool = False            # 当前消息像无效短词 (ff / asd / ...)
    is_delegation: bool = False               # 当前消息是授权默认 (看你安排 / 随便)

    # === 4. Final Output (输出) ===
    needs_user_input: bool = False            # True 时主流程中断, 向用户提问
    user_question: str = ""                   # needs_user_input=True 时填入追问文本 (LLM 生成)
    user_choice: str = ""                     # 用户从选项中选的结果 (新版, 暂未启用)
    skip_remaining_steps: bool = False        # 跳过剩余步骤直接输出结果

    final_report: str = ""                    # 最终给用户看的报告 (narrative 软装)
    structured_plan: dict = {}                # P1.1: 确定性组装出的结构化行程 (data inventory + warnings)
    assumptions: list[str] = []               # Planner 在信息不足时所做的合理假设
    warnings: list[str] = []                  # 流程中产生的告警 (硬约束违例 / 工具失败等)
    asked_fields: list[str] = []              # 本轮已经问过用户的字段名 (防重复问)
    scoping_offered: bool = False             # 是否已向用户展示过"领域菜单" (主动 scoping 引导, 只推一次)
