# agent_test0/workflow/nodes.py
"""
状态机 6 个节点的业务逻辑 (方案 A1: LangGraph state 透传).

【与 agent_practice 版的差异】
主项目版保留了几个生产必需的护栏, agent_practice 版为简化没抄:
  - guard 系列: 强制追问天数/预算/人数; assumptions 校验; LLM 生成追问文本
  - assemble_structured_plan / _check_deterministic_rules: P1.1/1.2 结构化 plan + 确定性规则
  - _generate_final_report: narrative 翻译, 优先用结构化 plan 而非自由文本

【节点标准三步】
  ① 从 state 组装 slots (dict, key 与 YAML 中的占位符一一对应)
  ② prompt = load_task_prompt(task_name, slots)
     out    = call_structured(prompt, model_cls=XxxOutput)
  ③ 把 out 的字段 reduce 回 state, 返回 _dirty(state) 全量增量

【关键词对齐】
slots dict 的 key 必须与 config/*.yaml 中的 {xxx} 占位符完全一致.
_safe_render 在缺 slot 时会一次列出所有缺失名, 定位准确.

【AskUser】
节点不再抛异常; 只写 state.needs_user_input, 条件边判定 → END.
重试计数 step_retry_counts / 上限 max_step_retries 已入 state.
"""
from __future__ import annotations

import json
import re

from agent_test0.workflow.state import (
    StepPlan,
    StepResult,
    ToolCall,
    ToolResult,
    PlannerOutput,
    StepPreparerOutput,
    StepVerifierOutput,
    ReplanOutput,
    FinalVerifierOutput,
    TravelState,
)
from agent_test0.workflow.structured import call_structured, load_task_prompt, StructuredCallError
from agent_test0.workflow.llm import zhipu_llm
from agent_test0.workflow.trace import timed
from agent_test0.workflow.ask_user import request_user_input, build_field_pool
from agent_test0.tools.registry import execute_tool_calls, format_for_llm


# 死循环保护: 单轮 Flow 内步骤迭代总次数上限 (含 retry / replan 重跑)
MAX_STEP_ITERATIONS = 30

# 用户明确授权系统自行默认/假设的表达. 只有第二轮 (已问过) 且命中这些表达, 才允许假设.
_USER_DELEGATION_PATTERN = re.compile(r"(看你安排|你安排|随便|无所谓|都可以|都行|默认|按常规|帮我定|你决定)")
_WEATHER_ONLY_PATTERN = re.compile(r"(天气|气温|下雨|温度|穿什么)")


# ============================================================
# 通用工具函数
# ============================================================

def _dirty(state: TravelState) -> dict:
    """把 state 全量 dump 成 dict, 作为 LangGraph 节点的返回增量.

    节点函数就地改多字段时, LangGraph 需要显式返回 dict 才会合并回全局 state.
    """
    return state.model_dump(mode="json")


def _set_ask_user(state: TravelState, question: str) -> None:
    """AskUser 写入 (方案A1: 内联)."""
    state.needs_user_input = True
    state.user_question = question
    state.final_report = question  # final_report 兜底也带上问题


def _check_ask_user(state: TravelState) -> bool:
    return bool(state.needs_user_input)


def _notify(config, text: str) -> None:
    """从 RunnableConfig 取 notify 回调并调用."""
    if config:
        notify = config.get("configurable", {}).get("notify")
        if notify:
            try:
                notify(text)
            except Exception as e:
                print(f"[notify] 回调异常 (忽略): {e}")
    print(f"[Flow] {text}")


def _format_tool_results_text(results: list[ToolResult]) -> str:
    """把一组 ToolResult 拼成 LLM 友好的文本 (registry 只提供单工具的 format_for_llm)."""
    if not results:
        return "(无工具输出)"
    lines = []
    for r in results:
        if r.error:
            lines.append(f"[{r.tool_name}] ERROR: {r.error}")
        else:
            lines.append(r.output_text or f"[{r.tool_name}] {r.output}")
    return "\n".join(lines)


# ============================================================
# Planner 护栏 (只在 planner_node 用)
# ============================================================

def _has_asked_trip_constraints(state: TravelState) -> bool:
    """最近几轮 assistant 是否已经问过天数/预算/人数等次要约束."""
    assistant_lines = []
    in_assistant = False
    for line in (state.focus or "").splitlines():
        if line.startswith("assistant:"):
            in_assistant = True
            assistant_lines.append(line.removeprefix("assistant:").strip())
            continue
        if line.startswith("user:") or line.startswith("system:"):
            in_assistant = False
            continue
        if in_assistant:
            assistant_lines.append(line.strip())
    asked_text = "\n".join(assistant_lines)
    return any(kw in asked_text for kw in ("计划玩几天", "大概预算", "几位出行", "几个人", "天数", "预算", "人数"))


def _user_delegated_defaults(state: TravelState) -> bool:
    """用户本轮是否明确授权系统自行安排."""
    return bool(_USER_DELEGATION_PATTERN.search(state.message or ""))


def _looks_like_travel_planning(state: TravelState, plan_data: dict) -> bool:
    """判断是否是行程规划类需求, 而非纯天气查询."""
    message = state.message or ""
    if plan_data.get("simple_answer") and not plan_data.get("steps"):
        return False
    if _WEATHER_ONLY_PATTERN.search(message):
        return False
    return bool(plan_data.get("steps"))


def _user_constraint_text(state: TravelState) -> str:
    """只看当前用户消息中的约束, 不看历史."""
    return state.message or ""


def _missing_secondary_constraints(state: TravelState, plan_data: dict) -> list[str]:
    """代码级判断行程规划中是否缺天数/预算/人数."""
    text = _user_constraint_text(state)
    missing = []
    if not re.search(r"\d+\s*天|一日|一天|两天|三天|四天|五天|周末", text):
        missing.append("天数")
    if not re.search(r"\d+\s*(元|块|k|K|千|万)|预算\s*[:：]?\s*\d+|人均\s*\d+|经济型|中等预算|高端|豪华", text):
        missing.append("预算")
    if not re.search(r"\d+\s*(人|位)|一个人|一人|两个人|两人|独自|情侣|夫妻|家庭|亲子|朋友|同学|团队", text):
        missing.append("人数")
    return missing


def _llm_followup_question(state: TravelState, plan_data: dict, reason: str, missing: list[str] | None = None) -> str:
    """让 LLM 根据当前输入和上下文生成追问文本; 代码只决定是否需要问."""
    missing_text = "、".join(missing or []) or "由你判断"
    prompt = f"""你是旅游规划助手. 现在不能继续生成行程, 需要向用户追问.

【当前用户输入】
{state.message}

【近期上下文】
{state.focus}

【Planner 初步判断】
{json.dumps(plan_data, ensure_ascii=False)[:2000]}

【不能继续的原因】
{reason}

【已检测到缺失/无效的信息】
{missing_text}

请生成一条自然、简洁、贴合当前输入的中文追问. 要求:
1. 不要机械套模板, 不要声称"某地行程没问题", 除非当前用户这句话确实表达了新目的地.
2. 如果当前输入像乱码、无意义短词或没有回答上一轮问题, 请先说明没有理解, 再引导用户补充有效信息.
3. 如果缺少天数/预算/人数, 请一次问全.
4. 可以提醒用户如果无特别要求, 也可以授权你按常规方案安排.
5. 只输出最终要发给用户的一段话, 不要 JSON, 不要解释."""
    try:
        with timed("LLM:AskUserQuestion"):
            question = zhipu_llm.call([{"role": "user", "content": prompt}]).strip()
        if question:
            return question[:500]
    except Exception as e:
        print(f"[PlannerGuard] LLM 生成追问失败: {e}")
    return "我还需要补充一些关键信息才能继续规划. 请说明计划玩几天、大概预算和几位出行; 如果没有特别要求, 也可以让我按常规方案安排."


def _enforce_first_turn_question(state: TravelState, plan_data: dict) -> bool:
    """Planner 代码级护栏: 行程规划第一次缺天数/预算/人数时, 强制提问.

    Returns True 表示已经写入 AskUser (真的中断), 调用方应立即 return.
             False 表示三层护栏拦截了 (已问过/超上限) 或本来就不需要问, 继续往下走.
    """
    location = (plan_data.get("location") or "").strip()
    if not location or location in ("未知", "未指定", "不明"):
        return False
    if not _looks_like_travel_planning(state, plan_data):
        return False

    missing = _missing_secondary_constraints(state, plan_data)
    if not missing:
        return False

    already_asked = _has_asked_trip_constraints(state)
    delegated = _user_delegated_defaults(state)
    if already_asked and delegated:
        print(f"[PlannerGuard] 已问过且用户授权默认, 允许假设: missing={missing}")
        return False

    reason = "用户本轮输入不足以继续规划, 不能直接沿用历史目的地或历史约束."
    if already_asked and not delegated:
        reason = "上一轮已经追问过关键信息, 但用户本轮没有明确授权默认, 也没有补全全部必要信息."
    question = _llm_followup_question(state, plan_data, reason, missing)

    # 缺失映射到候选池 key. 一次问全, 只以第一个缺失字段作为 asked_fields 记账.
    # (missing 里的中文标签 → 候选池 snake_case key)
    field_map = {"天数": "trip_days", "预算": "budget", "人数": "group_size"}
    primary_field = field_map.get(missing[0], "unknown")

    print(f"[PlannerGuard] 强制提问: 缺失 {missing}, already_asked={already_asked}, delegated={delegated}")
    # 走三层护栏: 命中拦截就返回 False, 让调用方继续 (走 assumptions)
    return request_user_input(state, primary_field, question)


# ============================================================
# 节点 ①  Planner
# ============================================================

def planner_node(state: TravelState, config=None) -> dict:
    """复杂度判定 + 步骤生成 (或追问 / 简单回复)."""
    if _check_ask_user(state):
        return _dirty(state)

    print(f"\n[Planner] 开始规划: message={state.message[:60]!r}")
    _notify(config, "📋 [Planner] 决策大脑正在建立行程执行策略...")

    if not state.steps:
        # ── ① 组装 slots (key 严格对齐 planning_task 的占位符) ──
        # field_pool: 动态候选池 (BASE + 触发的 DOMAIN + UNKNOWN), 供 LLM 选 missing_field
        #   由 build_field_pool 走"关键词硬匹配 → LLM 兜底分类"混合检测生成
        slots = {
            "message":       state.message,
            "user_id":       state.user_id,
            "focus":         state.focus or "无",
            "previous_plan": "无历史计划 (首次规划)",
            "current_step":  "无当前工单 (首次执行)",
            "current_draft": "无进度草案",
            "asked_fields":  ", ".join(state.asked_fields) if state.asked_fields else "(暂无)",
            "field_pool":    build_field_pool(state),
        }

        # ── ② prompt + 结构化调用 ──
        try:
            with timed("LLM:planner"):
                prompt = load_task_prompt("planning_task", slots)
                out: PlannerOutput = call_structured(prompt, model_cls=PlannerOutput)
        except StructuredCallError as e:
            print(f"[Planner] 结构化调用失败: {e}")
            state.final_report = "抱歉, 规划过程出现异常, 请稍后重试."
            return _dirty(state)

        plan_data = out.model_dump()
        print(f"[Planner] 结构化输出: {json.dumps(plan_data, ensure_ascii=False)[:500]}")

        # ── ③ reduce 回 state (guard 需要原始 focus, 先算完再覆盖) ──
        original_focus = state.focus
        already_asked = _has_asked_trip_constraints(state)
        delegated = _user_delegated_defaults(state)

        state.is_complex    = out.is_complex
        state.simple_answer = out.simple_answer
        state.location      = out.location or "未知"

        inferred_dest = (out.location or "").strip()
        if inferred_dest and inferred_dest not in ("未知", "未指定", "不明"):
            state.current_destination = inferred_dest

        # 非授权默认时, 不接受 Planner 自行编造的 assumptions
        if already_asked and delegated:
            state.assumptions = list(out.assumptions or [])
        else:
            state.assumptions = []

        # 代码级护栏 (需要 original_focus 判 already_asked, 所以 focus 先不覆盖)
        state.focus = original_focus
        if _enforce_first_turn_question(state, plan_data):
            return _dirty(state)
        state.focus = out.focus or state.focus

        # Planner 直接判定要追问 (场景 A / B 第一次)
        if out.needs_user_input:
            # 走三层护栏: 归一化候选池 + 去重 + max_asks 上限
            interrupted = request_user_input(
                state,
                out.missing_field or "unknown",     # LLM 输出的字段 key, 会归一化
                out.user_question or "请补充更多信息.",
            )
            if interrupted:
                return _dirty(state)
            # 三层拦截了 (已问过或超上限) → 走假设兜底, 继续下面的规划逻辑
            print(f"[Planner] AskUser 被三层拦截, 走假设继续规划")

        # 有 steps → 存进 state
        if out.steps:
            state.steps = list(out.steps)
            print(f"[Planner] 生成了 {len(state.steps)} 个粗粒度步骤")
        else:
            # 兜底: 未生成 steps, 直接返回 simple_answer
            fallback = out.simple_answer or "抱歉, 我暂时无法理解您的需求, 请补充更多信息."
            state.final_report = fallback
            print(f"[Planner] 兜底: 未生成 steps, 直接返回 simple_answer")
            _notify(config, "⚠️ [Planner] 未生成多步骤计划, 直接返回简要回答")
            return _dirty(state)

    state.current_step_index = 0
    return _dirty(state)


# ============================================================
# 节点 ②  StepPreparer
# ============================================================

def step_preparer_node(state: TravelState, config=None) -> dict:
    """为当前步骤决定调哪些工具、传什么参数. 填完 tools 即返回, 不串联下游."""
    if _check_ask_user(state):
        return _dirty(state)

    # Planner 已给出兜底 final_report → 直接结束
    if state.final_report and not state.steps:
        return _dirty(state)

    if not state.steps:
        state.final_report = "抱歉, 我没能为您的需求规划出执行步骤, 请补充更多信息后再试."
        return _dirty(state)

    idx = state.current_step_index
    if idx >= len(state.steps):
        return _dirty(state)

    step = state.steps[idx]
    if step.status == "completed":
        return _dirty(state)
    if step.prepared:
        # replan / retry 复用已有 tool_calls
        print(f"[StepPreparer] 跳过 LLM 规划 (已有小计划 {step.tools})")
        return _dirty(state)

    print(f"\n[StepPreparer] 为步骤 {idx} 生成工具调用计划: {step.description[:50]}")
    _notify(config, f"📋 [StepPreparer] 步骤 {idx} 规划中...")

    # ── ① 组装 slots ──
    previous_results = {
        str(r.step_index): (r.result_text or str(r.result))
        for r in state.step_results if r.passed
    }
    slots = {
        "step_index":            idx,
        "step_goal":             step.description,
        "previous_step_results": json.dumps(previous_results, ensure_ascii=False),
        "global_constraints":    json.dumps({
            "user_id":  state.user_id,
            "location": state.location,
            "focus":    state.focus,
        }, ensure_ascii=False),
    }

    # ── ② prompt + 结构化调用 ──
    try:
        with timed("LLM:step_preparer"):
            prompt = load_task_prompt("step_preparer_task", slots)
            out: StepPreparerOutput = call_structured(prompt, model_cls=StepPreparerOutput)
    except StructuredCallError as e:
        print(f"[StepPreparer] 结构化调用失败: {e}")
        # 天气类步骤做确定性兜底; 其它步骤按无需工具处理
        if "天气" in step.description and state.location not in ("", "未知", "未知地点"):
            step.tool_calls = [ToolCall(order=1, tool_name="weather_tool", parameters={"city": state.location})]
        else:
            step.tool_calls = []
        step.tools = [t.tool_name for t in step.tool_calls]
        step.prepared = True
        return _dirty(state)

    # ── ③ reduce 回 state ──
    step.tool_calls = list(out.tools_to_call or [])
    step.tools = [t.tool_name for t in step.tool_calls]
    step.prepared = True
    print(f"[StepPreparer] 步骤 {idx} 小计划: {step.tools or '(无需工具)'}")
    return _dirty(state)


# ============================================================
# 节点 ③  StepExecutor —— 不调 LLM, 纯 Python 工具执行
# ============================================================

def step_executor_node(state: TravelState, config=None) -> dict:
    """按 step.tool_calls 执行工具, 写入 step.result / step.result_text."""
    if _check_ask_user(state):
        return _dirty(state)

    if not state.steps:
        return _dirty(state)

    idx = state.current_step_index
    if idx >= len(state.steps):
        return _dirty(state)

    step = state.steps[idx]
    if step.status == "completed":
        return _dirty(state)
    if step.status == "executing":
        print(f"[StepExecutor] 步骤 {idx} 正在执行中, 跳过重复调用")
        return _dirty(state)

    # DAG 依赖检查 (当前线性执行, 未满足只 warn)
    if step.dependencies:
        unmet = [
            d for d in step.dependencies
            if d < 0 or d >= len(state.steps) or state.steps[d].status != "completed"
        ]
        if unmet:
            print(f"[StepExecutor] ⚠️ 步骤 {idx} 依赖 {unmet} 尚未完成, 但当前为线性执行")

    step.status = "executing"
    print(f"\n[StepExecutor] 执行步骤 {idx}: {step.description[:50]}")
    _notify(config, f"🛠️ [StepExecutor] 步骤 {idx} 调用工具中...")

    if not step.prepared:
        print(f"[StepExecutor] 警告: 步骤 {idx} 尚未 prepare, 按无需工具处理")
        step.prepared = True

    # 无 tool_calls → 直接标 completed
    if not step.tool_calls:
        step.tool_results = []
        step.result = None
        step.result_text = "(无外部工具调用)"
        step.status = "completed"
        state.step_results.append(StepResult(
            step_index=idx, step_description=step.description,
            result=None, result_text=step.result_text,
            passed=True, validation_feedback="",
        ))
        print(f"[StepExecutor] 步骤 {idx} 无需外部工具, 直接完成")
        return _dirty(state)

    # 有 tool_calls → 交给 registry 执行
    results: list[ToolResult] = execute_tool_calls(step.tool_calls)
    step.tool_results = results
    step.result_text = _format_tool_results_text(results)

    # 聚合结构化 output: 1 个 → 直接用; 多个 → 打包 dict
    outputs = [r.output for r in results if r.output is not None]
    if len(outputs) == 1:
        step.result = outputs[0]
    elif len(outputs) > 1:
        step.result = {"items": outputs}
    else:
        step.result = None

    has_error = any(r.error for r in results)
    step.status = "failed" if has_error else "completed"
    step.error = "; ".join(r.error for r in results if r.error)

    state.step_results.append(StepResult(
        step_index=idx, step_description=step.description,
        result=step.result, result_text=step.result_text,
        passed=not has_error, validation_feedback="",
    ))

    print(f"[StepExecutor] 步骤 {idx} 完成: {'失败' if has_error else '成功'}")
    return _dirty(state)


# ============================================================
# 节点 ④  StepVerifier
# ============================================================

def step_verifier_node(state: TravelState, config=None) -> dict:
    """审核单步骤结果, 决定 pass / retry / fail / ask_user (由条件边路由)."""
    if _check_ask_user(state):
        return _dirty(state)

    if not state.steps:
        return _dirty(state)

    idx = state.current_step_index
    if idx >= len(state.steps):
        return _dirty(state)

    step = state.steps[idx]

    # 跳过已处理的步骤 (防重复触发)
    if step.status in ("completed", "failed") and step.validation_feedback:
        print(f"[StepVerifier] 步骤 {idx} 已处理过, 跳过")
        return _dirty(state)

    print(f"\n[StepVerifier] 审核步骤 {idx}")
    _notify(config, f"🔍 [StepVerifier] 步骤 {idx} 审核中...")

    # 【确定性短路】已 completed 且有非空 result → 直接 pass
    _has_result = bool(
        step.result is not None and step.result != "" and step.result != {} and step.result != []
    )
    if step.status == "completed" and _has_result:
        print(f"[StepVerifier] 短路 pass: 步骤 {idx} 有非空结果")
        _notify(config, f"✅ [StepVerifier] 步骤 {idx} 直接通过 (有数据)")
        step.validation_feedback = "有非空结果, 直接通过"
        state.current_step_index += 1
        return _dirty(state)

    # ── ① 组装 slots ──
    slots = {
        "step_index":        idx,
        "step_goal":         step.description,
        "execution_plan":    json.dumps({
            "tools":      step.tools,
            "tool_calls": [t.model_dump() for t in step.tool_calls],
        }, ensure_ascii=False),
        "execution_results": json.dumps(
            [r.model_dump() for r in step.tool_results], ensure_ascii=False, default=str,
        ) if step.tool_results else (step.result_text or ""),
    }

    # ── ② prompt + 结构化调用 ──
    try:
        with timed("LLM:step_verifier"):
            prompt = load_task_prompt("step_validator_task", slots)
            out: StepVerifierOutput = call_structured(prompt, model_cls=StepVerifierOutput)
    except StructuredCallError as e:
        print(f"[StepVerifier] 结构化调用失败, 默认通过: {e}")
        step.status = "completed"
        step.validation_feedback = "审核异常, 默认通过"
        state.current_step_index += 1
        return _dirty(state)

    step.validation_feedback = out.reason

    # ── ③ 按 verdict reduce ──
    if out.verdict == "ask_user":
        # 走三层护栏; verifier 阶段的追问不知道具体字段, 用 unknown 兜底
        interrupted = request_user_input(state, "unknown", out.question or "信息不足, 请补充.")
        if interrupted:
            return _dirty(state)
        # 三层拦截 → 视为 pass 继续 (assumptions 已写入, 由 final_report 披露)
        step.status = "completed"
        step.validation_feedback = "追问被护栏拦截, 走假设通过"
        state.current_step_index += 1
        print(f"[StepVerifier] 步骤 {idx} 追问被护栏拦截, 走假设通过")
        return _dirty(state)

    if out.verdict == "pass":
        step.status = "completed"
        state.current_step_index += 1
        print(f"[StepVerifier] 步骤 {idx} 审核通过")
        _notify(config, f"✅ [StepVerifier] 步骤 {idx} 审核通过")
        return _dirty(state)

    if out.verdict == "retry":
        cnt = state.step_retry_counts.get(idx, 0)
        if cnt < state.max_step_retries:
            state.step_retry_counts[idx] = cnt + 1
            step.status = "pending"
            step.result = None
            step.result_text = ""
            step.error = ""
            step.tool_results = []
            step.validation_feedback = ""
            print(f"[StepVerifier] 步骤 {idx} 重试 ({cnt+1}/{state.max_step_retries})")
            _notify(config, f"🔄 [StepVerifier] 步骤 {idx} 重试中...")
            return _dirty(state)
        # 重试耗尽 → fail
        step.status = "failed"
        step.validation_feedback = f"重试 {state.max_step_retries} 次后仍失败"
        state.failed_steps_indices.append(idx)
        state.current_step_index += 1
        print(f"[StepVerifier] 步骤 {idx} 重试耗尽, 标记失败并跳过")
        return _dirty(state)

    # verdict == "fail" → 交给 partial_replanner (由 graph 条件边路由)
    step.status = "failed"
    state.failed_steps_indices.append(idx)
    print(f"[StepVerifier] 步骤 {idx} 判 fail, 将触发 PartialReplanner")
    return _dirty(state)


# ============================================================
# 节点 ⑤  PartialReplanner —— append-only 局部重规划
# ============================================================

def partial_replanner_node(state: TravelState, config=None, failure_feedback: dict | None = None) -> dict:
    """保留所有已有 steps (completed + failed), 仅追加补救任务."""
    if _check_ask_user(state):
        return _dirty(state)

    state.replan_count += 1
    if state.replan_count > state.max_replan_attempts:
        print(f"[PartialReplanner] 重规划超限, 强制结束")
        state.final_report = _generate_final_report(state)
        state.is_done = True
        return _dirty(state)

    print(f"\n[PartialReplanner] 第 {state.replan_count} 次重规划")
    _notify(config, f"🔄 [PartialReplanner] 追加补救任务中...")

    failed_indices = list(set(state.failed_steps_indices))
    state.failed_steps_indices = []
    if not failed_indices:
        print(f"[PartialReplanner] 没有失败步骤, 无法重规划")
        return _dirty(state)

    # 保留全部已有步骤 (append-only)
    preserved_steps = list(range(len(state.steps)))
    original_failed = [
        i for i in range(len(state.steps))
        if state.steps[i].status == "failed" or i in failed_indices
    ]
    print(f"[PartialReplanner] 保留: {preserved_steps}, 历史失败: {original_failed}")

    preserved_results = {}
    for i in preserved_steps:
        s = state.steps[i]
        text = s.result_text or (s.result if isinstance(s.result, str) else "")
        preserved_results[str(i)] = {
            "status":      s.status,
            "description": s.description,
            "result":      text or "(无输出)",
        }
        if i in original_failed:
            preserved_results[str(i)]["error"] = s.error

    # ── ① 组装 slots ──
    slots = {
        "failure_reason":           (failure_feedback or {}).get("reason", ""),
        "failed_step_indices":      json.dumps(failed_indices),
        "suggested_corrections":    json.dumps(
            (failure_feedback or {}).get("suggested_corrections", {}), ensure_ascii=False,
        ),
        "preserved_steps_results":  json.dumps(preserved_results, ensure_ascii=False, default=str),
        "original_remaining_steps": json.dumps(original_failed),
    }

    # ── ② prompt + 结构化调用 ──
    try:
        with timed("LLM:partial_replanner"):
            prompt = load_task_prompt("replan_task", slots)
            out: ReplanOutput = call_structured(prompt, model_cls=ReplanOutput)
    except StructuredCallError as e:
        print(f"[PartialReplanner] 结构化调用失败: {e}")
        return _dirty(state)

    # ── ③ append 新步骤到末尾 ──
    new_steps = list(out.new_appended_steps) or list(out.new_coarse_steps)
    if not new_steps:
        print(f"[PartialReplanner] 警告: 未生成新步骤")
        return _dirty(state)

    next_idx = len(state.steps)
    for s in new_steps:
        step_obj = s if isinstance(s, StepPlan) else StepPlan(**s)
        if step_obj.index < next_idx:
            print(f"[PartialReplanner] 修正 index {step_obj.index} → {next_idx}")
            step_obj.index = next_idx
        state.steps.append(step_obj)
        next_idx += 1

    state.current_step_index = len(state.steps) - len(new_steps)
    print(f"[PartialReplanner] 追加 {len(new_steps)} 步, 从 index={state.current_step_index} 继续")
    return _dirty(state)


# ============================================================
# 节点 ⑥  FinalVerifier —— 结构化 plan + 确定性规则 + LLM 软检查 + narrative
# ============================================================

def assemble_structured_plan(state: TravelState) -> dict:
    """P1.1: 确定性结构化行程组装器. 不调 LLM, 纯代码聚合已完成 steps."""
    plan = {
        "destination": state.location,
        "user_query":  state.message,
        "assumptions": list(state.assumptions or []),
        "steps":       [],
        "data_sources": [],
        "warnings":    [],
        "failed_steps": list(state.failed_steps_indices or []),
    }
    seen_tools = set()

    for i, s in enumerate(state.steps or []):
        if s.status not in ("completed", "failed"):
            plan["warnings"].append(f"步骤 {i} 未完成 (status={s.status}): {s.description}")
            continue

        plan["steps"].append({
            "index":       i,
            "description": s.description,
            "tools_used":  list(s.tools or []),
            "data":        s.result,  # 结构化 (来自 tool_results[].output)
            "status":      s.status,
        })

        if s.status == "failed":
            plan["warnings"].append(
                f"步骤 {i} 失败: {s.description} | error: {s.error[:200] if s.error else '(unknown)'}"
            )
            continue

        for tool_name in s.tools or []:
            if tool_name not in seen_tools:
                plan["data_sources"].append(tool_name)
                seen_tools.add(tool_name)

    if not plan["steps"]:
        plan["warnings"].append("没有完成的步骤, 无法组装结构化 plan")
    return plan


def _check_deterministic_rules(plan: dict) -> list[str]:
    """P1.2: 确定性规则检查器. 返回失败原因列表 (空 = 全部通过)."""
    failures = []
    completed = [s for s in plan.get("steps", []) if s.get("status") == "completed"]

    if not completed:
        failures.append("R1: 没有已完成的步骤")

    if plan.get("failed_steps"):
        failures.append(f"R2: 有 {len(plan['failed_steps'])} 个失败步骤: {plan['failed_steps']}")

    has_data = any(s.get("data") not in (None, "", [], {}) for s in completed)
    if completed and not has_data:
        failures.append("R3: 所有 completed 步骤都没有结构化数据")

    return failures


def final_verifier_node(state: TravelState, config=None) -> dict:
    """整体审核: 确定性规则 → LLM 软检查 → narrative 报告."""
    if state.final_verifier_done:
        return _dirty(state)
    if _check_ask_user(state):
        return _dirty(state)
    if state.current_step_index < len(state.steps):
        print(f"[FinalVerifier] 跳过: 还有 {len(state.steps) - state.current_step_index} 步未完成")
        return _dirty(state)

    print(f"\n[FinalVerifier] 开始整体审核")
    _notify(config, f"🔍 [FinalVerifier] 整体审核中...")
    state.final_verifier_done = True

    # P1.1: 确定性组装结构化 plan
    structured_plan = assemble_structured_plan(state)
    state.structured_plan = structured_plan
    print(f"[FinalVerifier] 结构化 plan: {len(structured_plan['steps'])} 步, "
          f"data_sources={structured_plan['data_sources']}, warnings={len(structured_plan['warnings'])}")

    # P1.2: 确定性规则检查, 不通过直接走 replan (不浪费 LLM)
    rule_failures = _check_deterministic_rules(structured_plan)
    if rule_failures:
        print(f"[FinalVerifier] 确定性规则不通过: {rule_failures}")
        _notify(config, f"⚠️ [FinalVerifier] 规则不通过: {rule_failures[0]}")
        feedback = {
            "verdict": "fail",
            "reason":  f"确定性规则不通过: {'; '.join(rule_failures)}",
            "failed_step_ids": list(structured_plan.get("failed_steps", [])),
        }
        if feedback["failed_step_ids"]:
            state.failed_steps_indices = feedback["failed_step_ids"]
        state.final_verifier_done = False  # 允许重规划后重跑
        return _dirty(state)

    # 规则通过 → LLM 做语义软检查 (用户硬约束 / 跨步矛盾)
    # ── ① 组装 slots ──
    slots = {
        "all_steps_with_results": json.dumps(structured_plan["steps"], ensure_ascii=False, default=str),
        "full_plan_document":     "\n".join(
            f"步骤 {i}: {s.description}" for i, s in enumerate(state.steps)
        ),
    }

    # ── ② prompt + 结构化调用 ──
    try:
        with timed("LLM:final_verifier"):
            prompt = load_task_prompt("final_validator_task", slots)
            out: FinalVerifierOutput = call_structured(prompt, model_cls=FinalVerifierOutput)
    except StructuredCallError as e:
        print(f"[FinalVerifier] 结构化调用失败, 默认通过: {e}")
        state.final_report = _generate_final_report(state)
        state.is_done = True
        return _dirty(state)

    print(f"[FinalVerifier] 结构化反馈: verdict={out.global_verdict}, reason={out.reason[:100]}")

    # ── ③ reduce ──
    if out.global_verdict == "pass":
        print(f"[FinalVerifier] 整体审核通过")
        _notify(config, f"🎉 [FinalVerifier] 整体审核通过")
        state.final_report = _generate_final_report(state)
        state.is_done = True
        return _dirty(state)

    # fail_with_patches → 触发局部重规划
    print(f"[FinalVerifier] LLM 软检查不通过, 触发局部重规划")
    _notify(config, f"⚠️ [FinalVerifier] 整体审核不通过")
    if out.failed_step_ids:
        state.failed_steps_indices = list(out.failed_step_ids)
    state.final_verifier_done = False  # 允许重规划后重跑
    return _dirty(state)


# ============================================================
# 报告生成 + finalize
# ============================================================

def _generate_final_report(state: TravelState) -> str:
    """LLM 把结构化 plan 或步骤结果翻译成用户友好的中文报告.

    优先用 state.structured_plan (P1.1 组装), 否则 fallback 到 result_text 聚合.
    禁止 LLM 创造结构化 plan 之外的数据.
    """
    structured = state.structured_plan if isinstance(state.structured_plan, dict) and state.structured_plan else None

    assumptions_block = ""
    if state.assumptions:
        bullets = "\n".join(f"- {a}" for a in state.assumptions)
        assumptions_block = f"\n【系统所做的关键假设 (必须在报告开头以 📌 形式向用户披露, 并提示用户可调整)】\n{bullets}\n"

    if structured:
        plan_json = json.dumps(structured, ensure_ascii=False, indent=2, default=str)[:6000]
        prompt = f"""你是一位资深旅游规划师, 负责把【结构化行程数据】翻译成用户友好的中文报告.

【用户需求】
- 目的地: {state.location}
- 关注重点: {state.focus}
- 用户原话: {state.message[:500]}
{assumptions_block}
【结构化行程数据 (JSON)】
{plan_json}

【撰写要求】
1. 严格基于结构化数据翻译, 不得编造数据中没有的景点/天气/预算数字.
2. 不要描述"执行步骤" (如"步骤 0 完成了查询").
3. 包含以下板块 (数据中没有的板块直接跳过, 不要硬编):
   - 目的地概况
   - 天气与最佳出行建议 (有 weather data 时)
   - 行程要点 (按 description 任务名)
   - 注意事项 (来自 warnings)
4. 使用清晰的小标题、emoji 和分段.
5. 总字数控制在 800 字以内.
6. 如果有 warnings 字段, 在报告末尾用 ⚠️ 列出前 3 条."""
    else:
        if not state.steps:
            return "未能生成行程报告"
        lines = []
        for s in state.steps:
            marker = "✅" if s.status == "completed" else "⚠️"
            text = s.result_text or (s.result if isinstance(s.result, str) else "")
            lines.append(f"[{marker}] {text or s.description + ' - ' + ('已完成' if s.status == 'completed' else '未完成')}")
        collected = "\n\n".join(lines)
        prompt = f"""你是一位资深旅游规划师. 请根据以下执行数据, 为用户撰写一份完整的旅行计划报告.

【用户需求】
- 目的地: {state.location}
- 关注重点: {state.focus}
- 用户原话: {state.message[:500]}
{assumptions_block}
【已收集的数据】
{collected}

【撰写要求】
1. 绝不要重复原始的执行步骤描述 (如"查询天气"、"获取偏好"等).
2. 把所有数据整合成一份自然流畅的旅行计划.
3. 总字数控制在 800 字以内."""

    try:
        with timed("LLM:generate_final_report"):
            report = zhipu_llm.call([{"role": "user", "content": prompt}]).strip()
        if report and len(report) > 20:
            return report
    except Exception as e:
        print(f"[_generate_final_report] LLM 生成报告失败: {e}")

    # 兜底: 不调 LLM
    if structured:
        return (
            f"📋 行程规划结果\n\n"
            f"目的地: {structured.get('destination', '未知')}\n"
            f"数据来源: {', '.join(structured.get('data_sources', [])) or '无'}\n"
            f"步骤数: {len(structured.get('steps', []))}\n"
            f"⚠️ 警告: {len(structured.get('warnings', []))} 条"
        )
    if state.steps:
        lines = []
        for s in state.steps:
            marker = "✅" if s.status == "completed" else "⚠️"
            text = s.result_text or (s.result if isinstance(s.result, str) else "")
            if text:
                lines.append(f"[{marker}] {text}")
        return f"📋 行程规划结果\n\n" + "\n\n".join(lines)
    return "未能生成行程报告"


# 兼容旧调用者 (外部可能 import generate_final_report)
generate_final_report = _generate_final_report


def finalize_node(state: TravelState, config=None) -> dict:
    """流程结束: 标记 is_done, 兜底补 final_report."""
    if _check_ask_user(state):
        return _dirty(state)

    print(f"\n[Finalize] 流程结束")
    if not state.final_report:
        state.final_report = _generate_final_report(state) or "未能生成报告"
    state.is_done = True
    return _dirty(state)
