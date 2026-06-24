# agent_test0/workflow/nodes.py
"""
状态机 6 个节点的业务逻辑 + 显式驱动循环。

设计要点（本次重构的核心修复）：
  - 不再依赖 CrewAI Flow 的 @listen 自动传播 + 手动 flow.step_xxx() 串联两套机制并存
    （那会导致整条链被同步跑一遍后，编排器又把下游 listener 重跑一遍 → 节点双触发、
     LLM 翻倍、final_report 被迟到的那次覆盖）。
  - 改为：flow.py 只保留 @start 入口，方法体调 run_state_machine(self)；
    本文件的 run_state_machine 用一个显式 while 循环驱动 6 状态机，
    天然支持 retry / replan 这种"循环"语义（满足"信息不足立即反馈"的硬性要求），
    且每个 run_xxx 是纯函数，只读写 flow.state、只返回 verdict，不再手动调下游。

每个节点函数 `run_xxx(flow, ...)`：
  - 接收 TravelWorkflow 实例，通过 flow.state 读写；结构化生成走 direct LLM + Pydantic，工具执行走 Python registry
  - 只做自己那一件事，推进逻辑上移到 run_state_machine
"""

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
)
from agent_test0.workflow.structured import call_structured, load_task_prompt, StructuredCallError
from agent_test0.workflow.llm import zhipu_llm
from agent_test0.workflow.trace import timed
from agent_test0.tools.registry import execute_tool_calls, format_tool_results


# 死循环保护：单轮 Flow 内步骤迭代总次数上限（含 retry / replan 重跑）
MAX_STEP_ITERATIONS = 30

# 用户明确授权系统自行默认/假设的表达。只有第二轮（已问过）且命中这些表达，才允许假设。
_USER_DELEGATION_PATTERN = re.compile(r"(看你安排|你安排|随便|无所谓|都可以|都行|默认|按常规|帮我定|你决定)")
_TRAVEL_INTENT_PATTERN = re.compile(r"(旅游|旅行|行程|攻略|游玩|玩|安排|规划|路线)")
_WEATHER_ONLY_PATTERN = re.compile(r"(天气|气温|下雨|温度|穿什么)")


def _has_asked_trip_constraints(flow) -> bool:
    """最近几轮 assistant 是否已经问过天数/预算/人数等次要约束。"""
    assistant_lines = []
    in_assistant = False
    for line in (flow.state.focus or "").splitlines():
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
    return any(
        kw in asked_text
        for kw in ("计划玩几天", "大概预算", "几位出行", "几个人", "天数", "预算", "人数")
    )


def _user_delegated_defaults(flow) -> bool:
    """用户本轮是否明确授权系统自行安排。"""
    return bool(_USER_DELEGATION_PATTERN.search(flow.state.message or ""))


def _looks_like_travel_planning(flow, plan_data: dict) -> bool:
    """判断是否是行程规划类需求，而非天气查询/闲聊。"""
    message = flow.state.message or ""
    focus = plan_data.get("focus", "") or ""
    intent_text = f"{message}\n{focus}"
    if plan_data.get("simple_answer") and not plan_data.get("steps"):
        return False
    if _WEATHER_ONLY_PATTERN.search(message) and not _TRAVEL_INTENT_PATTERN.search(message):
        return False
    return bool(_TRAVEL_INTENT_PATTERN.search(intent_text) or plan_data.get("steps"))


def _user_constraint_text(flow) -> str:
    """
    只看当前用户消息中的约束。

    之前把 focus 里的历史 user 行也算进来，会导致同一个 session 上一轮完整行程
    （如“重庆3天预算3000两人”）污染下一轮新需求（如“想去成都”），从而错误地认为
    本轮已经提供了天数/预算/人数，直接出最终答案而不追问。

    历史只用于 _has_asked_trip_constraints 判断 assistant 是否已经问过；
    不用于判断本轮用户是否提供了缺失信息。
    """
    return flow.state.message or ""


def _missing_secondary_constraints(flow, plan_data: dict) -> list[str]:
    """代码级判断行程规划中是否缺天数/预算/人数（只看用户真实输入，不看模型假设）。"""
    text = _user_constraint_text(flow)
    missing = []
    if not re.search(r"\d+\s*天|一日|一天|两天|三天|四天|五天|周末", text):
        missing.append("天数")
    if not re.search(r"\d+\s*(元|块|k|K|千|万)|预算\s*[:：]?\s*\d+|人均\s*\d+|经济型|中等预算|高端|豪华", text):
        missing.append("预算")
    if not re.search(r"\d+\s*(人|位)|一个人|一人|两个人|两人|独自|情侣|夫妻|家庭|亲子|朋友|同学|团队", text):
        missing.append("人数")
    return missing


def _llm_followup_question(flow, plan_data: dict, reason: str, missing: list[str] | None = None) -> str:
    """让 LLM 根据当前输入和上下文生成追问文本；代码只决定是否需要问。"""
    missing_text = "、".join(missing or []) or "由你判断"
    prompt = f"""你是旅游规划助手。现在不能继续生成行程，需要向用户追问。

【当前用户输入】
{flow.state.message}

【近期上下文】
{flow.state.focus}

【Planner 初步判断】
{json.dumps(plan_data, ensure_ascii=False)[:2000]}

【不能继续的原因】
{reason}

【已检测到缺失/无效的信息】
{missing_text}

请生成一条自然、简洁、贴合当前输入的中文追问。要求：
1. 不要机械套模板，不要声称“某地行程没问题”，除非当前用户这句话确实表达了新目的地。
2. 如果当前输入像乱码、无意义短词或没有回答上一轮问题，请先说明没有理解，再引导用户补充有效信息。
3. 如果缺少天数/预算/人数，请一次问全。
4. 可以提醒用户如果无特别要求，也可以授权你按常规方案安排。
5. 只输出最终要发给用户的一段话，不要 JSON，不要解释。"""
    try:
        with timed("LLM:AskUserQuestion"):
            question = zhipu_llm.call([{"role": "user", "content": prompt}]).strip()
        if question:
            return question[:500]
    except Exception as e:
        print(f"[PlannerGuard] LLM 生成追问失败: {e}")

    # LLM 不可用时的兜底只保证流程不中断；正常路径不依赖硬编码模板。
    return "我还需要补充一些关键信息才能继续规划。请说明计划玩几天、大概预算和几位出行；如果没有特别要求，也可以让我按常规方案安排。"


def _enforce_first_turn_question(flow, plan_data: dict) -> bool:
    """
    Planner 的代码级护栏：
    - 行程规划类需求，目的地明确，但天数/预算/人数缺失
    - 第一次遇到缺失 → 强制提问，禁止使用模型 assumptions 直接默认
    - 已经问过一次且用户明确说"看你安排"等 → 允许默认假设

    Returns True 表示已经写入 AskUser 并应立即 return。
    """
    location = (plan_data.get("location") or "").strip()
    if not location or location in ("未知", "未指定", "不明"):
        return False  # 目的地缺失由 prompt/plan_data 的 needs_user_input 处理
    if not _looks_like_travel_planning(flow, plan_data):
        return False

    missing = _missing_secondary_constraints(flow, plan_data)
    if not missing:
        return False

    already_asked = _has_asked_trip_constraints(flow)
    delegated = _user_delegated_defaults(flow)
    if already_asked and delegated:
        print(f"[PlannerGuard] 已问过且用户授权默认，允许假设: missing={missing}")
        return False

    reason = "用户本轮输入不足以继续规划，不能直接沿用历史目的地或历史约束。"
    if already_asked and not delegated:
        reason = "上一轮已经追问过关键信息，但用户本轮没有明确授权默认，也没有补全全部必要信息。"
    question = _llm_followup_question(flow, plan_data, reason, missing)
    print(f"[PlannerGuard] 强制提问：缺失 {missing}，already_asked={already_asked}, delegated={delegated}")
    flow._set_ask_user_question(question)
    return True


# ============================================================
# 显式驱动循环 —— 唯一的状态机推进者
# ============================================================

def run_state_machine(flow):
    """
    6 状态机的显式驱动循环。由 flow.plan_steps(@start) 调用。

    结构：
      1. run_planner：生成 steps；可能直接 set final_report（ask_user / 兜底简单回答）
      2. 主步骤循环：prepare → execute → verify，按 verdict 推进/重试/重规划
      3. 所有步骤完成 → run_final_verifier：不通过则 replan 后回到步骤循环

    任何阶段 needs_user_input=True 都立即 return（final_report 已写好问题文本）。
    """
    # ── 1. Planner ──
    run_planner(flow)
    if flow._check_ask_user_hook():
        return
    # Planner 已给出兜底 final_report（简单闲聊 / 解析失败），或没有 steps → 直接结束
    if flow.state.final_report or not flow.state.steps:
        return

    # ── 2 + 3. 步骤循环 ↔ 整体审核 ──
    while True:
        # 阶段 A：推进未完成的步骤
        while flow.state.current_step_index < len(flow.state.steps):
            flow.state.total_steps_counted += 1
            if flow.state.total_steps_counted > MAX_STEP_ITERATIONS:
                print(f"[StateMachine] 步骤迭代超限 ({MAX_STEP_ITERATIONS})，强制合成报告结束")
                flow.state.final_report = generate_final_report(flow)
                return

            run_step_preparer(flow)
            if flow._check_ask_user_hook():
                return
            run_step_executor(flow)
            if flow._check_ask_user_hook():
                return
            verdict = run_step_verifier(flow)
            if verdict == "ask_user":
                return
            # verdict 其它取值（pass/retry/fail）的索引推进/重置已在 run_step_verifier 内完成：
            #   pass  → current_step_index 已 +1
            #   retry → 索引不变，下一轮循环重跑同一步骤
            #   fail  → 已调 run_partial_replanner，索引已重置到失败处

        # 阶段 B：所有步骤完成 → 整体审核
        fv_verdict = run_final_verifier(flow)
        if fv_verdict == "ask_user":
            return
        if fv_verdict == "pass":
            return  # final_report 已在 run_final_verifier 内生成
        # fv_verdict == "fail"：run_final_verifier 内已触发 run_partial_replanner，
        # 索引已重置。若有新步骤可跑 → 回到阶段 A；否则强制结束。
        if flow.state.current_step_index >= len(flow.state.steps):
            print("[StateMachine] FinalVerifier 不通过但无步骤可重跑，强制结束")
            if not flow.state.final_report:
                flow.state.final_report = generate_final_report(flow)
            return


# ============================================================
# 状态 1: Planner —— 生成粗粒度步骤列表
# ============================================================

def assemble_structured_plan(flow) -> dict:
    """
    P1.1: 确定性结构化行程组装器。

    不调 LLM, 纯代码从已完成 steps 聚合结构化数据, 输出可机读 + 可被规则检查的 plan。

    Returns:
        {
            "destination": str,                    # 目的地
            "user_query": str,                     # 用户原始请求
            "assumptions": list[str],              # Planner 假设
            "steps": [                             # 每个已完成 step 的结构化记录
                {
                    "index": int,
                    "description": str,
                    "tools_used": list[str],
                    "data": <Any>,                 # 聚合 tool_results[].output (结构化)
                    "status": str,
                }
            ],
            "data_sources": list[str],             # 全部成功调用的工具名 (去重)
            "warnings": list[str],                 # 异常/缺失警告
            "failed_steps": list[int],             # 失败 step 索引
        }
    """
    plan = {
        "destination": flow.state.location,
        "user_query": flow.state.message,
        "assumptions": list(flow.state.assumptions or []),
        "steps": [],
        "data_sources": [],
        "warnings": [],
        "failed_steps": list(flow.state.failed_steps_indices or []),
    }
    seen_tools = set()

    for i, s in enumerate(flow.state.steps or []):
        if s.status not in ("completed", "failed"):
            plan["warnings"].append(f"步骤 {i} 未完成 (status={s.status}): {s.description}")
            continue

        step_entry = {
            "index": i,
            "description": s.description,
            "tools_used": list(s.tools or []),
            "data": s.result,  # P0.2: 结构化 (来自 tool_results[].output)
            "status": s.status,
        }
        plan["steps"].append(step_entry)

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
    """
    P1.2: 确定性规则检查器。返回失败原因列表 (空 = 全部通过)。

    能用代码判的绝不用 LLM 判。当前覆盖:
      R1: 至少 1 个 completed step
      R2: 不存在 failed step (data inventory 必须完整)
      R3: completed steps 至少有一个有结构化 data (None/空 = 失败)

    语义类规则 (用户硬约束、跨步矛盾) 留给 FinalVerifier LLM 做软检查。
    """
    failures = []
    completed = [s for s in plan.get("steps", []) if s.get("status") == "completed"]
    if not completed:
        failures.append("R1: 没有已完成的步骤")

    if plan.get("failed_steps"):
        failures.append(
            f"R2: 有 {len(plan['failed_steps'])} 个失败步骤: {plan['failed_steps']}"
        )

    has_data = any(s.get("data") not in (None, "", [], {}) for s in completed)
    if completed and not has_data:
        failures.append("R3: 所有 completed 步骤都没有结构化数据")

    return failures


def run_planner(flow):
    """复杂度判定 + 偏好提取 + 步骤生成。"""
    # 全局钩子：检查是否需要向用户提问
    if flow._check_ask_user_hook():
        return

    print(f"\n{'='*60}")
    print(f"[Planner] 决策官剖析需求中...")
    print(f"{'='*60}")
    flow.notify("📋 [Planner] 决策大脑正在建立行程执行策略...")

    # 如果还没有步骤列表，直接调用 LLM + Pydantic 生成 PlannerOutput
    if not flow.state.steps:
        inputs = {
            "message": flow.state.message,
            "user_id": flow.state.user_id,
            "focus": flow.state.focus,
            "previous_plan": "无历史计划（首次规划）",
            "current_step": "无当前工单（首次执行）",
            "current_draft": "无进度草案",
        }

        try:
            prompt = load_task_prompt("tasks.yaml", "planning_task", inputs)
            plan = call_structured("Planner", prompt, PlannerOutput)
            plan_data = plan.model_dump()
            raw_text = json.dumps(plan_data, ensure_ascii=False)
            print(f"[Planner] 结构化输出: {raw_text[:500]}")
        except StructuredCallError as e:
            print(f"[Planner] 结构化输出失败: {e}")
            plan_data = None
            raw_text = ""

        if plan_data:
            # 注意：guard 需要读取原始 focus 里的近期对话，不能先用 Planner 输出覆盖掉。
            original_focus = flow.state.focus
            already_asked = _has_asked_trip_constraints(flow)
            delegated = _user_delegated_defaults(flow)

            flow.state.is_complex = plan_data.get("is_complex", True)
            flow.state.simple_answer = plan_data.get("simple_answer", "")
            flow.state.location = plan_data.get("location", "未知")
            # 非授权默认时，不接受 Planner 自行编造的 assumptions；
            # 只有 assistant 已经问过一次且用户本轮明确授权默认，才保留模型假设。
            if already_asked and delegated:
                flow.state.assumptions = plan_data.get("assumptions", []) or []
            else:
                flow.state.assumptions = []

            # 代码级护栏：行程规划第一次缺天数/预算/人数时，强制提问，
            # 防止 Planner 违背 prompt 直接塞默认 assumptions 继续规划。
            flow.state.focus = original_focus
            if _enforce_first_turn_question(flow, plan_data):
                return
            flow.state.focus = plan_data.get("focus", "")

            # 信息不足时 Planner 可能直接发起结构化提问
            if plan_data.get("needs_user_input") or plan_data.get("verdict") == "ask_user":
                question = plan_data.get("user_question") or plan_data.get("question") or "信息不足，请补充。"
                flow._set_ask_user_question(question)
                return

            # 提取步骤列表
            steps = plan_data.get("steps", [])
            print(
                f"[Planner] 解析结果: is_complex={flow.state.is_complex}, "
                f"simple_answer='{flow.state.simple_answer[:50]}', steps={len(steps)}"
            )
            if steps:
                flow.state.steps = [StepPlan(**s) for s in steps]
                print(f"[Planner] 生成了 {len(flow.state.steps)} 个粗粒度步骤")
        else:
            print(f"[Planner] 警告: 输出中没有 JSON 块")
            flow.state.is_complex = True

        # 【兜底】Planner 没产出 steps（解析失败 / 模型直接闲聊）：
        # 把原始输出当成简单回答，直接终止流程，避免链路因 not steps 静默死掉。
        if not flow.state.steps:
            fallback_answer = (
                flow.state.simple_answer.strip()
                if flow.state.simple_answer
                else raw_text[:600] if raw_text else "抱歉，我暂时无法理解您的需求，请补充更多信息。"
            )
            print(f"[Planner] 兜底: 未生成 steps，直接返回 simple_answer / raw_text")
            flow.state.final_report = fallback_answer
            flow.notify("⚠️ [Planner] 未生成多步骤计划，直接返回简要回答")
            return

    # 初始化当前步骤索引
    if flow.state.steps:
        flow.state.current_step_index = 0


# ============================================================
# 状态 2: StepPreparer —— 为当前步骤生成执行计划
# ============================================================

def run_step_preparer(flow):
    """为当前步骤决定调哪些工具、传什么参数。纯函数：填完 tools 即返回，不串联下游。"""
    print(f"[StepPreparer] 被调用，检查是否从重规划来...")

    # 全局钩子：检查是否需要向用户提问
    if flow._check_ask_user_hook():
        return

    # Planner 已提前给出 final_report（兜底简单回答）：直接结束
    if flow.state.final_report and not flow.state.steps:
        print(f"[StepPreparer] 跳过: Planner 已给出兜底 final_report")
        return

    if not flow.state.steps:
        print(f"[StepPreparer] 跳过: 没有步骤，生成兜底报告")
        flow.state.final_report = "抱歉，我没能为您的需求规划出执行步骤，请补充更多信息后再试。"
        return

    step_idx = flow.state.current_step_index
    if step_idx >= len(flow.state.steps):
        print(f"[StepPreparer] 跳过: 索引超出范围")
        return

    current_step = flow.state.steps[step_idx]
    # 跳过已完成的步骤
    if current_step.status == "completed":
        print(f"[StepPreparer] 跳过: 步骤 {step_idx} 已完成")
        return

    # 已完成细粒度规划：跳过 LLM 规划（replan / retry 复用已有 tool_calls）
    if current_step.prepared:
        print(f"[StepPreparer] 跳过 LLM 规划（已有小计划 {current_step.tools}）")
        return

    print(f"\n{'='*60}")
    print(f"[StepPreparer] 为步骤 {step_idx} 生成执行计划...")
    print(f"{'='*60}")
    flow.notify(f"📋 [StepPreparer] 正在为步骤生成执行计划...")

    # 构建上下文信息
    previous_results = {
        str(r.step_index): r.result
        for r in flow.state.step_results if r.passed
    }

    inputs = {
        "step_index": step_idx,
        "step_goal": current_step.description,
        "previous_step_results": json.dumps(previous_results, ensure_ascii=False),
        "global_constraints": json.dumps({
            "user_id": flow.state.user_id,
            "location": flow.state.location,
            "focus": flow.state.focus
        }, ensure_ascii=False),
    }

    try:
        prompt = load_task_prompt("step_preparer_tasks.yaml", "step_preparer_task", inputs)
        plan = call_structured("StepPreparer", prompt, StepPreparerOutput)
        current_step.tool_calls = plan.tools_to_call
        current_step.tools = [t.tool_name for t in current_step.tool_calls]
        current_step.prepared = True
        if current_step.tool_calls:
            print(f"[StepPreparer] 为步骤 {step_idx} 填充了小计划: {current_step.tool_calls}")
        else:
            print(f"[StepPreparer] 步骤 {step_idx} 无需外部工具")
    except StructuredCallError as e:
        print(f"[StepPreparer] 解析执行计划失败")
        # 仅天气类步骤做确定性兜底；其它步骤作为无需工具的整合步骤处理。
        if "天气" in current_step.description and flow.state.location not in ("", "未知", "未知地点"):
            current_step.tool_calls = [ToolCall(order=1, tool_name="weather_tool", parameters={"city": flow.state.location})]
        else:
            current_step.tool_calls = []
        current_step.tools = [t.tool_name for t in current_step.tool_calls]
        current_step.prepared = True

    # 不再手动 flow.step_executor() —— 由 run_state_machine 驱动循环统一推进


# ============================================================
# 状态 3: StepExecutor —— 执行工具调用
# ============================================================

def run_step_executor(flow):
    """按 step.tools 执行工具调用，把结果写入 step.result。纯函数：写完即返回，不串联下游。"""
    print(f"[StepExecutor] 驱动循环调度执行")
    if not flow.state.steps:
        print(f"[StepExecutor] 跳过: 没有步骤")
        return

    # 全局钩子
    if flow._check_ask_user_hook():
        return

    step_idx = flow.state.current_step_index
    if step_idx >= len(flow.state.steps):
        return

    current_step = flow.state.steps[step_idx]
    if current_step.status == "completed":
        return
    # 幂等保护：如果正在执行中则跳过
    if current_step.status == "executing":
        print(f"[StepExecutor] 步骤 {step_idx} 正在执行中，跳过重复调用")
        return

    # DAG 依赖检查 (P0.1: StepPlan.dependencies 字段恢复)
    # 当前架构仍是线性串行执行, 依赖未满足只 warn, 不阻塞 (Planner 一般把 DAG 排成线性)
    # 未来可在此处实现真正的并行调度
    if current_step.dependencies:
        unmet = [
            d for d in current_step.dependencies
            if d < 0
            or d >= len(flow.state.steps)
            or flow.state.steps[d].status != "completed"
        ]
        if unmet:
            print(
                f"[StepExecutor] ⚠️ 步骤 {step_idx} 依赖 {unmet} 尚未完成，"
                f"但当前为线性执行模式 (将先尝试执行, 由 Planner DAG 顺序保证正确性)"
            )

    current_step.status = "executing"

    print(f"\n{'='*60}")
    print(f"[StepExecutor] 执行步骤 {step_idx}: {current_step.description[:50]}...")
    print(f"{'='*60}")
    flow.notify(f"🛠️ [StepExecutor] 正在执行工具调用...")

    # 纯 Python 工具执行器：StepPreparer 已经产出 tool_calls（小计划），这里不再调用 LLM。
    if not current_step.prepared:
        print(f"[StepExecutor] 警告: 步骤 {step_idx} 尚未完成 StepPreparer，按无需工具处理")
        current_step.prepared = True

    if not current_step.tool_calls:
        current_step.tool_results = []
        current_step.result_text = format_tool_results([])
        current_step.status = "completed"
        flow.state.step_results.append(StepResult(
            step_index=step_idx,
            step_description=current_step.description,
            result=None,
            result_text=current_step.result_text,
            passed=True,
            validation_feedback=""
        ))
        print(f"[StepExecutor] 步骤 {step_idx} 无需外部工具，直接完成")
        return

    result_dicts = execute_tool_calls(current_step.tool_calls)
    results = [ToolResult(**r) for r in result_dicts]
    current_step.tool_results = results
    # P0.2: 拆分结构化 (result) + 文本 (result_text)
    current_step.result_text = format_tool_results(results)
    # 聚合所有工具的结构化输出: 单一工具 → 直接用; 多工具 → 包成 dict
    structured_outputs = [r.output for r in results if r.output is not None]
    if len(structured_outputs) == 1:
        current_step.result = structured_outputs[0]
    elif len(structured_outputs) > 1:
        current_step.result = {"items": structured_outputs}
    else:
        current_step.result = None

    has_error = any(r.error for r in results)
    current_step.status = "failed" if has_error else "completed"
    current_step.error = "; ".join(r.error for r in results if r.error)

    flow.state.step_results.append(StepResult(
        step_index=step_idx,
        step_description=current_step.description,
        result=current_step.result,
        result_text=current_step.result_text,
        passed=not has_error,
        validation_feedback=""
    ))

    print(f"[StepExecutor] 步骤 {step_idx} Python 执行完成: {'成功' if not has_error else '失败'}")

    # 不再手动 flow.step_verifier() —— 由 run_state_machine 驱动循环统一推进


# ============================================================
# 状态 4: StepVerifier —— 审核单个步骤结果
# ============================================================

def run_step_verifier(flow) -> str:
    """
    审核 step.result 是否满足 step.description。

    Returns:
        "pass"    —— 通过，current_step_index 已推进
        "retry"   —— 重试当前步骤（未耗尽），索引不变，已重置步骤状态供下轮重跑
        "fail"    —— 失败，已触发 run_partial_replanner，索引已重置
        "ask_user"—— 需向用户提问，final_report 已写好
    """
    if not flow.state.steps:
        return "pass"

    if flow._check_ask_user_hook():
        return "ask_user"

    step_idx = flow.state.current_step_index
    if step_idx >= len(flow.state.steps):
        return "pass"

    current_step = flow.state.steps[step_idx]

    # 跳过已处理的步骤（防重复触发）——驱动循环单线程下一般不会到这
    if current_step.status in ("completed", "failed") and current_step.validation_feedback:
        print(f"[StepVerifier] 步骤 {step_idx} 已处理过，跳过")
        return "pass"

    print(f"\n{'='*60}")
    print(f"[StepVerifier] 审核步骤 {step_idx} 结果...")
    print(f"{'='*60}")
    flow.notify(f"🔍 [StepVerifier] 正在审核步骤结果...")

    # 【确定性短路】步骤已有非空 result 且 status==completed → 直接 pass，
    # 不浪费 LLM 调用，避免 LLM 因"数据不够丰富"挑刺退回 retry。
    # P0.2: result 可能是 dict (结构化) 或 str, 用统一的非空判断
    _has_result = bool(
        current_step.result is not None
        and current_step.result != ""
        and current_step.result != {}
        and current_step.result != []
    )
    if current_step.status == "completed" and _has_result:
        print(f"[StepVerifier] 短路 pass：步骤 {step_idx} 有非空结果且 StepExecutor 已标记 completed")
        flow.notify(f"✅ [StepVerifier] 步骤 {step_idx} 直接通过（有数据）")
        current_step.validation_feedback = "有非空结果，直接通过"
        flow.state.current_step_index += 1
        return "pass"

    # 构建审核输入
    # P0.2: 用 result_text (格式化文本) 给 LLM, 结构化 result 留给后续规则检查
    inputs = {
        "step_index": step_idx,
        "step_goal": current_step.description,
        "execution_plan": json.dumps({
            "tools": current_step.tools,
            "tool_calls": [t.model_dump() for t in current_step.tool_calls],
        }, ensure_ascii=False),
        "execution_results": (
            json.dumps([r.model_dump() for r in current_step.tool_results], ensure_ascii=False)
            if current_step.tool_results
            else (current_step.result_text or (current_step.result if isinstance(current_step.result, str) else ""))
        ),
    }

    try:
        prompt = load_task_prompt("step_validator_tasks.yaml", "step_validator_task", inputs)
        feedback = call_structured("StepVerifier", prompt, StepVerifierOutput).model_dump()
    except StructuredCallError as e:
        print(f"[StepVerifier] 结构化审核失败，默认通过: {e}")
        feedback = {"verdict": "pass", "reason": "结构化审核失败，默认通过"}

    current_step.validation_feedback = feedback.get("reason", "")

    # 处理用户提问
    if feedback.get("verdict") == "ask_user":
        print(f"[StepVerifier] 检测到用户提问指令")
        return "ask_user"

    if feedback.get("verdict") == "pass":
        print(f"[StepVerifier] 步骤 {step_idx} 审核通过")
        flow.notify(f"✅ [StepVerifier] 步骤 {step_idx} 审核通过")
        current_step.status = "completed"
        current_step.validation_feedback = feedback.get("reason", "通过")
        flow.state.current_step_index += 1
        return "pass"

    if feedback.get("verdict") == "retry":
        retry_count = flow.step_retry_counts.get(step_idx, 0)
        if retry_count < flow.max_step_retries:
            flow.step_retry_counts[step_idx] = retry_count + 1
            print(f"[StepVerifier] 步骤 {step_idx} 重试中 ({retry_count + 1}/{flow.max_step_retries})")
            flow.notify(f"🔄 [StepVerifier] 步骤 {step_idx} 重试中...")
            # 重置步骤状态，供驱动循环下轮重新 prepare→execute
            current_step.status = "pending"
            # P0.2: result 拆分为 result (Any) + result_text (str)
            current_step.result = None
            current_step.result_text = ""
            current_step.error = ""
            current_step.tool_results = []
            current_step.validation_feedback = ""
            return "retry"
        else:
            print(f"[StepVerifier] 步骤 {step_idx} 重试耗尽，标记为失败并跳过")
            current_step.status = "failed"
            current_step.validation_feedback = f"重试 {flow.max_step_retries} 次后失败"
            flow.state.failed_steps_indices.append(step_idx)
            flow.state.current_step_index += 1
            return "pass"  # 推进到下一步骤（失败步骤留给 FinalVerifier 兜底）

    # verdict == "fail" —— 触发局部重规划
    print(f"[StepVerifier] 步骤 {step_idx} 审核失败，触发 PartialReplanner")
    current_step.status = "failed"
    flow.state.failed_steps_indices.append(step_idx)
    run_partial_replanner(flow, feedback)
    return "fail"


# ============================================================
# 状态 5: PartialReplanner —— 局部重规划
# ============================================================

def run_partial_replanner(flow, failure_feedback: dict):
    """P2.1: append-only 重规划。保留所有已有 steps (completed + failed), 仅追加补救任务。

    旧语义: 失败处之后整体替换 (new_coarse_steps)
    新语义: 已完成步骤保留, 失败步骤保留为历史, 新任务追加到末尾 (new_appended_steps)
    """
    if flow._check_ask_user_hook():
        return

    # 防止无限重规划（replan_count 是 state 正式字段，跨 replan 累计）
    flow.state.replan_count += 1
    if flow.state.replan_count > flow.max_replan_attempts:
        print(f"[PartialReplanner] 重规划次数超限 ({flow.state.replan_count}/{flow.max_replan_attempts})，强制结束")
        flow.state.final_report = generate_final_report(flow)
        run_finalize(flow)
        return

    print(f"\n{'='*60}")
    print(f"[PartialReplanner] 触发 append-only 重规划 (第 {flow.state.replan_count} 次)...")
    print(f"{'='*60}")
    flow.notify(f"🔄 [PartialReplanner] 正在追加补救任务...")

    failed_indices = list(set(flow.state.failed_steps_indices))
    flow.state.failed_steps_indices = []  # 清空，重新开始
    print(f"[PartialReplanner] 失败步骤索引: {failed_indices}")
    if not failed_indices:
        print(f"[PartialReplanner] 没有失败的步骤，无法重规划")
        return

    # P2.1: 保留全部已有步骤 (completed + failed), 不再截断到 min(failed_indices)
    preserved_steps = list(range(len(flow.state.steps)))
    print(f"[PartialReplanner] 保留全部步骤 (append-only): {preserved_steps}")

    # 原始失败步骤索引 (供 LLM 参考, 提示"这些已失败, 不要重新规划")
    original_failed = [i for i in range(len(flow.state.steps))
                       if flow.state.steps[i].status == "failed" or i in failed_indices]
    print(f"[PartialReplanner] 原始失败步骤 (保留为历史): {original_failed}")

    # 收集 preserved steps 的 result_text (LLM 看的文本, 不是 result 字段)
    preserved_results = {}
    for i in preserved_steps:
        s = flow.state.steps[i]
        text = s.result_text or (s.result if isinstance(s.result, str) else "")
        if text or s.status == "completed":
            preserved_results[str(i)] = {
                "status": s.status,
                "description": s.description,
                "result": text or "(无输出)",
            }
    # 把失败步骤的 error 信息也带上, 让 LLM 知道失败原因
    for i in original_failed:
        if str(i) in preserved_results:
            preserved_results[str(i)]["error"] = flow.state.steps[i].error

    inputs = {
        "failure_reason": failure_feedback.get("reason", ""),
        "failed_step_indices": json.dumps(failed_indices),
        "suggested_corrections": json.dumps(failure_feedback.get("suggested_corrections", {}), ensure_ascii=False),
        "preserved_steps_results": json.dumps(preserved_results, ensure_ascii=False),
        "original_remaining_steps": json.dumps(original_failed),
    }

    try:
        prompt = load_task_prompt("replan_tasks.yaml", "replan_task", inputs)
        replan_data = call_structured("PartialReplanner", prompt, ReplanOutput).model_dump()
    except StructuredCallError as e:
        print(f"[PartialReplanner] 结构化重规划失败: {e}")
        replan_data = None

    # 解析重规划结果 (优先 new_appended_steps, fallback new_coarse_steps 兼容旧 prompt)
    new_steps_raw = []
    if replan_data:
        new_steps_raw = replan_data.get("new_appended_steps") or replan_data.get("new_coarse_steps") or []

    if new_steps_raw:
        next_index = len(flow.state.steps)
        appended = []
        for s in new_steps_raw:
            step_obj = StepPlan(**s) if isinstance(s, dict) else s
            # 强制修正 index 为 next available, 防止 LLM 给出重复/冲突 index
            if step_obj.index < next_index:
                print(f"[PartialReplanner] 修正 LLM 输出的 index {step_obj.index} → {next_index} (避免冲突)")
                step_obj.index = next_index
            appended.append(step_obj)
            next_index += 1
        flow.state.steps.extend(appended)
        flow.state.current_step_index = len(flow.state.steps) - len(appended)
        print(f"[PartialReplanner] append-only 完成, 共 {len(flow.state.steps)} 个步骤")
        print(f"[PartialReplanner] 新追加索引: {[s.index for s in appended]}")
        print(f"[PartialReplanner] 当前步骤索引: {flow.state.current_step_index}")
    else:
        print(f"[PartialReplanner] 警告: 重规划未返回新步骤 (appended=0)")

    # 不再手动 flow.step_preparer() —— 由 run_state_machine 驱动循环续跑


# ============================================================
# 状态 6: FinalVerifier —— 整体审核
# ============================================================

def run_final_verifier(flow) -> str:
    """
    P1.2: 结构化 plan → 确定性规则 → LLM 软检查 → narrative 翻译。

    Returns:
        "pass"     —— 通过，final_report 已生成
        "fail"     —— 不通过，已触发 run_partial_replanner，索引已重置
        "ask_user" —— 需向用户提问
    """
    print(f"[FinalVerifier] 被调用，检查是否已执行...")

    # 重入保护（final_verifier_done 是 state 正式字段）
    if flow.state.final_verifier_done:
        print(f"[FinalVerifier] 已执行过，跳过")
        return "pass"

    if flow._check_ask_user_hook():
        return "ask_user"

    # 只有在所有步骤都完成时才执行最终审核
    if flow.state.current_step_index < len(flow.state.steps):
        print(f"[FinalVerifier] 跳过: 还有 {len(flow.state.steps) - flow.state.current_step_index} 个步骤未完成")
        return "pass"

    print(f"\n{'='*60}")
    print(f"[FinalVerifier] 开始执行...")
    print(f"{'='*60}")
    flow.notify(f"🔍 [FinalVerifier] 正在进行整体审核...")

    flow.state.final_verifier_done = True

    # P1.2 step 1: 确定性组装结构化 plan
    structured_plan = assemble_structured_plan(flow)
    flow.state.structured_plan = structured_plan
    print(f"[FinalVerifier] 结构化 plan 已组装: {len(structured_plan['steps'])} 步, "
          f"data_sources={structured_plan['data_sources']}, warnings={len(structured_plan['warnings'])}")

    # P1.2 step 2: 确定性规则检查
    rule_failures = _check_deterministic_rules(structured_plan)
    if rule_failures:
        print(f"[FinalVerifier] 确定性规则不通过: {rule_failures}")
        flow.notify(f"⚠️ [FinalVerifier] 规则检查不通过: {rule_failures[0]}")
        # 直接走重规划, 不浪费 LLM
        feedback = {
            "verdict": "fail",
            "reason": f"确定性规则不通过: {'; '.join(rule_failures)}",
            "failed_step_ids": list(structured_plan.get("failed_steps", [])),
        }
        if feedback["failed_step_ids"]:
            flow.state.failed_steps_indices = feedback["failed_step_ids"]
        run_partial_replanner(flow, feedback)
        return "fail"

    # P1.2 step 3: 规则通过, LLM 仅做语义软检查 (用户硬约束/跨步矛盾)
    # 把结构化 plan 也喂给 LLM, 让它做最终判断
    inputs = {
        "all_steps_with_results": json.dumps(structured_plan["steps"], ensure_ascii=False, default=str),
        "structured_plan": json.dumps(structured_plan, ensure_ascii=False, default=str),
        "full_plan_document": "\n".join([f"步骤 {i}: {s.description}" for i, s in enumerate(flow.state.steps)]),
    }

    try:
        prompt = load_task_prompt("final_validator_tasks.yaml", "final_validator_task", inputs)
        fv = call_structured("FinalVerifier", prompt, FinalVerifierOutput)
        feedback = fv.model_dump()
        feedback["verdict"] = "pass" if feedback.get("global_verdict") == "pass" else "fail_with_patches"
    except StructuredCallError as e:
        print(f"[FinalVerifier] 结构化整体审核失败，默认通过: {e}")
        feedback = {"verdict": "pass", "reason": "结构化整体审核失败，默认通过", "failed_step_ids": []}

    print(f"[FinalVerifier] 结构化反馈: {feedback}")

    if feedback.get("verdict") == "ask_user":
        return "ask_user"

    if feedback.get("verdict") == "pass":
        print(f"[FinalVerifier] 整体审核通过")
        flow.notify(f"🎉 [FinalVerifier] 整体审核通过")
        # P1.2 step 4: LLM 仅做 narrative 翻译, 输入是结构化 plan (不是自由文本)
        flow.state.final_report = generate_final_report(flow)
        report_len = len(flow.state.final_report) if flow.state.final_report else 0
        print(f"[FinalVerifier] 生成的最终报告长度: {report_len}")
        print(f"[FinalVerifier] 生成的最终报告内容: {flow.state.final_report[:200] if flow.state.final_report else 'None'}")
        run_finalize(flow)
        return "pass"

    # LLM 软检查不通过 —— 触发局部重规划
    print(f"[FinalVerifier] LLM 软检查不通过，触发局部重规划")
    flow.notify(f"⚠️ [FinalVerifier] 整体审核不通过")
    failed_indices = feedback.get("failed_step_ids", [])
    if failed_indices:
        flow.state.failed_steps_indices = failed_indices
    run_partial_replanner(flow, feedback)
    return "fail"


# ============================================================
# 报告生成 + finalize
# ============================================================

def generate_final_report(flow) -> str:
    """
    P1.2: LLM 仅做 narrative 翻译, 输入是结构化 plan (而非自由文本 result)。

    步骤:
      1. 优先用 state.structured_plan (P1.1 确定性组装) 作为 LLM 输入
      2. fallback: 旧路径 (聚合 result_text)
      3. 严格禁止 LLM 创造结构化 plan 之外的数据 (没有的工具结果不能瞎编)
    """
    structured = flow.state.structured_plan if isinstance(flow.state.structured_plan, dict) and flow.state.structured_plan else None

    # 把规划阶段做出的假设带给报告生成 LLM，让它在开头明确披露
    assumptions_block = ""
    if flow.state.assumptions:
        bullets = "\n".join(f"- {a}" for a in flow.state.assumptions)
        assumptions_block = f"\n【系统所做的关键假设（必须在报告开头以 📌 形式向用户披露，并提示用户可调整）】\n{bullets}\n"

    if structured:
        plan_json = json.dumps(structured, ensure_ascii=False, indent=2, default=str)[:6000]
        prompt = f"""你是一位资深旅游规划师, 负责把【结构化行程数据】翻译成用户友好的中文报告。

【用户需求】
- 目的地: {flow.state.location}
- 关注重点: {flow.state.focus}
- 用户原话: {flow.state.message[:500]}
{assumptions_block}
【结构化行程数据 (JSON)】
{plan_json}

【撰写要求】
1. 严格基于结构化数据翻译, 不得编造数据中没有的景点/天气/预算数字。
2. 不要描述"执行步骤" (如"步骤 0 完成了查询")。
3. 包含以下板块 (数据中没有的板块直接跳过, 不要硬编):
   - 目的地概况
   - 天气与最佳出行建议 (有 weather data 时)
   - 行程要点 (按 description 任务名, 不要拼凑 day-by-day)
   - 注意事项 (来自 warnings)
4. 使用清晰的小标题、emoji 和分段, 方便用户在飞书上阅读。
5. 总字数控制在 800 字以内。
6. 如果有 warnings 字段, 在报告末尾用 ⚠️ 列出前 3 条。"""
    else:
        # 旧路径 fallback: 聚合 result_text
        if not flow.state.steps:
            return "未能生成行程报告"
        results_text = []
        for s in flow.state.steps:
            label = "✅" if s.status == "completed" else "⚠️"
            text = s.result_text or (s.result if isinstance(s.result, str) else "")
            if text:
                results_text.append(f"[{label}] {text}")
            elif s.status == "completed":
                results_text.append(f"[{label}] {s.description} - 已完成")
            else:
                results_text.append(f"[{label}] {s.description} - 未完成")
        collected = "\n\n".join(results_text)
        prompt = f"""你是一位资深旅游规划师。请根据以下执行数据，为用户撰写一份完整的旅行计划报告。

【用户需求】
- 目的地: {flow.state.location}
- 关注重点: {flow.state.focus}
- 用户原话: {flow.state.message[:500]}
{assumptions_block}
【已收集的数据】
{collected}

【撰写要求】
1. 绝不要重复原始的执行步骤描述（如"查询天气"、"获取偏好"等）。
2. 把所有数据整合成一份自然流畅的旅行计划。
3. 总字数控制在 800 字以内。"""
        structured = None  # for fallback path

    try:
        with timed("LLM:generate_final_report"):
            report = zhipu_llm.call([{"role": "user", "content": prompt}]).strip()
        if report and len(report) > 20:
            return report
    except Exception as e:
        print(f"[generate_final_report] LLM 生成报告失败: {e}")

    # fallback: 旧路径 - 返回收集到的原始结果
    if structured:
        return (
            f"📋 行程规划结果\n\n"
            f"目的地: {structured.get('destination', '未知')}\n"
            f"数据来源: {', '.join(structured.get('data_sources', [])) or '无'}\n"
            f"步骤数: {len(structured.get('steps', []))}\n"
            f"⚠️ 警告: {len(structured.get('warnings', []))} 条"
        )
    if flow.state.steps:
        results_text = []
        for s in flow.state.steps:
            label = "✅" if s.status == "completed" else "⚠️"
            text = s.result_text or (s.result if isinstance(s.result, str) else "")
            if text:
                results_text.append(f"[{label}] {text}")
        return f"📋 行程规划结果\n\n" + "\n\n".join(results_text)
    return "未能生成行程报告"


def run_finalize(flow):
    """流程结束方法 - 手动调用，不需要 listen 装饰器"""
    if flow._check_ask_user_hook():
        return

    print(f"\n{'='*60}")
    print(f"[结束] 流程结束")
    print(f"{'='*60}")
    if not flow.state.final_report:
        flow.state.final_report = generate_final_report(flow) or '未能生成报告'
