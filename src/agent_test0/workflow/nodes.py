# agent_test0/workflow/nodes.py
"""
状态机 6 个节点的业务逻辑。

每个节点是一个纯函数 `run_xxx(flow, ...)`，接收 TravelWorkflow 实例，
通过 flow.state 读写状态、通过 flow._run_crew_with_callback 调用 Crew。

为什么不直接做成 TravelWorkflow 的方法？
  - 让 flow.py 只剩"调度骨架"，方法体超过几行的部分都搬到这里。
  - 方便单独阅读：要看 Planner 业务逻辑，只看 run_planner 函数即可。
  - 方便后续替换：想把 Planner 换成纯代码实现（不用 LLM），只改 run_planner。

【@listen 装饰器在哪？】
  在 flow.py 的方法上。本文件的函数是被 flow.py 方法直接调用的。
  Flow 节点之间的串联（手动 self.step_executor() 调用链）在原代码里就这么做的，
  这里保留原行为。
"""

import json

from agent_test0.workflow.state import StepPlan, StepResult
from agent_test0.workflow.crews import (
    PlannerCrew,
    StepPreparerCrew,
    StepExecutorCrew,
    StepVerifierCrew,
    PartialReplannerCrew,
    FinalVerifierCrew,
)
from agent_test0.workflow.parsing import extract_json_object, parse_step_feedback
from agent_test0.workflow.llm import zhipu_llm


# ============================================================
# 状态 1: Planner —— 生成粗粒度步骤列表
# ============================================================

def run_planner(flow):
    """复杂度判定 + 偏好提取 + 步骤生成。"""
    # 全局钩子：检查是否需要向用户提问
    if flow._check_ask_user_hook():
        return

    print(f"\n{'='*60}")
    print(f"[Planner] 决策官剖析需求中...")
    print(f"{'='*60}")
    flow.notify("📋 [Planner] 决策大脑正在建立行程执行策略...")

    # 如果还没有步骤列表，调用 PlannerCrew 生成
    if not flow.state.steps:
        inputs = {
            "message": flow.state.message,
            "user_id": flow.state.user_id,
            "focus": flow.state.focus,
            "previous_plan": "无历史计划（首次规划）",
            "current_step": "无当前工单（首次执行）",
            "current_draft": "无进度草案",
        }

        result = flow._run_crew_with_callback(PlannerCrew, inputs)
        raw_text = result.raw.strip()
        print(f"[Planner] 原始输出: {raw_text[:500]}")

        plan_data = extract_json_object(raw_text)
        if plan_data:
            flow.state.is_complex = plan_data.get("is_complex", True)
            flow.state.simple_answer = plan_data.get("simple_answer", "")
            flow.state.location = plan_data.get("location", "未知")
            flow.state.focus = plan_data.get("focus", "")
            flow.state.assumptions = plan_data.get("assumptions", []) or []

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
    """为当前步骤决定调哪些工具、传什么参数。"""
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

    # 已有工具：跳过 LLM 规划，但仍然推进到 step_executor
    if current_step.tools:
        print(f"[StepPreparer] 跳过 LLM 规划（已有工具 {current_step.tools}），直接推进到 step_executor")
        flow.step_executor()
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

    result = flow._run_crew_with_callback(StepPreparerCrew, inputs)
    raw_text = result.raw.strip()

    # 解析执行计划（填充工具调用序列）
    plan_data = extract_json_object(raw_text)
    if plan_data:
        tools_to_call = plan_data.get("tools_to_call", [])
        if tools_to_call:
            current_step.tools = [t.get("tool_name", "") for t in tools_to_call]
            print(f"[StepPreparer] 为步骤 {step_idx} 填充了工具: {current_step.tools}")
        else:
            print(f"[StepPreparer] 警告: 未找到 tools_to_call")
    else:
        print(f"[StepPreparer] 解析执行计划失败")
        # 默认使用 weather_tool（最常用）
        current_step.tools = ["weather_tool"]

    # 显式调用 step_executor（Flow 的 @listen 不会对直接调用传播）
    flow.step_executor()


# ============================================================
# 状态 3: StepExecutor —— 执行工具调用
# ============================================================

def run_step_executor(flow):
    """按 step.tools 执行工具调用，把结果写入 step.result。"""
    print(f"[StepExecutor] 监听到 step_preparer 完成")
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
    current_step.status = "executing"

    print(f"\n{'='*60}")
    print(f"[StepExecutor] 执行步骤 {step_idx}: {current_step.description[:50]}...")
    print(f"{'='*60}")
    flow.notify(f"🛠️ [StepExecutor] 正在执行工具调用...")

    # 构建执行计划
    execution_plan = {
        "step_index": step_idx,
        "step_goal": current_step.description,
        "tools": current_step.tools
    }

    inputs = {
        "step_index": step_idx,
        "step_goal": current_step.description,
        "execution_plan": json.dumps(execution_plan, ensure_ascii=False),
    }

    result = flow._run_crew_with_callback(StepExecutorCrew, inputs)
    raw_text = result.raw.strip()

    # 解析执行结果
    exec_data = extract_json_object(raw_text)
    if exec_data:
        results = exec_data.get("execution_results", [])

        # 记录执行结果
        step_result = ""
        has_error = False
        for r in results:
            step_result += f"\n[{r.get('tool_name', 'unknown')}] {str(r.get('output', ''))[:500]}"
            if r.get('error'):
                has_error = True

        current_step.result = step_result
        current_step.status = "failed" if has_error else "completed"

        # 记录到历史
        flow.state.step_results.append(StepResult(
            step_index=step_idx,
            step_description=current_step.description,
            result=step_result,
            passed=not has_error,
            validation_feedback=""
        ))

        print(f"[StepExecutor] 步骤 {step_idx} 执行完成: {'成功' if not has_error else '失败'}")
    else:
        print(f"[StepExecutor] 解析结果失败")
        current_step.result = raw_text[:1000]
        current_step.status = "completed"

    # 显式调用 step_verifier（Flow 的 @listen 不会对直接调用传播）
    flow.step_verifier()


# ============================================================
# 状态 4: StepVerifier —— 审核单个步骤结果
# ============================================================

def run_step_verifier(flow):
    """审核 step.result 是否满足 step.description。pass / retry / fail 三种结果。"""
    if not flow.state.steps:
        return

    if flow._check_ask_user_hook():
        return

    step_idx = flow.state.current_step_index
    if step_idx >= len(flow.state.steps):
        return

    current_step = flow.state.steps[step_idx]

    # 跳过已处理的步骤（防止 Flow 重复触发）
    if current_step.status in ("completed", "failed") and current_step.validation_feedback:
        print(f"[StepVerifier] 步骤 {step_idx} 已处理过，跳过")
        return

    print(f"\n{'='*60}")
    print(f"[StepVerifier] 审核步骤 {step_idx} 结果...")
    print(f"{'='*60}")
    flow.notify(f"🔍 [StepVerifier] 正在审核步骤结果...")

    # 【确定性短路】如果步骤已有非空 result 且 status==completed，
    # 直接 pass，不浪费 LLM 调用。这避免 LLM 因"数据不够丰富"挑刺退回 retry。
    if current_step.status == "completed" and current_step.result and current_step.result.strip():
        print(f"[StepVerifier] 短路 pass：步骤 {step_idx} 有非空结果且 StepExecutor 已标记 completed")
        flow.notify(f"✅ [StepVerifier] 步骤 {step_idx} 直接通过（有数据）")
        current_step.validation_feedback = "有非空结果，直接通过"
        flow.state.current_step_index += 1
        if flow.state.current_step_index >= len(flow.state.steps):
            flow.final_verifier()
        else:
            flow.step_preparer()
        return

    # 构建审核输入
    inputs = {
        "step_index": step_idx,
        "step_goal": current_step.description,
        "execution_plan": json.dumps({"tools": current_step.tools}, ensure_ascii=False),
        "execution_results": current_step.result,
    }

    result = flow._run_crew_with_callback(StepVerifierCrew, inputs)
    raw_text = result.raw.strip()

    # 解析验证反馈
    feedback = parse_step_feedback(flow, raw_text)
    current_step.validation_feedback = feedback.get("reason", "")

    # 处理用户提问
    if feedback.get("verdict") == "ask_user":
        print(f"[StepVerifier] 检测到用户提问指令")
        flow.notify(f"🙋 [AskUser] {flow.state.user_question}")
        flow.state.final_report = flow.state.user_question
        return

    if feedback.get("verdict") == "pass":
        print(f"[StepVerifier] 步骤 {step_idx} 审核通过")
        flow.notify(f"✅ [StepVerifier] 步骤 {step_idx} 审核通过")

        # 标记完成
        current_step.status = "completed"
        current_step.validation_feedback = feedback.get("reason", "通过")

        # 推进到下一个步骤
        flow.state.current_step_index += 1
        if flow.state.current_step_index >= len(flow.state.steps):
            flow.final_verifier()
        else:
            print(f"[StepVerifier] 准备执行步骤 {flow.state.current_step_index}")
            flow.step_preparer()
    elif feedback.get("verdict") == "retry":
        # 重试当前步骤
        retry_count = flow.step_retry_counts.get(step_idx, 0)
        if retry_count < flow.max_step_retries:
            flow.step_retry_counts[step_idx] = retry_count + 1
            print(f"[StepVerifier] 步骤 {step_idx} 重试中 ({retry_count + 1}/{flow.max_step_retries})")
            flow.notify(f"🔄 [StepVerifier] 步骤 {step_idx} 重试中...")
            flow.step_executor()
        else:
            print(f"[StepVerifier] 步骤 {step_idx} 重试耗尽，标记为失败")
            current_step.status = "failed"
            current_step.validation_feedback = f"重试 {flow.max_step_retries} 次后失败"
            flow.state.failed_steps_indices.append(step_idx)
            # 推进到下一个步骤
            flow.state.current_step_index += 1
            if flow.state.current_step_index >= len(flow.state.steps):
                flow.final_verifier()
            else:
                flow.step_preparer()
                flow.step_executor()
    else:  # fail - 需要 PartialReplanner
        print(f"[StepVerifier] 步骤 {step_idx} 审核失败，触发 PartialReplanner")
        current_step.status = "failed"
        flow.state.failed_steps_indices.append(step_idx)
        run_partial_replanner(flow, feedback)


# ============================================================
# 状态 5: PartialReplanner —— 局部重规划
# ============================================================

def run_partial_replanner(flow, failure_feedback: dict):
    """仅修复失败步骤，不重新规划全局。"""
    if flow._check_ask_user_hook():
        return

    # 防止无限重规划
    replan_count = getattr(flow.state, '_replan_count', 0) + 1
    flow.state._replan_count = replan_count
    if replan_count > flow.max_replan_attempts:
        print(f"[PartialReplanner] 重规划次数超限 ({replan_count}/{flow.max_replan_attempts})，强制结束")
        flow.state.final_report = generate_final_report(flow)
        run_finalize(flow)
        return

    print(f"\n{'='*60}")
    print(f"[PartialReplanner] 触发局部重规划 (第 {replan_count} 次)...")
    print(f"{'='*60}")
    flow.notify(f"🔄 [PartialReplanner] 正在局部重规划...")

    failed_indices = list(set(flow.state.failed_steps_indices))
    flow.state.failed_steps_indices = []  # 清空，重新开始
    print(f"[PartialReplanner] 失败步骤索引: {failed_indices}")
    if not failed_indices:
        print(f"[PartialReplanner] 没有失败的步骤，无法重规划")
        return

    # 保留已完成的步骤
    preserved_steps = [i for i in range(len(flow.state.steps)) if i < min(failed_indices)]
    print(f"[PartialReplanner] 保留步骤: {preserved_steps}")

    # 原始剩余步骤
    original_remaining = [i for i in range(len(flow.state.steps)) if i >= min(failed_indices)]
    print(f"[PartialReplanner] 原始剩余步骤: {original_remaining}")

    inputs = {
        "failure_reason": failure_feedback.get("reason", ""),
        "failed_step_indices": json.dumps(failed_indices),
        "suggested_corrections": json.dumps(failure_feedback.get("suggested_corrections", {}), ensure_ascii=False),
        "preserved_steps_results": json.dumps({
            str(i): flow.state.steps[i].result
            for i in preserved_steps if i < len(flow.state.steps)
        }, ensure_ascii=False),
        "original_remaining_steps": json.dumps(original_remaining),
    }

    result = flow._run_crew_with_callback(PartialReplannerCrew, inputs)
    raw_text = result.raw.strip()

    # 解析重规划结果
    replan_data = extract_json_object(raw_text)
    if replan_data:
        new_steps = replan_data.get("new_coarse_steps", [])
        if new_steps:
            new_step_objects = [StepPlan(**s) for s in new_steps]
            preserved = [flow.state.steps[i] for i in preserved_steps if i < len(flow.state.steps)]
            flow.state.steps = preserved + new_step_objects
            flow.state.current_step_index = len(preserved)
            print(f"[PartialReplanner] 重规划完成，共 {len(flow.state.steps)} 个步骤")
            print(f"[PartialReplanner] 当前步骤索引: {flow.state.current_step_index}")
            flow.step_preparer()
        else:
            print(f"[PartialReplanner] 警告: 重规划未返回新步骤")
    else:
        print(f"[PartialReplanner] 解析重规划结果失败")


# ============================================================
# 状态 6: FinalVerifier —— 整体审核
# ============================================================

def run_final_verifier(flow):
    """整体审核 + 调用 LLM 合成最终报告 + finalize。"""
    print(f"[FinalVerifier] 被调用，检查是否已执行...")

    # 检查是否已经执行过（避免重复执行）
    if getattr(flow.state, '_final_verifier_executed', False):
        print(f"[FinalVerifier] 已执行过，跳过")
        return

    if flow._check_ask_user_hook():
        return

    # 只有在所有步骤都完成时才执行最终审核
    if flow.state.current_step_index < len(flow.state.steps):
        print(f"[FinalVerifier] 跳过: 还有 {len(flow.state.steps) - flow.state.current_step_index} 个步骤未完成")
        return

    print(f"\n{'='*60}")
    print(f"[FinalVerifier] 开始执行...")
    print(f"{'='*60}")
    flow.notify(f"🔍 [FinalVerifier] 正在进行整体审核...")

    flow.state._final_verifier_executed = True

    all_steps_with_results = [
        {"index": i, "description": s.description, "status": s.status, "result": s.result}
        for i, s in enumerate(flow.state.steps)
    ]

    inputs = {
        "all_steps_with_results": json.dumps(all_steps_with_results, ensure_ascii=False),
        "full_plan_document": "\n".join([f"步骤 {i}: {s.description}" for i, s in enumerate(flow.state.steps)]),
    }

    result = flow._run_crew_with_callback(FinalVerifierCrew, inputs)
    raw_text = result.raw.strip()
    print(f"[FinalVerifier] 原始输出: {raw_text[:500]}")

    feedback = parse_step_feedback(flow, raw_text)
    print(f"[FinalVerifier] 解析反馈: {feedback}")

    if feedback.get("verdict") == "pass":
        print(f"[FinalVerifier] 整体审核通过")
        flow.notify(f"🎉 [FinalVerifier] 整体审核通过")

        flow.state.final_report = generate_final_report(flow)
        report_len = len(flow.state.final_report) if flow.state.final_report else 0
        print(f"[FinalVerifier] 生成的最终报告长度: {report_len}")
        print(f"[FinalVerifier] 生成的最终报告内容: {flow.state.final_report[:200] if flow.state.final_report else 'None'}")
        run_finalize(flow)
    else:
        print(f"[FinalVerifier] 整体审核不通过，触发局部重规划")
        flow.notify(f"⚠️ [FinalVerifier] 整体审核不通过")

        failed_indices = feedback.get("failed_step_ids", [])
        if failed_indices:
            flow.state.failed_steps_indices = failed_indices
            run_partial_replanner(flow, feedback.get("global_feedback", {}))
            flow.step_preparer()


# ============================================================
# 报告生成 + finalize
# ============================================================

def generate_final_report(flow) -> str:
    """
    将全部步骤的执行结果交给 LLM 合成为用户可读的旅游计划。

    步骤的 description 是执行指令（如"查询天气"），result 才是实际数据。
    这里只收集 result，让 LLM 生成最终的用户旅行计划。
    """
    if not flow.state.steps:
        return "未能生成行程报告"

    # 收集所有步骤的结果（实际数据，不是执行指令）
    results_text = []
    for s in flow.state.steps:
        label = "✅" if s.status == "completed" else "⚠️"
        if s.result:
            results_text.append(f"[{label}] {s.result}")
        elif s.status == "completed":
            results_text.append(f"[{label}] {s.description} - 已完成")
        else:
            results_text.append(f"[{label}] {s.description} - 未完成")

    collected = "\n\n".join(results_text)

    # 把规划阶段做出的假设带给报告生成 LLM，让它在开头明确披露
    assumptions_block = ""
    if flow.state.assumptions:
        bullets = "\n".join(f"- {a}" for a in flow.state.assumptions)
        assumptions_block = f"\n【系统所做的关键假设（必须在报告开头以 📌 形式向用户披露，并提示用户可调整）】\n{bullets}\n"

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
2. 把所有数据整合成一份自然流畅的旅行计划，像真人旅游顾问写的那样。
3. 包含以下板块（根据数据情况可省略无数据的板块）：
   - 目的地概况
   - 天气与最佳出行建议
   - 行程安排（按天列出）
   - 预算参考
   - 注意事项

4. 如果某个板块完全没有数据，直接跳过不要硬编。
5. 使用清晰的小标题、emoji 和分段，方便用户在飞书上阅读。
6. 总字数控制在 800 字以内。"""

    try:
        report = zhipu_llm.call([{"role": "user", "content": prompt}]).strip()
        if report and len(report) > 20:
            return report
    except Exception as e:
        print(f"[generate_final_report] LLM 生成报告失败: {e}")

    # fallback: 返回收集到的原始结果
    if collected:
        return f"📋 行程规划结果\n\n{collected}"
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
