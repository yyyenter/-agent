# agent_test0/workflow/flow.py
"""
TravelWorkflow —— Flow 编排骨架。

本文件只负责"调度结构"：
  - @start / @listen 装饰的方法体只调用 nodes.py 里的对应业务函数
  - 通知/回调/Crew 包装器等基础设施薄薄地代理到 callbacks / parsing 模块
  - 外部统一入口 run_for_user：管理记忆生命周期 + Flow 执行 + final_report 兜底

业务逻辑请去 workflow/nodes.py 看。
"""

import sys
from crewai.flow import Flow, start
from langgraph.types import Command
from agent_test0.memory import MemoryManager, get_redis_or_fallback
from agent_test0.workflow.state import TravelState
from agent_test0.workflow.ask_user import (
    AskUserInterrupt,
)
from agent_test0.workflow.llm import zhipu_llm
from agent_test0.workflow import nodes
from agent_test0.workflow.graph import get_travel_graph, _shared_checkpointer
from agent_test0.workflow.trace import (
    timed, reset as trace_reset, report as trace_report,
    quiet_crewai, dump_json, span,
)


# 修复 stdout 编码（Windows 终端）
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


# ─── 全局 Redis 客户端：进程级单例 ───
_redis_client, _is_redis_fallback = get_redis_or_fallback()


# ============================================================
# Flow 主类
# ============================================================

class TravelWorkflow(Flow[TravelState]):
    """
    新 6 状态机：Planner → StepPreparer → StepExecutor → StepVerifier
                       → (PartialReplanner) → FinalVerifier
    """

    # 类常量：重试与重规划上限
    DEFAULT_MAX_STEP_RETRIES = 3
    DEFAULT_MAX_REPLAN_ATTEMPTS = 3

    def __init__(self, status_callback=None, content_callback=None):
        super().__init__()
        self.max_step_retries = self.DEFAULT_MAX_STEP_RETRIES
        self.max_replan_attempts = self.DEFAULT_MAX_REPLAN_ATTEMPTS
        self.step_retry_counts: dict[int, int] = {}
        self.status_callback = status_callback
        self.content_callback = content_callback

    # ============================================================
    # 通知/回调
    # ============================================================

    def notify(self, text: str):
        if self.status_callback:
            self.status_callback(text)
        print(f"[Flow] {text}")

    def _notify_content(self, text: str, content_type: str = "status"):
        # ⚠️ 强制指定给前端的 type 为 "status"
        if self.content_callback:
            self.content_callback(text, content_type)

    # ============================================================
    # 基础设施代理（薄包装到 workflow.* 模块）

    # ============================================================
    # 状态节点
    # 状态节点
    # ============================================================
    #
    # 【架构说明】方案 A1: 编排由 LangGraph StateGraph (workflow/graph.py) 驱动,
    # 不再用 CrewAI Flow 的 @start/@listen。run_for_user 里直接 graph.invoke。
    # 旧的 @start plan_steps / partial_replanner / finalize 方法已删除
    # (它们调用的 nodes.run_state_machine / run_partial_replanner / run_finalize 已不存在)。
    # notify 回调通过 RunnableConfig['configurable']['notify'] 注入到节点。

    # ============================================================
    # 外部统一入口
    # ============================================================

    @classmethod
    def run_for_user(
        cls,
        user_text: str,
        user_id: str,
        session_id: str | None = None,
        memory: "MemoryManager | None" = None,
        status_callback=None,
        content_callback=None,
    ) -> str:
        """
        统一的"对一句用户输入跑完一轮 Flow"入口。

        连接层（飞书、FastAPI、CLI）只需传 user_text + user_id 即可：
        memory / redis 客户端 / prompt 构造 / 状态机调度 / final_report
        生成 / 记忆写回（episodic→working、shortterm→semantic）全部封装在此处。

        Args:
            user_text: 用户输入
            user_id: 用户 id（飞书 open_id / 网页 user_id）
            session_id: 可选会话 id；不传时自动派生
            memory: 可选已有的 MemoryManager（极少用，仅在外部要复用同一会话上下文时）
            status_callback / content_callback: 流式回调

        Returns:
            最终给用户的回复文本（旅行计划 / 用户提问 / 错误信息）
        """
        try:
            trace_reset()  # 每轮清空计时，避免跨轮累计
            quiet_crewai()  # 关掉 CrewAI 自带的 ┌─...└─ 框
            sid = session_id or f"sess_{user_id}"

            # 由 crew 自己构造 MemoryManager（连接层无需关心 redis 客户端）
            if memory is None:
                memory = MemoryManager(sid, user_id, _redis_client, _is_redis_fallback)

            # 1) 写入用户输入到 episodic 记忆 (方案 C: 拿 msg_id, 供 graph 后回写业务字段)
            user_msg_id = memory.add_message("user", user_text)

            # 2) 短期记忆直接使用 episodic 原文，不再做 LLM 蒸馏 summary
            #    （蒸馏会漏字段/残留字段/示例污染，导致默认值或旧目的地泄露）。

            # 3) 跑 Flow (方案 A1: LangGraph StateGraph 驱动, 不再用 CrewAI Flow kickoff)
            flow = cls(status_callback=status_callback, content_callback=content_callback)
            flow.state.message = user_text
            flow.state.user_id = user_id
            flow.state.session_id = sid
            flow.state.focus = memory.get_global_context_prompt(user_text)

            # 3a) 把 MemoryManager 跨轮业务字段同步到 TravelState
            #     (current_task_id / current_destination / current_topic)
            #     让 Planner 节点能直接读 flow.state.current_destination。
            memory.bind_to_state(flow.state)

            # 3b) 检查 checkpointer 里当前 thread_id 是否有未完成的 interrupt.
            #     ★ Reactive Clarification: 上一轮某节点抛 interrupt 时, 全 state 已由
            #     checkpointer 持久化. 本轮 user_text 就是对那个 interrupt 的回答, 用
            #     Command(resume=user_text) 恢复, interrupt() 会返回 user_text, 节点从
            #     中断处继续 —— steps / result / current_step_index 都保留.
            graph_config = {
                "configurable": {
                    "notify": lambda text: flow.notify(text),
                    "thread_id": sid,   # ★ checkpointer 按 thread_id 隔离状态
                },
            }
            graph = get_travel_graph()
            checkpoint = _shared_checkpointer.get(graph_config)
            has_pending_interrupt = False
            if checkpoint:
                # LangGraph 把 pending interrupt 放在 next 集合里, 也可以通过
                # graph.get_state(config).next 判断; 简单起见看 __interrupt__
                try:
                    snapshot = graph.get_state(graph_config)
                    has_pending_interrupt = bool(snapshot.next) and any(
                        it.value for it in (snapshot.tasks or [])
                        for iv in [getattr(it, "interrupts", None)] if iv
                    )
                except Exception as e:
                    print(f"[run_for_user] 读取 checkpoint 状态失败 (视为无 interrupt): {e}")
                    has_pending_interrupt = False

            if has_pending_interrupt:
                # 走恢复路径: 不新建 state, 用 Command(resume=user_text)
                print(f"[run_for_user] 检测到 pending interrupt, 恢复中: reply={user_text[:80]!r}")
                graph_input = Command(resume=user_text)
            else:
                # 新起一轮: 用 pure_state (TravelState from flow.state, memory bind 结果)
                # ⚠️ 不能直接传 flow.state: CrewAI Flow 的 self.state 是 StateProxy 包装,
                #    model_dump() 会带隐藏的 id 字段, 导致 LangGraph 误判为 checkpoint
                #    state (StateWithId) → InvalidUpdateError. 先 dump 成纯 TravelState.
                pure_state = TravelState(**{k: v for k, v in flow.state.model_dump(mode="json").items()
                                            if k in TravelState.model_fields})
                graph_input = pure_state

            # 顶层 span: 整轮 Flow 的入口, 记录 input (用户消息) + output (最终报告)
            try:
                with span("run_for_user",
                          user_id=user_id, session_id=sid,
                          message_len=len(user_text),
                          message_preview=user_text[:80]) as run_span:
                    with timed("Flow:state_machine_total"):
                        result_state = graph.invoke(graph_input, config=graph_config)

                    # ★ Reactive: 检查本轮是否有新的 interrupt (节点抛出中断需要用户回答)
                    #   result_state 里会有 "__interrupt__" 键 (LangGraph 1.x 结构)
                    #   或通过 graph.get_state(config).tasks[*].interrupts 判断
                    interrupt_payload = None
                    if isinstance(result_state, dict) and result_state.get("__interrupt__"):
                        # LangGraph 1.x 返回结构里可能带 __interrupt__ 列表
                        raw = result_state["__interrupt__"]
                        if raw:
                            first = raw[0] if isinstance(raw, list) else raw
                            interrupt_payload = getattr(first, "value", None) or first
                    else:
                        try:
                            snap = graph.get_state(graph_config)
                            for t in (snap.tasks or []):
                                its = getattr(t, "interrupts", None) or []
                                for it in its:
                                    interrupt_payload = getattr(it, "value", None) or it
                                    break
                                if interrupt_payload:
                                    break
                        except Exception:
                            pass

                    # graph 返回最终 state (dict 或 TravelState), 写回 flow.state 供下游使用
                    if isinstance(result_state, dict):
                        for k, v in result_state.items():
                            if k == "__interrupt__":
                                continue
                            if k in TravelState.model_fields:
                                setattr(flow.state, k, v)
                    elif isinstance(result_state, TravelState):
                        flow.state = result_state

                    # 若本轮有 interrupt, 把问题作为 final_report / needs_user_input
                    if interrupt_payload:
                        question_text = (
                            interrupt_payload.get("question")
                            if isinstance(interrupt_payload, dict)
                            else str(interrupt_payload)
                        )
                        flow.state.needs_user_input = True
                        flow.state.user_question = question_text or "请补充信息."
                        flow.state.final_report = flow.state.user_question
                        print(f"[run_for_user] 本轮以 interrupt 结束, 等待用户回答")

                    # 写 output
                    run_span.set_output(
                        final_report_len=len(flow.state.final_report or ""),
                        steps=len(flow.state.steps or []),
                        needs_user_input=flow.state.needs_user_input,
                    )
            except AskUserInterrupt as e:
                # 保留兼容: 个别路径可能仍抛 AskUserInterrupt (旧 ask_user_and_exit)
                print(f"[run_for_user] AskUser 中断: {e.question} (field={e.blocking_field})")

            # 4) 取最终输出，按优先级兜底
            final_report = (flow.state.final_report or "").strip()

            if not final_report:
                if flow.state.needs_user_input and flow.state.user_question:
                    final_report = flow.state.user_question
                else:
                    # P0.2: 用 result_text (格式化文本), 避免 dict 拼接到 str 列表里炸
                    completed_results = [
                        s.result_text for s in (flow.state.steps or [])
                        if s.status == "completed" and s.result_text
                    ]
                    if completed_results:
                        final_report = "\n\n".join(completed_results)
                    elif flow.state.simple_answer:
                        final_report = flow.state.simple_answer
                    else:
                        final_report = "抱歉，本次未能成功生成行程，请稍后重试或补充更多信息。"

            print(f"[run_for_user] final_report 长度: {len(final_report)}, "
                  f"steps: {len(flow.state.steps or [])}, "
                  f"needs_user_input: {flow.state.needs_user_input}")

            # 5) 把单轮结果回写到 MemoryManager 跨轮业务字段
            #    (Planner 推断出的 current_destination / current_topic)
            memory.sync_from_state(flow.state)

            # 5a) 方案 C: 确定本轮 task_id (新目的地 → 开新 task; 否则继承),
            #     回写 user 索引条目的 task_id/destination (add_message 时未知),
            #     让下一轮 retrieve_short_term_context 能按 task_id 召回历史。
            inferred_dest = (flow.state.current_destination or "").strip()
            task_id = memory.current_task_id
            if inferred_dest and inferred_dest != memory.current_destination:
                # 新目的地 (或首次) → 开新 task
                task_id = memory.new_task_id()
                memory.current_task_id = task_id
                memory.current_destination = inferred_dest
                flow.state.current_task_id = task_id
            elif task_id is None and inferred_dest:
                # 首次有 destination 但还没 task_id
                task_id = memory.new_task_id()
                memory.current_task_id = task_id
                flow.state.current_task_id = task_id

            # 5b) 回写 user 索引条目 (Planner 跑完才知道 destination, 此时补上)
            if task_id or inferred_dest:
                try:
                    memory.update_index_entry(
                        user_msg_id,
                        task_id=task_id,
                        destination=inferred_dest or None,
                        topic=flow.state.current_topic if flow.state.current_topic != "general" else None,
                    )
                except Exception as e:
                    print(f"[run_for_user] 回写 user 索引失败（可忽略）: {e}")

            # 5c) 写助手回复到 episodic, 带上本轮 task_id/destination (索引条目完整)
            memory.add_message(
                "assistant", final_report,
                task_id=task_id,
                destination=inferred_dest or None,
                topic=flow.state.current_topic if flow.state.current_topic != "general" else None,
                is_completed_task=bool(flow.state.final_report and not flow.state.needs_user_input),
            )

            # 5d) 若本轮已生成 final_report (即非 AskUser 中断), 把当前 task 标记为已完成
            #     下次 retrieve_short_term_context 会自动把该 task 归入 excluded_history。
            if flow.state.final_report and not flow.state.needs_user_input:
                try:
                    memory.mark_current_task_completed()
                except Exception as e:
                    print(f"[run_for_user] mark_current_task_completed 失败（可忽略）: {e}")

            # 6) 异步把短期摘要蒸馏到 semantic（长期偏好）
            try:
                with timed("Memory:convert_to_semantic"):
                    memory.convert_to_semantic(zhipu_llm)
            except Exception as e:
                print(f"[run_for_user] shortterm→semantic 失败（可忽略）: {e}")

            trace_report()  # 打印本轮 span 树 (新格式: 树形 + 数据流)
            return final_report

        except Exception as e:
            print(f"[TravelWorkflow.run_for_user] 调用失败: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return f"抱歉，处理您的请求时出现了错误：{str(e)}"

    @classmethod
    def simple_reply(
        cls,
        user_text: str,
        user_id: str,
        session_id: str | None = None,
        memory: "MemoryManager | None" = None,
    ) -> str:
        """闲聊/非旅游意图的轻量回复：走 LLM + 记忆生命周期，不跑 6 状态机。

        与 run_for_user 对称：同样构造 MemoryManager、写 episodic、蒸馏 semantic，
        但不 kickoff Flow，只做一次 LLM 回答。供飞书 / CLI 的非旅游分支复用，
        避免在连接层重复写 LLM + 记忆逻辑。
        """
        try:
            trace_reset()
            quiet_crewai()
            sid = session_id or f"sess_{user_id}"

            if memory is None:
                memory = MemoryManager(sid, user_id, _redis_client, _is_redis_fallback)

            memory.add_message("user", user_text)

            with span("simple_reply",
                      user_id=user_id, session_id=sid,
                      message_len=len(user_text),
                      message_preview=user_text[:80]) as s:
                context_payload = memory.get_global_context_prompt(user_text)
                system_prompt = "你是一个亲切的旅游管家。请根据上下文自然地回答用户。"
                with timed("LLM:simple_reply"):
                    reply = zhipu_llm.call([
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": context_payload},
                    ]).strip()
                s.set_output(reply_len=len(reply))

            if not reply:
                reply = "抱歉，我暂时无法理解您的需求，请补充更多信息。"

            memory.add_message("assistant", reply)

            try:
                with timed("Memory:convert_to_semantic"):
                    memory.convert_to_semantic(zhipu_llm)
            except Exception as e:
                print(f"[simple_reply] shortterm→semantic 失败（可忽略）: {e}")

            trace_report()
            return reply

        except Exception as e:
            print(f"[TravelWorkflow.simple_reply] 调用失败: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return f"抱歉，处理您的请求时出现了错误：{str(e)}"


__all__ = [
    "TravelWorkflow",
    "_redis_client",
    "_is_redis_fallback",
]
