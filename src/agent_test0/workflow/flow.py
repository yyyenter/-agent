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

from crewai.flow import Flow, listen, start

from agent_test0.memory import MemoryManager, get_redis_or_fallback
from agent_test0.workflow.state import TravelState
from agent_test0.workflow.callbacks import run_crew_with_callback
from agent_test0.workflow.parsing import parse_step_feedback
from agent_test0.workflow.ask_user import (
    AskUserInterrupt,
    check_ask_user_hook,
    set_ask_user_question,
)
from agent_test0.workflow.llm import zhipu_llm
from agent_test0.workflow import nodes


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

    def _run_crew_with_callback(self, crew_class, inputs):
        return run_crew_with_callback(self, crew_class, inputs)

    def _check_ask_user_hook(self):
        return check_ask_user_hook(self)

    def _set_ask_user_question(self, question: str = None):
        set_ask_user_question(self, question)

    def _parse_step_feedback(self, raw_text: str) -> dict:
        return parse_step_feedback(self, raw_text)

    def _generate_final_report(self) -> str:
        return nodes.generate_final_report(self)

    # ============================================================
    # 状态节点（@start / @listen 装饰，方法体只调 nodes 里的业务函数）
    # ============================================================

    @start()
    def plan_steps(self):
        return nodes.run_planner(self)

    @listen(plan_steps)
    def step_preparer(self):
        return nodes.run_step_preparer(self)

    @listen(step_preparer)
    def step_executor(self):
        return nodes.run_step_executor(self)

    @listen(step_executor)
    def step_verifier(self):
        return nodes.run_step_verifier(self)

    @listen(step_verifier)
    def final_verifier(self):
        return nodes.run_final_verifier(self)

    def partial_replanner(self, failure_feedback: dict):
        """非 @listen 节点：由 step_verifier / final_verifier 直接调用"""
        return nodes.run_partial_replanner(self, failure_feedback)

    def finalize(self):
        return nodes.run_finalize(self)

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
            sid = session_id or f"sess_{user_id}_{abs(hash(user_text)) % 1000000:06d}"

            # 由 crew 自己构造 MemoryManager（连接层无需关心 redis 客户端）
            if memory is None:
                memory = MemoryManager(sid, user_id, _redis_client, _is_redis_fallback)

            # 1) 写入用户输入到 episodic 记忆
            memory.add_message("user", user_text)

            # 2) 把原始对话蒸馏到 working memory（短期约束提取）
            try:
                memory.convert_episodic_to_working(zhipu_llm)
            except Exception as e:
                print(f"[run_for_user] episodic→working 失败（可忽略）: {e}")

            # 3) 跑 Flow
            flow = cls(status_callback=status_callback, content_callback=content_callback)
            flow.state.message = user_text
            flow.state.user_id = user_id
            flow.state.session_id = sid
            flow.state.focus = memory.get_global_context_prompt(user_text)

            # 顶层捕获 AskUserInterrupt：节点用新 API 抛出时，安静吞掉，
            # state.final_report 已经在 ask_user_and_exit 里写好了
            try:
                flow.kickoff()
            except AskUserInterrupt as e:
                print(f"[run_for_user] AskUser 中断: {e.question} (field={e.blocking_field})")

            # 4) 取最终输出，按优先级兜底
            final_report = (flow.state.final_report or "").strip()

            if not final_report:
                if flow.state.needs_user_input and flow.state.user_question:
                    final_report = flow.state.user_question
                else:
                    completed_results = [
                        s.result for s in (flow.state.steps or [])
                        if s.status == "completed" and s.result
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

            # 5) 写助手回复到 episodic
            memory.add_message("assistant", final_report)

            # 6) 异步把短期摘要蒸馏到 semantic（长期偏好）
            try:
                memory.convert_to_semantic(zhipu_llm)
            except Exception as e:
                print(f"[run_for_user] shortterm→semantic 失败（可忽略）: {e}")

            return final_report

        except Exception as e:
            print(f"[TravelWorkflow.run_for_user] 调用失败: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return f"抱歉，处理您的请求时出现了错误：{str(e)}"


__all__ = [
    "TravelWorkflow",
    "_redis_client",
    "_is_redis_fallback",
]
