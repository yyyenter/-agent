# agent_test0/workflow/graph.py
"""
LangGraph StateGraph —— 6 状态机的图声明 (方案 A1)。

替代旧 nodes.run_state_machine(flow) 的手写 while 循环。
图状态 = TravelState (Pydantic), 节点 = nodes.xxx_node(state, config) -> dict。

图结构:
  START
    ↓
  planner ─┬─ needs_user_input ─────► END (追问)
           ├─ final_report(兜底)/no-steps ─► END (简单回答)
           └─ 有 steps ─► step_preparer
                                ↓
                          step_executor
                                ↓
                          step_verifier ─┬─ ask_user ──► END
                                         ├─ retry ─────► step_preparer (同 index 重跑)
                                         ├─ fail ──────► replanner ─► step_preparer
                                         └─ pass & 全完成 ─► final_verifier
                                                            ├─ ask_user ─► END
                                                            ├─ final_report ─► END
                                                            └─ fail ─► replanner ─► step_preparer

retry/fail 的回边通过读 state 判定 (见下方条件边函数)。
"""

from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from agent_test0.workflow.state import TravelState
from agent_test0.workflow import nodes


# 全局单例 checkpointer: 存活在进程内, 按 thread_id (= session_id) 隔离.
# 用途: 让节点里的 interrupt() 能在下轮 Command(resume=...) 恢复继续.
_shared_checkpointer = InMemorySaver()


# ============================================================
# 条件边: 读 state 决定下一个节点
# ============================================================

def _after_planner(state: TravelState) -> str:
    """planner 之后路由。"""
    # AskUser (planner guard 或 LLM 主动提问)
    if state.needs_user_input:
        return "end"
    # 兜底简单回答 (无 steps, final_report 已写)
    if not state.steps or state.final_report:
        return "end"
    return "step_preparer"


def _after_verifier(state: TravelState) -> str:
    """step_verifier 之后路由。

    verifier_node 内部已根据 verdict 改 state:
      - pass: current_step_index += 1 (推进)
      - retry: 步骤状态重置为 pending, index 不变, step_retry_counts 更新
      - fail: failed_steps_indices 已追加, 已调 partial_replanner_node 追加新步骤
      - ask_user: needs_user_input=True
    """
    if state.needs_user_input:
        return "end"
    # fail: replanner 已被 verifier 内部调用并追加步骤 → 回 prepare 跑新步骤
    #   (失败原步骤状态=failed, 新步骤已 extend 到 steps 末尾, current_step_index 指向新步骤)
    # retry: 步骤状态 pending → 回 prepare 重跑同一步骤
    # 两者都表现为「还有未完成步骤」→ 回 prepare
    # 但若 fail 后 replanner 没追加新步骤 (appended=0) 且无更多 pending, 走 final_verifier 兜底
    if state.current_step_index < len(state.steps):
        # 检查当前步骤是否需要重跑 (retry/pending) 或是新步骤 (replanner 追加)
        cur = state.steps[state.current_step_index]
        if cur.status in ("pending", "executing") or not cur.prepared:
            return "step_preparer"
        # 已 completed/failed 但 index 还没推进到这里 → 不该发生, 兜底推进
        return "step_preparer"
    # 所有步骤 index 走完 → final_verifier
    return "final_verifier"


def _after_final_verifier(state: TravelState) -> str:
    """final_verifier 之后路由。

    final_verifier_node 内部已处理:
      - pass: final_report 已生成, is_done=True
      - fail: 已调 partial_replanner_node 追加新步骤
      - ask_user: needs_user_input=True
    """
    if state.needs_user_input:
        return "end"
    if state.final_report and state.is_done:
        return "end"
    # fail: replanner 追加了新步骤 → 回 prepare
    if state.current_step_index < len(state.steps):
        return "step_preparer"
    # replanner 未追加新步骤也没 final_report → 兜底结束
    return "end"


# ============================================================
# 图构建
# ============================================================

def build_travel_graph():
    """构建并编译 TravelWorkflow 的 StateGraph, 带 InMemorySaver checkpointer.

    checkpointer 让 interrupt() / Command(resume=...) 生效:
      - 节点抛 interrupt 时, 图暂停, 全 state 保存在 checkpointer 里 (按 thread_id 分)
      - 下轮同 thread_id 再 invoke 时用 Command(resume=user_answer), 从中断处继续
    单进程 InMemorySaver 够用 (每个 session 一个 thread_id).
    生产要跨进程持久化时可换 langgraph-checkpoint-redis (需另装).
    """
    g = StateGraph(TravelState)

    # 节点
    g.add_node("planner", nodes.planner_node)
    g.add_node("step_preparer", nodes.step_preparer_node)
    g.add_node("step_executor", nodes.step_executor_node)
    g.add_node("step_verifier", nodes.step_verifier_node)
    g.add_node("final_verifier", nodes.final_verifier_node)
    # replanner 作为独立节点: verifier/final_verifier fail 时经条件边进入,
    # 内部从 state.failed_steps_indices 取失败步骤 + 追加补救步骤。
    # (注: step_verifier/final_verifier 内部也直接调 partial_replanner_node,
    #  这里的独立节点用于「fail 但 verifier 没内部调」的兜底路径。)
    g.add_node("replanner", lambda state, config=None: nodes.partial_replanner_node(state, config, None))
    g.add_node("finalize", nodes.finalize_node)

    # 边
    g.add_edge(START, "planner")

    g.add_conditional_edges(
        "planner",
        _after_planner,
        {
            "end": END,
            "step_preparer": "step_preparer",
        },
    )

    # prepare → execute → verify 固定链
    g.add_edge("step_preparer", "step_executor")
    g.add_edge("step_executor", "step_verifier")

    # verify 之后条件路由
    g.add_conditional_edges(
        "step_verifier",
        _after_verifier,
        {
            "end": END,
            "step_preparer": "step_preparer",
            "final_verifier": "final_verifier",
        },
    )

    # final_verifier 之后条件路由
    g.add_conditional_edges(
        "final_verifier",
        _after_final_verifier,
        {
            "end": END,
            "step_preparer": "step_preparer",
        },
    )

    # replanner 之后回 step_preparer (追加的新步骤从这里跑)
    g.add_edge("replanner", "step_preparer")
    # finalize → END
    g.add_edge("finalize", END)

    return g.compile(checkpointer=_shared_checkpointer)


# 模块级单例: 图只编译一次, 每轮 invoke 复用 (checkpointer 在 invoke config 里给)
_compiled_graph = None


def get_travel_graph():
    """惰性编译并缓存图单例。"""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_travel_graph()
    return _compiled_graph


__all__ = ["build_travel_graph", "get_travel_graph"]
