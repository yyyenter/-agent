#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""阶段0验证：TravelState(含 StepPlan/ToolResult 嵌套) 在 LangGraph MemorySaver 能 round-trip。

这是草图「风险1」：嵌套 Pydantic 模型经 checkpoint(JSON 序列化) 后类型是否保留。
若失败，方案A 的 checkpoint 跨轮恢复会炸，需改用 TypedDict 或自定义 serializer。
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agent_test0.workflow.state import TravelState, StepPlan, ToolResult


def test_nested_state_roundtrip():
    """构造含嵌套 StepPlan 的 TravelState，跑图，从 checkpoint 取回后断言类型。"""

    # 一个会修改 state 的节点
    def planner_node(state: TravelState) -> dict:
        # 模拟 run_planner 写入嵌套字段
        state.steps = [
            StepPlan(index=0, description="查询天气", tools=["weather_tool"]),
            StepPlan(index=1, description="整合报告", dependencies=[0]),
        ]
        state.location = "重庆"
        state.current_step_index = 0
        # 返回增量（LangGraph 合并）
        return {
            "steps": state.steps,
            "location": state.location,
            "current_step_index": state.current_step_index,
        }

    def verifier_node(state: TravelState) -> dict:
        # 读取上一步写入的嵌套字段，验证类型
        assert isinstance(state.steps, list), f"steps 不是 list: {type(state.steps)}"
        assert len(state.steps) == 2, f"steps 长度不对: {len(state.steps)}"
        s0 = state.steps[0]
        assert isinstance(s0, StepPlan), f"steps[0] 不是 StepPlan: {type(s0)}"
        assert s0.description == "查询天气"
        assert s0.tools == ["weather_tool"]
        assert state.steps[1].dependencies == [0], "嵌套 dependencies 丢了"
        assert state.location == "重庆"
        # 再写一个嵌套 ToolResult
        state.steps[0].tool_results = [
            ToolResult(tool_name="weather_tool", input={"city": "重庆"},
                      output={"天气": "多云"}, output_text="多云", error=""),
        ]
        return {"steps": state.steps}

    graph = StateGraph(TravelState)
    graph.add_node("planner", planner_node)
    graph.add_node("verifier", verifier_node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "verifier")
    graph.add_edge("verifier", END)

    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)

    # 跑一遍
    config = {"configurable": {"thread_id": "test-thread-001"}}
    initial = TravelState(message="想去重庆玩3天", user_id="u1", session_id="s1")
    result = compiled.invoke(initial, config=config)

    # 关键：从 checkpoint 取回，验证嵌套类型在序列化后仍保留
    saved = compiled.get_state(config)
    state_values = saved.values

    print("=== round-trip 后的字段检查 ===")
    print("location:", state_values.get("location"), type(state_values.get("location")).__name__)
    print("steps 数量:", len(state_values.get("steps", [])))
    for i, s in enumerate(state_values.get("steps", [])):
        print(f"  steps[{i}] type={type(s).__name__} desc={getattr(s,'description',None)!r}")
        tr = getattr(s, "tool_results", None)
        if tr:
            print(f"    tool_results[0] type={type(tr[0]).__name__} output={getattr(tr[0],'output',None)!r}")

    # 断言嵌套类型保留
    assert isinstance(state_values["steps"], list)
    assert isinstance(state_values["steps"][0], StepPlan), \
        f"checkpoint 后 steps[0] 类型变了: {type(state_values['steps'][0])}"
    tr = state_values["steps"][0].tool_results
    assert isinstance(tr[0], ToolResult), \
        f"checkpoint 后 tool_results[0] 类型变了: {type(tr[0])}"
    assert tr[0].output == {"天气": "多云"}, "嵌套 dict 值丢了"

    print("\n✅ 风险1验证通过：Pydantic 嵌套 state 在 MemorySaver round-trip 类型保留")
    print("   → 方案A 的 checkpoint 跨轮恢复可行")


def test_messages_reducer_warning():
    """TravelState 若有 messages 字段，LangGraph 默认 reducer 行为预检。"""
    from agent_test0.workflow.state import TravelState
    fields = TravelState.model_fields
    print("\n=== TravelState 字段 reducer 预检 ===")
    list_fields = [n for n, f in fields.items()
                   if "list" in str(f.annotation) or "List" in str(f.annotation)]
    print("list 类型字段:", list_fields)
    # steps 索引寻址 → 整体覆盖（默认 last-write-wins）即可
    print("steps 用整体覆盖(默认) → OK（索引寻址非追加）")
    print("asked_fields/assumptions 若跨节点追加 → 需配 add reducer（阶段2b 确认）")


if __name__ == "__main__":
    test_nested_state_roundtrip()
    test_messages_reducer_warning()
