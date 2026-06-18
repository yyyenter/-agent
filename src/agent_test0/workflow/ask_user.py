# agent_test0/workflow/ask_user.py
"""
AskUser 中断机制 —— 让任何状态节点都能在发现信息缺失时立即中断本轮 Flow。

【两个原 Bug 的修复】

  Bug 1：原 _check_ask_user_hook 不"中断"，只"返回 True"——靠调用方写 return
  ----------------------------------------------------------------------
  原代码：
      if self._check_ask_user_hook(): return  # 调用方守纪律才能终止

  问题：任意一个节点忘写 return，hook 形同虚设。
  修复：新增 ask_user_and_exit() 用 raise AskUserInterrupt 强制中断，
        Flow 顶层捕获后立即停止整轮，不依赖任何节点的纪律。

  Bug 2：没有"已问字段"记录，下一轮可能重复问同一字段
  ----------------------------------------------------------------------
  原代码：根本没有 asked_fields 这个字段（阶段 A 已加，默认值修为 []）。
  修复：ask_user_and_exit 接受 blocking_field 参数，自动写入 state.asked_fields。
        节点在生成下一个问题前调用 has_already_asked(field)，问过的就不再问，
        改为做合理假设（写入 state.assumptions，由 final_report 披露给用户）。

【两套 API 并存的原因】

  - 旧 API（check_ask_user_hook / set_ask_user_question）：原节点入口已经在用，
    搬运时不动它们，保证阶段 D 的搬家不带行为变化（除了顺手修一个 final_report
    同步时机的小 bug）。
  - 新 API（ask_user_and_exit / has_already_asked / AskUserInterrupt）：给后续
    增量改造用——节点可以从"set 字段 + return"逐步升级到"raise 异常"。

【未来怎么用】

  节点内部发现信息缺失：
      from agent_test0.workflow.ask_user import ask_user_and_exit, has_already_asked

      if not has_already_asked(flow, "trip_days"):
          ask_user_and_exit(flow, "请问想去几天？", blocking_field="trip_days")
      else:
          # 已经问过一次，用户没补充 → 自己假设
          flow.state.assumptions.append("默认 3 天行程")

  Flow 顶层捕获：
      try:
          flow.kickoff()
      except AskUserInterrupt:
          # state.final_report 已经写好问题文本，直接返回即可
          pass
"""


# ============================================================
# 异常：用于硬中断 Flow
# ============================================================

class AskUserInterrupt(Exception):
    """
    节点抛出此异常后，Flow 顶层捕获，立即终止本轮，不再调用任何下游节点。

    与 NotImplementedError / RuntimeError 等通用异常不同，这是业务正常分支，
    Flow 顶层会安静吞掉它（不打印 traceback），并保证 state.final_report
    已经写入了问题文本。
    """

    def __init__(self, question: str, blocking_field: str = ""):
        self.question = question
        self.blocking_field = blocking_field
        super().__init__(question)


# ============================================================
# 新 API：硬中断 + 防重复问
# ============================================================

def ask_user_and_exit(flow, question: str, blocking_field: str = "") -> None:
    """
    在任何状态节点中调用此函数即可"硬中断"本轮 Flow。

    做的事：
      1. 写 state.needs_user_input / user_question / final_report
      2. 把 blocking_field 加入 asked_fields（防下一轮重复问，跨轮持久化到 Redis）
      3. 抛 AskUserInterrupt → Flow 顶层捕获 → 不再调任何节点

    Args:
        flow: TravelWorkflow 实例
        question: 给用户看的提问文本（最终也写入 final_report）
        blocking_field: 缺哪个具体字段，可枚举：
            "destination" | "trip_days" | "budget" | "group_size" |
            "transport_mode" | "accommodation_pref" | "meal_pref"
            空字符串表示不记录（不推荐）。
    """
    flow.state.needs_user_input = True
    flow.state.user_question = question
    flow.state.final_report = question

    if blocking_field and blocking_field not in flow.state.asked_fields:
        flow.state.asked_fields.append(blocking_field)

    flow.notify(f"🙋 [AskUser] {question}")
    print(f"[AskUser] blocking_field={blocking_field!r}, asked_fields={flow.state.asked_fields}")

    raise AskUserInterrupt(question, blocking_field)


def has_already_asked(flow, field: str) -> bool:
    """节点决策辅助：这个字段（含跨轮）是否已经问过？问过就别再问，自己假设。"""
    return field in flow.state.asked_fields


# ============================================================
# 旧 API：保留以兼容现有节点入口检查
# ============================================================

def check_ask_user_hook(flow) -> bool:
    """
    旧式 hook：检查 state.needs_user_input。如为 True 则同步 final_report 并返回 True。
    调用方负责自己 return。

    保留原因：原代码每个节点入口都调用了它，搬运时不动它们。
    """
    if flow.state.needs_user_input:
        flow.notify(f"🙋 [AskUser] {flow.state.user_question}")
        flow.state.final_report = flow.state.user_question
        return True
    return False


def set_ask_user_question(flow, question: str = None) -> None:
    """
    旧式：设置中断标记（不抛异常，调用方负责自己 return）。

    【顺手修的小 bug】原代码这里没写 final_report，要等下一个节点的 hook 才同步。
    现在直接在这里写，避免"延迟一步"的窗口期。
    """
    if question is None:
        question = "信息不足，无法继续规划。"
    flow.state.needs_user_input = True
    flow.state.user_question = question
    # ★ Bug 修复：原代码这里没写 final_report，导致下一节点之前的兜底取不到
    flow.state.final_report = question

    print(f"\n{'='*60}")
    print(f"[AskUser] 信息不足，需要向用户提问...")
    print(f"{'='*60}")
    flow.notify(f"🙋 [AskUser] {question}")
