# agent_test0/workflow/parsing.py
"""
统一 JSON 解析工具。

原 crew.py 在 6 个节点里散落 6 处 `re.search(r'\\{.*\\}', raw_text, re.DOTALL)`，
脆弱且重复。本文件集中 JSON 提取与节点反馈解析逻辑。

提供：
- extract_json_object: 从 LLM 原始输出中尽力解析出第一个 JSON 对象
- parse_step_feedback: 解析 StepVerifier / FinalVerifier 的反馈结构
"""

import re
import json


def extract_json_object(raw_text: str) -> dict | None:
    """
    从 LLM 原始输出中提取第一个 JSON 对象。

    解析顺序（越前面越准）：
      1. ```json ... ``` 代码块
      2. ``` ... ``` 普通代码块
      3. 裸 {...}（贪婪匹配，能容纳嵌套大括号）

    返回 dict（成功）或 None（无法解析）。永远不抛异常。
    """
    if not raw_text:
        return None

    # 1) ```json 代码块
    m = re.search(r'```json\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 2) ``` 普通代码块
    m = re.search(r'```\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 3) 裸 {...}：贪婪匹配以保证嵌套大括号也能拿到完整对象
    m = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None


def parse_step_feedback(flow, raw_text: str) -> dict:
    """
    解析步骤验证反馈。

    优先识别结构化 JSON：
      - {"verdict": "pass" | "retry" | "fail" | "ask_user", "reason": "...", "question": "..."}
      - FinalVerifier 兼容：global_verdict / failed_step_ids / suggested_corrections

    命中 ask_user 时，会通过 flow._set_ask_user_question 设置结构化中断标记。
    无 JSON 或解析失败时，默认返回 pass（保守策略，避免无谓重试）。

    Args:
        flow: TravelWorkflow 实例（需要其 _set_ask_user_question 能力）
        raw_text: LLM 原始输出
    """
    feedback = extract_json_object(raw_text)
    if feedback is not None:
        # FinalVerifier 兼容：global_verdict -> verdict
        if "global_verdict" in feedback and "verdict" not in feedback:
            feedback["verdict"] = "pass" if feedback["global_verdict"] == "pass" else "fail_with_patches"

        # 结构化用户提问
        if feedback.get("verdict") == "ask_user":
            question = feedback.get("question") or feedback.get("reason") or "信息不足，请补充。"
            flow._set_ask_user_question(question)
        return feedback

    # 没有可解析 JSON：默认通过
    print(f"[parse_step_feedback] 无法解析 JSON，默认通过: {raw_text[:120]}")
    return {"verdict": "pass", "reason": "无法解析反馈，默认通过"}
