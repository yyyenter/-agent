# agent_test0/workflow/structured.py
"""
直接 LLM + Pydantic 的结构化调用工具。

【定位】
- 单一职责: (system + user prompt, Pydantic 类) → Pydantic 实例
- 不关心 prompt 从哪来 (那是 prompt.py 的事)
- 不关心 LLM 客户端怎么建 (那是 llm.py 的事)

【替代关系】
替代 CrewAI Agent 的 ReAct 解析层. 当前节点 (Planner / StepPreparer /
Verifier / Replanner / FinalVerifier) 都不需要工具调用, 只需要
"输入上下文 → 结构化对象", 因此不应让 CrewAI Agent 先按
Thought/Action/Final Answer 解析。

【向后兼容】
历史 nodes.py 里习惯从 structured 一起 import load_task_prompt / StructuredCallError,
所以在这里 re-export, 避免大规模改 import。
"""
from __future__ import annotations

import instructor
from pydantic import BaseModel, ValidationError

from agent_test0.workflow.llm import GLM_MODEL, openai_client
# 向后兼容 re-export (原本 nodes.py 从 structured 直接 import 这两个)
from agent_test0.workflow.prompt import load_task_prompt, StructuredCallError  # noqa: F401


# instructor 把 openai_client 包一层, 加上 schema 注入 + json.loads + Pydantic 校验 + 自动 retry
client = instructor.from_openai(openai_client, mode=instructor.Mode.JSON)


def call_structured(
    prompt: str,
    model_cls: type[BaseModel],
    *,
    system: str = "",
    max_retries: int = 2,
    temperature: float = 0.2,
) -> BaseModel:
    """调 LLM 拿结构化输出 (instructor Mode.JSON).

    参数:
      prompt       : user message 内容
      model_cls    : Pydantic 类, 用作 response_model, 决定输出结构
      system       : 可选 system message (身份 + 规则 + 硬约束)
      max_retries  : Pydantic 校验失败时的重试次数
      temperature  : 采样温度, 结构化调用建议低温 (0.1-0.3)

    返回: model_cls 的实例, 已通过 Pydantic 校验

    异常: StructuredCallError (统一包装 instructor / Pydantic / OpenAI 层的异常)
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        return client.chat.completions.create(
            model=GLM_MODEL,
            messages=messages,
            response_model=model_cls,
            max_retries=max_retries,
            temperature=temperature,
        )
    except ValidationError as e:
        raise StructuredCallError(f"Pydantic 校验失败: {e}") from e
    except Exception as e:
        raise StructuredCallError(f"LLM 调用失败: {type(e).__name__}: {e}") from e
