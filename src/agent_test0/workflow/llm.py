# agent_test0/workflow/llm.py
"""
共享 LLM 客户端。

【两个客户端并存的原因】
- zhipu_llm     : CrewAI 的 LLM 对象, 给 CrewAI Agent 用 (LiteLLM 路由)
- openai_client : 原生 openai.OpenAI 客户端, 给 instructor.from_openai 用
两者都指向 GLM 智谱, 通过 OpenAI 兼容 API 接入。

【为什么不复用 zhipu_llm 给 instructor】
CrewAI LLM 对象不实现 openai.OpenAI 的 chat.completions.create 接口,
instructor 拿不到需要的入口, 必须单独建一个原生 openai SDK 客户端。
"""
from __future__ import annotations

import os

from crewai import LLM
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# ─── 环境变量集中处 (structured.py 也会 import GLM_MODEL) ───
GLM_MODEL = os.getenv("GLM_MODEL_NAME") or "glm-4-flash"
GLM_BASE  = os.getenv("GLM_API_BASE") or "https://open.bigmodel.cn/api/paas/v4/"
GLM_KEY   = os.getenv("GLM_API_KEY") or ""


# ─── 给 CrewAI Agent 用 (走 LiteLLM 路由) ───
zhipu_llm = LLM(
    model=GLM_MODEL,
    base_url=GLM_BASE,
    api_key=GLM_KEY,
)

# ─── 给 instructor / 手写节点 用 (原生 openai SDK 接口) ───
openai_client = OpenAI(
    api_key=GLM_KEY,
    base_url=GLM_BASE,
)
