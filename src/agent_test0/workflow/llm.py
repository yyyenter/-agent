# agent_test0/workflow/llm.py
"""
共享 LLM 实例与共享工具。

所有 Crew/Node 通过 import 此模块来共用同一个 GLM 连接，避免重复配置。
GLM 智谱 API 通过 OpenAI 兼容路由接入，凭据从 .env 读取（GLM_API_KEY/GLM_API_BASE/GLM_MODEL_NAME）。
"""

import os
from crewai import LLM
from crewai_tools import TavilySearchTool
from dotenv import load_dotenv

load_dotenv()


# ─── 共享 LLM 实例 ───
zhipu_llm = LLM(
    model=os.getenv("GLM_MODEL_NAME") or "glm-4-flash",
    base_url=os.getenv("GLM_API_BASE") or "",
    api_key=os.getenv("GLM_API_KEY") or "",
)


# ─── 共享搜索工具 ───
# 在模块级实例化一次，所有 Crew 共用同一个对象
search_tool = TavilySearchTool()
