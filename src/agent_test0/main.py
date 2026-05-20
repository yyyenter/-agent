#!/usr/bin/env python
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# 配置 logging，确保子线程的输出也能看到
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

import asyncio
import json
import os
import uuid
import uvicorn
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import StreamingResponse
from typing import Optional
from crewai import LLM
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# ---- 意图路由: 加载 routes.json + 构建旅行关键词索引 ----
ROUTES_PATH = Path(__file__).resolve().parent.parent.parent / "knowledge" / "routes.json"
with open(ROUTES_PATH, "r", encoding="utf-8") as _f:
    ROUTES = json.load(_f)

from semantic_router import Route, SemanticRouter
from semantic_router.encoders.ollama import OllamaEncoder


travel_route = Route(
    name="travel",
    utterances=ROUTES.get("travel", []),
    description="旅游规划相关请求",
)
chat_route = Route(
    name="default_chat",
    utterances=ROUTES.get("chitchat", []),
    description="闲聊和日常对话",
)

intent_router = SemanticRouter(
    routes=[travel_route, chat_route],
    encoder=OllamaEncoder(
        name="nomic-embed-text",
        base_url="http://localhost:11434",
        score_threshold=0.3,
    ),
)


def classify_intent(message: str) -> str:
    """基于 semantic_router + Ollama 的语义意图分类"""
    try:
        result = intent_router(message)
        if result and result.name:
            return result.name
    except Exception:
        pass
    return "default_chat"

# 环境变量隔离与注入
os.environ["OPENAI_API_KEY"] = os.getenv("GLM_API_KEY", "dummy_key")
os.environ["OPENAI_API_BASE"] = os.getenv("GLM_API_BASE", "")
glm_model = os.getenv("GLM_MODEL_NAME", "glm-4-flash")
os.environ["OPENAI_MODEL_NAME"] = f"openai/{glm_model}"
os.environ["LITELLM_LOG"] = "ERROR" 
os.environ["SUPPRESS_LITELLM_LOGS"] = "True"

# ✅ 从我们全新拆分的 harness 导入隔离机制与记忆管理器
from agent_test0.harness import MemoryManager, get_redis_or_fallback
from agent_test0.crew import TravelWorkflow

# 初始化 Redis 物理连接（自动回退到内存存储）
redis_client, is_redis_fallback = get_redis_or_fallback()

zhipu_llm = LLM(
    model=os.getenv("GLM_MODEL_NAME") or "glm-4-flash",
    base_url=os.getenv("GLM_API_BASE") or "",
    api_key=os.getenv("GLM_API_KEY") or "",
)

app: FastAPI = FastAPI(title="智能旅游规划 Agent API", description="基于多智能体的旅游规划服务")

class ChatRequest(BaseModel):
    user_id: str = "yyy"
    session_id: Optional[str] = None
    message: str

def rewrite_query_lightweight(memory: MemoryManager, current_message: str) -> str:
    """
    【路由前专用】轻量级指代消解
    - 只看最近5轮对话，不含长期记忆
    - 绝对不越界污染路由判断
    """
    history = memory.get_chat_history()[-10:]  # 最近5轮
    if not history:
        return current_message

    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
    
    rewrite_prompt = f"""你是一个极其严格的"指代消解"组件。
    任务：将用户的【当前回复】重写为一句独立、明确的句子。

    【绝对规则】：
    1. 仅仅替换代词（如把"那里"替换为上文提到的地点）。
    2. 如果当前回复是明确的转移话题（如"你好"、"你是谁"、"讲个笑话"），【必须100%原样保留】。
    3. 只输出一句话，不加任何解释。

    【最近对话】：
    {history_text}

    【当前回复】：
    user: {current_message}

    【重写结果】：
    """
    try:
        response = zhipu_llm.call([{"role": "user", "content": rewrite_prompt}])
        return response.strip()
    except Exception as e:
        return current_message

# --- 流式 API 核心交互接口 ---
@app.post("/api/chat_stream")
async def chat_endpoint_stream(request: ChatRequest):
    actual_user_id: str = request.user_id
    actual_session_id = request.session_id or f"sess_{uuid.uuid4().hex[:6]}"
    
    # 实例化当前会话的记忆管理器
    memory = MemoryManager(actual_session_id, actual_user_id, redis_client, is_redis_fallback)

    # 路由前先追加消息，保持上下文连贯
    memory.add_message("user", request.message)

    # 转换历史对话为短期约束
    memory.convert_episodic_to_working(zhipu_llm)
    
    # 【路由前专用】轻量级指代消解 - 绝对不传入长期记忆
    contextual_message = rewrite_query_lightweight(memory, request.message)
    
    # 多路意图分发: semantic_router + Ollama 语义路由
    intent_name = classify_intent(contextual_message)
    
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()

    def run_crewai_task():
        try:
            if intent_name == "travel":
                def workflow_status_listener(status_text: str):
                    asyncio.run_coroutine_threadsafe(
                        queue.put({"type": "status", "content": status_text}), loop
                    )

                def workflow_content_listener(content: str, content_type: str):
                    asyncio.run_coroutine_threadsafe(
                        queue.put({"type": content_type, "content": content}), loop
                    )

                travel_flow = TravelWorkflow(status_callback=workflow_status_listener, content_callback=workflow_content_listener)
                
                # 保留原始消息 + 完整上下文，让模型充分发挥能力
                # ReadMemoryTool 会在 Agent 执行时自动注入长期偏好
                travel_flow.state.message = request.message  # 原始消息，保留完整语义
                travel_flow.state.focus = memory.get_global_context_prompt(request.message)
                travel_flow.state.user_id = actual_user_id
                travel_flow.state.session_id = actual_session_id
                
                # 直接执行 TravelWorkflow
                result = travel_flow.kickoff()
                
                # 发送最终结果
                asyncio.run_coroutine_threadsafe(
                    queue.put({"type": "finish", "content": result.state.final_report}), loop
                )
            else:
                # ---------- 闲聊分支 ----------
                context_payload = memory.get_global_context_prompt(request.message)
                system_prompt = "你是一个亲切的旅游管家。请根据以下上下文自然地回答用户。"
                
                response = zhipu_llm.call([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context_payload}
                ])
                reply_text = response.strip()
                memory.add_message("assistant", reply_text)
                # 先提取本轮对话的短期约束，再基于短期摘要提炼长期偏好
                memory.convert_episodic_to_working(zhipu_llm)
                memory.convert_to_semantic(zhipu_llm)
                
                asyncio.run_coroutine_threadsafe(
                    queue.put({"type": "finish", "content": reply_text}), loop
                )
        except Exception as e:
            asyncio.run_coroutine_threadsafe(
                queue.put({"type": "error", "content": f"系统运行出错: {str(e)}"}), loop
            )

    # 在线程池中执行 CrewAI 任务
    import concurrent.futures

    def run_in_thread():
        run_crewai_task()

    loop.run_in_executor(None, run_in_thread)

    async def event_generator():
        while True:
            msg = await queue.get()
            yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
            if msg["type"] in ["finish", "error"]:
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")

def run():
    print("====================================")
    print("[GLOBAL] 智能旅游规划系统 - Agent API 服务已启动")
    print("====================================")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    run()
