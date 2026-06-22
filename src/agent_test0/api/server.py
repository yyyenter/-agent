#!/usr/bin/env python
import os
import sys
import logging
import asyncio
import json
import uuid
import uvicorn
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import StreamingResponse
from typing import Optional
from crewai import LLM
from dotenv import load_dotenv

def hard_print(text):
    """无视任何劫持，直接把字刻在终端屏幕上"""
    try:
        os.write(1, (str(text) + "\n").encode('utf-8'))
    except Exception:
        pass

# 加载 .env 文件
load_dotenv()

# =========================================
# 暴力过滤器：彻底屏蔽烦人的 Token 消耗日志
# =========================================
class TokenUsageFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if "OpenAI API usage" in msg or "litellm" in msg.lower():
            return False
        return True

console_handler = logging.StreamHandler(sys.stdout)
console_handler.addFilter(TokenUsageFilter())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[console_handler]
)

# 压制底层库日志
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# ---- 意图路由: 加载 routes.json + 构建旅行关键词索引 ----
# 路径锚定到仓库根的 knowledge/routes.json：
#   server.py 位于  <repo>/src/agent_test0/api/server.py
#   parents[3]   →  <repo>/
ROUTES_PATH = Path(__file__).resolve().parents[3] / "knowledge" / "routes.json"
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
    try:
        result = intent_router(message)
        if result and result.name:
            return result.name
    except Exception:
        pass
    return "default_chat"

# 环境变量隔离
os.environ["OPENAI_API_KEY"] = os.getenv("GLM_API_KEY", "dummy_key")
os.environ["OPENAI_API_BASE"] = os.getenv("GLM_API_BASE", "")
glm_model = os.getenv("GLM_MODEL_NAME", "glm-4-flash")
os.environ["OPENAI_MODEL_NAME"] = f"openai/{glm_model}"
os.environ["LITELLM_LOG"] = "ERROR" 
os.environ["SUPPRESS_LITELLM_LOGS"] = "True"

from agent_test0.memory import MemoryManager, get_redis_or_fallback
from agent_test0.workflow import TravelWorkflow, TravelState
import json

redis_client, is_redis_fallback = get_redis_or_fallback()

zhipu_llm = LLM(
    model=os.getenv("GLM_MODEL_NAME") or "glm-4-flash",
    base_url=os.getenv("GLM_API_BASE") or "",
    api_key=os.getenv("GLM_API_KEY") or "",
)

app: FastAPI = FastAPI(title="智能旅游规划 Agent API")

class ChatRequest(BaseModel):
    user_id: str = "yyy"
    session_id: Optional[str] = None
    message: str

def rewrite_query_lightweight(memory: MemoryManager, current_message: str) -> str:
    history = memory.get_chat_history()[-10:]
    if not history: return current_message
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
    rewrite_prompt = f"""任务：将用户的【当前回复】重写为独立明确的句子。仅替换代词，只输出一句话。
    【最近对话】：\n{history_text}\n【当前回复】：user: {current_message}\n【重写结果】："""
    try:
        return zhipu_llm.call([{"role": "user", "content": rewrite_prompt}]).strip()
    except Exception:
        return current_message

@app.post("/api/chat_stream")
async def chat_endpoint_stream(request: ChatRequest):
    actual_user_id: str = request.user_id
    actual_session_id = request.session_id or f"sess_{uuid.uuid4().hex[:6]}"
    memory = MemoryManager(actual_session_id, actual_user_id, redis_client, is_redis_fallback)
    memory.add_message("user", request.message)
    # 短期记忆直接使用 episodic 原文，不再做 LLM 蒸馏 summary
    contextual_message = rewrite_query_lightweight(memory, request.message)
    intent_name = classify_intent(contextual_message)
    
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()

    def run_crewai_task():
        # 【核弹级打印】验证子线程执行
        hard_print(f"\n🚀 [核弹级测试] run_crewai_task 真的开始执行了！")
        hard_print(f"🚀 [参数检查] intent={intent_name}, user_id={actual_user_id}, session_id={actual_session_id}")

        try:
            if intent_name == "travel":
                # ⚠️ 这里修改了回调，确保把消息推送到前端识别的 type 里
                def workflow_status_listener(status_text: str):
                    asyncio.run_coroutine_threadsafe(queue.put({"type": "status", "content": status_text}), loop)

                def workflow_content_listener(content: str, content_type: str):
                    asyncio.run_coroutine_threadsafe(queue.put({"type": content_type, "content": content}), loop)

                travel_flow = TravelWorkflow(status_callback=workflow_status_listener, content_callback=workflow_content_listener)
                travel_flow.state.message = request.message
                travel_flow.state.focus = memory.get_global_context_prompt(request.message)
                travel_flow.state.user_id = actual_user_id
                travel_flow.state.session_id = actual_session_id

                # ===== 从 Redis 恢复 Flow 状态（跨请求保持计划进度）=====
                flow_state_key = f"session:{actual_session_id}:flow_state"
                saved_state = redis_client.hgetall(flow_state_key)
                if saved_state:
                    hard_print("📦 [状态恢复] 从 Redis 恢复 Flow 状态")
                    try:
                        steps_json = saved_state.get("steps", "[]")
                        if steps_json:
                            from agent_test0.workflow.state import StepPlan
                            travel_flow.state.steps = [StepPlan(**s) for s in json.loads(steps_json)]
                        travel_flow.state.current_step_index = int(saved_state.get("current_step_index", "0"))
                        travel_flow.state.location = saved_state.get("location", "未知地点")
                        # focus 每轮由 MemoryManager 基于原始对话重新组装，避免恢复旧 summary 造成上下文泄露
                    except Exception as e:
                        hard_print(f"⚠️ [状态恢复] 解析失败: {e}")

                result = travel_flow.kickoff()

                # ===== 保存 Flow 状态到 Redis（24h TTL）=====
                flow_state = {
                    "steps": json.dumps([s.model_dump() for s in (result.state.steps or [])], ensure_ascii=False),
                    "current_step_index": str(result.state.current_step_index),
                    "location": result.state.location or "",
                    "focus": result.state.focus or "",
                    "final_report": result.state.final_report or "",
                }
                redis_client.hset(flow_state_key, mapping=flow_state)
                redis_client.expire(flow_state_key, 86400)
                hard_print("💾 [状态保存] Flow 状态已持久化到 Redis")

                asyncio.run_coroutine_threadsafe(queue.put({"type": "finish", "content": result.state.final_report}), loop)
            else:
                context_payload = memory.get_global_context_prompt(request.message)
                system_prompt = "你是一个亲切的旅游管家。请根据以下上下文自然地回答用户。"
                response = zhipu_llm.call([{"role": "system", "content": system_prompt}, {"role": "user", "content": context_payload}])
                reply_text = response.strip()
                memory.add_message("assistant", reply_text)
                memory.convert_to_semantic(zhipu_llm)
                asyncio.run_coroutine_threadsafe(queue.put({"type": "finish", "content": reply_text}), loop)
        except Exception as e:
            hard_print(f"💥 [子线程异常] {type(e).__name__}: {str(e)}")
            asyncio.run_coroutine_threadsafe(queue.put({"type": "error", "content": f"系统运行出错: {str(e)}"}), loop)
        finally:
            hard_print(f"🏁 [核弹级测试] run_crewai_task 执行完毕")

    import concurrent.futures
    def run_in_thread(): run_crewai_task()
    loop.run_in_executor(None, run_in_thread)

    async def event_generator():
        while True:
            msg = await queue.get()
            hard_print(f"📡 [SSE发送] type={msg['type']}, content={str(msg['content'])[:100]}")
            yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
            if msg["type"] in ["finish", "error"]: break

    return StreamingResponse(event_generator(), media_type="text/event-stream")

def run():
    hard_print("====================================")
    hard_print("[GLOBAL] 智能旅游规划系统 - Agent API 服务已启动")
    hard_print("====================================")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    run()