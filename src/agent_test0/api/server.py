#!/usr/bin/env python
import os
import sys
import logging
import asyncio
import json
import uuid
import uvicorn
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

# ---- 意图路由: 统一从共享模块（API / 飞书 / CLI 共用 routes.json + Ollama）----
# 懒构造 + 关键词降级，详见 agent_test0.workflow.intent
from agent_test0.workflow.intent import classify_intent

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
                # ⚠️ 回调推送到前端 SSE queue
                def workflow_status_listener(status_text: str):
                    asyncio.run_coroutine_threadsafe(queue.put({"type": "status", "content": status_text}), loop)

                def workflow_content_listener(content: str, content_type: str):
                    asyncio.run_coroutine_threadsafe(queue.put({"type": content_type, "content": content}), loop)

                # 方案A1: 走 run_for_user (= graph.invoke), 与飞书/CLI 统一入口。
                # 跨轮上下文由 MemoryManager (Redis 记忆系统) 管理;
                # 不再手动 kickoff + 手动 Redis 存取 steps (graph 每轮重跑 Planner)。
                final_report_text = TravelWorkflow.run_for_user(
                    user_text=request.message,
                    user_id=actual_user_id,
                    session_id=actual_session_id,
                    memory=memory,
                    status_callback=workflow_status_listener,
                    content_callback=workflow_content_listener,
                )

                asyncio.run_coroutine_threadsafe(queue.put({"type": "finish", "content": final_report_text}), loop)
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