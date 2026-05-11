#!/usr/bin/env python
import asyncio
import json
import os
import uuid
import redis
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import StreamingResponse
from crewai import LLM

# 环境变量隔离与注入
os.environ["OPENAI_API_KEY"] = os.getenv("GLM_API_KEY", "dummy_key")
os.environ["OPENAI_API_BASE"] = os.getenv("GLM_API_BASE", "")
glm_model = os.getenv("GLM_MODEL_NAME", "glm-4-flash")
os.environ["OPENAI_MODEL_NAME"] = f"openai/{glm_model}"

# ✅ 从我们全新拆分的 harness 导入隔离机制与记忆管理器
from harness import MemoryManager
from agent_test0.crew import TravelWorkflow

# 初始化 Redis 物理连接
redis_client = redis.Redis(host='localhost', port=6373, db=0, decode_responses=True)

zhipu_llm = LLM(
    model=os.getenv("GLM_MODEL_NAME") or "glm-4-flash",
    base_url=os.getenv("GLM_API_BASE") or "",
    api_key=os.getenv("GLM_API_KEY") or "",
)

app: FastAPI = FastAPI(title="智能旅游规划 Agent API", description="基于多智能体的旅游规划服务")

class ChatRequest(BaseModel):
    user_id: str = "yyy"
    session_id: str = ""
    message: str

def rewrite_query_with_context(memory: MemoryManager, current_message: str) -> str:
    if not memory.get_chat_history():
        return current_message

    # 获取 CC 层级拼装好的 Context (包含长期特征、短期总结、历史对话)
    context_payload = memory.get_global_context_prompt(current_message)
    
    rewrite_prompt = f"""
你是一个极其严格的“指代消解”与“意图补全”组件。
你的唯一任务是：根据【用户长期偏好画像】、【当前行程核心约束】和【近期对话上下文】，将用户的【当前最新指令】重写为一句独立、明确、包含所有关键实体（如地点、人数、约束）的完整句子。

【绝对规则】：
1. 必须以用户的口吻进行重写！绝不能使用 AI (assistant) 的视角或复述 AI 的自我介绍。
2. 如果用户的指令中包含“那里”、“这个”、“明天”等代词，请从上下文替换为具体名词。
3. 即使重写，也【必须】保留用户最新的动作意图。如果用户说“你好”，重写结果就是“你好”。
4. 只输出最终句子，没有任何解释。

{context_payload}

【最终重写结果】：
"""
    try:
        response = zhipu_llm.call([{"role": "user", "content": rewrite_prompt}])
        return response.strip()
    except Exception as e:
        print(f"⚠️ 重写失败: {e}")
        return current_message


# --- 流式 API 核心交互接口 ---
@app.post("/api/chat_stream")
async def chat_endpoint_stream(request: ChatRequest):
    actual_user_id: str = request.user_id
    actual_session_id = request.session_id or f"sess_{uuid.uuid4().hex[:6]}"
    
    # 实例化当前会话的记忆管理器
    memory = MemoryManager(actual_session_id, actual_user_id, redis_client)
    
    # 🔄 记忆转换链路 1: 启动前，快速进行 历史对话(Episodic) -> 短期约束(Working Memory) 的同步转化
    memory.convert_episodic_to_working(zhipu_llm)
    
    # 结合分层上下文进行高精度 Query 重写
    contextual_message = rewrite_query_with_context(memory, request.message)
    
    # 将用户最新消息追加进 Episodic (桶3)
    memory.add_message("user", request.message)
    
    # 简易多路意图分发
    is_travel_related = any(kw in contextual_message for kw in ["旅游", "规划", "行程", "攻略", "玩", "去", "景点"])
    intent_name = "travel" if is_travel_related else "default_chat"
    
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()

    def run_crewai_task():
        try:
            if intent_name == "travel":
                # 再次执行一次提纯转换，捕获当下输入暴露的短期指标
                memory.convert_episodic_to_working(zhipu_llm)

                def workflow_status_listener(status_text: str):
                    asyncio.run_coroutine_threadsafe(
                        queue.put({"type": "status", "content": status_text}), loop
                    )

                travel_flow = TravelWorkflow(status_callback=workflow_status_listener)
                
                # 透传重写 Query，并将 Working Memory 桶中的约束 JSON 包转化为 Focus 的约束注入
                travel_flow.state.message = contextual_message 
                travel_flow.state.focus = json.dumps(memory.get_short_term_summary(), ensure_ascii=False)
                travel_flow.state.user_id = actual_user_id
                travel_flow.state.session_id = actual_session_id
                
                travel_flow.kickoff()
                final_result = travel_flow.state.final_report
                
                # 记录 AI 的结案报告
                memory.add_message("assistant", final_result)
                
                # 🔄 记忆转换链路 2: 会话达成重大进展时，快速进行 Working/Episodic -> Semantic (长期记忆) 转化持久化
                memory.convert_to_semantic(zhipu_llm)
                
                asyncio.run_coroutine_threadsafe(
                    queue.put({"type": "finish", "content": final_result}), loop
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
                
                # 闲聊同样进行画像特征捕获转化
                memory.convert_to_semantic(zhipu_llm)
                
                asyncio.run_coroutine_threadsafe(
                    queue.put({"type": "finish", "content": reply_text}), loop
                )
        except Exception as e:
            asyncio.run_coroutine_threadsafe(
                queue.put({"type": "error", "content": f"系统运行出错: {str(e)}"}), loop
            )

    async def event_generator():
        task = loop.run_in_executor(None, run_crewai_task)
        while True:
            msg = await queue.get()
            yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
            if msg["type"] in ["finish", "error"]:
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")

def run():
    print("====================================")
    print("🌍 智能旅游规划系统 - Agent API 服务已启动 ")
    print("====================================")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    run()
