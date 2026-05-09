#!/usr/bin/env python
import asyncio
import hashlib
from   .harness import MemoryManager
from fastapi.applications import FastAPI
import json
import os
import redis
import uuid
from crewai import LLM
# from dotenv import load_dotenv
from pydantic import BaseModel
from semantic_router import Route
from semantic_router.encoders import OllamaEncoder
from semantic_router.routers import SemanticRouter
from starlette.responses import StreamingResponse
os.environ["OPENAI_API_KEY"] = os.getenv("GLM_API_KEY", "dummy_key")
os.environ["OPENAI_API_BASE"] = os.getenv("GLM_API_BASE", "")
glm_model = os.getenv("GLM_MODEL_NAME", "glm-4-flash")
os.environ["OPENAI_MODEL_NAME"] = f"openai/{glm_model}"
from agent_test0.crew import TravelWorkflow
import uvicorn


# --- 初始化 Redis 连接 ---
# 建议通过环境变量配置：os.getenv("REDIS_HOST", "localhost")
redis_client = redis.Redis(host='localhost', port=6373, db=0, decode_responses=True)
MAX_HISTORY_TURNS = 10  # 限制存储最近 10 轮对话，防止 Token 爆炸
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
routes_path = os.path.join(base_dir, "knowledge", "routes.json")

zhipu_llm = LLM(
    model=os.getenv("GLM_MODEL_NAME") or "glm-4-flash",
    base_url=os.getenv("GLM_API_BASE") or "",
    api_key=os.getenv("GLM_API_KEY") or "",
)
# --- 初始化 FastAPI 应用 ---
app: FastAPI = FastAPI(title="智能旅游规划 Agent API", description="基于多智能体的旅游规划服务")
class ChatRequest(BaseModel):
    user_id: str = "yyy"
    session_id: str = ""
    message: str
    

def rewrite_query_with_context(memory: MemoryManager, current_message: str) -> str:
    # 如果完全没有历史，直接返回原话
    if not memory.get_chat_history():
        return current_message

    # 获取分层组装好的 Context
    context_payload = memory.get_global_context_prompt(current_message)
    
    rewrite_prompt = f"""
你是一个极其严格的“指代消解”与“意图补全”组件。
你的唯一任务是：根据【当前行程核心约束】和【近期对话上下文】，将用户的【当前最新指令】重写为一句独立、明确、包含所有关键实体（如地点、人数、约束）的完整句子。

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

def build_router():
    print("正在从 JSON 加载语义路由例句库...")
    encoder = OllamaEncoder(name="nomic-embed-text")
    
    # 读取 JSON 文件
    with open(routes_path, "r", encoding="utf-8") as f:
        routes_data = json.load(fp=f)
    
    # 动态生成 Route 对象列表
    routes = []
    for name, utterances in routes_data.items():
        routes.append(Route(name=name, utterances=utterances))
    
    return SemanticRouter(encoder=encoder, routes=routes,auto_sync="local")

# 全局初始化路由器
fast_intent_router = build_router()
print("✅ Semantic Router 初始化完成！")

# --- 2. 核心交互接口 ---
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    # 1. 会话分配逻辑 (复用我们之前的逻辑)
    actual_user_id = request.user_id
    actual_session_id = request.session_id or f"sess_{uuid.uuid4().hex[:6]}"
    
    # 2.  核心修复：上下文重写
    # 我们不拿 request.message 去路由，而是拿重写后的 contextual_message 去路由！
    contextual_message = rewrite_query_with_context(actual_session_id, request.message)

    # 3. 将用户的原话加入短期记忆历史
    if actual_session_id not in SESSION_HISTORY:
        SESSION_HISTORY[actual_session_id] = []
    SESSION_HISTORY[actual_session_id].append({"role": "user", "content": request.message})

    # 4. 路由判断 (使用重写后的话)
    route_choice = fast_intent_router(contextual_message)
    intent_name = route_choice.name

    response_payload = {
        "user_id": actual_user_id,
        "session_id": actual_session_id,
        "intent": intent_name
    }

    reply_text = ""

    # 5. 分流执行
    if intent_name == "travel":
        print("🚀 触发复杂工作流...")
        travel_flow = TravelWorkflow()
        # 注意：这里传给 Flow 的仍然是重写后、带有明确地点的话
        travel_flow.state.message = contextual_message 
        travel_flow.state.user_id = actual_user_id
        travel_flow.state.session_id = actual_session_id
        
        travel_flow.kickoff()
        reply_text = travel_flow.state.final_report
    else:
        # 闲聊处理...
        reply_text = "收到你的消息啦！" 

    # 6. 将 AI 的回复存入短期记忆，完成闭环
    SESSION_HISTORY[actual_session_id].append({"role": "assistant", "content": reply_text})
    # 维持历史记录长度，避免爆内存
    if len(SESSION_HISTORY[actual_session_id]) > MAX_HISTORY_TURNS * 2:
        SESSION_HISTORY[actual_session_id] = SESSION_HISTORY[actual_session_id][-MAX_HISTORY_TURNS*2:]

    return {**response_payload, "reply": reply_text}


@app.post("/api/chat_stream")
async def chat_endpoint_stream(request: ChatRequest):
    actual_user_id: str = request.user_id
    actual_session_id = request.session_id or f"sess_{uuid.uuid4().hex[:6]}"
    
    # 实例化当前会话的记忆管家
    memory = MemoryManager(actual_session_id)
    
    # 1. 结合 CC 层级记忆进行重写
    contextual_message = rewrite_query_with_context(memory, request.message)
    
    # 2. 将用户原话存入 桶3 (ChatHistory)
    memory.add_message("user", request.message)
    
    # 3. 路由判断
    route_choice = fast_intent_router(contextual_message)
    intent_name = route_choice.name if route_choice is not None else "default_chat"
    
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()

    def run_crewai_task():
        try:
            if intent_name == "travel":
                # --- 🧠 动态提取短期摘要 (Short-Term Memory) ---
                # 在触发昂贵的 CrewAI 之前，用极快的速度提取当前对话的关键约束
                extract_prompt = f"根据用户需求：'{contextual_message}'，提取JSON格式的核心约束，如地名、天数、偏好等。仅输出JSON格式，例如：{{\"destination\": \"杭州\", \"preferences\": \"不吃辣\"}}。如果没有明确约束则输出 {{}}"
                try:
                    summary_resp = zhipu_llm.call([{"role": "user", "content": extract_prompt}])
                    import re
                    json_match = re.search(r'\{.*\}', summary_resp, re.DOTALL)
                    if json_match:
                        new_constraints = json.loads(json_match.group(0))
                        # 更新到 桶5
                        memory.update_short_term_summary(new_constraints)
                except Exception as e:
                    print(f"短期约束提取失败: {e}")

                def workflow_status_listener(status_text: str):
                    asyncio.run_coroutine_threadsafe(
                        queue.put({"type": "status", "content": status_text}), loop
                    )

                travel_flow = TravelWorkflow(status_callback=workflow_status_listener)
                
                # ✅ 这里极其关键：把提纯后的 context 而不是简单的 message 传给 Flow
                # 这样 Planner 就能直接看到 {"destination": "上海", "preferences": "不爬山"}
                travel_flow.state.message = contextual_message 
                travel_flow.state.focus = json.dumps(memory.get_short_term_summary(), ensure_ascii=False)
                travel_flow.state.user_id = actual_user_id
                travel_flow.state.session_id = actual_session_id
                
                travel_flow.kickoff()
                final_result = travel_flow.state.final_report
                
                # 记录 AI 回复到 桶3
                memory.add_message("assistant", final_result)
                
                asyncio.run_coroutine_threadsafe(
                    queue.put({"type": "finish", "content": final_result}), loop
                )
            else:
                # ---------- 闲聊分支 ----------
                # 直接获取完整的全局 Context 给 LLM 进行日常对话
                context_payload = memory.get_global_context_prompt(request.message)
                system_prompt = "你是一个亲切的旅游管家。请根据以下上下文自然地回答用户。"
                
                response = zhipu_llm.call([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context_payload}
                ])
                reply_text = response.strip()
                
                memory.add_message("assistant", reply_text)
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
    print("🌍 欢迎使用智能旅游规划系统 ")
    print("====================================")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    print("====================================")
    print("测试读取到的模型地址：", os.getenv("GLM_API_BASE"))
    print("测试读取到的 API Key:", os.getenv("GLM_API_KEY"))
    print("====================================")
    run()
