#!/usr/bin/env python
"""
旅行规划 Agent 纯同步本地调试器 (交互循环版)
- 不连接前端 Streamlit，不启动 FastAPI / Uvicorn 服务器
- 没有任何子线程，全部代码在主线程线性顺序执行
- 支持连续对话、长期记忆积累、优雅退出与错误隔离
- 包含意图路由机制（旅游规划 vs 闲聊），与 main.py 保持一致
- 保留 CrewAI verbose=True 的完整调试输出
"""

import time
import sys
import os
from pathlib import Path
import logging
import traceback
import json
import uuid

# ==================== 1. 自动化环境变量与路径修复 ====================
# 把 src/ 加入 sys.path，让 `from agent_test0.xxx import ...` 能解析。
#   debug_cli.py 位于  <repo>/src/agent_test0/api/debug_cli.py
#   parents[2]      →  <repo>/src/
current_dir = Path(__file__).resolve().parent
src_path = str(current_dir.parents[1])
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from dotenv import load_dotenv
load_dotenv()

# ==================== 2. 修复 stdout 编码问题 ====================
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# 配置标准日志输出（Token 过滤 + 压制底层库日志）
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

logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# ==================== 3. 意图路由机制（与 main.py 完全一致） ====================
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

# def get_router_with_retry(max_retries=5, delay=2):
#     """带重试机制的路由初始化"""
#     encoder = OllamaEncoder(
#         name="nomic-embed-text",
#         base_url="http://localhost:11434"
#     )

#     for i in range(max_retries):
#         try:
#             print(f"[路由] 正在尝试连接模型 (第 {i+1}/{max_retries} 次)...")
#             encoder(["test"])
#             print("[路由] 模型连接成功！")

#             # 创建路由
#             all_utterances = ROUTES.get("travel", []) + ROUTES.get("chitchat", [])
#             print(f"[路由] 共加载 {len(all_utterances)} 条 utterances")

#             # 使用 SemanticRouter.fit() 方法，传入 utterances 和对应的 route_names
#             router = SemanticRouter(encoder=encoder)
#             # 创建 utterance 到 route_name 的映射
#             travel_utts = ROUTES.get("travel", [])
#             chat_utts = ROUTES.get("chitchat", [])
#             router.fit(
#                 X=travel_utts + chat_utts,
#                 y=["travel"] * len(travel_utts) + ["default_chat"] * len(chat_utts)
#             )
#             print("[路由] 索引建立完成！")

#             return router
#         except Exception as e:
#             print(f"[路由] 第 {i+1} 次失败: {e}，将在 {delay} 秒后重试...")
#             time.sleep(delay)

#     print("[路由] 严重错误：Ollama 模型初始化重试失败！")
#     return None

# 替换你原本的路由初始化逻辑
print("[路由] 正在初始化语义路由...")
# intent_router = get_router_with_retry()
intent_router = None
if intent_router is None:
    print("[路由] 路由初始化失败，将使用关键词 fallback 模式")
    # 降级方案：使用简单的关键词匹配
    def classify_intent(message: str) -> str:
        travel_keywords = ["旅游", "玩", "去", "旅行", "景点", "攻略", "行程", "出差", "度假", "游玩"]
        if any(kw in message for kw in travel_keywords):
            return "travel"
        return "travel"
else:
    def classify_intent(message: str) -> str:
        """意图分类：返回 'travel' 或 'default_chat'"""
        try:
            result = intent_router(message)
            if result and result.name:
                return result.name
        except Exception as e:
            print(f"[警告] 语义路由异常: {e}，降级为闲聊模式")
        return "travel"

# ==================== 4. 引入核心模块 ====================
# 环境变量隔离（与 main.py 一致）
os.environ["OPENAI_API_KEY"] = os.getenv("GLM_API_KEY", "dummy_key")
os.environ["OPENAI_API_BASE"] = os.getenv("GLM_API_BASE", "")
glm_model = os.getenv("GLM_MODEL_NAME", "glm-4-flash")
os.environ["OPENAI_MODEL_NAME"] = f"openai/{glm_model}"
os.environ["LITELLM_LOG"] = "ERROR"
os.environ["SUPPRESS_LITELLM_LOGS"] = "True"

from agent_test0.crew import TravelWorkflow
from agent_test0.harness import MemoryManager, get_redis_or_fallback
from crewai import LLM


def print_divider(title):
    """辅助排版函数"""
    print("\n" + "=" * 60)
    print(f" {title} ".center(60, "="))
    print("=" * 60 + "\n")


def run_standalone_debug():
    print_divider("欢迎进入旅行智能体【连续交互】调试控制台")

    # ==================== 5. 核心资源全局初始化 ====================
    print("[系统初始化] 正在建立 Redis 连接与实例化 LLM 模型...")
    redis_client, is_redis_fallback = get_redis_or_fallback()

    zhipu_llm = LLM(
        model=os.getenv("GLM_MODEL_NAME") or "glm-4-flash",
        base_url=os.getenv("GLM_API_BASE") or "",
        api_key=os.getenv("GLM_API_KEY") or "",
    )

    # ==================== 6. 固化用户身份与会话 ====================
    default_user_id = "debug_master_001"
    input_user_id = input(f"请输入测试用户ID (直接回车保持长期身份 '{default_user_id}'): ").strip()

    actual_user_id = input_user_id if input_user_id else default_user_id
    # actual_session_id = f"session_{actual_user_id}_" + str(uuid.uuid4())[:4]
    actual_session_id = f"session_{actual_user_id}_" 

    memory = MemoryManager(actual_session_id, actual_user_id, redis_client, is_redis_fallback)

    print("\n[载入参数检查]")
    print(f" ├─ 用户标识 (User ID): {actual_user_id} (长期记忆累积目标)")
    print(f" ├─ 会话标识 (Session ID): {actual_session_id} (当前连续对话上下文)")
    print(f" └─ Redis 模式: {'内存回退' if is_redis_fallback else 'Redis 连接'}")

    def local_status_callback(status_text):
        print(f"[Workflow]: {status_text}")

    print_divider("系统就绪，随时可以开始规划 (输入 'q' 或 'exit' 退出)")

    # ==================== 7. REPL 连续交互循环 ====================
    turn_count = 0
    while True:
        try:
            # 7.1 优雅退出判断
            user_input = input(f"\n[{actual_user_id} @ Round {turn_count+1}] 请输入需求: ").strip()
            if user_input.lower() in ['q', 'quit', 'exit', '退出']:
                print("\n[系统] 收到退出指令，正在保存记忆并安全关闭...")
                break

            if not user_input:
                continue

            turn_count += 1
            print_divider(f"开始执行第 {turn_count} 轮规划")

            # 7.2 记忆流转 A：记录当前对话 -> 提取短期约束
            memory.add_message("user", user_input)
            memory.convert_episodic_to_working(zhipu_llm)
            print("[记忆系统] 正在结合过往对话理解您的最新意图...")

            # 7.3 意图路由
            intent_name = classify_intent(user_input)
            print(f"[路由决策] 意图识别为: {intent_name}")
            print("=" * 60)

            # 7.4 根据意图分发处理
            if intent_name == "travel":
                # ========== 旅游规划模式 ==========
                workflow = TravelWorkflow(status_callback=local_status_callback)
                workflow.state.message = user_input
                workflow.state.user_id = actual_user_id
                workflow.state.session_id = actual_session_id
                workflow.state.focus = memory.get_global_context_prompt(user_input)

                workflow.kickoff()

                final_output = workflow.state.final_report
                adjust_count = getattr(workflow, 'current_adjust_count', 0)
                print("\n[📝 本轮最终输出]:")
                print("-" * 60)
                print(final_output)
                print("-" * 60)
                print(f"[状态机指标] 内部质检打回重改次数: {adjust_count} 次")

                # 记忆流转 B：提炼长期偏好
                print("\n[记忆系统] 正在将本轮关键信息提炼为长期偏好...")
                memory.convert_to_semantic(zhipu_llm)

            else:
                # ========== 闲聊模式 ==========
                print("\n[闲聊模式] 正在为您生成自然回复...")
                context_payload = memory.get_global_context_prompt(user_input)
                system_prompt = "你是一个亲切的旅游管家。请根据以下上下文自然地回答用户。"
                response = zhipu_llm.call([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context_payload}
                ])
                reply_text = response.strip()
                print("\n[📝 本轮最终输出]:")
                print("-" * 60)
                print(reply_text)
                print("-" * 60)

                # 闲聊也记录到记忆中
                memory.add_message("assistant", reply_text)
                memory.convert_to_semantic(zhipu_llm)

            # 显示实时更新的用户画像
            profile = memory.get_user_profile()
            print(f"[最新用户长期偏好]: {profile}")

        # 7.5 优雅捕获 Ctrl+C
        except KeyboardInterrupt:
            print("\n\n[系统] 捕获到 Ctrl+C，正在安全退出...")
            break

        # 7.6 错误隔离：本轮报错不影响下一轮对话
        except Exception:
            print_divider("❌ 本轮执行发生致命崩溃")
            traceback.print_exc()
            print("\n[系统] 错误已被隔离，您可以继续输入或输入 'q' 退出。")
            continue

    print_divider("调试控制台已关闭，期待下次使用")


if __name__ == "__main__":
    run_standalone_debug()
