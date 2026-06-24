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

# ==================== 3. 意图路由机制（统一共享模块） ====================
# 之前 debug_cli 重复了一份路由代码，且 intent_router 硬编码 None 导致 fallback
# 永远 return "travel"，闲聊分支永远到不了。现统一走 agent_test0.workflow.intent。
from agent_test0.workflow.intent import classify_intent

# ==================== 4. 引入核心模块 ====================
# 环境变量隔离（与 main.py 一致）
os.environ["OPENAI_API_KEY"] = os.getenv("GLM_API_KEY", "dummy_key")
os.environ["OPENAI_API_BASE"] = os.getenv("GLM_API_BASE", "")
glm_model = os.getenv("GLM_MODEL_NAME", "glm-4-flash")
os.environ["OPENAI_MODEL_NAME"] = f"openai/{glm_model}"
os.environ["LITELLM_LOG"] = "ERROR"
os.environ["SUPPRESS_LITELLM_LOGS"] = "True"

from agent_test0.workflow import TravelWorkflow
from agent_test0.memory import MemoryManager, get_redis_or_fallback
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

            # 7.2 记录当前对话；短期上下文直接使用原始对话，不再做 LLM 蒸馏 summary
            memory.add_message("user", user_input)
            print("[记忆系统] 正在结合过往原始对话理解您的最新意图...")

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
