#!/usr/bin/env python
"""
本地 Agent 测试脚本 - 用于隔离测试 CrewAI Flow
剥离飞书和 FastAPI，直接测试 Agent 是否正常工作
"""
import os
import sys
import logging
import json

# =========================================
# 【核弹级打印】无视所有框架劫持，直接向 OS 文件描述符 1 写入
# =========================================
def hard_print(text):
    print(text)
    # """无视任何劫持，直接把字刻在终端屏幕上"""
    # try:
    #     os.write(1, (str(text) + "\n").encode('utf-8'))
    # except Exception:
    #     pass

# 加载 .env 文件
from dotenv import load_dotenv
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

# 压制 CrewAI 事件总线的编码警告
logging.getLogger("CrewAIEventsBus").setLevel(logging.ERROR)

# =========================================
# 环境变量配置
# =========================================
os.environ["OPENAI_API_KEY"] = os.getenv("GLM_API_KEY", "dummy_key")
os.environ["OPENAI_API_BASE"] = os.getenv("GLM_API_BASE", "")
glm_model = os.getenv("GLM_MODEL_NAME", "glm-4-flash")
os.environ["OPENAI_MODEL_NAME"] = f"openai/{glm_model}"
os.environ["LITELLM_LOG"] = "ERROR"
os.environ["SUPPRESS_LITELLM_LOGS"] = "True"

# =========================================
# 导入项目模块
# =========================================
from agent_test0.memory import MemoryManager, get_redis_or_fallback
from agent_test0.workflow import TravelWorkflow, TravelState
from crewai import LLM

# =========================================
# 测试配置
# =========================================
TEST_CASES = [
    {
        "name": "简单天气查询",
        "user_id": "test_user_001",
        "session_id": "sess_test_001",
        "message": "杭州天气",
        "expected_steps": 1
    },
    {
        "name": "简单自我介绍",
        "user_id": "test_user_002",
        "session_id": "sess_test_002",
        "message": "介绍一下你自己",
        "expected_steps": 1
    },
    {
        "name": "复杂旅行规划",
        "user_id": "test_user_003",
        "session_id": "sess_test_003",
        "message": "我想去杭州玩3天",
        "expected_steps": 3
    }
]

# =========================================
# Redis 连接
# =========================================
redis_client, is_redis_fallback = get_redis_or_fallback()
hard_print(f"📦 [Redis] 连接状态: {'内存模式(回退)' if is_redis_fallback else 'Redis模式'}")

# =========================================
# 测试函数
# =========================================
def run_test(test_case):
    """运行单个测试用例"""
    name = test_case["name"]
    user_id = test_case["user_id"]
    session_id = test_case["session_id"]
    message = test_case["message"]
    expected_steps = test_case["expected_steps"]

    hard_print("\n" + "="*60)
    hard_print(f"🧪 测试: {name}")
    hard_print(f"   用户: {user_id}")
    hard_print(f"   会话: {session_id}")
    hard_print(f"   消息: {message}")
    hard_print("="*60)

    # 清理 Redis 中的旧状态
    flow_state_key = f"session:{session_id}:flow_state"
    if not is_redis_fallback:
        redis_client.delete(flow_state_key)
        hard_print(f"🗑️  [Redis] 清理旧状态: {flow_state_key}")

    # 创建 MemoryManager
    memory = MemoryManager(session_id, user_id, redis_client, is_redis_fallback)
    memory.add_message("user", message)

    # 创建 LLM 实例
    zhipu_llm = LLM(
        model=os.getenv("GLM_MODEL_NAME") or "glm-4-flash",
        base_url=os.getenv("GLM_API_BASE") or "",
        api_key=os.getenv("GLM_API_KEY") or "",
    )

    # 转换 episodic 记忆到 working memory
    try:
        memory.convert_episodic_to_working(zhipu_llm)
        hard_print("🧠 [记忆] Episodic → Working 转换完成")
    except Exception as e:
        hard_print(f"⚠️  [记忆] Episodic → Working 转换失败: {e}")

    # 创建 Workflow 并执行
    try:
        # 回调函数
        status_messages = []
        content_messages = []

        def workflow_status_listener(status_text: str):
            status_messages.append(status_text)
            hard_print(f">Status> {status_text}")

        def workflow_content_listener(content: str, content_type: str):
            content_messages.append((content_type, content))
            hard_print(f">Content> [{content_type}] {content[:100]}...")

        # 创建 Workflow
        travel_flow = TravelWorkflow(
            status_callback=workflow_status_listener,
            content_callback=workflow_content_listener
        )

        # 设置状态
        travel_flow.state.message = message
        travel_flow.state.focus = memory.get_global_context_prompt(message)
        travel_flow.state.user_id = user_id
        travel_flow.state.session_id = session_id

        hard_print("\n🚀 [Workflow] 开始执行...")

        # 执行 - 传入初始状态作为 inputs
        inputs = {
            "message": message,
            "focus": memory.get_global_context_prompt(message),
            "user_id": user_id,
            "session_id": session_id
        }
        travel_flow.kickoff(inputs=inputs)

        hard_print("\n✅ [Workflow] 执行完成")

        # Flow 的状态通过 flow.state 访问，而不是返回值
        result_state = travel_flow.state

        hard_print(f"\n📊 [结果] 最终报告:")
        hard_print("-" * 60)
        hard_print(result_state.final_report)
        hard_print("-" * 60)

        # 验证结果
        hard_print(f"\n🔍 [验证]")
        hard_print(f"  - 计划步骤数: {len(getattr(result_state, 'steps', []))}")
        hard_print(f"  - 是否复杂任务: {result_state.is_complex}")
        hard_print(f"  - 简单回答: {result_state.simple_answer[:50] if result_state.simple_answer else '无'}")

        return True

    except Exception as e:
        hard_print(f"\n💥 [异常] {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# =========================================
# 主函数
# =========================================
def main():
    hard_print("\n" + "="*60)
    hard_print("🧪 Agent 本地测试脚本")
    hard_print("="*60)

    # 选择测试用例
    if len(sys.argv) > 1:
        # 命令行指定测试索引
        test_index = int(sys.argv[1])
        if 0 <= test_index < len(TEST_CASES):
            results = [run_test(TEST_CASES[test_index])]
        else:
            hard_print(f"❌ 测试索引超出范围: 0-{len(TEST_CASES)-1}")
            return
    else:
        # 运行所有测试
        results = []
        for test_case in TEST_CASES:
            result = run_test(test_case)
            results.append(result)
            # 休息一下，避免太快
            import time
            time.sleep(1)

    # 总结
    hard_print("\n" + "="*60)
    hard_print("📊 测试总结")
    hard_print("="*60)
    passed = sum(results)
    total = len(results)
    hard_print(f"通过: {passed}/{total}")

    if passed == total:
        hard_print("🎉 所有测试通过!")
    else:
        hard_print(f"❌ {total - passed} 个测试失败")

if __name__ == "__main__":
    main()
