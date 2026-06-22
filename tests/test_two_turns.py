#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
两轮对话测试：模拟"先问一次，第二轮还缺就自己假设"。

轮 1: 用户说"想去重庆" → Planner 应该 needs_user_input=true，问天数/预算/人数。
轮 2: 同 session_id 下用户说"看你安排" → Planner 看到上一轮自己问过 →
     不再追问，做合理假设 → 生成完整旅行计划，开头有"📌 假设：..."。
"""
import os
import sys
import logging

# 强制行缓冲，避免被 Windows 块缓冲卡住
try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

from dotenv import load_dotenv
load_dotenv()

# 屏蔽烦人的 Token 日志
class TokenUsageFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return not ("OpenAI API usage" in msg or "litellm" in msg.lower())

console_handler = logging.StreamHandler(sys.stdout)
console_handler.addFilter(TokenUsageFilter())
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=[console_handler])
for name in ("LiteLLM", "httpx", "openai", "CrewAIEventsBus"):
    logging.getLogger(name).setLevel(logging.WARNING)

# GLM 兼容 OpenAI 协议
os.environ["OPENAI_API_KEY"] = os.getenv("GLM_API_KEY", "dummy_key")
os.environ["OPENAI_API_BASE"] = os.getenv("GLM_API_BASE", "")
os.environ["OPENAI_MODEL_NAME"] = f"openai/{os.getenv('GLM_MODEL_NAME', 'glm-4-flash')}"
os.environ["LITELLM_LOG"] = "ERROR"

from agent_test0.workflow import TravelWorkflow, _redis_client, _is_redis_fallback
from agent_test0.memory import MemoryManager


def hr(title=""):
    print("\n" + "=" * 70)
    if title:
        print(f"  {title}")
        print("=" * 70)


def main():
    user_id = "test_two_turns_user"
    session_id = "sess_two_turns_001"

    # 清理旧状态：保证两轮测试是干净的
    if not _is_redis_fallback:
        for pattern in [
            f"sess:{session_id}:*",
            f"session:{session_id}:*",
            f"cc:session:{session_id}:*",
        ]:
            for key in _redis_client.keys(pattern):
                _redis_client.delete(key)
        # 也清掉用户级的 working memory
        for pattern in [f"cc:user:{user_id}:*", f"user:{user_id}:*"]:
            for key in _redis_client.keys(pattern):
                _redis_client.delete(key)
        print(f"🗑️  已清空 session={session_id} 的 redis 历史")

    # ─── 轮 1 ───
    hr("轮 1：用户说『想去重庆』")
    msg1 = "想去重庆"
    print(f"👤 用户: {msg1}")

    reply1 = TravelWorkflow.run_for_user(
        user_text=msg1,
        user_id=user_id,
        session_id=session_id,
    )
    hr("轮 1 回复")
    print(f"🤖 助手:\n{reply1}")

    # ─── 检查记忆里确实留痕了 ───
    mem = MemoryManager(session_id, user_id, _redis_client, _is_redis_fallback)
    history = mem.get_chat_history()
    hr("轮 1 之后 memory 中的对话历史")
    for m in history[-6:]:
        print(f"  {m['role']}: {m['content'][:120]}")

    # ─── 轮 2 ───
    hr("轮 2：用户说『看你安排』")
    msg2 = "看你安排"
    print(f"👤 用户: {msg2}")

    reply2 = TravelWorkflow.run_for_user(
        user_text=msg2,
        user_id=user_id,
        session_id=session_id,
    )
    hr("轮 2 回复")
    print(f"🤖 助手:\n{reply2}")

    # ─── 自动判定 ───
    hr("自动判定")
    r1_is_question = any(
        kw in reply1
        for kw in ["几天", "天数", "预算", "几位", "几个人", "请问", "?", "？"]
    )
    r2_is_plan = ("假设" in reply2) or ("📌" in reply2) or len(reply2) > 200
    r2_not_question = not (
        reply2.strip().startswith("请问")
        or reply2.strip().endswith("？")
        or reply2.strip().endswith("?")
    )
    r2_no_hangzhou_leak = "杭州" not in reply2

    print(f"  轮 1 是问句? {'✅ 是' if r1_is_question else '❌ 否'}")
    print(f"  轮 2 是计划（包含假设说明 / 长度 > 200）? {'✅ 是' if r2_is_plan else '❌ 否'}")
    print(f"  轮 2 不是问句? {'✅ 是' if r2_not_question else '❌ 否'}")
    print(f"  轮 2 没有泄露杭州? {'✅ 是' if r2_no_hangzhou_leak else '❌ 否'}")

    if r1_is_question and r2_is_plan and r2_not_question and r2_no_hangzhou_leak:
        print("\n🎉 测试通过：先问一次 → 第二轮自己假设并出计划")
    else:
        print("\n⚠️  测试未完全通过，请人工核对上面回复内容")


if __name__ == "__main__":
    main()
