#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单轮 trace 测试：跑一次 run_for_user，看耗时分布 + 反复调用证据。
不依赖 Redis/飞书，用回退内存模式即可。
"""
import os
import sys
import logging

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

from dotenv import load_dotenv
load_dotenv()

class _F(logging.Filter):
    def filter(self, r):
        m = r.getMessage()
        return not ("OpenAI API usage" in m or "litellm" in m.lower())

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
for n in ("LiteLLM", "httpx", "openai", "CrewAIEventsBus"):
    logging.getLogger(n).setLevel(logging.WARNING)

os.environ["OPENAI_API_KEY"] = os.getenv("GLM_API_KEY", "dummy_key")
os.environ["OPENAI_API_BASE"] = os.getenv("GLM_API_BASE", "")
os.environ["OPENAI_MODEL_NAME"] = f"openai/{os.getenv('GLM_MODEL_NAME', 'glm-4-flash')}"
os.environ["LITELLM_LOG"] = "ERROR"

from agent_test0.workflow import TravelWorkflow


def main():
    user_id = "trace_user"
    msg = sys.argv[1] if len(sys.argv) > 1 else "杭州天气"
    print(f"\n{'#'*70}\n# 测试消息: {msg}\n{'#'*70}")

    reply = TravelWorkflow.run_for_user(
        user_text=msg,
        user_id=user_id,
        session_id=f"trace_{user_id}",
    )
    print(f"\n{'='*70}\n[最终回复] (长度 {len(reply)})\n{reply}\n{'='*70}")


if __name__ == "__main__":
    main()
