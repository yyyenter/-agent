# -*- coding: utf-8 -*-
"""
飞书长连接机器人 — 只负责飞书 WebSocket 连接和消息收发。
所有 LLM 提示词、步骤规划、报告生成等业务逻辑在 crew.py 中。
"""

# 强制 stdout/stderr 行缓冲并使用 utf-8——这一步必须在任何 import 之前执行，
# 否则当被重定向到文件/pipe 时，Windows 默认的块缓冲会让日志几十秒看不到一行。
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass

import lark_oapi as lark
from lark_oapi.api.im.v1 import *
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import os

from agent_test0.crew import TravelWorkflow

load_dotenv()

# 后台执行 Agent 的线程池：避免阻塞 WebSocket 回调线程导致 ping_timeout
_agent_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="agent")

# 防重入锁：飞书 WebSocket 在 ping_timeout 重连后会重发同一条消息事件，
# 用 message_id 做幂等，避免同一条用户消息被处理多次。
_processed_message_ids: set[str] = set()
_processed_lock = threading.Lock()
_PROCESSED_MAX_SIZE = 1000  # 防止集合无限增长


# ══════════════════════════════════════════════════════════════════════════════
# 飞书事件处理
# ══════════════════════════════════════════════════════════════════════════════

def do_p2_im_message_receive_v1(data: lark.im.v1.P2ImMessageReceiveV1) -> None:
    """
    收到飞书消息事件时触发（长连接推送到这里）。

    这个函数运行在飞书 WebSocket 接收线程上，必须立刻返回，否则会卡 ping
    心跳并触发 ping_timeout，导致连接断开 + 事件重投递（同一条消息被处理多次）。

    我们的策略：在此函数里只发"正在处理"提示，再把真正的 Agent 调用扔到
    后台线程池，由后台线程负责发最终回复。
    """
    start_time = time.time()
    print("\n" + "=" * 60)
    print("[飞书] 收到消息事件")

    event = data.event
    message = event.message
    sender = event.sender

    # 幂等：同一 message_id 在重连/重投递时只处理一次
    msg_id = getattr(message, "message_id", None)
    if msg_id:
        with _processed_lock:
            if msg_id in _processed_message_ids:
                print(f"  [飞书] 跳过重复消息: message_id={msg_id}")
                return
            # 控制集合大小
            if len(_processed_message_ids) >= _PROCESSED_MAX_SIZE:
                _processed_message_ids.clear()
            _processed_message_ids.add(msg_id)

    content_str = message.content
    try:
        content_json = json.loads(content_str)
        user_text = content_json.get("text", "")
    except json.JSONDecodeError:
        user_text = content_str

    sender_open_id = sender.sender_id.open_id
    chat_id = message.chat_id

    print(f"  发送者: {sender_open_id}")
    print(f"  群聊: {chat_id or '(私聊)'}")
    print(f"  内容: {user_text}")
    print(f"  收到时间: {time.strftime('%H:%M:%S', time.localtime(start_time))}")

    # 1) 立刻发"正在处理"提示，避免用户等待无反馈
    print("  [飞书] 发送'正在处理'提示...")
    _send_reply_immediate(sender_open_id, "正在处理您的请求，请稍候...")

    # 2) 把耗时的 Agent 调用 + 最终回复扔到后台线程，立即放回 ws 接收线程
    _agent_executor.submit(
        _process_message_async, user_text, sender_open_id, chat_id, start_time
    )
    print("  [飞书] 已交给后台线程处理，立即返回以维持 ping 心跳")


def _process_message_async(user_text: str, sender_open_id: str, chat_id: str, start_time: float) -> None:
    """后台线程：跑 Agent + 发送最终回复。任何异常都吞下并发用户友好的错误消息。"""
    try:
        reply_text = _call_agent(user_text, sender_open_id, chat_id)
    except Exception as e:
        print(f"  [Agent] 后台执行异常: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        reply_text = f"抱歉，处理您的请求时出现了错误：{e}"

    elapsed = time.time() - start_time
    print(f"  总耗时: {elapsed:.2f} 秒")

    try:
        _send_reply(sender_open_id, reply_text)
        print(f"[飞书] 已回复: {reply_text[:80]}...")
    except Exception as e:
        print(f"  [飞书] 最终回复发送失败: {e}")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# 事件注册
# ══════════════════════════════════════════════════════════════════════════════

def _silent_handler(_data) -> None:
    """空白处理器：吃掉飞书推送但本机器人不关心的事件，避免框架日志报错。"""
    return None


event_handler = (
    lark.EventDispatcherHandler.builder("", "")
    .register_p2_im_message_receive_v1(do_p2_im_message_receive_v1)
    # 飞书还会推送"消息已读回执"和"用户进入与机器人的私聊"事件，
    # 我们不需要它们，但必须注册占位处理器，否则 lark-oapi 会刷
    # "processor not found" ERROR 日志。
    .register_p2_im_message_message_read_v1(_silent_handler)
    .register_p2_im_chat_access_event_bot_p2p_chat_entered_v1(_silent_handler)
    .build()
)


# ══════════════════════════════════════════════════════════════════════════════
# Agent 调用 — 纯连接层，不包含任何 LLM 提示词逻辑
# ══════════════════════════════════════════════════════════════════════════════

def _call_agent(user_text: str, user_id: str, chat_id: str = None) -> str:
    """
    调用 CrewAI TravelWorkflow 处理飞书用户消息。
    本函数只负责连接层适配：把 user_text + user_id 透传给 TravelWorkflow.run_for_user。
    memory / redis / prompt / final_report 生成全部在 crew 中。
    """
    print("\n[Agent] 调用 CrewAI TravelWorkflow...")

    session_id = f"feishu_{user_id}_{abs(hash(user_text)) % 1000000:06d}"

    final_report = TravelWorkflow.run_for_user(
        user_text=user_text,
        user_id=user_id,
        session_id=session_id,
    )

    print(f"  [Agent] 回复长度: {len(final_report)}")
    print(f"  [Agent] 回复预览: {final_report[:100]}")
    return final_report


# ══════════════════════════════════════════════════════════════════════════════
# 消息发送
# ══════════════════════════════════════════════════════════════════════════════

def _send_reply_immediate(open_id: str, text: str) -> None:
    """立即发送一条消息，用于"正在处理"等提示"""
    try:
        client = (
            lark.Client.builder()
            .app_id(os.getenv("FEISHU_APP_ID"))
            .app_secret(os.getenv("FEISHU_APP_SECRET"))
            .log_level(lark.LogLevel.ERROR)
            .build()
        )
        msg_content = json.dumps({"text": text}, ensure_ascii=False)
        request = (
            CreateMessageRequest.builder()
            .receive_id_type("open_id")
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id(open_id)
                .msg_type("text")
                .content(msg_content)
                .build()
            )
            .build()
        )
        response = client.im.v1.message.create(request)
        if not response.success():
            print(f"  [飞书] 提示消息发送失败: code={response.code}, msg={response.msg}")
    except Exception as e:
        print(f"  [飞书] 提示消息发送异常: {e}")


def _send_reply(open_id: str, text: str):
    """通过飞书 API 发送回复消息"""
    client = (
        lark.Client.builder()
        .app_id(os.getenv("FEISHU_APP_ID"))
        .app_secret(os.getenv("FEISHU_APP_SECRET"))
        .log_level(lark.LogLevel.WARNING)
        .build()
    )

    msg_content = json.dumps({"text": text}, ensure_ascii=False)

    print(f"  [飞书] 消息长度: {len(text)}")
    print(f"  [飞书] 消息预览: {text[:100]}...")

    request = (
        CreateMessageRequest.builder()
        .receive_id_type("open_id")
        .request_body(
            CreateMessageRequestBody.builder()
            .receive_id(open_id)
            .msg_type("text")
            .content(msg_content)
            .build()
        )
        .build()
    )

    response = client.im.v1.message.create(request)

    if not response.success():
        print(f"  [飞书] 发送失败: code={response.code}, msg={response.msg}")
    else:
        print(f"  [飞书] 发送成功: msg_id={response.data.message_id}")


# ══════════════════════════════════════════════════════════════════════════════
# WebSocket 长连接启动
# ══════════════════════════════════════════════════════════════════════════════

def start():
    """启动飞书长连接机器人（阻塞）"""
    print("=" * 60)
    print("飞书长连接机器人启动")
    print("=" * 60)
    print(f"  App ID: {os.getenv('FEISHU_APP_ID')}")
    print(f"  模式:  WebSocket 长连接")
    print(f"  Agent: CrewAI TravelWorkflow")
    print("=" * 60)
    print()

    cli = lark.ws.Client(
        os.getenv("FEISHU_APP_ID"),
        os.getenv("FEISHU_APP_SECRET"),
        event_handler=event_handler,
        log_level=lark.LogLevel.WARNING,
    )
    cli.start()  # 阻塞


if __name__ == "__main__":
    start()
