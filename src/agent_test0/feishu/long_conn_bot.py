# -*- coding: utf-8 -*-
"""
飞书长连接机器人 — 按官方示例实现
参考文档:
  https://open.feishu.cn/document/server-side-sdk/python--sdk/invoke-server-api
  https://open.feishu.cn/document/server-side-sdk/python--sdk/handle-events
"""

import lark_oapi as lark
from lark_oapi.api.im.v1 import *
import json
import asyncio
from dotenv import load_dotenv
import os

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  🔗🔗🔗  【Agent 连接点 1】导入你的 CrewAI Agent 相关模块              ║
# ╚══════════════════════════════════════════════════════════════════════════╝
from agent_test0.crew import TravelWorkflow        # CrewAI Flow（多智能体调度核心）
from agent_test0.harness import MemoryManager, get_redis_or_fallback  # 记忆管理
from agent_test0.main import zhipu_llm              # GLM 模型实例

load_dotenv()

# ─── 全局初始化（只启动一次） ───
_redis_client, _is_fallback = get_redis_or_fallback()

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  第 1 步：定义事件处理函数                                              ║
# ║  这是收到飞书消息后的入口，                                         ║
# ║  🔗 在这里调用你的 CrewAI Agent                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def do_p2_im_message_receive_v1(data: lark.im.v1.P2ImMessageReceiveV1) -> None:
    """
    收到飞书消息事件时触发（长连接推送到这里）
    这是飞书 → Agent 的唯一入口

    🔗 【Agent 连接点 2】: 提取消息内容 → 调用 Agent → 发送回复
    """
    print("\n" + "=" * 60)
    print("📩 [飞书] 收到消息事件")

    # ─── 解析飞书事件数据 ───
    event = data.event
    message = event.message
    sender = event.sender

    # 消息内容（飞书是 JSON 字符串）
    content_str = message.content
    try:
        content_json = json.loads(content_str)
        user_text = content_json.get("text", "")
    except json.JSONDecodeError:
        user_text = content_str

    # 发送者 open_id（用于回复）
    sender_open_id = sender.sender_id.open_id
    chat_id = message.chat_id

    print(f"   发送者: {sender_open_id}")
    print(f"   群聊: {chat_id or '(私聊)'}")
    print(f"   内容: {user_text}")

    # ╔══════════════════════════════════════════════════════════════════╗
    # ║  🔗🔗🔗  【Agent 连接点 3 - 核心！】                            ║
    # ║  调用 CrewAI TravelWorkflow 处理消息，获取回复                  ║
    # ╚══════════════════════════════════════════════════════════════════╝
    reply_text = _call_agent(user_text, sender_open_id, chat_id)

    # ─── 发送回复 ───
    _send_reply(sender_open_id, reply_text)
    print(f"📤 [飞书] 已回复: {reply_text[:80]}...")
    print("=" * 60)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  第 2 步：注册事件处理器                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# 长连接模式下 encrypt_key 和 verification_token 传空字符串
event_handler = (
    lark.EventDispatcherHandler.builder("", "")
    # 注册消息接收事件 → 触发 do_p2_im_message_receive_v1
    .register_p2_im_message_receive_v1(do_p2_im_message_receive_v1)
    .build()
)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  🔗🔗🔗  【Agent 连接点 4 - 核心函数】                                ║
# ║  接收用户消息字符串，调用 CrewAI TravelWorkflow，返回规划结果          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _call_agent(user_text: str, user_id: str, chat_id: str = None) -> str:
    """
    将飞书用户消息送入 CrewAI Agent 进行旅游规划

    输入: 用户在飞书输入的原始文本，如 "我想去杭州玩3天"
    输出: CrewAI 多智能体协作生成的旅游规划方案

    内部流程（参见 crew.py 的 TravelWorkflow）：
      PlannerCrew (决策) → TravelExpertCrew (执行) → ValidatorCrew (质检) → 最终报告
    """
    print("\n🔗 [Agent] 正在调用 CrewAI 多智能体旅游规划系统...")

    try:
        # Step 1: 生成会话 ID
        session_id = f"feishu_{user_id}_{abs(hash(user_text)) % 1000000:06d}"

        # Step 2: 创建 MemoryManager（四级记忆：Redis + MySQL）
        memory = MemoryManager(session_id, user_id, _redis_client, _is_fallback)
        memory.add_message("user", user_text)
        memory.convert_episodic_to_working(zhipu_llm)

        # Step 3: 重写查询（处理代词指代，如 "还有别的吗" → "还有别的杭州景点推荐吗"）
        contextual_msg = _rewrite_query(memory, user_text)

        # Step 4: 🔗 初始化并运行 CrewAI Flow（核心！）
        flow = TravelWorkflow()
        flow.state.message = contextual_msg
        flow.state.user_id = user_id
        flow.state.session_id = session_id
        flow.state.focus = memory.get_global_context_prompt(contextual_msg)

        print("   🚀 [Agent] TravelWorkflow.kickoff() 开始执行...")
        flow.kickoff()  # kickoff() 返回 None，结果存储在 flow.state 中
        print(f"   ✅ [Agent] TravelWorkflow.kickoff() 完成")

        # Step 5: 获取最终报告
        final_report = flow.state.final_report

        # Step 6: 保存助手回复到记忆
        memory.add_message("assistant", final_report)

        return final_report

    except Exception as e:
        print(f"   ❌ [Agent] 调用失败: {type(e).__name__}: {e}")
        return f"抱歉，处理您的请求时出现了错误：{str(e)}"


def _rewrite_query(memory: MemoryManager, current_message: str) -> str:
    """
    轻量级查询重写
    解决代词指代问题（如用户说"还有别的吗" → 补全上下文）
    """
    history = memory.get_chat_history()[-10:]
    if not history:
        return current_message

    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
    rewrite_prompt = (
        "任务：将用户的【当前回复】重写为独立明确的句子。仅替换代词，只输出一句话。\n"
        f"【最近对话】：\n{history_text}\n"
        f"【当前回复】：user: {current_message}\n"
        "【重写结果】："
    )

    try:
        return zhipu_llm.call([{"role": "user", "content": rewrite_prompt}]).strip()
    except Exception:
        return current_message


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  第 3 步：发送回复                                                      ║
# ║  使用飞书 API 将 Agent 生成的结果发送回飞书                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _send_reply(open_id: str, text: str):
    """
    通过飞书 API 发送消息给用户

    使用 lark.Client (同步 HTTP 客户端) 调用飞书发送消息接口
    """
    # 创建飞书 API 客户端（用于发送消息，不同于 WebSocket 长连接客户端）
    client = (
        lark.Client.builder()
        .app_id(os.getenv("FEISHU_APP_ID"))
        .app_secret(os.getenv("FEISHU_APP_SECRET"))
        .log_level(lark.LogLevel.WARNING),  # 生产环境用 WARNING，调试时改 DEBUG
        .build()
    )

    # 构建消息体（飞书消息内容是 JSON 字符串）
    msg_content = json.dumps({"text": text}, ensure_ascii=False)

    # 构建请求
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

    # 发送！
    response = client.im.v1.message.create(request)

    if not response.success():
        print(f"   ❌ [飞书] 发送消息失败: code={response.code}, msg={response.msg}")
    else:
        print(f"   ✅ [飞书] 消息发送成功: msg_id={response.data.message_id}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  第 4 步：启动长连接（WebSocket）                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def start():
    """
    启动飞书长连接机器人

    这个函数会阻塞，通过 WebSocket 持续接收飞书事件
    收到消息事件时自动调用 do_p2_im_message_receive_v1 → _call_agent
    """
    print("=" * 60)
    print("🤖 飞书长连接机器人启动")
    print("=" * 60)
    print(f"  App ID: {os.getenv('FEISHU_APP_ID')}")
    print(f"  模式:  WebSocket 长连接")
    print(f"  Agent: CrewAI TravelWorkflow（多智能体旅游规划）")
    print()
    print("  📡 等待飞书推送事件...")
    print("  用户在飞书 @机器人 发消息 → 这里自动收到 → CrewAI 处理 → 自动回复")
    print("=" * 60)
    print()

    # 创建 WebSocket 长连接客户端（按官方示例）
    cli = lark.ws.Client(
        os.getenv("FEISHU_APP_ID"),
        os.getenv("FEISHU_APP_SECRET"),
        event_handler=event_handler,       # ← 消息到达时触发这里的处理函数
        log_level=lark.LogLevel.WARNING,
    )
    cli.start()  # 阻塞，持续监听


if __name__ == "__main__":
    start()
