# 飞书长连接机器人 — 集成指南

## 架构（一句话）

```
飞书消息 → WebSocket → do_p2_im_message_receive_v1()
→ 🔗 _call_agent() → TravelWorkflow.kickoff()
→ 🔗 _send_reply() → 飞书用户收到回复
```

## 文件说明

| 文件 | 作用 |
|------|------|
| `long_conn_bot.py` | **唯一入口** — 长连接机器人，所有功能集中于此 |
| `CONNECTION_GUIDE.md` | 连接关系可视化说明（推荐先看这个） |

## 启动

```bash
uv run python -m agent_test0.feishu.long_conn_bot
```

## 在飞书开放平台的配置

1. 进入 https://open.feishu.cn/app/cli_aaa0222323fa9ce4
2. 点击「事件订阅」→ 开通
3. 订阅事件：勾选 `im.message.receive_v1`（接收消息）
4. 发布应用

## 技术原理

使用 `lark.ws.Client` (WebSocket 长连接) 保持与飞书服务器的持续连接。
收到消息事件时自动触发 `do_p2_im_message_receive_v1()` → 调用 CrewAI Agent → 通过 `lark.Client` HTTP API 发送回复。

参考文档:
- https://open.feishu.cn/document/server-side-sdk/python--sdk/invoke-server-api
- https://open.feishu.cn/document/server-side-sdk/python--sdk/handle-events
