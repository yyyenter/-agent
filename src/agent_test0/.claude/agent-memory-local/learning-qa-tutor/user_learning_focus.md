---
name: user-learning-focus
description: 用户当前学习重点与已覆盖的知识点，用于后续对话针对性讲解
metadata:
  type: user
---

用户正在学习本项目（基于 CrewAI 的旅游规划系统）的**飞书长连接接口**，重点在 `lark-oapi` SDK 的 WebSocket 模式。

已讲解过的知识点（2026-06-22）：
- `lark.ws.Client(app_id, app_secret, event_handler=, log_level=)` 的参数含义与选择标准
- `event_handler` 由 `EventDispatcherHandler.builder(encrypt_key, verification_token, level).register_p2_im_xxx_v1(func).build()` 构造
  - 注意 builder 参数顺序是 `encrypt_key, verification_token`（不是反过来）
  - 长连接模式下两者留空串即可
- `lark.LogLevel` 共 5 个枚举：CRITICAL / DEBUG / ERROR / INFO / WARNING
  - `log_level` 只控制 SDK 自身日志，不影响业务 logging
  - 项目选 `WARNING` 的理由：避免 INFO 刷屏但保留断连/重连可见性

已讲解过的知识点（2026-06-23）：
- `do_p2_im_message_receive_v1(data: lark.im.v1.P2ImMessageReceiveV1)` 的 data 嵌套结构
  - `data.event`（类型 P2ImMessageReceiveV1Data，中间多一层，日常当 event 容器用）
  - `event.message: EventMessage`、`event.sender: EventSender`
  - `message.content` 是 JSON **字符串**不是 dict，text 消息结构 `{"text":"..."}`，需 json.loads
  - `message.message_id`（"om_" 开头）用于幂等去重；`getattr` 防御缺字段
  - `message.chat_id`（"oc_" 开头）私聊为 None，统一用 open_id 回复最省心
  - `sender.sender_id.open_id`（"ou_" 开头）用作 user_id；session_id 按 user 派生 `feishu_{open_id}`，不能按消息
  - 字段定义在 `lark_oapi/api/im/v1/model/` 下，按类名同名文件查找
  - 易错点：不判断 message_type 就取 text 会在图片消息时拿到空串；open_id 跨应用会变
- bot.py 待讲：`_send_reply` 的 CreateMessageRequest.builder 构建链、`_agent_executor.submit` 为何必须异步

用户偏好（推断）：
- 偏好"先核心结论、再展开、再给最小示例、再点易错点"的教学结构
- 希望回答基于真实签名验证，不要凭记忆编造（已用源码文件验证后再讲）
- 偏好给出"怎么自己查到答案"的路径（IDE 跳转、inspect、官方文档、源码文件路径）
