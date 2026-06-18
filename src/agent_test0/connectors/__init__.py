# agent_test0/connectors/__init__.py
"""
第三方连接器包。

每个子目录是一个连接器（如飞书、企业微信、钉钉），把外部消息通道适配到
TravelWorkflow.run_for_user 统一入口。

子模块：
    feishu/  —— 飞书长连接 bot（WebSocket 模式）
"""
