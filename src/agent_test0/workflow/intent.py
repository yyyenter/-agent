# agent_test0/workflow/intent.py
"""
意图路由共享模块 —— API / 飞书 / CLI 三个入口统一从这里判 travel / default_chat。

之前路由分散在三处且互相不一致：
  - server.py    : 真·semantic_router（routes.json + Ollama），但异常时回 default_chat
  - debug_cli.py : 重复了一份路由代码，且 intent_router 硬编码 None，fallback 永远 return "travel"
  - feishu bot   : 完全没有路由，每条消息都跑全 Flow

本模块统一为单一来源：
  1. 优先用 semantic_router + Ollama(nomic-embed-text) + knowledge/routes.json 样本库
  2. Ollama 不可用 / 路由异常时降级关键词匹配（CLAUDE.md 承诺的降级路径，原 server.py 没实现）
  3. 懒构造：import 不触发 Ollama 连接，飞书/CLI 不会因 Ollama 没起而启动崩

返回值契约：'travel' 或 'default_chat'（与原 server.py 保持一致）。
"""

import json
from pathlib import Path


# routes.json 锚定到仓库根：<repo>/knowledge/routes.json
#   intent.py 位于 <repo>/src/agent_test0/workflow/intent.py
#   parents[0]=workflow  [1]=agent_test0  [2]=src  [3]=<repo>
ROUTES_PATH = Path(__file__).resolve().parents[3] / "knowledge" / "routes.json"

ROUTES: dict = {}
try:
    with open(ROUTES_PATH, "r", encoding="utf-8") as _f:
        ROUTES = json.load(_f)
except FileNotFoundError:
    print(f"[intent] routes.json 未找到: {ROUTES_PATH}（将只用关键词降级）")

# 懒构造的路由器；None 表示尚未构建或已失效
_intent_router = None
# 是否已尝试构建过（避免每次调用都重试 import / 构造）
_router_attempted = False

# 关键词降级表（Ollama 不可用时的兜底，覆盖常见旅游意图词）
# 取舍：Ollama 挂时宁可把闲聊误判进 Flow（Planner 会兜底 simple_answer），
# 也别漏判旅游（漏判 = 用户拿不到行程）。所以保留裸 "玩" 等宽匹配。
_TRAVEL_KEYWORDS = (
    "旅游", "旅行", "行程", "攻略", "游玩", "景点", "度假", "出差",
    "自驾", "跟团", "自由行", "周边游", "短途", "长途",
    "出去玩", "去玩", "去哪玩", "几天", "预算", "人均",
    "玩",  # 裸字宽匹配：覆盖"想去北京玩3天"这类省略词
)


def _build_router():
    """构建 semantic_router；失败（库缺失 / Ollama 不可达）返回 None。"""
    try:
        from semantic_router import Route, SemanticRouter
        from semantic_router.encoders.ollama import OllamaEncoder
    except ImportError as e:
        print(f"[intent] semantic_router 未安装，降级关键词匹配: {e}")
        return None
    try:
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
        return SemanticRouter(
            routes=[travel_route, chat_route],
            encoder=OllamaEncoder(
                name="nomic-embed-text",
                base_url="http://localhost:11434",
                score_threshold=0.3,
            ),
        )
    except Exception as e:
        print(f"[intent] 语义路由初始化失败，降级关键词匹配: {e}")
        return None


def classify_intent(message: str) -> str:
    """判断用户消息意图。

    Returns:
        'travel'       —— 旅游规划类，入口层应跑 TravelWorkflow
        'default_chat' —— 闲聊/非旅游，入口层走轻量 LLM 回答

    Ollama 不可用时降级关键词匹配（不会抛异常）。
    """
    global _intent_router, _router_attempted
    if not _router_attempted:
        _router_attempted = True
        _intent_router = _build_router()

    if _intent_router is not None:
        try:
            result = _intent_router(message)
            if result and result.name:
                return result.name
        except Exception as e:
            # 调用期失败（如 Ollama 断连）→ 本次降级关键词，并标记路由器失效
            print(f"[intent] 路由调用异常，本次降级关键词匹配: {e}")
            _intent_router = None

    # 关键词降级
    if any(kw in message for kw in _TRAVEL_KEYWORDS):
        return "travel"
    return "default_chat"


__all__ = ["classify_intent", "ROUTES"]
