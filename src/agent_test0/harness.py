import json
import hashlib
import math
import sqlite3
import os
import re
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

from crewai import LLM

DB_PATH = "knowledge/user_profiles.db"
os.makedirs("knowledge", exist_ok=True)

# ==================== 内存回退存储 ====================
class InMemoryFallback:
    """Redis 不可用时的内存回退存储"""
    def __init__(self):
        self.data: dict[str, list[str]] = {}
        self.hashes: dict[str, dict[str, str]] = {}
        self.kv: dict[str, str] = {}
    
    def rpush(self, key: str, value: str) -> None:
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(value)
    
    def ltrim(self, key: str, start: int, end: int) -> None:
        if key in self.data:
            self.data[key] = self.data[key][start:]
    
    def lrange(self, key: str, start: int, end: int) -> list[str]:
        if key not in self.data:
            return []
        if end == -1:
            return self.data[key][start:]
        return self.data[key][start:end+1]
    
    def expire(self, key: str, ttl: int) -> None:
        pass  # 内存存储不需要 TTL
    
    def hset(self, key: str, mapping: dict[str, str] | None = None, **kwargs: str) -> None:
        if key not in self.hashes:
            self.hashes[key] = {}
        if mapping:
            self.hashes[key].update(mapping)
        self.hashes[key].update(kwargs)

    def hget(self, key: str, field: str) -> Optional[str]:
        return self.hashes.get(key, {}).get(field)

    def hgetall(self, key: str) -> dict[str, str]:
        return self.hashes.get(key, {})
    
    def get(self, key: str) -> Optional[str]:
        return self.kv.get(key)

    def setex(self, key: str, expire: int, value: str) -> None:
        self.kv[key] = value

    def delete(self, key: str) -> None:
        self.kv.pop(key, None)
        self.hashes.pop(key, None)
        self.data.pop(key, None)

# 全局内存回退实例
_memory_fallback = InMemoryFallback()

def get_redis_or_fallback():  # -> tuple[Any, bool]
    """
    获取 Redis 客户端，如果连接失败或模块不存在则返回内存回退存储
    返回: (client, is_fallback: bool)
    """
    if not REDIS_AVAILABLE:
        print("[Redis] Module not available, using in-memory storage")
        return _memory_fallback, True

    try:
        client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        client.ping()  # 测试连接
        print("[Redis] Connected successfully")
        return client, False  # type: ignore[return-value]
    except Exception as e:
        print(f"[Redis] Connection failed: {e}")
        print("[Memory] Falling back to in-memory storage")
        return _memory_fallback, True

def init_db():
    """初始化 SQLite 动态 Key-Value 长期记忆表"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_memory (
                user_id TEXT,
                memory_key TEXT,         -- 动态键，如 dietary_restrictions, physical_limits
                memory_value TEXT,       -- 动态值，如 对海鲜严重过敏, 不能爬山
                last_updated TIMESTAMP,
                PRIMARY KEY (user_id, memory_key)
            )
        """)
init_db()

class MemoryManager:
    """工业级 Session 记忆管理器 (CC 架构) - 支持 Redis 和内存回退"""
    
    def __init__(self, session_id: str, user_id: str, redis_client, is_fallback: bool = False):
        self.session_id = session_id
        self.user_id = user_id
        self.redis = redis_client
        self.is_fallback = is_fallback
        
        self.chat_key = f"session:{session_id}:chat"          # 桶3: 原始对话轮次 (Episodic)
        self.summary_key = f"session:{session_id}:summary"    # 桶5: 短期精炼约束 (Working Memory)
        self.ttl = 86400  # 24小时

    # ==================== 桶3: Episodic Memory (Redis) ====================
    def add_message(self, role: str, content: str, max_turns: int = 8):
        """追加原始对话并控制长度"""
        msg = json.dumps({"role": role, "content": content}, ensure_ascii=False)
        self.redis.rpush(self.chat_key, msg)
        self.redis.ltrim(self.chat_key, -(max_turns * 2), -1)
        self.redis.expire(self.chat_key, self.ttl)

    def get_chat_history(self) -> list[dict[str, str]]:
        """获取最近的原始对话"""
        raw = self.redis.lrange(self.chat_key, 0, -1)
        return [json.loads(m) for m in raw]

    # ==================== 桶5: Working Memory (Redis Summary) ====================
    def update_short_term_summary(self, new_constraints: dict[str, str]):
        """更新当前行程的硬约束"""
        if new_constraints:
            self.redis.hset(self.summary_key, mapping=new_constraints)
            self.redis.expire(self.summary_key, self.ttl)

    def get_short_term_summary(self) -> dict[str, str]:
        """获取当前行程的核心约束字典"""
        summary = self.redis.hgetall(self.summary_key)
        return {k: v for k, v in summary.items()} if summary else {}

    # ==================== 桶6: Semantic Memory (SQLite KV 长期偏好) ====================
    def save_user_memory(self, memory_key: str, memory_value: str):
        """写入 SQLite KV 长期偏好库 (内置去重合并逻辑)"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO user_memory (user_id, memory_key, memory_value, last_updated)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, memory_key) 
                DO UPDATE SET memory_value = excluded.memory_value, 
                              last_updated = excluded.last_updated
            """, (self.user_id, memory_key, memory_value, timestamp))

    def get_user_profile(self) -> str:
        """从 SQLite 提取该用户的所有长期特征，组装为可读文本"""
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.execute(
                    "SELECT memory_key, memory_value FROM user_memory WHERE user_id = ?", 
                    (self.user_id,)
                )
                rows = cursor.fetchall()
            if not rows:
                return "暂无长期偏好"
            profile_lines = [f"{row[0]}: {row[1]}" for row in rows]
            return "\n".join(profile_lines)
        except Exception:
            return "暂无长期偏好"

    # ==================== 🧠 记忆生命周期自动流转机制 ====================
    
    def convert_episodic_to_working(self, llm: LLM):
        """
        核心流转 A：Episodic -> Working Memory (提取当前行程的临时硬约束)
        快速分析最近对话，捕获用户只针对"当前这一次出行"提出的限制。
        """
        history = self.get_chat_history()
        if not history:
            return
        
        # 仅拿最新对话进行快速解析，降低延迟与 token 成本
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-3:]])
        
        prompt = f"""
        请分析以下对话，提取出属于【当前具体行程】的最新硬性指标（如目的地、出行天数、出行预算、特定的临时偏好等）。
        仅提取属于本次行程的临时限制或规划目标，忽略用户的通用长期习惯。
        必须以纯 JSON 格式输出，不要包含 ```json 等任何 Markdown 标记或多余的文字解释。
        格式样例：
        {{"destination": "杭州", "days": "3", "budget": "高预算", "preferences": "想去西湖"}}
        如果未分析出任何具体约束，直接输出 {{}}。
        
        【对话历史】：
        {history_text}
        """
        try:
            response = llm.call([{"role": "user", "content": prompt}]).strip()
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                new_constraints = json.loads(json_match.group(0))
                if new_constraints:
                    self.update_short_term_summary(new_constraints)
        except Exception as e:
            print(f"[Memory Conversion] Episodic -> Working failed: {e}")

    def convert_to_semantic(self, llm: LLM):
        """
        核心流转 B：Short-term Summary -> Semantic Memory (提炼并永久持久化长期偏好)
        【关键设计】只使用短期摘要作为输入，用于控制上下文长度和提取长期价值特征
        """
        short_term = self.get_short_term_summary()
        if not short_term:
            return
        
        short_term_text = json.dumps(short_term, ensure_ascii=False)
        
        prompt = f"""你是一个资深用户画像分析师。
以下是用户本次行程的【临时约束摘要】：
{short_term_text}

请从中剥离出：
1. 属于本次临时的约束（如去哪玩、几天、预算多少）。 -> 【忽略】
2. 属于用户长期的、通用的个人偏好（如：忌口、身体状况、强烈的品牌偏好）。 -> 【提取】

如果没有长期偏好，输出 []。
如果有，请输出 JSON 数组，例如：
[{{"key": "dietary_restrictions", "value": "海鲜过敏"}}, {{"key": "travel_style", "value": "不安排早起"}}]
"""
        try:
            response = llm.call([{"role": "user", "content": prompt}]).strip()
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                profile_list = json.loads(json_match.group(0))
                for item in profile_list:
                    key = item.get("key")
                    value = item.get("value")
                    if key and value:
                        self.save_user_memory(key, value)
        except Exception as e:
            print(f"[Memory Conversion] Conversion to Semantic failed: {e}")

    def get_global_context_prompt(self, current_message: str = "") -> str:
        """
        组装完整上下文 Prompt（供旅游意图时使用）
        仅在路由判断为 travel 后调用
        """
        parts = []
        
        # 长期偏好
        profile = self.get_user_profile()
        if profile != "暂无长期偏好":
            parts.append(f"【用户长期偏好画像】：\n{profile}")
        
        # 短期约束
        summary = self.get_short_term_summary()
        if summary:
            parts.append(f"【当前行程核心约束】：\n{json.dumps(summary, ensure_ascii=False)}")
        
        # 对话历史（限制条数）
        history = self.get_chat_history()
        if history:
            history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-6:]])
            parts.append(f"【近期对话上下文】：\n{history_text}")
        
        if current_message:
            parts.append(f"【当前最新指令】：{current_message}")
        
        return "\n\n".join(parts) if parts else ""

class ToolCacheManager:
    """桶4: 全局工具缓存 — 三层匹配: 精确(L1) → 归一(L2) → 语义向量(L3)

    L1 精确哈希: 参数完全相同时毫秒级命中
    L2 归一哈希: 自动去空格/转小写/排序, 处理格式变体
    L3 语义向量: 字符 bigram 余弦相似度, 处理同义换说 (阈值 0.45)
    """

    CACHE_PREFIX = "cc:global:tool_cache"
    # L3: 语义匹配配置
    SEMANTIC_THRESHOLD = 0.45   # 余弦相似度阈值 (越低越宽松)
    MAX_SCAN = 200              # L3 扫描上限, 防性能退化

    # ---- 参数标准化 (L2) ----

    @staticmethod
    def _normalize_params(params):
        """参数文本标准化: 去空格 / 转小写 / 排序"""
        if isinstance(params, str):
            return re.sub(r'\s+', '', params).strip().lower()
        if isinstance(params, dict):
            return {k: ToolCacheManager._normalize_params(v)
                    for k, v in sorted(params.items())}
        if isinstance(params, list):
            return sorted(ToolCacheManager._normalize_params(p) for p in params)
        return params

    @staticmethod
    def _params_to_text(params) -> str:
        """从参数中提取可搜索的文本, 生成语义向量输入"""
        if isinstance(params, str):
            return params
        if isinstance(params, dict):
            parts = []
            for v in params.values():
                if isinstance(v, str):
                    parts.append(v)
                elif isinstance(v, (int, float)):
                    parts.append(str(v))
            return ' '.join(parts)
        return json.dumps(params, sort_keys=True, ensure_ascii=False)

    # ---- 键生成 (L1/L2) ----

    @staticmethod
    def _make_key(tool_name: str, params, *, normalized: bool = False) -> str:
        p = ToolCacheManager._normalize_params(params) if normalized else params
        param_str = json.dumps(p, sort_keys=True, ensure_ascii=False)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()
        return f"{ToolCacheManager.CACHE_PREFIX}:{tool_name}:{param_hash}"

    # ---- 后端读写 ----

    @staticmethod
    def _read_field(redis_client, key: str) -> Optional[str]:
        if hasattr(redis_client, 'hget'):
            return redis_client.hget(key, "result")
        return redis_client.get(key)

    # ---- 语义向量 (L3) ----

    @staticmethod
    def _bigram_vector(text: str) -> dict[str, float]:
        """字符 bigram TF 向量 — 零依赖语义指纹.
        中文: "杭州旅游" → [杭州,州旅,旅游] 捕获词语模式
        """
        text = re.sub(r'\s+', '', text).lower()
        vec: dict[str, float] = {}
        for i in range(len(text) - 1):
            gram = text[i:i+2]
            vec[gram] = vec.get(gram, 0.0) + 1.0
        return vec

    @staticmethod
    def _cosine_sim(a: dict[str, float], b: dict[str, float]) -> float:
        """稀疏向量余弦相似度"""
        if not a or not b:
            return 0.0
        keys = set(a) & set(b)
        if not keys:
            return 0.0
        dot = sum(a[k] * b[k] for k in keys)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    @staticmethod
    def _scan_semantic(redis_client, prefix: str, query_vec: dict[str, float],
                       threshold: float) -> Optional[str]:
        """扫描同工具缓存, 返回语义最匹配 (且超阈值) 的结果."""
        best_score = 0.0
        best_result = None
        scanned = 0

        if hasattr(redis_client, 'hashes'):
            for key, entry in redis_client.hashes.items():
                if not key.startswith(prefix):
                    continue
                emb_str = entry.get("embedding", "")
                if not emb_str:
                    continue
                try:
                    vec = json.loads(emb_str)
                    score = ToolCacheManager._cosine_sim(query_vec, vec)
                    if score > best_score:
                        best_score = score
                        best_result = entry.get("result")
                except (json.JSONDecodeError, TypeError):
                    continue
                scanned += 1
                if scanned >= ToolCacheManager.MAX_SCAN:
                    break
        elif hasattr(redis_client, 'scan_iter'):
            try:
                for key in redis_client.scan_iter(match=f"{prefix}*", count=50):
                    emb_str = redis_client.hget(key, "embedding")
                    if not emb_str:
                        continue
                    try:
                        vec = json.loads(emb_str)
                        score = ToolCacheManager._cosine_sim(query_vec, vec)
                        if score > best_score:
                            best_score = score
                            best_result = redis_client.hget(key, "result")
                    except (json.JSONDecodeError, TypeError):
                        continue
                    scanned += 1
                    if scanned >= ToolCacheManager.MAX_SCAN:
                        break
            except Exception:
                pass

        return best_result if best_score >= threshold else None

    # ---- Public API ----

    @staticmethod
    def get_tool_result(redis_client, tool_name: str, params) -> Optional[str]:
        # L1: 精确哈希
        exact_key = ToolCacheManager._make_key(tool_name, params)
        result = ToolCacheManager._read_field(redis_client, exact_key)
        if result is not None:
            return result

        # L2: 归一哈希 (空格/大小写/排序变体)
        norm_key = ToolCacheManager._make_key(tool_name, params, normalized=True)
        if norm_key != exact_key:
            result = ToolCacheManager._read_field(redis_client, norm_key)
            if result is not None:
                return result

        # L3: 语义向量 (同义换说, 字符 bigram 余弦相似度)
        query_text = ToolCacheManager._params_to_text(params)
        query_vec = ToolCacheManager._bigram_vector(query_text)
        prefix = f"{ToolCacheManager.CACHE_PREFIX}:{tool_name}:"
        return ToolCacheManager._scan_semantic(
            redis_client, prefix, query_vec, ToolCacheManager.SEMANTIC_THRESHOLD
        )

    @staticmethod
    def set_tool_result(redis_client, tool_name: str, params,
                        result: str, expire: int = 3600) -> None:
        key = ToolCacheManager._make_key(tool_name, params)
        param_str = json.dumps(params, sort_keys=True, ensure_ascii=False)
        # 存储语义向量 (L3 检索用)
        query_text = ToolCacheManager._params_to_text(params)
        embedding_str = json.dumps(
            ToolCacheManager._bigram_vector(query_text), ensure_ascii=False)

        if hasattr(redis_client, 'hset'):
            redis_client.hset(key, mapping={
                "tool_name": tool_name,
                "params": param_str,
                "result": result,
                "embedding": embedding_str,
            })
            redis_client.expire(key, expire)
        else:
            redis_client.setex(key, expire, result)