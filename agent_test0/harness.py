import json
import hashlib
import sqlite3
import os
import re
from datetime import datetime
import redis
from crewai import LLM

DB_PATH = "knowledge/user_profiles.db"
os.makedirs("knowledge", exist_ok=True)

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
    """工业级 Session 记忆管理器 (CC 架构)"""
    
    def __init__(self, session_id: str, user_id: str, redis_client: redis.Redis):
        self.session_id = session_id
        self.user_id = user_id
        self.redis = redis_client
        
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

    def get_chat_history(self) -> list[dict]:
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
        快速分析最近对话，捕获用户只针对“当前这一次出行”提出的限制。
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
        核心流转 B：Working/Episodic -> Semantic Memory (提炼并永久持久化长期偏好)
        过滤分析当前上下文，将用户的“恒定人设与禁忌”（如过敏源、偏好酒店等）永久固化进 SQLite DB。
        """
        history = self.get_chat_history()
        if not history:
            return
        
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
        
        prompt = f"""
        你是一个资深用户画像分析师。请分析用户在对话中表现出来的【长期、通用且恒定】个人特质或偏好（例如：严重饮食忌口、身体健康限制、常住地、习惯性出行工具偏好、恒定住宿标准等）。
        注意：排除仅针对某一次具体行程的临时性安排（例如：“我明天去杭州”是临时安排，不是长期偏好；而“我对海鲜过敏”或“我习惯住高档星级酒店”属于长期偏好）。
        
        如果找到了，请将其抽象为 Key-Value 键值对：
        - Key: 必须是下划线英文命名，如 'dietary_restrictions', 'physical_limits', 'preferred_hotel_brands'。
        - Value: 具体特征描述。
        
        请严格按照纯 JSON 数组格式输出，不要包含 ```json 等任何 Markdown 标记或多余解释。
        格式样例：
        [
          {{"key": "dietary_restrictions", "value": "不能吃任何海鲜，过敏"}},
          {{"key": "physical_limits", "value": "膝盖有旧伤，避免大量登山和剧烈徒徒步运动"}}
        ]
        如果未分析出任何恒定的长期偏好，直接输出 []。
        
        【对话历史】：
        {history_text}
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

class ToolCacheManager:
    """桶4: Tool Results (全局工具缓存，跨会话、跨用户共享)"""
    @staticmethod
    def get_tool_result(redis_client: redis.Redis, tool_name: str, params: dict) -> str | None:
        param_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
        key = f"cc:global:tool_cache:{tool_name}:{param_hash}"
        return redis_client.get(key)

    @staticmethod
    def set_tool_result(redis_client: redis.Redis, tool_name: str, params: dict, result: str, expire: int = 3600):
        param_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
        key = f"cc:global:tool_cache:{tool_name}:{param_hash}"
        redis_client.setex(key, expire, result)
