import json
import hashlib
import math
import os
import re
from datetime import datetime
from typing import Any, Optional
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

try:
    import pymysql
    pymysql.install_as_MySQLdb()
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False
    pymysql = None

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

from crewai import LLM

# ==================== MySQL 数据库配置 ====================
def get_mysql_connection():
    """获取 MySQL 连接（自动创建数据库和表，类似 SQLite 的自动创建行为）"""
    host = os.getenv("MYSQL_HOST", "localhost")
    port = int(os.getenv("MYSQL_PORT", "3306"))
    user = os.getenv("MYSQL_USER", "root")
    password = os.getenv("MYSQL_PASSWORD", "")
    database = os.getenv("MYSQL_DATABASE", "agent_test0")

    if not MYSQL_AVAILABLE:
        raise ImportError("pymysql is not installed. Please install it with: pip install pymysql")

    # 先连接到 MySQL（不指定数据库，用于创建）
    conn = pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        charset='utf8mb4'
    )

    try:
        with conn.cursor() as cursor:
            # 自动创建数据库（类似 SQLite 自动创建 .db 文件）
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{database}` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            # 选择数据库
            cursor.execute(f"USE `{database}`")
            # 自动创建表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS `user_memory` (
                    `user_id` VARCHAR(255) NOT NULL,
                    `memory_key` VARCHAR(255) NOT NULL,
                    `memory_value` TEXT,
                    `context_tag` VARCHAR(100) DEFAULT 'global',
                    `scope` VARCHAR(50) DEFAULT 'long_term',
                    `last_updated` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    PRIMARY KEY (`user_id`, `memory_key`, `context_tag`),
                    INDEX `idx_user_id` (`user_id`),
                    INDEX `idx_context_tag` (`context_tag`),
                    INDEX `idx_scope` (`scope`)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
        conn.commit()
    finally:
        conn.close()

    # 再连接到指定数据库（类似 SQLite 的 sqlite3.connect）
    return pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

def init_db():
    """初始化 MySQL 动态 Key-Value 长期记忆表（含作用域隔离）

    类似 SQLite 的行为，数据库和表会在第一次连接时自动创建。
    此函数可用于手动检查/重建表结构。
    """
    conn = get_mysql_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS `user_memory` (
                    `user_id` VARCHAR(255) NOT NULL,
                    `memory_key` VARCHAR(255) NOT NULL,
                    `memory_value` TEXT,
                    `context_tag` VARCHAR(100) DEFAULT 'global',
                    `scope` VARCHAR(50) DEFAULT 'long_term',
                    `last_updated` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    PRIMARY KEY (`user_id`, `memory_key`, `context_tag`),
                    INDEX `idx_user_id` (`user_id`),
                    INDEX `idx_context_tag` (`context_tag`),
                    INDEX `idx_scope` (`scope`)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
        conn.commit()
        print("[MySQL] 表 user_memory 创建成功或已存在")
    finally:
        conn.close()
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

    def lset(self, key: str, index: int, value: str) -> None:
        # 对齐 Redis LSET: 按索引就地覆写; 越界抛错, 与 redis-py 行为一致。
        if key not in self.data or not (-len(self.data[key]) <= index < len(self.data[key])):
            raise IndexError("index out of range")
        self.data[key][index] = value
    
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
    """初始化 MySQL 动态 Key-Value 长期记忆表（含作用域隔离）"""
    conn = get_mysql_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_memory (
                    user_id VARCHAR(255) NOT NULL,
                    memory_key VARCHAR(255) NOT NULL,
                    memory_value TEXT,
                    context_tag VARCHAR(100) DEFAULT 'global',
                    scope VARCHAR(50) DEFAULT 'long_term',
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, memory_key, context_tag),
                    INDEX idx_user_id (user_id),
                    INDEX idx_context_tag (context_tag),
                    INDEX idx_scope (scope)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
        conn.commit()
    finally:
        conn.close()


class MemoryManager:
    """工业级 Session 记忆管理器 (CC 架构) - 支持 Redis 和内存回退"""

    def __init__(self, session_id: str, user_id: str, redis_client, is_fallback: bool = False):
        self.session_id = session_id
        self.user_id = user_id
        self.redis = redis_client
        self.is_fallback = is_fallback
        self._db_connection = None
        # 进程内自增 msg_id 后缀, 保证同一毫秒多次 add_message 也不会撞 id
        self._msg_seq = 0

        # chat_key  : 短期会话原文日志（Full Short-Term Log）
        #              —— 完整保存 user/assistant 原始消息, 永远是审计/检索的权威源。
        # index_key : 短期会话索引（Retrieved Short-Term Index）
        #              —— 同步生成的轻量条目 (task_id/destination/topic/has_slots/...),
        #                 用于从大量历史中按业务字段定位相关 turns, 避免污染当前任务。
        # summary_key: 旧版 working summary, 不再作为权威短期记忆, 仅作兼容保留。
        self.chat_key = f"session:{session_id}:chat"
        self.index_key = f"session:{session_id}:index"
        self.summary_key = f"session:{session_id}:summary"
        self.ttl = 86400  # 24小时

        # === 跨轮业务字段 (Multi-turn Persistence) ===
        # 这些字段跨飞书多轮对话持续存在, 跟 session_id 同生命周期。
        # 在 run_for_user() 入口 bind_to_state(flow.state) 时同步到 TravelState,
        # 让节点能直接读 flow.state.current_destination 而不必每次反查 memory。
        # 设计上 memory 是权威源, TravelState 是单轮缓存。
        self.current_task_id: str | None = None
        self.current_destination: str | None = None
        self.current_topic: str = "general"
        # 跨轮持久计数器 (用于在 task 切换时识别"上一轮", 而不是"本轮新开")
        self._task_seq: int = 0

    def _get_db_connection(self):
        """获取数据库连接（复用连接）"""
        if self._db_connection is None or not self._db_connection.open:
            self._db_connection = get_mysql_connection()
        return self._db_connection

    def _close_db_connection(self):
        """关闭数据库连接"""
        if self._db_connection and self._db_connection.open:
            self._db_connection.close()
            self._db_connection = None

    def __del__(self):
        """析构时关闭数据库连接"""
        self._close_db_connection()

    # ==================== 短期会话记忆：原文 + 索引 ====================
    def add_message(self, role: str, content: str,
                    max_turns: int | None = 100,
                    *,
                    task_id: str | None = None,
                    topic: str = "general",
                    destination: str | None = None,
                    has_slots: bool = False,
                    extracted_slots: dict[str, Any] | None = None,
                    is_completed_task: bool = False):
        """
        追加短期会话原文（Full Short-Term Log），并同步写入检索索引。

        - chat_key 始终保留完整原文, 方便审计 / 回放 / 重新检索。
        - index_key 保存业务级索引条目 (task_id/destination/topic/has_slots/...),
          供 retrieve_short_term_context() 在规划前精准召回相关 turns,
          避免把旧任务字段 (重庆/3天/3000/两人) 误用于新任务 (想去成都)。
        - max_turns=100 仅作为防爆上限 (200 条消息); 传 None 不裁剪。
        """
        msg_id = self._next_msg_id()
        msg = json.dumps(
            {"msg_id": msg_id, "role": role, "content": content},
            ensure_ascii=False,
        )
        self.redis.rpush(self.chat_key, msg)
        if max_turns is not None:
            self.redis.ltrim(self.chat_key, -(max_turns * 2), -1)
        self.redis.expire(self.chat_key, self.ttl)

        index_entry = self._build_index_entry(
            msg_id=msg_id,
            role=role,
            content=content,
            task_id=task_id,
            topic=topic,
            destination=destination,
            has_slots=has_slots,
            extracted_slots=extracted_slots,
            is_completed_task=is_completed_task,
        )
        if index_entry is not None:
            self.redis.rpush(self.index_key, json.dumps(index_entry, ensure_ascii=False))
            # 索引与原文同生命周期, 裁剪保持一致
            if max_turns is not None:
                self.redis.ltrim(self.index_key, -(max_turns * 2), -1)
            self.redis.expire(self.index_key, self.ttl)

        return msg_id

    def update_index_entry(self, msg_id: str, *,
                           task_id: str | None = None,
                           destination: str | None = None,
                           topic: str | None = None,
                           has_slots: bool | None = None,
                           extracted_slots: dict[str, Any] | None = None,
                           is_completed_task: bool | None = None) -> bool:
        """按 msg_id 定位 Redis 索引条目并就地更新业务字段 (方案 C)。

        用途: add_message 写入时业务字段 (task_id/destination/slots) 未知
        (user message 进来时 Planner 还没跑), 等 Planner 推断出 destination
        后用本方法回写索引, 让下一轮 retrieve_short_term_context 能按
        task_id/destination 召回历史。

        只更新传了非 None 的字段。返回是否找到并更新。
        """
        raw = self.redis.lrange(self.index_key, 0, -1)
        updated = False
        for i, item in enumerate(raw):
            try:
                entry = json.loads(item)
            except (json.JSONDecodeError, TypeError):
                continue
            if entry.get("msg_id") != msg_id:
                continue
            if task_id is not None:
                entry["task_id"] = task_id
            if destination is not None:
                entry["destination"] = destination
            if topic is not None:
                entry["topic"] = topic
            if has_slots is not None:
                entry["has_slots"] = has_slots
            if extracted_slots is not None:
                entry["extracted_slots"] = extracted_slots
            if is_completed_task is not None:
                entry["is_completed_task"] = is_completed_task
            # Redis List 没有 lset 按值更新, 用 lset 按索引写回
            self.redis.lset(self.index_key, i, json.dumps(entry, ensure_ascii=False))
            updated = True
            break
        return updated

    def get_chat_history(self) -> list[dict[str, str]]:
        """获取最近完整短期对话原文。"""
        raw = self.redis.lrange(self.chat_key, 0, -1)
        return [json.loads(m) for m in raw]

    def get_short_term_index(self) -> list[dict[str, Any]]:
        """获取完整短期会话索引。"""
        raw = self.redis.lrange(self.index_key, 0, -1)
        out: list[dict[str, Any]] = []
        for item in raw:
            try:
                out.append(json.loads(item))
            except json.JSONDecodeError:
                continue
        return out

    # ============================================================
    # 跨轮业务字段: bind / mark_completed / new_task_id
    # ============================================================
    def bind_to_state(self, state) -> None:
        """把 MemoryManager 的跨轮业务字段同步到 TravelState (单轮缓存)。

        流程:
          1. run_for_user() 在 kickoff 前调一次, 把上一轮的 task/destination/topic
             同步到 flow.state, 让 Planner 节点能直接读 flow.state.current_destination。
          2. 节点跑完后, 通过 flow.state.current_task_id 反向回写 memory。

        字段语义对照:
          memory.current_task_id  <->  state.current_task_id  (跨轮, 字符串 ID)
          memory.current_destination <-> state.current_destination
          memory.current_topic    <->  state.current_topic
        """
        if state.current_task_id is None:
            state.current_task_id = self.current_task_id
        if state.current_destination is None:
            state.current_destination = self.current_destination
        if not state.current_topic or state.current_topic == "general":
            if self.current_topic and self.current_topic != "general":
                state.current_topic = self.current_topic

    def sync_from_state(self, state) -> None:
        """把 TravelState 单轮结果回写到 MemoryManager (权威源)。

        在 FinalVerifier 完成 / Planner 推断出新任务时调用,
        让下一轮 run_for_user() 能从 memory 读到最新的 task/destination/topic。
        """
        if state.current_task_id:
            self.current_task_id = state.current_task_id
        if state.current_destination:
            self.current_destination = state.current_destination
        if state.current_topic and state.current_topic != "general":
            self.current_topic = state.current_topic

    def new_task_id(self) -> str:
        """分配一个新任务 ID (用于新任务开启, 例如想去成都→上轮是重庆)。

        ID 格式: t_{session_id_suffix}_{seq} 避免不同 session 撞 ID。
        """
        self._task_seq += 1
        suffix = (self.session_id or "anon")[-8:]
        return f"t_{suffix}_{self._task_seq}"

    def mark_current_task_completed(self) -> None:
        """把当前 task 在索引中标记为已完成 (is_completed_task=True)。

        在 FinalVerifier 完成后调用, 让下一轮 retrieve_short_term_context()
        自动把该 task 归入 excluded_history, 避免污染新一轮。
        """
        if not self.current_task_id:
            return
        # 索引是用 LPUSH 顺序存储的 List, 倒序遍历找到第一条当前 task 的 user
        # 条目, 用 LSET 改 is_completed_task 标记。Redis 6.2+ 原生支持, 否则
        # 走读-改-写回退。InMemoryFallback 直接读-改-写。
        index = self.get_short_term_index()
        changed = False
        for i, entry in enumerate(index):
            if (entry.get("task_id") == self.current_task_id
                    and entry.get("role") == "user"):
                entry["is_completed_task"] = True
                changed = True
                # 只标记第一条 user (任务开启那条)
                break

        if changed:
            # 重建索引列表
            try:
                self.redis.delete(self.index_key)
            except Exception:
                pass
            for entry in index:
                self.redis.rpush(self.index_key,
                                 json.dumps(entry, ensure_ascii=False))
            self.redis.expire(self.index_key, self.ttl)
            # 任务完成后清空 task 指针和 destination, 下一轮若新目的地会分配新 task_id
            # 保留 destination 会让 bind_to_state 把旧目的地"复活"到新 state,
            # 看起来像"已完成的任务还在污染下一轮"。
            self.current_task_id = None
            self.current_destination = None
            self.current_topic = "general"

    def _next_msg_id(self) -> str:
        """生成会话内唯一的 msg_id。

        用 (毫秒时间戳, 自增序号) 拼接, 避免同一毫秒多次 add_message 撞 id,
        进而导致 history_by_id / 索引合并时把不同消息当成同一条处理。
        """
        self._msg_seq += 1
        return f"msg_{int(datetime.now().timestamp() * 1000)}_{self._msg_seq}"

    @staticmethod
    def _build_index_entry(*, msg_id: str, role: str, content: str,
                           task_id: str | None, topic: str,
                           destination: str | None, has_slots: bool,
                           extracted_slots: dict[str, Any] | None,
                           is_completed_task: bool) -> dict[str, Any] | None:
        """构建一条短期索引条目。

        设计原则:
        - 索引只保留业务级结构化字段, 不存原始长文本。
        - 不做向量, 不做 embedding, 只做业务键匹配。
        - destination 缺省时尝试从消息中常见城市名简单抽取, 仅作为提示。
        """
        return {
            "msg_id": msg_id,
            "role": role,
            "content_preview": content[:50],
            "task_id": task_id,
            "topic": topic or "general",
            "destination": destination,
            "has_slots": has_slots,
            "extracted_slots": extracted_slots or {},
            "is_completed_task": is_completed_task,
        }

    # ==================== 桶5: Working Memory (Redis Summary) ====================
    def update_short_term_summary(self, new_constraints: dict[str, str]):
        """更新当前行程的硬约束（兼容旧调用；主路径已不再依赖该 summary）。"""
        # 先清再写，避免旧 destination/days/budget 等字段残留到新请求。
        self.redis.delete(self.summary_key)
        if new_constraints:
            self.redis.hset(self.summary_key, mapping=new_constraints)
            self.redis.expire(self.summary_key, self.ttl)

    def get_short_term_summary(self) -> dict[str, str]:
        """获取当前行程的核心约束字典"""
        summary = self.redis.hgetall(self.summary_key)
        return {k: v for k, v in summary.items()} if summary else {}

    # ==================== 桶6: Semantic Memory (MySQL KV 长期偏好) ====================
    def save_user_memory(self, memory_key: str, memory_value: str,
                         context_tag: str = "global", scope: str = "long_term"):
        """写入 MySQL KV 长期偏好库（支持作用域隔离）
        scope: 'permanent' 永久约束(过敏/疾病), 'long_term' 长期偏好, 'trip_scoped' 本次行程临时约束
        """
        # trip_scoped 绝不能写入 global，否则会被下一轮全局 profile 读出造成行程参数泄露。
        if scope == "trip_scoped" and context_tag == "global":
            context_tag = self.session_id

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = self._get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO user_memory (user_id, memory_key, memory_value,
                                             context_tag, scope, last_updated)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        memory_value = VALUES(memory_value),
                        scope = VALUES(scope),
                        last_updated = VALUES(last_updated)
                """, (self.user_id, memory_key, memory_value, context_tag, scope, timestamp))
            conn.commit()
        except Exception as e:
            print(f"[MySQL] save_user_memory failed: {e}")
            conn.rollback()

    def get_user_profile(self, context_tag: str = None) -> str:
        """从 MySQL 提取用户长期特征（支持作用域过滤）
        若提供 context_tag，返回该 context + global + permanent scope 的记忆。
        不提供则返回 global + permanent scope 的记忆。
        trip_scoped 记忆仅在 context_tag 完全匹配时返回。
        """
        try:
            conn = self._get_db_connection()
            with conn.cursor() as cursor:
                if context_tag:
                    cursor.execute(
                        """SELECT memory_key, memory_value, scope, context_tag
                           FROM user_memory
                           WHERE user_id = %s
                             AND (
                               scope = 'permanent'
                               OR (scope = 'long_term' AND context_tag = 'global')
                               OR (scope = 'trip_scoped' AND context_tag = %s)
                             )
                           ORDER BY
                             CASE scope WHEN 'permanent' THEN 0 WHEN 'long_term' THEN 1 ELSE 2 END,
                             last_updated DESC""",
                        (self.user_id, context_tag)
                    )
                else:
                    cursor.execute(
                        """SELECT memory_key, memory_value, scope, context_tag
                           FROM user_memory
                           WHERE user_id = %s
                             AND (scope = 'permanent' OR (scope = 'long_term' AND context_tag = 'global'))
                           ORDER BY
                             CASE scope WHEN 'permanent' THEN 0 WHEN 'long_term' THEN 1 ELSE 2 END,
                             last_updated DESC""",
                        (self.user_id,)
                    )
                rows = cursor.fetchall()
            if not rows:
                return "暂无长期偏好"
            profile_lines = []
            for row in rows:
                key = row.get("memory_key")
                value = row.get("memory_value")
                scope = row.get("scope", "long_term")
                scope_label = {"permanent": "[永久]", "trip_scoped": "[本次]", "long_term": ""}.get(scope, "")
                profile_lines.append(f"- {key}{scope_label}: {value}")
            return "\n".join(profile_lines)
        except Exception as e:
            print(f"[MySQL] get_user_profile failed: {e}")
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
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-5:]])
        
        prompt = f"""
        请分析以下对话，提取出属于【当前具体行程】的最新硬性指标（如目的地、出行天数、出行预算、特定的临时偏好等）。
        仅提取属于本次行程的临时限制或规划目标，忽略用户的通用长期习惯。
        必须以纯 JSON 格式输出，不要包含 ```json 等任何 Markdown 标记或多余的文字解释。
        格式样例（仅演示 JSON 结构，严禁照抄占位符）：
        {{"destination": "<目的地>", "days": "<天数>", "budget": "<预算>", "preferences": "<临时偏好>"}}
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
        核心流转 B：Episodic -> Semantic Memory。

        只沉淀真正长期/永久偏好；目的地、天数、预算、人数、日期等一次性行程参数
        不得进入 global profile，避免下一次请求被旧行程污染。
        """
        history = self.get_chat_history()
        if not history:
            return

        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-20:]])

        prompt = f"""你是一个严格的用户长期画像分析师。
以下是用户近期原始对话（未蒸馏）：
{history_text}

请只提取【跨行程也长期成立】或【健康安全永久成立】的信息。

允许保存：
1. permanent：过敏、疾病、行动限制、宗教/健康禁忌等。
2. long_term：反复出现或用户明确表达为长期偏好的饮食口味、住宿偏好、交通偏好、节奏偏好。

禁止保存：
- 具体目的地、城市、景点
- 出行天数、日期、预算、人数、出发地
- 本次行程安排、酒店、交通班次等一次性参数

输出 JSON 数组，例如：
[
  {{"key": "allergy", "value": "<过敏信息>", "scope": "permanent"}},
  {{"key": "travel_pace", "value": "<长期节奏偏好>", "scope": "long_term"}}
]
如果没有可提取的长期/永久信息，输出 []。
"""
        deny_keys = {
            "destination", "city", "location", "days", "dates", "date", "budget", "headcount",
            "group_size", "people", "origin_city", "hotel", "attraction", "route", "itinerary"
        }
        deny_value_pattern = re.compile(r"(\d+\s*天|\d+\s*(人|位|元|块)|预算|目的地|出发地|景点|酒店|行程)")

        try:
            response = llm.call([{"role": "user", "content": prompt}]).strip()
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                profile_list = json.loads(json_match.group(0))
                for item in profile_list:
                    key = (item.get("key") or "").strip()
                    value = (item.get("value") or "").strip()
                    scope = (item.get("scope") or "long_term").strip()
                    if not key or not value:
                        continue
                    if key.lower() in deny_keys or deny_value_pattern.search(value):
                        print(f"[Memory Conversion] drop trip-scoped semantic candidate: {key}={value}")
                        continue
                    if scope not in ("permanent", "long_term"):
                        # 本次行程临时参数不进入 global profile；如需保留，也必须绑 session_id。
                        self.save_user_memory(key, value, context_tag=self.session_id, scope="trip_scoped")
                    else:
                        self.save_user_memory(key, value, context_tag="global", scope=scope)
        except Exception as e:
            print(f"[Memory Conversion] Conversion to Semantic failed: {e}")

    def retrieve_short_term_context(self, current_message: str) -> dict[str, Any]:
        """
        基于短期索引 (session:{sid}:index) 检索与当前输入相关的上下文。

        返回结构:
        {
            "current_message": ...,
            "current_task_id": ... | None,
            "relevant_turns": [原文 messages],          # 当前任务相关, 可注入 Planner
            "excluded_history": ["重庆3天3000..."],    # 旧任务 / 不同目的地的历史, 不得作为当前事实
            "last_assistant_question": "...",          # 最近一条 assistant 追问
            "is_invalid_reply": bool,                  # 当前 message 是否像无效回复 (ff/asd/...)
            "is_delegation": bool,                     # 当前 message 是否为授权默认
            "recent_turns": [原文 messages],           # 最近少量原文, 仅做兜底
        }

        检索规则 (轻量, 无向量, 无 LLM):
        1. current_task_id 优先: 若当前 message 命中已有 task_id, 沿用;
           否则若 message 中含新目的地, 视为新任务, 沿用最近 task_id 中目的地不变者;
           再否则使用最近非 None 的 task_id (例如回答上一轮追问/无效回复/授权默认)。
        2. relevant_turns 召回: 同一 task_id 的原文 + 同 destination 的最近追问。
        3. excluded_history: 索引中 is_completed_task=True 的条目, 或 destination != current_destination 的旧任务条目。
        """
        index = self.get_short_term_index()
        history = self.get_chat_history()
        history_by_id = {h.get("msg_id"): h for h in history if h.get("msg_id")}

        # 找最近一条非 None task_id
        last_task_id = None
        for entry in reversed(index):
            if entry.get("task_id"):
                last_task_id = entry["task_id"]
                break

        # 当前 task 判定: 默认继承最近 task_id (回答追问 / 授权默认 / 无效回复都继承)
        current_task_id = last_task_id

        # 当前 destination 判定
        current_destination = None
        if index:
            for entry in reversed(index):
                if entry.get("task_id") == current_task_id and entry.get("destination"):
                    current_destination = entry["destination"]
                    break
        if current_destination is None:
            for entry in reversed(index):
                if entry.get("destination"):
                    current_destination = entry["destination"]
                    break

        msg_norm = (current_message or "").strip().lower()
        invalid_tokens = {"ff", "??", "。。。", "不知道", "随便说"}
        delegation_tokens = {"看你安排", "你安排", "随便", "默认", "你来", "看着办"}
        is_invalid = len(msg_norm) <= 2 or msg_norm in invalid_tokens
        is_delegation = msg_norm in delegation_tokens

        # 相关 turns: 同一 task_id 的索引条目
        relevant_turns: list[dict[str, str]] = []
        for entry in index:
            if entry.get("task_id") and entry.get("task_id") == current_task_id:
                raw = history_by_id.get(entry.get("msg_id"))
                if raw is not None:
                    relevant_turns.append(raw)

        # recent_turns 仅作为同 task_id 的兜底, 避免污染
        recent_turns = relevant_turns[-6:] if relevant_turns else []

        # 旧任务排除清单 (用预览, 不返回完整原文, 避免污染)
        excluded: list[str] = []
        for entry in index:
            if entry.get("is_completed_task"):
                excluded.append(
                    f"{entry.get('destination') or '未知'}: {entry.get('content_preview')}"
                )
            elif (
                entry.get("task_id")
                and current_destination
                and entry.get("destination")
                and entry["destination"] != current_destination
            ):
                excluded.append(
                    f"{entry.get('destination')}: {entry.get('content_preview')}"
                )

        # 最近 assistant 追问
        last_question = ""
        for entry in reversed(index):
            if entry.get("role") == "assistant" and entry.get("topic") in ("ask_user", "ask_slots"):
                msg = history_by_id.get(entry.get("msg_id"))
                if msg:
                    last_question = msg.get("content", "")
                break

        return {
            "current_message": current_message,
            "current_task_id": current_task_id,
            "current_destination": current_destination,
            "relevant_turns": relevant_turns,
            "recent_turns": recent_turns,
            "excluded_history": excluded,
            "last_assistant_question": last_question,
            "is_invalid_reply": is_invalid,
            "is_delegation": is_delegation,
        }

    def get_global_context_prompt(self, current_message: str = "") -> str:
        """
        组装规划前上下文 Prompt (供旅游意图时使用)。

        不再无差别拼接最近 20 条原文, 改为基于 retrieve_short_term_context() 的
        检索结果渲染, 显式隔离旧任务。
        """
        parts: list[str] = []

        # 长期偏好
        profile = self.get_user_profile()
        if profile != "暂无长期偏好":
            parts.append(f"【用户长期偏好画像】：\n{profile}")

        if current_message:
            ctx = self.retrieve_short_term_context(current_message)

            # 当前任务槽位 (来自 relevant_turns 中 extracted_slots 合并)
            merged_slots: dict[str, Any] = {}
            for raw in ctx["relevant_turns"]:
                # 找对应索引条目
                for entry in self.get_short_term_index():
                    if entry.get("msg_id") == raw.get("msg_id"):
                        merged_slots.update(entry.get("extracted_slots") or {})
                        break
            if merged_slots:
                slot_lines = "\n".join(f"- {k}: {v}" for k, v in merged_slots.items())
                parts.append(f"【当前任务已知槽位】：\n{slot_lines}")

            # 相关近期对话
            if ctx["recent_turns"]:
                history_text = "\n".join(
                    f"{msg.get('role', '')}: {msg.get('content', '')}"
                    for msg in ctx["recent_turns"]
                )
                parts.append(f"【相关近期对话】：\n{history_text}")

            # 上一轮追问
            if ctx["last_assistant_question"]:
                parts.append(f"【上一轮追问】：\n{ctx['last_assistant_question']}")

            # 旧任务排除清单 (防止 Planner 误复用)
            if ctx["excluded_history"]:
                excluded_text = "\n".join(f"- {line}" for line in ctx["excluded_history"])
                parts.append(
                    f"【历史任务参考, 不得作为当前任务事实】：\n{excluded_text}\n"
                    "注意: 上述历史只用于了解用户风格, 其中的 destination/days/budget/people "
                    "等一次性参数禁止自动填入当前任务。"
                )

            # 标志位
            flags: list[str] = []
            if ctx["is_invalid_reply"]:
                flags.append("当前回复像无效短词, 需重新引导用户补充有效信息。")
            if ctx["is_delegation"]:
                flags.append("用户已授权默认安排, 可以在 assumptions 中给常规默认值。")
            if flags:
                parts.append("【对话状态】：" + " ".join(flags))

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