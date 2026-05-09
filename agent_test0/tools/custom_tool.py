from re import T
import sqlite3
from typing import Any, Type
from crewai_tools.tools.jina_scrape_website_tool.jina_scrape_website_tool import JinaScrapeWebsiteTool
from datetime import datetime
import json
import os
import uuid
from chromadb import Documents, EmbeddingFunction, Embeddings
import chromadb
from crewai.rag.embeddings.providers.custom.embedding_callable import CustomEmbeddingFunction
from crewai.tools import BaseTool
from langchain_core.chat_history import InMemoryChatMessageHistory
import ollama
from pydantic import BaseModel, Field
import requests

from agent_test0.main import MemoryManager
search_tool: JinaScrapeWebsiteTool = JinaScrapeWebsiteTool()

class weatherReport(BaseModel):
    title: str
    temp: int

class MyCustomToolInput(BaseModel):
    """Input schema for MyCustomTool."""
    argument: str = Field(..., description="Description of the argument.")

class MyCustomTool(BaseTool):
    name: str = "Name of my tool"
    description: str = (
        "Clear description for what this tool is useful for, your agent will need this information to use it."
    )
    args_schema: Type[BaseModel] = MyCustomToolInput

    def _run(self, argument: str) -> str:
        # Implementation goes here
        return "this is an example of a tool output, ignore it and move along."

class VannaInput(BaseModel):
    message: str = Field(..., description="要查询的自然语言问题")

class VannaQueryTool(BaseTool):
    name: str = "vanna_query_tool"
    description: str = "用于查询数据库信息"
    args_schema: Type[BaseModel] = VannaInput

    def _run(self, message: str) -> str:
        # ！！！核心：只拿 Vanna 的钥匙 ！！！
        vanna_key = os.getenv("VANNA_API_KEY")
        agent_id = os.getenv("VANNA_AGENT_ID")
        
        # 拼接请求发给 Vanna
        url = "https://app.vanna.ai/api/v2/chat_sse"
        headers = {"VANNA-API-KEY": vanna_key, "Content-Type": "application/json"}
        # ... 后续发送请求的逻辑 ...
        return "Vanna的查询结果"

class WeatherInput(BaseModel):
    location: str = Field(..., description="需要查询天气的城市名称")

class WeatherTool(BaseTool):
    name: str="GetWeatherTool"
    description:str="当你需要查询某个城市的实时天气时，请调用此工具。"
    args_schema: type[BaseModel]=WeatherInput
    def _run(self, location: str) -> str:
        # 1. 尝试从 Redis 缓存获取结果
        cached = MemoryManager.get_tool_result("qweather", {"loc": location})
        if cached: return cached

        # 2. 调用和风天气 (QWeather) API
        api_key = os.getenv("QWEATHER_API_KEY")
        # 先通过城市名获取 Location ID (这里建议直接搜或者传 ID，示例用最直接的查询)
        url = "https://devapi.qweather.com/v7/weather/now"
        
        # 首先需要把地名转为坐标/ID，这里假设传的是 ID 或直接搜地名（QWeather 支持模糊地名）
        params = {"location": location, "key": api_key, "lang": "zh"}
        
        try:
            resp = requests.get(url, params=params)
            data = resp.json()
            if data["code"] != "200": return f"查询失败: {data['code']}"
            
            now = data["now"]
            clean_res = {
                "城市": location,
                "天气": now["text"],
                "温度": f"{now['temp']}°C",
                "体感": f"{now['feelsLike']}°C",
                "湿度": f"{now['humidity']}%",
                "风向": now["windDir"],
                "更新时间": now["obsTime"]
            }
            res_str = json.dumps(clean_res, ensure_ascii=False)
            
            # 3. 写入缓存 (保存 30 分钟)
            MemoryManager.set_tool_result("qweather", {"loc": location}, res_str, 1800)
            return res_str
        except Exception as e:
            return f"天气查询出错: {str(e)}"

# --- 1. 本地嵌入函数 (Ollama) ---
class OllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name="nomic-embed-text"):  # pyright: ignore[reportMissingSuperCall]
        self.model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            response = ollama.embeddings(model=self.model_name, prompt=text)
            embeddings.append(response["embedding"])
        return embeddings

# ==========================================
# 长期记忆数据库初始化 (使用 SQLite 模拟结构化 DB)
# ==========================================
DB_PATH = "knowledge/user_profiles.db"
os.makedirs("knowledge", exist_ok=True)

def init_db():
    """初始化用户长期画像表"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                preferences TEXT,        -- 用户的通用偏好 (如：喜欢自然风光)
                dietary_rules TEXT,      -- 饮食禁忌 (如：不吃海鲜)
                system_notes TEXT,       -- AI 总结的其他特征
                last_updated TIMESTAMP
            )
        """)
init_db()

# ==========================================
# Tool 1: 灵活读取记忆
# ==========================================
class ReadMemoryInput(BaseModel):
    user_id: str = Field(..., description="当前对话的用户 ID")

class ReadMemoryTool(BaseTool):
    name: str = "read_memory_tool"
    description: str = "任务开始时，必须调用此工具检索指定用户的所有历史偏好和特征画像。"
    args_schema: type[BaseModel] = ReadMemoryInput

    def _run(self, user_id: str) -> str:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute(
                "SELECT memory_key, memory_value FROM user_memory WHERE user_id = ?", 
                (user_id,)
            )
            rows = cursor.fetchall()
            
        if not rows:
            return f"未找到用户 {user_id} 的历史记忆，请将其视为新用户。"
        
        # 动态组装所有查到的 Key-Value
        profile_lines = [f"- {row[0]}: {row[1]}" for row in rows]
        profile_text = "\n".join(profile_lines)
        
        return f"【用户 {user_id} 的长期画像】\n{profile_text}\n请在规划本次行程时，严格遵守上述特征！"

# ==========================================
# Tool 2: 动态保存记忆 (不再校验固定 Category)
# ==========================================
class SaveMemoryInput(BaseModel):
    user_id: str = Field(..., description="当前对话的用户 ID")
    memory_key: str = Field(..., description="提炼出的偏好维度，使用英文下划线命名法，例如：'dietary_habit', 'favorite_transport', 'budget_preference'")
    memory_value: str = Field(..., description="具体的偏好内容，例如：'不吃香菜', '尽量避免红眼航班'")

class SaveMemoryTool(BaseTool):
    name: str = "save_memory_tool"
    description: str = "如果在对话中发现了用户新的硬性偏好或禁忌，提取维度名称(key)和具体内容(value)，调用此工具持久化。"
    args_schema: type[BaseModel] = SaveMemoryInput

    def _run(self, user_id: str, memory_key: str, memory_value: str) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with sqlite3.connect(DB_PATH) as conn:
            # SQLite 的 UPSERT 语法：遇到主键冲突则更新，否则插入
            conn.execute("""
                INSERT INTO user_memory (user_id, memory_key, memory_value, last_updated)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, memory_key) 
                DO UPDATE SET memory_value = memory_value || '；' || excluded.memory_value, 
                              last_updated = excluded.last_updated
            """, (user_id, memory_key, memory_value, timestamp))
                
        return f"✅ 成功将【{memory_key}: {memory_value}】保存至用户【{user_id}】的长期记忆中。"
