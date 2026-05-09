from typing import Type
from crewai_tools.tools.jina_scrape_website_tool.jina_scrape_website_tool import JinaScrapeWebsiteTool
import os
import json
import sqlite3
import requests
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

# 1. 引用全局搜索工具
search_tool: JinaScrapeWebsiteTool = JinaScrapeWebsiteTool()
DB_PATH = "knowledge/user_profiles.db"

class WeatherInput(BaseModel):
    location: str = Field(..., description="需要查询天气的城市名称")

class WeatherTool(BaseTool):
    name: str = "GetWeatherTool"
    description: str = "当你需要查询某个城市的实时天气时，请调用此工具。"
    args_schema: type[BaseModel] = WeatherInput

    def _run(self, location: str) -> str:
        import redis
        from harness import ToolCacheManager
        redis_client = redis.Redis(host='localhost', port=6373, db=0, decode_responses=True)
        
        # 1. 先查缓存
        cached = ToolCacheManager.get_tool_result(redis_client, "qweather", {"loc": location})
        if cached:
            return f"[Cache Hit] {cached}"

        # 2. 调用 QWeather API
        api_key = os.getenv("QWEATHER_API_KEY")
        url = "https://devapi.qweather.com/v7/weather/now"
        params = {"location": location, "key": api_key, "lang": "zh"}
        
        try:
            resp = requests.get(url, params=params)
            data = resp.json()
            if data["code"] != "200":
                return f"天气查询失败: {data['code']}"
            
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
            
            # 3. 写入缓存 (1800 秒 / 30分钟)
            ToolCacheManager.set_tool_result(redis_client, "qweather", {"loc": location}, res_str, 1800)
            return res_str
        except Exception as e:
            return f"天气查询出错: {str(e)}"


# ==================== 长期记忆工具 (SQLite KV 架构) ====================

class ReadMemoryInput(BaseModel):
    user_id: str = Field(..., description="当前对话的用户 ID，用于提取该用户的长期偏好")

class ReadMemoryTool(BaseTool):
    name: str = "read_memory_tool"
    description: str = "任务开始时，必须调用此工具检索指定用户的所有历史偏好、特征画像或严重禁忌。"
    args_schema: type[BaseModel] = ReadMemoryInput

    def _run(self, user_id: str) -> str:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute(
                "SELECT memory_key, memory_value FROM user_memory WHERE user_id = ?", 
                (user_id,)
            )
            rows = cursor.fetchall()
            
        if not rows:
            return f"未找到用户 {user_id} 的任何历史长期偏好，请将其视为新用户。"
        
        profile_lines = [f"- {row[0]}: {row[1]}" for row in rows]
        profile_text = "\n".join(profile_lines)
        
        return f"【用户 {user_id} 的长期偏好画像】：\n{profile_text}\n规划行程时，请绝对且无条件满足上述特质！"


class SaveMemoryInput(BaseModel):
    user_id: str = Field(..., description="当前对话的用户 ID")
    memory_key: str = Field(..., description="提炼出的特征维度（英文下划线，如 dietary_restrictions, favorite_transport, physical_limits）")
    memory_value: str = Field(..., description="具体的特征描述，如 '对芒果严重过敏', '腰不好不能爬山'")

class SaveMemoryTool(BaseTool):
    name: str = "save_memory_tool"
    description: str = "如果在对话中发现了用户新暴露的通用、硬性、长久特征（偏好或禁忌），提炼 key 和 value 调用此工具持久化。"
    args_schema: type[BaseModel] = SaveMemoryInput

    def _run(self, user_id: str, memory_key: str, memory_value: str) -> str:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO user_memory (user_id, memory_key, memory_value, last_updated)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, memory_key) 
                DO UPDATE SET memory_value = memory_value || '；' || excluded.memory_value, 
                              last_updated = excluded.last_updated
            """, (user_id, memory_key, memory_value, timestamp))
                
        return f"✅ 成功将用户【{user_id}】的长期特征【{memory_key}: {memory_value}】保存到 SQLite 数据库中。"