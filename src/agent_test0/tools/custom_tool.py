from typing import Type
from crewai_tools.tools.jina_scrape_website_tool.jina_scrape_website_tool import JinaScrapeWebsiteTool
import os
import json
import requests
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
import redis
import pymysql
# 假设 ToolCacheManager 在你的项目中正确导入
from ..memory import ToolCacheManager

# ==================== 工具输出压缩 ====================
MAX_TOOL_OUTPUT_CHARS = 200

def compress_tool_result(result: str, max_chars: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    """两阶段处理：完整数据已入缓存，仅将压缩版喂给 LLM"""
    if len(result) <= max_chars:
        return result

    # 尝试 JSON 解析 → 提取核心字段
    try:
        data = json.loads(result)
        if isinstance(data, dict):
            # 保留前 3 个关键字段
            key_fields = list(data.items())[:3]
            short = ", ".join(f"{k}: {v}" for k, v in key_fields)
            if len(short) <= max_chars:
                return f"{short} [完整数据已缓存]"
    except (json.JSONDecodeError, TypeError):
        pass

    # 文本：在句子/换行边界截断
    truncated = result[:max_chars]
    last_break = max(truncated.rfind("。"), truncated.rfind("\n"), truncated.rfind(". "))
    if last_break > max_chars // 2:
        truncated = truncated[:last_break + 1]

    omitted = len(result) - len(truncated)
    return f"{truncated}\n[完整数据已缓存，此处为摘要，省略 {omitted} 字符]"

# 1. 引用全局搜索工具
search_tool: JinaScrapeWebsiteTool = JinaScrapeWebsiteTool()


def get_mysql_connection():
    """获取 MySQL 连接"""
    host = os.getenv("MYSQL_HOST", "localhost")
    port = int(os.getenv("MYSQL_PORT", "3306"))
    user = os.getenv("MYSQL_USER", "root")
    password = os.getenv("MYSQL_PASSWORD", "")
    database = os.getenv("MYSQL_DATABASE", "agent_test0")

    return pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )


class WeatherInput(BaseModel):
    location: str = Field(..., description="需要查询天气的城市名称，例如：北京、上海")


class WeatherTool(BaseTool):
    name: str = "GetWeatherTool"
    description: str = "当你需要查询某个城市的实时天气时，请调用此工具。"
    args_schema: type[BaseModel] = WeatherInput

    def _run(self, location: str) -> str:
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

        # 1. 先查缓存 (基于用户输入的城市名作为 Key)
        cached = ToolCacheManager.get_tool_result(redis_client, "qweather", {"loc": location})
        if cached:
            return f"[Cache Hit] {cached}"

        api_key = os.getenv("QWEATHER_API_KEY")
        api_host = os.getenv("QWEATHER_API_HOST")
        if not api_key or not api_host:
            return "工具调用失败：未配置 QWEATHER_API_KEY 或 QWEATHER_API_HOST 环境变量"

        try:
            headers = {"X-QW-Api-Key": api_key}

            # ==========================================
            # 2. 调用 GeoAPI: 城市名称 -> LocationID
            # ==========================================
            geo_url = f"https://{api_host}/geo/v2/city/lookup"
            geo_params = {"location": location}

            geo_resp = requests.get(geo_url, params=geo_params, headers=headers)
            geo_data = geo_resp.json()

            if geo_data.get("code") != "200" or not geo_data.get("location"):
                return f"天气查询失败: 无法解析城市 '{location}' 的位置信息 (Code: {geo_data.get('code')})"

            # 提取排名第一的城市 LocationID 和标准城市名
            location_id = geo_data["location"][0]["id"]
            std_city_name = geo_data["location"][0]["name"] # 获取标准名字，比如"北京"

            # ==========================================
            # 3. 调用 WeatherAPI: LocationID -> 实时天气
            # ==========================================
            weather_url = f"https://{api_host}/v7/weather/now"
            weather_params = {"location": location_id, "lang": "zh"}

            weather_resp = requests.get(weather_url, params=weather_params, headers=headers)
            data = weather_resp.json()

            if data.get("code") != "200":
                return f"天气查询失败: 接口返回错误码 {data.get('code')}"

            now = data["now"]
            clean_res = {
                "城市": std_city_name, # 使用标准名称替代用户可能输入的别名
                "天气": now["text"],
                "温度": f"{now['temp']}°C",
                "体感": f"{now['feelsLike']}°C",
                "湿度": f"{now['humidity']}%",
                "风向": now["windDir"],
                "更新时间": now["obsTime"]
            }
            res_str = json.dumps(clean_res, ensure_ascii=False)

            # 4. 写入缓存 (18000 秒 / 5小时)，完整数据
            ToolCacheManager.set_tool_result(redis_client, "qweather", {"loc": location}, res_str, 18000)
            # 返回压缩版给 LLM
            return compress_tool_result(res_str)

        except Exception as e:
            return f"天气查询出错: {str(e)}"


# ==================== 长期记忆工具 (MySQL KV 架构) ====================

class ReadMemoryInput(BaseModel):
    user_id: str = Field(..., description="当前对话的用户 ID，用于提取该用户的长期偏好")
    context_tag: str = Field(default="global", description="行程上下文标签，用于隔离不同行程的记忆")


class ReadMemoryTool(BaseTool):
    name: str = "read_memory_tool"
    description: str = "检索指定用户在特定行程上下文中的历史偏好。context_tag 用于隔离不同行程的记忆。"
    args_schema: type[BaseModel] = ReadMemoryInput

    def _run(self, user_id: str, context_tag: str = "global") -> str:
        conn = get_mysql_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """SELECT memory_key, memory_value, scope
                       FROM user_memory
                       WHERE user_id = %s
                         AND (context_tag = %s OR context_tag = 'global' OR scope = 'permanent')
                       ORDER BY CASE scope WHEN 'permanent' THEN 0 WHEN 'long_term' THEN 1 ELSE 2 END""",
                    (user_id, context_tag)
                )
                rows = cursor.fetchall()
        finally:
            conn.close()

        if not rows:
            return f"未找到用户 {user_id} (context: {context_tag}) 的任何相关长期偏好。"

        profile_lines = [f"- {row['memory_key']}: {row['memory_value']}" for row in rows]
        profile_text = "\n".join(profile_lines)

        full_result = f"【用户 {user_id} 的长期偏好画像 (context: {context_tag})】：\n{profile_text}\n规划行程时，请绝对且无条件满足上述特质！"
        return compress_tool_result(full_result, max_chars=300)


class SaveMemoryInput(BaseModel):
    user_id: str = Field(..., description="当前对话的用户 ID")
    memory_key: str = Field(..., description="提炼出的特征维度（英文下划线，如 dietary_restrictions, favorite_transport, physical_limits）")
    memory_value: str = Field(..., description="具体的特征描述，如 '对芒果严重过敏', '腰不好不能爬山'")
    context_tag: str = Field(default="global", description="行程上下文标签，用于隔离不同行程的记忆")
    scope: str = Field(default="long_term", description="记忆作用域: permanent(永久如过敏), long_term(长期偏好), trip_scoped(本次行程)")


class SaveMemoryTool(BaseTool):
    name: str = "save_memory_tool"
    description: str = "持久化用户特征。scope='permanent'用于健康/过敏等不可变约束，'long_term'用于长期偏好，'trip_scoped'用于本次行程临时约束。"
    args_schema: type[BaseModel] = SaveMemoryInput

    def _run(self, user_id: str, memory_key: str, memory_value: str,
             context_tag: str = "global", scope: str = "long_term") -> str:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = get_mysql_connection()
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
                """, (user_id, memory_key, memory_value, context_tag, scope, timestamp))
            conn.commit()
        finally:
            conn.close()

        scope_label = {"permanent": "永久约束", "long_term": "长期偏好", "trip_scoped": "本次行程"}.get(scope, scope)
        return f"✅ 已保存 {scope_label}【{memory_key}: {memory_value}】到 context: {context_tag}"
