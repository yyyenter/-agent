from re import T
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

# --- 2. 持久化初始化 ---
# 确保目录存在
os.makedirs("knowledge", exist_ok=True)
chroma_client = chromadb.PersistentClient(path="knowledge/chroma_db")
collection = chroma_client.get_or_create_collection(
    name="agent_long_term_memory",
    embedding_function=OllamaEmbeddingFunction()
)

history_store = InMemoryChatMessageHistory()

class ReadMemoryInput(BaseModel):
    query: str = Field(..., description="检索关键词，用于寻找历史偏好")
    user_id: str = Field(..., description="当前对话的用户 ID，用于隔离查询") # ✅ 新增

class ReadMemoryTool(BaseTool):
    name: str = "read_memory_tool"
    description: str = "任务开始时，必须调用此工具检索指定用户的历史对话或偏好。"
    args_schema: Type[BaseModel] = ReadMemoryInput

    def _run(self, query: str, user_id: str) -> str: # ✅ 接收 user_id
        # 使用 where 条件实现真正的隔离！
        results = collection.query(
            query_texts=[query], 
            n_results=1000,
            where={"user_id": user_id} 
        )
        if not results['documents'] or not results['documents'][0]:
            return f"未找到用户 {user_id} 的相关历史记录。"
        
        formatted = [f"[{r['role']}] {doc}" for doc, r in zip(results['documents'][0], results['metadatas'][0])]
        return f"检索到用户 {user_id} 的历史记忆：\n" + "\n".join(formatted)


class SaveMemoryInput(BaseModel):
    content: str = Field(..., description="需要持久化保存的分析结论")
    user_id: str = Field(..., description="当前对话的用户 ID") # ✅ 新增

class SaveMemoryTool(BaseTool):
    name: str = "save_memory_tool"
    description: str = "获得关键结论后，必须调用此工具将信息持久化到指定用户库。"
    args_schema: type[BaseModel] = SaveMemoryInput
    fixed_role: str = "assistant"

    def _run(self, content: str, user_id: str) -> str: # ✅ 接收 user_id
        timestamp = str(datetime.now())
        collection.add(
            documents=[content],
            ids=[str(uuid.uuid4())],
            metadatas=[{"role": self.fixed_role, "timestamp": timestamp, "user_id": user_id}] # ✅ 写入隔离标签
        )
        return "关键信息已成功保存至持久化库。"



