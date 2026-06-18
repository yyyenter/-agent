import asyncio
import sys
import json
from pathlib import Path

# 把仓库根的 src/ 加入 sys.path（无论从哪个 cwd 运行都生效）
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from agent_test0.main import app
from fastapi.testclient import TestClient

# Mock the LLM to avoid API calls
class MockLLM:
    def call(self, messages):
        # Return a mock response for rewrite_query_lightweight
        return "帮我规划一个去北京的3天旅游行程"

# Mock the TravelWorkflow to simulate streaming
class MockTravelWorkflow:
    def __init__(self, status_callback=None, content_callback=None):
        self.status_callback = status_callback
        self.content_callback = content_callback
        self.state = type('State', (), {})()
        self.state.final_report = "最终审核结果：方案可行"
    
    def kickoff(self):
        # Simulate streaming messages
        if self.content_callback:
            self.content_callback("决策阶段完成：分析用户需求，确定目的地为北京", "planning")
            self.content_callback("执行阶段完成：生成旅游方案 - 第一天故宫，第二天长城，第三天天安门", "execution")
            self.content_callback("质检阶段完成：方案逻辑合理，符合要求", "validation")
        return self

# Monkey patch the imports
import agent_test0.main
agent_test0.main.zhipu_llm = MockLLM()
agent_test0.main.TravelWorkflow = MockTravelWorkflow

async def test_streaming_response():
    """测试流式响应功能"""
    client = TestClient(app)
    
    # 设置测试环境变量
    import os
    os.environ["GLM_API_KEY"] = "test_key"
    os.environ["GLM_API_BASE"] = "http://test.com"
    os.environ["GLM_MODEL_NAME"] = "test-model"
    
    # 发送流式请求
    response = client.post(
        "/api/chat_stream",
        json={
            "user_id": "test_user",
            "session_id": "test_session",
            "message": "帮我规划一个去北京的3天旅游行程"
        },
        headers={"Accept": "text/event-stream"}
    )
    
    print(f"Response status: {response.status_code}")
    print(f"Response headers: {dict(response.headers)}")
    
    if response.status_code == 200:
        # 解析 SSE 响应
        content = response.content.decode('utf-8')
        print(f"Raw response content:\n{content}")
        
        # 解析数据行
        lines = content.strip().split('\n')
        messages = []
        
        for line in lines:
            if line.startswith('data: '):
                try:
                    data = json.loads(line[6:])  # 去掉 'data: ' 前缀
                    messages.append(data)
                    print(f"Received: {data['type']} - {data['content'][:50]}...")
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON: {line} - Error: {e}")
        
        print(f"\nTotal messages received: {len(messages)}")
        for i, msg in enumerate(messages):
            print(f"{i+1}. {msg['type']}: {msg['content'][:100]}...")
            
        # 检查是否包含期望的消息类型
        message_types = [msg['type'] for msg in messages]
        expected_types = ['planning', 'execution', 'validation', 'finish']
        
        print(f"\nMessage types received: {message_types}")
        print(f"Expected types: {expected_types}")
        
        if all(t in message_types for t in expected_types):
            print("✅ 流式响应测试成功！所有期望的消息类型都已接收。")
        else:
            print("❌ 流式响应测试失败！缺少某些消息类型。")
            missing = [t for t in expected_types if t not in message_types]
            print(f"缺少的消息类型: {missing}")
    else:
        print(f"❌ 请求失败: {response.status_code}")
        print(f"Response content: {response.content.decode('utf-8')}")

if __name__ == "__main__":
    asyncio.run(test_streaming_response())