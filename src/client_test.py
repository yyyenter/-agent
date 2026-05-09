import requests
import json

def test_agent(message: str):
    url = "http://127.0.0.1:8000/api/chat"
    payload = {
        "user_id": "test_user_888",
        "session_id": "session_001",
        "message": message
    }
    
    print(f"\n提问: {message}")
    try:
        response = requests.post(url, json=payload)
        res_data = response.json()
        print(f"判定意图: {res_data.get('intent')}")
        print(f"AI 回复: {res_data.get('reply')}")
    except Exception as e:
        print(f"请求失败: {e}")

if __name__ == "__main__":
    # 你可以在这里一次性测试多个场景
    test_cases = [
        "你好啊", 
        # "我想去徐州吃烧烤", 
    ]
    
    for case in test_cases:
        test_agent(case)