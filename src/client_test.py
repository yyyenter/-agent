import requests
import json


def test_agent(message: str):
    url = "http://127.0.0.1:8000/api/chat_stream"
    payload = {
        "user_id": "test_user_888",
        "session_id": "session_001",
        "message": message,
    }

    print(f"\n提问: {message}")
    try:
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data_str = line[len("data: "):]
            try:
                msg = json.loads(data_str)
            except json.JSONDecodeError:
                print(f"  [原始] {data_str}")
                continue

            msg_type = msg.get("type")
            content = msg.get("content", "")
            if msg_type == "status":
                print(f"  [状态] {content}")
            elif msg_type == "finish":
                print(f"  [完成] {content}")
            elif msg_type == "error":
                print(f"  [错误] {content}")
    except Exception as e:
        print(f"请求失败: {e}")


if __name__ == "__main__":
    test_cases = [
        "你好啊",
        # "我想去徐州吃烧烤",
    ]

    for case in test_cases:
        test_agent(case)
