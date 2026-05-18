"""GLM 智谱 API 连通性测试"""
import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GLM_API_KEY")
API_BASE = os.getenv("GLM_API_BASE")
MODEL_NAME = os.getenv("GLM_MODEL_NAME") or "glm-4-flash"

if not API_KEY or not API_BASE:
    print("[FAIL] 未配置 GLM_API_KEY 或 GLM_API_BASE 环境变量")
    exit(1)


def test_glm_api() -> bool:
    """测试 GLM API 调用"""
    url = f"{API_BASE}/chat/completions"
    headers = {"Authorization": API_KEY, "Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "你好，请简短回复"}],
        "temperature": 0.7,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
    except requests.exceptions.ConnectionError as e:
        print(f"  [FAIL] 连接失败: {e}")
        return False
    except requests.exceptions.Timeout:
        print(f"  [FAIL] 请求超时")
        return False

    print(f"  HTTP {resp.status_code}, Content-Type: {resp.headers.get('Content-Type', 'N/A')}")

    if not resp.text.strip():
        print(f"  [FAIL] 响应体为空")
        return False

    try:
        data = resp.json()
    except json.JSONDecodeError:
        print(f"  [FAIL] 响应不是合法 JSON: {resp.text[:200]}")
        return False

    choices = data.get("choices", [])
    if not choices:
        print(f"  [FAIL] 无返回 choices: {json.dumps(data, ensure_ascii=False)[:200]}")
        return False

    content = choices[0].get("message", {}).get("content", "")
    print(f"  [OK] 模型={MODEL_NAME}")
    print(f"  [OK] 响应: {content[:100]}")
    return True


if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=" * 60)
    print("GLM 智谱 API 连通性测试")
    print(f"API Base: {API_BASE}")
    print(f"API Key : {API_KEY[:4]}...{API_KEY[-4:] if len(API_KEY)>4 else ''}")
    print(f"Model   : {MODEL_NAME}")
    print("=" * 60)

    if test_glm_api():
        print("\n" + "=" * 60)
        print("结果: GLM API 调用成功")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("结果: GLM API 调用失败")
        print("=" * 60)
