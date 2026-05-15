"""和风天气 API 连通性测试 — 适配 2026 新版独享 Host 与 Header 鉴权"""
import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("QWEATHER_API_KEY")
API_HOST = os.getenv("QWEATHER_API_HOST")

if not API_KEY or not API_HOST:
    print("[FAIL] 未配置 QWEATHER_API_KEY 或 QWEATHER_API_HOST 环境变量")
    exit(1)

TEST_CITIES = ["北京", "上海", "广州"]


def test_geo_api(city: str) -> dict | None:
    """步骤1: 城市名称 -> LocationID"""
    # 新版独享 Host 的 Geo 路径必须包含 /geo 前缀
    url = f"https://{API_HOST}/geo/v2/city/lookup"
    headers = {"X-QW-Api-Key": API_KEY}
    params = {"location": city}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
    except requests.exceptions.ConnectionError as e:
        print(f"  [FAIL] 连接失败: {e}")
        return None
    except requests.exceptions.Timeout:
        print(f"  [FAIL] 请求超时")
        return None

    print(f"  HTTP {resp.status_code}, Content-Type: {resp.headers.get('Content-Type', 'N/A')}")
    
    if not resp.text.strip():
        print(f"  [FAIL] 响应体为空 — 请检查 API_HOST 是否填写正确")
        return None

    try:
        data = resp.json()
    except json.JSONDecodeError:
        print(f"  [FAIL] 响应不是合法 JSON: {resp.text[:200]}")
        return None

    if data.get("code") != "200" or not data.get("location"):
        print(f"  [FAIL] GeoAPI code={data.get('code')}")
        return None

    loc = data["location"][0]
    print(f"  [OK] {city} -> LocationID={loc['id']}, 标准名={loc['name']}, 所属={loc.get('adm2', 'N/A')}")
    return {"id": loc["id"], "name": loc["name"]}


def test_weather_api(loc_info: dict) -> bool:
    """步骤2: LocationID -> 实时天气"""
    url = f"https://{API_HOST}/v7/weather/now"
    headers = {"X-QW-Api-Key": API_KEY}
    params = {"location": loc_info["id"], "lang": "zh"}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
    except requests.exceptions.ConnectionError as e:
        print(f"  [FAIL] WeatherAPI 连接失败: {e}")
        return False
    except requests.exceptions.Timeout:
        print(f"  [FAIL] WeatherAPI 请求超时")
        return False

    if not resp.text.strip():
        print(f"  [FAIL] WeatherAPI 响应体为空")
        return False

    try:
        data = resp.json()
    except json.JSONDecodeError:
        print(f"  [FAIL] WeatherAPI 响应不是合法 JSON: {resp.text[:200]}")
        return False

    if data.get("code") != "200":
        print(f"  [FAIL] WeatherAPI code={data.get('code')}: {json.dumps(data, ensure_ascii=False)[:200]}")
        return False

    now = data["now"]
    print(f"  [OK] {loc_info['name']}: {now['text']}, {now['temp']}°C "
          f"(体感 {now['feelsLike']}°C), 湿度 {now['humidity']}%, "
          f"{now['windDir']}风, 更新于 {now['obsTime']}")
    return True


if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=" * 60)
    print("和风天气 API 连通性测试 (新版)")
    print(f"API Host: {API_HOST}")
    print(f"API Key : {API_KEY[:4]}...{API_KEY[-4:] if len(API_KEY)>4 else ''}")
    print("=" * 60)

    passed = 0
    for city in TEST_CITIES:
        print(f"\n>> 测试: {city}")
        loc = test_geo_api(city)
        if loc and test_weather_api(loc):
            passed += 1

    print(f"\n{'=' * 60}")
    print(f"结果: {passed}/{len(TEST_CITIES)} 通过")
    print("=" * 60)