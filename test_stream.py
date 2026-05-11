import requests
import json

# 测试流式 API
url = 'http://127.0.0.1:8001/api/chat_stream'
data = {
    'user_id': 'test_user',
    'session_id': 'test_session',
    'message': '帮我规划一个去北京的3天旅游行程'
}

print('发送请求...')
response = requests.post(url, json=data, stream=True)

print('接收流式响应:')
for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            try:
                data = json.loads(line[6:])  # 去掉 'data: '
                content_preview = data["content"][:100] + "..." if len(data["content"]) > 100 else data["content"]
                print(f'{data["type"]}: {content_preview}')
            except Exception as e:
                print(f'解析失败: {line}, 错误: {e}')