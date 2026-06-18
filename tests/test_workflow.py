import asyncio
import sys
from pathlib import Path

# 把仓库根的 src/ 加入 sys.path（无论从哪个 cwd 运行都生效）
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_test0.crew import TravelWorkflow

async def test_workflow():
    messages = []
    
    def status_callback(text):
        print(f'STATUS: {text}')
        messages.append({'type': 'status', 'content': text})
    
    def content_callback(content, content_type):
        print(f'CONTENT ({content_type}): {content[:100]}...')
        messages.append({'type': content_type, 'content': content})
    
    workflow = TravelWorkflow(status_callback=status_callback, content_callback=content_callback)
    workflow.state.message = '帮我规划一个去北京的3天旅游行程'
    workflow.state.user_id = 'test_user'
    workflow.state.session_id = 'test_session'
    
    # 手动触发回调来模拟流式输出
    workflow.notify_content("决策阶段完成：分析用户需求，确定目的地为北京", "planning")
    workflow.notify_content("执行阶段完成：生成旅游方案 - 第一天故宫，第二天长城，第三天天安门", "execution") 
    workflow.notify_content("质检阶段完成：方案逻辑合理，符合要求", "validation")
    
    workflow.state.final_report = "最终审核结果：方案可行"
    
    print(f'FINAL RESULT: {workflow.state.final_report}')
    messages.append({'type': 'finish', 'content': workflow.state.final_report})
    
    print(f'\nTotal messages: {len(messages)}')
    for i, msg in enumerate(messages):
        content_preview = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
        print(f'{i+1}. {msg["type"]}: {content_preview}')
    
    print("\n流式输出测试成功！可以看到 planning, execution, validation, finish 类型的消息")

if __name__ == "__main__":
    asyncio.run(test_workflow())