# agent_test0/workflow/callbacks.py
"""
Agent 步骤回调与日志记录。

包含：
- agent_step_logger: 解析 CrewAI 各种格式的 step_output，硬打印到控制台
- make_step_callback: 给指定 flow 实例创建 step_callback（同时打日志 + 推 SSE）
- run_crew_with_callback: 黑科技包装器，强制把回调挂到 Crew 与所有 Agent 上

分离原因：
  这部分是工具性的、跨节点共用的，没有业务逻辑。
  原 crew.py 把它和状态机逻辑揉在一起，影响阅读。
"""


def agent_step_logger(step_output):
    """兼容解析 CrewAI 各种格式的 step_output，使用硬打印确保输出可见"""
    try:
        step_list = step_output if isinstance(step_output, list) else [step_output]
        for step in step_list:
            # 格式1: Tuple (AgentAction, Observation)
            if isinstance(step, tuple) and len(step) >= 2:
                action, obs = step[0], step[1]
                thought = getattr(action, 'log', getattr(action, 'thought', ''))
                tool = getattr(action, 'tool', '')
                tool_input = getattr(action, 'tool_input', '')

                if thought: print(f"🤔 [Agent思考] {thought.strip()}")
                if tool: print(f"🛠️ [Agent工具] {tool} | 参数: {tool_input}")
                if obs: print(f"✅ [Agent结果] {str(obs)[:300]}...")

            # 格式2: AgentStep Object
            else:
                thought = getattr(step, 'thought', getattr(step, 'log', ''))
                tool = getattr(step, 'tool', '')
                tool_input = getattr(step, 'tool_input', '')
                result = getattr(step, 'result', getattr(step, 'observation', ''))

                if thought: print(f"🤔 [Agent思考] {thought.strip()}")
                if tool: print(f"🛠️ [Agent工具] {tool} | 参数: {tool_input}")
                if result: print(f"✅ [Agent结果] {str(result)[:300]}...")

                if not thought and not tool and not result:
                    print(f"🤖 [Agent追踪] {str(step)[:300]}")
    except Exception:
        print(f"💥 [Agent原始动作异常] {str(step_output)[:300]}")


def make_step_callback(flow):
    """
    给指定 flow 实例创建 step_callback：同时写日志 + 截取关键信息推 SSE。

    flow 必须是 TravelWorkflow（或带有 _notify_content 方法的对象）。
    """
    def callback(step_output):
        # 1. 强制写日志到控制台和文件
        agent_step_logger(step_output)

        # 2. 提取精简文本推给前端的 Web 界面
        try:
            step_list = step_output if isinstance(step_output, list) else [step_output]
            for step in step_list:
                if isinstance(step, tuple) and len(step) >= 2:
                    thought = getattr(step[0], 'log', '')
                    tool = getattr(step[0], 'tool', '')
                else:
                    thought = getattr(step, 'thought', getattr(step, 'log', ''))
                    tool = getattr(step, 'tool', '')

                if thought:
                    flow._notify_content(f"🤔 {thought.strip()}", "status")
                elif tool:
                    flow._notify_content(f"🛠️ 正在调用工具查询: {tool}", "status")
        except Exception:
            pass
    return callback


def run_crew_with_callback(flow, crew_class, inputs):
    """
    黑科技包装器：绕过 CrewBase 强行将 callback 绑到所有对象上。

    这样能保证不管 CrewBase 内部怎么实例化 Agent，回调都能挂上去。
    """
    from agent_test0.workflow.trace import timed

    crew_instance = crew_class().crew()
    cb = make_step_callback(flow)
    crew_instance.step_callback = cb
    # 强制遍历，连每个 Agent 身上都挂上回调，确保万无一失
    for a in crew_instance.agents:
        a.step_callback = cb
    # 计时：每个 Crew 调用都是一次 LLM 往返，是耗时大头
    with timed(f"Crew:{crew_class.__name__}"):
        result = crew_instance.kickoff(inputs=inputs)
    # Crew 输出安全截断
    if hasattr(result, 'raw') and len(result.raw) > 8000:
        print(f"[Truncation] Crew 输出从 {len(result.raw)} 截断至 8000 字符")
        result.raw = result.raw[:8000] + "\n\n[输出过长已截断]"
    return result
