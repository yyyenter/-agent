# agent_test0/crew.py
import sys
import json
import os
import re
import logging
from crewai import Agent, Crew, Process, Task, LLM
from crewai.flow import Flow, listen, start
from crewai.project import CrewBase, agent, task, crew
from pydantic import BaseModel
from crewai_tools import TavilySearchTool
from dotenv import load_dotenv

# 修复 stdout 编码（Windows 终端）
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from agent_test0.tools.custom_tool import ReadMemoryTool, SaveMemoryTool, WeatherTool

load_dotenv()

# =========================================
# 【终端打印】处理 Windows 终端编码问题
# =========================================
def terminal_print(text):
    """兼容 Windows 终端的打印函数"""
    try:
        # 尝试使用 utf-8 编码
        os.write(1, (str(text) + "\n").encode('utf-8'))
    except UnicodeEncodeError:
        # 如果 utf-8 失败，尝试使用 gbk（Windows 终端默认）
        try:
            os.write(1, (str(text) + "\n").encode('gbk', errors='ignore'))
        except Exception:
            pass
    except Exception:
        pass

# 使用别名保持代码一致性
hard_print = terminal_print

# ============================================
# 终极版 Agent 动作解析函数 (处理 CrewAI Tuple 变体)
# ============================================
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
    except Exception as e:
        print(f"💥 [Agent原始动作异常] {str(step_output)[:300]}")


zhipu_llm = LLM(
    model=os.getenv("GLM_MODEL_NAME") or "glm-4-flash",
    base_url=os.getenv("GLM_API_BASE") or "",
    api_key=os.getenv("GLM_API_KEY") or "",
)

# 实例化搜索工具
search_tool = TavilySearchTool()

class TravelState(BaseModel):
    message: str = ''
    user_id: str = 'default_user'
    session_id: str = 'default_sess'
    is_complex: bool = True
    simple_answer: str = ""
    location: str = "未知地点"
    focus: str = ""
    tools_needed: list[str] = []
    plan_document: str = ""
    draft_report: str = ""
    previous_draft: str = ""  # 上一版本草案，用于增量修改
    last_validation_feedback: str = ""  # 最后一次质检反馈
    final_report: str = ""
    current_step_instruction: str = "" # 专门存放当前步骤的具体工单

@CrewBase
class PlannerCrew:
    agents_config = 'config/agent.yaml'
    tasks_config = 'config/tasks.yaml'
    # ⚠️ 去掉了 __init__ 重写，保持框架原生状态

    @agent
    def planner_agent(self) -> Agent:
        return Agent(config=self.agents_config['planner_agent'], tools=[ReadMemoryTool(), SaveMemoryTool()], llm=zhipu_llm, verbose=True)
    @task
    def planning_task(self) -> Task:
        return Task(config=self.tasks_config['planning_task'])
    @crew
    def crew(self) -> Crew:
        return Crew(agents=self.agents, tasks=self.tasks, verbose=True)


@CrewBase
class TravelExpertCrew:
    agents_config = 'config/agent.yaml'
    tasks_config  = 'config/research_tasks.yaml'

    @agent
    def info_search_agent(self) -> Agent:
        return Agent(config=self.agents_config['info_search_agent'], tools=[WeatherTool(), ReadMemoryTool()] if search_tool is None else [search_tool, WeatherTool(), ReadMemoryTool()], llm=zhipu_llm, verbose=True)
    @agent
    def itinerary_planner_agent(self) -> Agent:
        return Agent(config=self.agents_config['itinerary_planner_agent'], tools=[], llm=zhipu_llm, verbose=True)

    @task
    def research_task(self) -> Task:
        return Task(config=self.tasks_config['execution_task'])

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents, 
            tasks=self.tasks, 
            process=Process.hierarchical, # 开启层级动态拆解机制
            manager_llm=zhipu_llm,        # 必须指定一个模型作为“包工头”来做路由分配
            verbose=True
        )


@CrewBase
class ValidatorCrew:
    agents_config = 'config/agent.yaml'
    tasks_config = 'config/logic_validator_tasks.yaml'

    @agent
    def logic_validator_agent(self) -> Agent:
        return Agent(config=self.agents_config['logic_validator_agent'], tools=[ReadMemoryTool()] if search_tool is None else [search_tool, ReadMemoryTool()], llm=zhipu_llm, verbose=True)
    @task
    def validation_task(self) -> Task:
        return Task(config=self.tasks_config['validation_task'])
    @crew
    def crew(self) -> Crew:
        return Crew(agents=self.agents, tasks=self.tasks, verbose=True)


# ==================== Flow 调度（自适应 ReAct 收敛闭环） ====================
class TravelWorkflow(Flow[TravelState]):
    def __init__(self, status_callback=None, content_callback=None):
        super().__init__()
        self.max_error_retries = 3
        self.max_adjustments = 20
        self.current_error_count = 0
        self.current_adjust_count = 0
        self.feedback_history: list[str] = []
        self.status_callback = status_callback
        self.content_callback = content_callback

    def notify(self, text: str):
        if self.status_callback:
            self.status_callback(text)
        # 【核弹级打印】确保 Flow 状态输出可见
        print(f"[Flow] {text}")

    def _notify_content(self, text: str, content_type: str = "status"):
        # ⚠️ 强制指定给前端的 type 为 "status"
        if self.content_callback:
            self.content_callback(text, content_type)

    def _make_step_callback(self):
        """创建 step_callback：同时写日志 + 截取关键信息推 SSE"""
        flow = self
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
        
    def _run_crew_with_callback(self, crew_class, inputs):
        """黑科技包装器：绕过 CrewBase 强行将 Callback 绑到所有对象上"""
        crew_instance = crew_class().crew()
        cb = self._make_step_callback()
        crew_instance.step_callback = cb
        # 强制遍历，连每个 Agent 身上都挂上回调，确保万无一失
        for a in crew_instance.agents:
            a.step_callback = cb
        return crew_instance.kickoff(inputs=inputs)

    def _parse_feedback(self, feedback: str) -> tuple[str, str]:
        if feedback.startswith("[提问]"): return "ask_user", feedback.replace("[提问]", "").strip()
        if feedback.startswith("[重做]"): return "error", feedback.replace("[重做]", "").strip()
        if feedback.startswith("[继续]"): return "continue", feedback.replace("[继续]", "").strip()
        if feedback.startswith("[通过]"): return "pass", feedback.replace("[通过]", "").strip()
        
        feedback_lower = feedback.lower()
        if any(kw in feedback_lower for kw in ["通过", "合格", "满意", "完美", "pass", "ok"]):
            if "错误" not in feedback_lower and "不足" not in feedback_lower and "打回" not in feedback_lower: return "pass", ""
        if any(kw in feedback_lower for kw in ["逻辑错误", "严重", "错误", "矛盾", "不可行", "打回修正", "打回"]): return "error", feedback
        if any(kw in feedback_lower for kw in ["不足", "缺少", "不完整", "建议补充", "需要更多信息"]): return "incomplete", feedback
        if any(kw in feedback_lower for kw in ["建议", "优化", "可以调整", "改进", "更好"]): return "adjust", feedback
        return "pass", feedback

    @start()
    def plan_steps(self):
        print(f"\n{'='*50}")
        print(f"[Plan] 决策官剖析需求中（error重试: {self.current_error_count}, adjust微调: {self.current_adjust_count}）")
        print(f"{'='*50}")
        self.notify("📋 [决策阶段] 决策大脑正在建立行程执行策略...")

        inputs = {
            "message": self.state.message,
            "user_id": self.state.user_id,
            "focus": self.state.focus
        }
        if self.feedback_history:
            inputs["feedback_history"] = "\n".join(self.feedback_history[:])

        # 使用封装后的函数启动！
        result = self._run_crew_with_callback(PlannerCrew, inputs)
        raw_text = result.raw.strip()
        self.state.plan_document = raw_text

        try:
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group(0))
                self.state.is_complex = plan_data.get("is_complex", True)
                self.state.simple_answer = plan_data.get("simple_answer", "")
                self.state.location = plan_data.get("location", "未知")
                self.state.focus = plan_data.get("focus", "")
                self.state.tools_needed = plan_data.get("tools_needed", [])
            else:
                self.state.is_complex = True
        except Exception:
            self.state.is_complex = True

    @listen(plan_steps)
    def execute_step(self):
        if not self.state.is_complex:
            self.notify("[解答] 正在直接回答您的问题...")
            self.state.final_report = self.state.simple_answer
            return

        self.notify(f"[执行阶段] 专家团队正在规划【{self.state.location}】...")
        print(f"\n{'='*50}")
        print(f"[Act] 执行生成（focus: {self.state.focus}）...")
        print(f"{'='*50}")

        # 构建执行输入
        inputs = {
            "plan_document": self.state.plan_document,
            "draft": self.state.draft_report or "",
            "location": self.state.location,
            "message": self.state.message,
            "user_id": self.state.user_id,
            "current_step_instruction": self.state.current_step_instruction or "初始化行程",
            "modification_instructions": self.state.last_validation_feedback or "无特别修改要求，请按原计划和当前步骤指示执行",
        }

        # 如果存在质检反馈，将其作为修改指令加入
        if self.state.last_validation_feedback:
            inputs["modification_instructions"] = self.state.last_validation_feedback
            print(f"[执行阶段] 检测到质检反馈，将按要求进行增量修改...")

        # 调用 TravelExpertCrew 执行规划任务
        result = self._run_crew_with_callback(TravelExpertCrew, inputs=inputs)
        self.state.draft_report = result.raw

    @listen(execute_step)
    def validate_router(self):
        if not self.state.is_complex:
            return

        while True:
            print(f"\n{'='*50}")
            print(f"[Reason] 质检评估中（error重试: {self.current_error_count}, adjust微调: {self.current_adjust_count}）")
            print(f"{'='*50}")

            self.notify("🔍 [质检阶段] 正在严苛审查行程...")

            # 1. 唤醒质检团队，⚠️ 注意这里新增了 current_step_instruction
            result = self._run_crew_with_callback(ValidatorCrew, inputs={
                "plan_document": self.state.plan_document,
                "current_step_instruction": getattr(self.state, 'current_step_instruction', '初始化行程'), 
                "draft": self.state.draft_report,
                "location": self.state.location,
                "user_id": self.state.user_id,
            })
            validation_feedback = result.raw
            self.feedback_history.append(validation_feedback)

            # 2. 🌟 唯一解析入口：使用健壮的解析器，一次性拿到标签类型和内容
            feedback_type, adjustment_hint = self._parse_feedback(validation_feedback)

            # 3. 🌟 纯净的状态分发 (不再有任何 startswith 或 replace)
            
            # 分支 A：触发交互钩子
            if feedback_type == "ask_user":
                self.notify(f"🙋 [询问] {adjustment_hint}")
                self.state.final_report = adjustment_hint
                return

            # 分支 B：当前步骤完美，推进下一步
            elif feedback_type == "continue":
                self.notify(f"▶️ [推进] 当前步骤达标，继续往下规划...")
                
                # 宏观进度条更新
                self.state.plan_document += f"\n\n✅ 已完成上一步。下一步指示: {adjustment_hint}"
                # 将 Validator 说的“下一步指示”变成 Executor 下一轮的“精准工单”
                self.state.current_step_instruction = adjustment_hint
                
                self.execute_step()
                continue

            # 分支 C：有问题，打回重做或微调
            elif feedback_type in ("adjust", "incomplete", "error"):
                if self.current_adjust_count < self.max_adjustments:
                    self.current_adjust_count += 1
                    self.notify(f"🔄 [打回重做 {self.current_adjust_count}/{self.max_adjustments}] 发现瑕疵，正在修正...")
                    
                    self.state.previous_draft = self.state.draft_report
                    # 将质检员的斥责单独存为修改指令，传给 Executor
                    self.state.last_validation_feedback = adjustment_hint
                    
                    self.execute_step()
                    continue
                else:
                    self.notify("⚠️ [警告] 修改次数耗尽，强制输出当前草案")
                    self.state.final_report = self.state.draft_report
                    return

            # 分支 D：全局完美完结
            elif feedback_type == "pass":
                self.notify("🎉 [终审通过] 您的完美行程已全部分步规划完毕！")
                self.state.final_report = self.state.draft_report
                return

    @listen(validate_router)
    def finalize(self):
        print(f"\n{'='*50}")
        print(f"[结束] 流程结束")
        print(f"{'='*50}")
        if not self.state.final_report:
            self.state.final_report = self.state.draft_report or '未能生成报告'