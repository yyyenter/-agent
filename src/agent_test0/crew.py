# agent_test0/crew.py
import json
import os
import re
from crewai import Agent, Crew, Process, Task, LLM
from crewai.flow import Flow, listen, start
from crewai.project import CrewBase, agent, task, crew
from pydantic import BaseModel
from crewai_tools import TavilySearchTool
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# ✅ 引入拆分后的原子工具 (SQLite 动态 KV 版本)
from agent_test0.tools.custom_tool import WeatherTool, ReadMemoryTool, SaveMemoryTool

zhipu_llm = LLM(
    model=os.getenv("GLM_MODEL_NAME") or "glm-4-flash",
    base_url=os.getenv("GLM_API_BASE") or "",
    api_key=os.getenv("GLM_API_KEY") or "",
)

search_tool = None  # TavilySearchTool(api_key=os.getenv('TAVILY_API_KEY'))

class TravelState(BaseModel):
    message: str = ''
    user_id: str = 'default_user'    
    session_id: str = 'default_sess' 
    is_complex: bool = True
    simple_answer: str = ""
    tools_needed: list[str] = []
    focus: str = ""
    location: str = "未知地点"
    topic: str = "旅游规划"
    food: str = ""
    plan: list[str] = []
    weather_info: str = "未查询天气"
    food_info: str = "未查询特色"
    draft_report: str = "" # 用于暂存生产团队的草案，供质检员审查
    final_report: str = ""
    
    # ========== ReAct 循环状态 ==========
    current_step: int = 0              # 当前执行到第几步
    retry_count: int = 0                # 当前步骤重试次数
    max_retries: int = 3                # 最大重试次数
    validation_feedback: str = ""       # 质检反馈内容
    needs_replan: bool = False          # 是否需要重新规划
    adjustment_hint: str = ""            # 调整建议（用于 Replan）


@CrewBase
class PlannerCrew:
    """负责 Flow 第一步的决策与长期偏好分析"""
    agents_config = 'config/agent.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def planner_agent(self) -> Agent:
        # 直接使用 YAML 中的配置，仅在运行时动态绑定 LLM 和记忆读写工具
        return Agent(
            config=self.agents_config['planner_agent'],
            tools=[ReadMemoryTool(), SaveMemoryTool()], # 决策官拥有读写长期偏好权限
            llm=zhipu_llm,
            verbose=True
        )

    @task
    def planning_task(self) -> Task:
        return Task(config=self.tasks_config['planning_task'])

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents, 
            tasks=self.tasks, 
            verbose=True
        )


@CrewBase
class TravelExpertCrew:
    """自治团队：收编了旅游情报、旅游规划、内部质检等所有 SOP 执行能力"""
    agents_config = 'config/agent.yaml'
    tasks_config  = 'config/research_tasks.yaml' 

    @agent
    def info_search_agent(self) -> Agent:
        # 严格对应 agent.yaml 的 info_search_agent，直接动态注入工具与模型
        return Agent(
            config=self.agents_config['info_search_agent'],
            tools=[WeatherTool(), ReadMemoryTool()] if search_tool is None else [search_tool, WeatherTool(), ReadMemoryTool()], # 搜集专家只有 Read 长期记忆权限，防止污染 DB
            llm=zhipu_llm,
            verbose=True
        )

    @agent
    def itinerary_planner_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['itinerary_planner_agent'],
            tools=[], # 纯逻辑规划，不需要绑定外部搜索或记忆工具
            llm=zhipu_llm,
            verbose=True
        )

    @agent
    def logic_validator_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['logic_validator_agent'],
            tools=[search_tool] if search_tool else [], # 内部快速质检需要时空距离校准
            llm=zhipu_llm,
            verbose=True
        )

    @task
    def research_task(self) -> Task:
        return Task(config=self.tasks_config['research_task'])

    @task
    def drafting_task(self) -> Task:
        return Task(config=self.tasks_config['drafting_task'])

    @task
    def validation_task(self) -> Task:
        return Task(config=self.tasks_config['validation_task'])

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents, 
            tasks=self.tasks, 
            process=Process.hierarchical, # 开启层级管理
            manager_llm=zhipu_llm,         # 指定经理大脑
            verbose=True
        )


@CrewBase
class ValidatorCrew:
    """终审重关卡：负责最终逻辑审核，不参与生产"""
    agents_config = 'config/agent.yaml'
    tasks_config = 'config/logic_validator_tasks.yaml'

    @agent
    def logic_validator_agent(self) -> Agent:
        # 对应 YAML 中的 logic_validator_agent Key
        return Agent(
            config=self.agents_config['logic_validator_agent'],
            tools=[ReadMemoryTool()] if search_tool is None else [search_tool, ReadMemoryTool()], # 终审官拥有核对历史长期偏好的权限
            llm=zhipu_llm,
            verbose=True
        )

    @task
    def validation_task(self) -> Task:
        return Task(config=self.tasks_config['validation_task'])

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents, 
            tasks=self.tasks, 
            verbose=True
        )
     
# ==================== Flow 调度（自适应 ReAct 收敛闭环） ====================
class TravelWorkflow(Flow[TravelState]):
    def __init__(self, status_callback = None):
        super().__init__()
        self.max_error_retries = 3
        self.max_adjustments = 20
        self.current_error_count = 0
        self.current_adjust_count = 0
        self.feedback_history: list[str] = []
        self.status_callback = status_callback

    def notify(self, text: str):
        if self.status_callback:
            self.status_callback(text)

    def _parse_feedback(self, feedback: str) -> tuple[str, str]:
        feedback_lower = feedback.lower()
        if any(kw in feedback_lower for kw in ["通过", "合格", "满意", "完美", "pass", "ok"]):
            if "错误" not in feedback_lower and "不足" not in feedback_lower and "打回" not in feedback_lower:
                return "pass", ""
        if any(kw in feedback_lower for kw in ["逻辑错误", "严重", "错误", "矛盾", "不可行", "打回修正", "打回"]):
            return "error", feedback
        if any(kw in feedback_lower for kw in ["不足", "缺少", "不完整", "建议补充", "需要更多信息"]):
            return "incomplete", feedback
        if any(kw in feedback_lower for kw in ["建议", "优化", "可以调整", "改进", "更好"]):
            return "adjust", feedback
        return "pass", feedback

    @start()
    def plan_steps(self):
        "Step 1: Plan — 决策官剖析需求"
        print(f"\n{'='*50}")
        print(f"[Plan] 决策官剖析需求中（error重试: {self.current_error_count}, adjust微调: {self.current_adjust_count}）")
        print(f"{'='*50}")
        self.notify("[决策阶段] 决策大脑正在读取您的长期特征并建立行程执行 Focus 指引...")

        inputs = {
            "message": self.state.message,
            "user_id": self.state.user_id,
            "focus": self.state.focus
        }
        if self.feedback_history:
            inputs["feedback_history"] = "\n".join(self.feedback_history[-3:])

        result = PlannerCrew().crew().kickoff(inputs=inputs)
        raw_text = result.raw.strip()

        try:
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                clean_json_str = json_match.group(0)
                plan_data = json.loads(clean_json_str)
                self.state.is_complex = plan_data.get("is_complex", True)
                self.state.simple_answer = plan_data.get("simple_answer", "")
                self.state.location = plan_data.get("location", "未知")
                self.state.focus = plan_data.get("focus", "")
                self.state.tools_needed = plan_data.get("tools_needed", [])
            else:
                raise ValueError("未在模型输出中检测到 JSON 结构")
        except Exception as e:
            print(f"[Planner] 提取失败: {str(e)}")
            self.state.is_complex = True

    @listen(plan_steps)
    def execute_step(self):
        "Step 2: Act — 核心 SOP 执行层"
        if not self.state.is_complex:
            self.notify("[决策阶段] 判定为简单任务，正在直接解答...")
            self.state.final_report = self.state.simple_answer
            return

        self.notify(f"[执行阶段] 专家团队已集结！正在深度检索与规划【{self.state.location}】...")
        print(f"\n{'='*50}")
        print(f"[Act] 执行生成（focus: {self.state.focus}）...")
        print(f"{'='*50}")

        result = TravelExpertCrew().crew().kickoff(inputs={
            "location": self.state.location,
            "focus": self.state.focus or "常规规划",
            "message": self.state.message,
            "user_id": self.state.user_id
        })
        self.state.draft_report = result.raw
        print(f"[生成] 草案: {len(result.raw)} 字符")

    @listen(execute_step)
    def validate_router(self):
        """
        Step 3: Reason — 硬性质检 + 内部 ReAct 闭环
        不再依赖 CrewAI Flow 的路由返回值（已被证实不工作），
        改为内部 while 循环直接调用 plan_steps / execute_step。
        """
        if not self.state.is_complex:
            return

        while True:
            print(f"\n{'='*50}")
            print(f"[Reason] 质检评估中（error重试: {self.current_error_count}, adjust微调: {self.current_adjust_count}）")
            print(f"{'='*50}")

            self.notify("[质检阶段] 逻辑质检员正在进行严格的通勤距离和长期画像合规审查...")

            result = ValidatorCrew().crew().kickoff(inputs={
                "draft": self.state.draft_report,
                "location": self.state.location,
                "user_id": self.state.user_id
            })
            validation_feedback = result.raw
            self.feedback_history.append(validation_feedback)

            feedback_type, adjustment_hint = self._parse_feedback(validation_feedback)
            print(f"[反馈] 类型: {feedback_type}")
            if adjustment_hint:
                print(f"   提示: {adjustment_hint[:200]}")

            if feedback_type == "pass":
                print("[Reason] 方案通过终审！");
                self.state.final_report = self.state.draft_report
                return

            elif feedback_type in ("adjust", "incomplete"):
                if self.current_adjust_count < self.max_adjustments:
                    self.current_adjust_count += 1
                    print(f"[Reason] 需要优化/补充（第{self.current_adjust_count}次微调），重新执行...");
                    self.notify(f"[重试 {self.current_adjust_count}/{self.max_adjustments}] 根据质检反馈微调行程...");
                    self.state.focus = f"{self.state.focus}；(质检整改要求: {adjustment_hint})"
                    self.execute_step()
                    continue
                else:
                    print(f"[警告] 微调次数耗尽，强制结束")
                    self.state.final_report = self.state.draft_report
                    return

            elif feedback_type == "error":
                if self.current_error_count < self.max_error_retries:
                    self.current_error_count += 1
                    print(f"[Reason] 严重逻辑矛盾（第{self.current_error_count}次），回滚到 Plan 重新规划...")
                    self.notify(f"[重试 {self.current_error_count}/{self.max_error_retries}] 发现严重问题，回滚重新规划...")
                    self.state.focus = f"{self.state.focus}；(历史质检致命错误: {adjustment_hint})"
                    self.plan_steps()
                    self.execute_step()
                    continue
                else:
                    print(f"[警告] 重试次数耗尽，强制结束")
                    self.state.final_report = f"[无法收敛修正]\n{self.state.draft_report}"
                    return

            # 未知反馈类型，保守处理为通过
            self.state.final_report = self.state.draft_report
            return

    @listen(validate_router)
    def finalize(self):
        print(f"\n{'='*50}")
        print(f"[结束] 流程结束")
        print(f"{'='*50}")
        if not self.state.final_report:
            self.state.final_report = self.state.draft_report or '未能生成报告'