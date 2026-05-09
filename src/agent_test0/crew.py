import json
import os
from typing import Literal
from crewai import Agent, Crew, Process, Task, LLM
from crewai.flow import Flow, listen, start
from crewai.project import CrewBase, agent, task, crew
from pydantic import BaseModel
from crewai_tools import TavilySearchTool

# ✅ 引入拆分后的原子工具
from agent_test0.tools.custom_tool import  WeatherTool, ReadMemoryTool, SaveMemoryTool

zhipu_llm = LLM(
    model=os.getenv("GLM_MODEL_NAME") or "glm-4-flash",
    base_url=os.getenv("GLM_API_BASE") or "",
    api_key=os.getenv("GLM_API_KEY") or "",
)

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
    draft_report: str = "" # ✅ 补充：用于暂存生产团队的草案，供质检员审查
    final_report: str = ""
    
    # ========== ReAct 循环状态 ==========
    current_step: int = 0              # 当前执行到第几步
    retry_count: int = 0                # 当前步骤重试次数
    max_retries: int = 3                # 最大重试次数
    validation_feedback: str = ""       # 质检反馈内容
    needs_replan: bool = False          # 是否需要重新规划
    adjustment_hint: str = ""            # 调整建议（用于 Replan）

# ==================== 共享 Agent  ====================
def create_info_search_agent() -> Agent:
    """情报搜集 Agent - 可被多个 Crew 复用"""
    return Agent(
        config={'role': '全能情报搜集专家', 'goal': '利用网络搜索获取旅游数据', 'backstory': '先看天气，再看景点和费用'},
        tools=[search_tool, WeatherTool()],
        llm=zhipu_llm
    )

def create_itinerary_planner_agent() -> Agent:
    """行程规划 Agent - 可被多个 Crew 复用"""
    return Agent(
        config={'role': '高级旅行架构师', 'goal': '将零散情报转化为结构化行程', 'backstory': '统筹安排，最优路线'},
        llm=zhipu_llm
    )

def create_logic_validator_agent(include_memory: bool = False) -> Agent:
    """逻辑质检 Agent - 可被多个 Crew 复用
    Args:
        include_memory: 是否包含记忆工具（最终质检需要，简单内部质检不需要）
    """
    tools: list[object] = [search_tool]
    if include_memory:
        tools.append(ReadMemoryTool())
    return Agent(
        config={'role': '逻辑与合规性评估员', 'goal': '审查行程逻辑矛盾', 'backstory': '严苛著称，立即驳回错误'},
        tools=tools,
        llm=zhipu_llm
    )

def create_planner_agent() -> Agent:
    """路径决策 Agent - 专门用于判断任务复杂度"""
    return Agent(
        config={'role': '旅行路径决策官', 'goal': '分析需求，判断复杂度，制定计划', 'backstory': '追求效率，一句话能说清就不浪费算力'},
        tools=[ReadMemoryTool()],
        llm=zhipu_llm
    )

# ---  复杂任务执行团队 (自治团队) ---
search_tool= TavilySearchTool(api_key = os.getenv('TAVILY_API_KEY'))
@CrewBase
class TravelExpertCrew:
    """自治团队：收编了旅游情报、旅游规划 所有能力"""
    agents_config = 'config/agent.yaml'
    tasks_config  = 'config/research_tasks.yaml' 

    # --- 定义 Agent 池（引用共享工厂） ---
    @agent
    def info_search_agent(self) -> Agent:
        return create_info_search_agent()

    @agent
    def itinerary_planner_agent(self) -> Agent:
        return create_itinerary_planner_agent()

    @agent
    def logic_validator_agent(self) -> Agent:
        return create_logic_validator_agent(include_memory=False)

    # --- 定义 Task 池 ---
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
        # 这里的 self.agents 和 self.tasks 会自动收集上面装饰的所有对象
        return Crew(
            agents=self.agents, 
            tasks=self.tasks, 
            process=Process.hierarchical, # 开启层级管理
            manager_llm=zhipu_llm,         # 必须指定经理大脑
            verbose=True
        )
    
@CrewBase
class PlannerCrew:
    """负责 Flow 第一步的决策 Crew"""
    agents_config = 'config/agent.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def planner_agent(self) -> Agent:
        return create_planner_agent()

    @task
    def planning_task(self) -> Task:
        return Task(config=self.tasks_config['planning_task'])

    @crew
    def crew(self) -> Crew:
        return Crew(agents=self.agents, tasks=self.tasks, verbose=True)

@CrewBase
class ValidatorCrew:
    """硬关卡：负责最终逻辑审核，不参与生产"""
    agents_config = 'config/agent.yaml'
    tasks_config = 'config/logic_validator_tasks.yaml'

    @agent
    def validator_agent(self) -> Agent:
        return create_logic_validator_agent(include_memory=True)

    @task
    def validation_task(self) -> Task:
        return Task(config=self.tasks_config['validation_task'])

    @crew
    def crew(self) -> Crew:
        return Crew(agents=self.agents, tasks=self.tasks, verbose=True)
     
# ==================== Flow 调度（ReAct 模式） ====================
class TravelWorkflow(Flow[TravelState]):
    def __init__(self, status_callback = None):
        super().__init__()
        self.max_error_retries = 3   # 针对【严重错误】的重试次数
        self.max_adjustments = 20     # 针对【优化建议】的打磨次数
        self.current_error_count = 0
        self.current_adjust_count = 0
        self.feedback_history: list[str] = []  # 记录反馈历史
        self.status_callback = status_callback
    def notify(self, text: str):
        """✅ 辅助方法：只要调用它，就能把进度实时送给前端"""
        if self.status_callback:
            self.status_callback(text)

    def _parse_feedback(self, feedback: str) -> tuple[str, str]:
        """解析质检反馈，判断类型和调整建议
        Returns: (feedback_type, adjustment_hint)
        - "pass": 通过
        - "error": 错误/矛盾（需重试）
        - "incomplete": 信息不足（需补充）
        - "adjust": 优化建议（可选调整）
        """
        feedback_lower = feedback.lower()
        
        # 直接通过
        if any(kw in feedback_lower for kw in ["通过", "合格", "满意", "完整", "pass", "ok", "✅"]):
            if "错误" not in feedback_lower and "不足" not in feedback_lower:
                return "pass", ""
        
        # 严重错误
        if any(kw in feedback_lower for kw in ["逻辑错误", "严重", "错误", "矛盾", "不可行", "❌"]):
            return "error", feedback
        
        # 信息不足/需要补充
        if any(kw in feedback_lower for kw in ["不足", "缺少", "不完整", "建议补充", "需要更多信息"]):
            return "incomplete", feedback
        
        # 优化建议
        if any(kw in feedback_lower for kw in ["建议", "优化", "可以调整", "改进", "更好"]):
            return "adjust", feedback
        
        # 无法判断，默认通过
        return "pass", feedback

    @start()
    def plan_steps(self) -> TravelState:
        """🚀 Step 1: Plan - 生成执行计划"""
        print(f"\n{'='*50}")
        print(f"📋 [Plan] 生成执行计划（重试次数: {self.current_error_count}）")
        print(f"{'='*50}")
        self.notify("📋 [决策阶段] 决策官正在分析您的需求，检索您的长期偏好并生成执行计划...")
        inputs = {
            "message": self.state.message,
            "user_id": self.state.user_id
        }
        
        if self.feedback_history:
            inputs["feedback_history"] = "\n".join(self.feedback_history[-3:]) # 注意这里不要传 -100 那么多，容易超 token
        
        result = PlannerCrew().crew().kickoff(inputs=inputs)
        raw_text = result.raw.strip()
        
        try:
            # ✅ 核心修复：使用正则表达式暴力提取大括号内的内容
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            
            if json_match:
                clean_json_str = json_match.group(0)
                plan_data = json.loads(clean_json_str)
                
                self.state.is_complex = plan_data.get("is_complex", True)
                self.state.simple_answer = plan_data.get("simple_answer", "")
                self.state.location = plan_data.get("location", "未知")
                self.state.topic = plan_data.get("topic", "旅游")
                self.state.food = plan_data.get("food", "")
                self.state.plan = plan_data.get("plan", ["research", "draft"])
            else:
                raise ValueError("未在模型输出中检测到 JSON 结构")
                
        except Exception as e:
            # ✅ 新增调试日志：如果还报错，打印出大模型到底胡言乱语了什么，方便你调 Prompt
            print(f"⚠️ Planner 解析失败，异常原因: {str(e)}")
            print(f"🛑 模型实际输出内容:\n{raw_text}")
            self.state.is_complex = True # 兜底逻辑
            
        print(f"✅ 计划: {self.state.plan}")
        return self.state

    @listen(plan_steps)
    def execute_step(self) -> TravelState:
        """🚀 Step 2: Act - 执行当前计划"""
        if not self.state.is_complex:
            self.notify("⚡ [决策阶段] 判定为简单任务，正在为您直接解答...")
            self.state.final_report = self.state.simple_answer
            return self.state
        self.notify(f"🔎 [执行阶段] 专家团队已集结！情报专家正在为您深度调研【{self.state.location}】的天气、门票和酒店价格...")
        print(f"\n{'='*50}")
        print(f"⚡ [Act] 执行生成...")
        print(f"{'='*50}")
        
        result = TravelExpertCrew().crew().kickoff(inputs={
            "location": self.state.location,
            "topic": self.state.topic,
            "food": self.state.food,
            "focus": self.state.focus or "",
            "message": self.state.message,
            "user_id": self.state.user_id
        })
        
        self.state.draft_report = result.raw
        print(f"📝 生成草案: {len(result.raw)} 字符")
        return self.state

    @listen(execute_step)
    def validate_router(self) -> Literal["execute_step", "plan_steps", "END"]:
        """🚀 Step 3: Reason - 分析反馈，决定下一步
        
        ReAct 核心：不是"只有错误才重试"，而是"任何反馈都要判断"
        """
        print(f"\n{'='*50}")
        print(f"🔍 [Reason] 分析反馈...")
        print(f"{'='*50}")
        
        # 简单任务跳过质检
        if not self.state.is_complex:
            self.state.final_report = self.state.simple_answer
            return "END"

        self.notify("🔍 [质检阶段] 严苛的逻辑质检员正在对行程草案进行合规性审查（交通、天气、预算）...")
        # 执行质检
        result = ValidatorCrew().crew().kickoff(inputs={
            "draft": self.state.draft_report,
            "location": self.state.location,
            "user_id": self.state.user_id
        })
        
        validation_feedback = result.raw
        self.feedback_history.append(validation_feedback)
        
        # 解析反馈类型
        feedback_type, adjustment_hint = self._parse_feedback(validation_feedback)
        
        print(f"📊 反馈类型: {feedback_type}")
        print(f"📋 反馈摘要: {validation_feedback[:150]}...")
        
        # ========== Reason：根据反馈类型决定下一步 ==========
        
        if feedback_type == "pass":
            # ✅ 方案通过
            print("✅ [Reason] 方案通过质检，任务完成！")
            self.state.final_report = validation_feedback
            return "END"
        
        elif feedback_type == "adjust":
            # 📝 优化建议：可选是否采纳
            print("📝 [Reason] 有优化建议，判断是否采纳...")
            if self.current_adjust_count < self.max_adjustments:
                self.current_adjust_count += 1
                print("🔄 采用建议，重新优化方案")
                return "execute_step"  # 重新执行
            else:
                print("⚠️ 已达最大优化次数，接受当前方案")
                self.state.final_report = validation_feedback
                return "END"
        
        elif feedback_type == "incomplete":
            # 📋 信息不足：补充信息继续执行
            print("📋 [Reason] 信息不足，需要补充")
            if self.current_adjust_count < self.max_adjustments:
                self.current_adjust_count += 1
                print("🔄 补充信息继续执行")
                return "execute_step"
            else:
                print("⚠️ 已达最大重试次数，标记为不完整")
                self.state.final_report = f"[信息不完整]\n{validation_feedback}"
                return "END"
        
        elif feedback_type == "error":
            # ❌ 错误：重新规划
            print(f"❌ [Reason] 发现错误，需要重新规划")
            if self.current_error_count < self.max_error_retries:
                self.current_error_count += 1
                print(f"🔄 重新规划中...（第 {self.current_retry} 次）")
                return "plan_steps"  # ← 回到计划阶段，传入反馈
            else:
                print(f"🚫 达到最大重试次数（{self.max_error_retries}），终止")
                self.state.final_report = f"[未能通过审查]\n{validation_feedback}"
                return "END"
        
        # 默认结束
        self.state.final_report = validation_feedback
        return "END"

    @listen(validate_router)
    def finalize(self) -> TravelState:
        """最终处理"""
        print(f"\n{'='*50}")
        print(f"🏁 流程结束")
        print(f"{'='*50}")
        if not self.state.final_report:
            self.state.final_report = self.state.draft_report or "未能生成报告"
        return self.state