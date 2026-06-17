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

from agent_test0.tools.custom_tool import WeatherTool
from agent_test0.harness import MemoryManager, get_redis_or_fallback

load_dotenv()

# ─── 全局 Redis 客户端：进程级单例，crew 自己持有 ───
_redis_client, _is_redis_fallback = get_redis_or_fallback()

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

# ============================================
# 【新状态机】辅助数据结构
# ============================================

class StepPlan(BaseModel):
    """单个执行步骤计划"""
    index: int
    description: str      # 步骤描述（不指定工具）
    tools: list[str] = [] # 需要的工具列表（StepPreparer 填充）
    status: str = "pending"  # pending | executing | completed | failed
    result: str = ""
    error: str = ""
    validation_feedback: str = ""  # 验证反馈

class StepResult(BaseModel):
    """步骤执行结果记录"""
    step_index: int
    step_description: str
    result: str
    passed: bool
    validation_feedback: str

class ValidationFeedback(BaseModel):
    """统一反馈结构"""
    is_valid: bool
    feedback_type: str  # pass | retry | partial_fail | full_replan
    failed_indices: list[int] = []  # 失败的步骤索引
    reason: str = ""  # 为什么失败
    suggested_corrections: dict[int, str] = {}  # 每个失败步骤的修正建议


class TravelState(BaseModel):
    # === 基础元数据 ===
    message: str = ''
    user_id: str = 'default_user'
    session_id: str = 'default_sess'

    # === 状态机核心字段 ===
    steps: list[StepPlan] = []          # 步骤列表
    current_step_index: int = 0         # 当前执行步骤索引
    step_results: list[StepResult] = [] # 步骤执行结果历史
    failed_steps_indices: list[int] = [] # 失败步骤索引列表

    # === 流程控制字段（结构化，替代文本前缀解析） ===
    needs_user_input: bool = False      # True 时主流程中断，向用户提问
    user_question: str = ""             # 当 needs_user_input=True 时填入问题
    skip_remaining_steps: bool = False  # 跳过剩余步骤直接输出结果

    # === 业务字段 ===
    is_complex: bool = True
    simple_answer: str = ""
    location: str = "未知地点"
    focus: str = ""
    assumptions: list[str] = []   # Planner 在信息不足时所做的合理假设
    final_report: str = ""

@CrewBase
class PlannerCrew:
    agents_config = 'config/agent.yaml'
    tasks_config = 'config/tasks.yaml'
    # ⚠️ 去掉了 __init__ 重写，保持框架原生状态

    @agent
    def planner_agent(self) -> Agent:
        return Agent(config=self.agents_config['planner_agent'], tools=[], llm=zhipu_llm, verbose=True)
    @task
    def planning_task(self) -> Task:
        return Task(config=self.tasks_config['planning_task'])
    @crew
    def crew(self) -> Crew:
        return Crew(agents=self.agents, tasks=self.tasks, verbose=True)


@CrewBase
class ValidatorCrew:
    agents_config = 'config/agent.yaml'
    tasks_config = 'config/logic_validator_tasks.yaml'

    @agent
    def logic_validator_agent(self) -> Agent:
        tools = [search_tool] if search_tool is not None else []
        return Agent(config=self.agents_config['logic_validator_agent'], tools=tools, llm=zhipu_llm, verbose=True)
    @task
    def validation_task(self) -> Task:
        return Task(config=self.tasks_config['validation_task'])
    @crew
    def crew(self) -> Crew:
        return Crew(agents=self.agents, tasks=self.tasks, verbose=True)


@CrewBase
class StepPreparerCrew:
    """StepPreparer 状态专用 Crew"""
    agents_config = 'config/agent.yaml'
    tasks_config = 'config/step_preparer_tasks.yaml'

    @agent
    def step_preparer_agent(self) -> Agent:
        return Agent(config=self.agents_config['info_search_agent'], tools=[], llm=zhipu_llm, verbose=True)

    @task
    def step_preparer_task(self) -> Task:
        return Task(config=self.tasks_config['step_preparer_task'])

    @crew
    def crew(self) -> Crew:
        return Crew(agents=self.agents, tasks=self.tasks, verbose=True)


@CrewBase
class StepExecutorCrew:
    """StepExecutor 状态专用 Crew"""
    agents_config = 'config/agent.yaml'
    tasks_config = 'config/executor_tasks.yaml'

    @agent
    def step_executor_agent(self) -> Agent:
        return Agent(config=self.agents_config['info_search_agent'], tools=[WeatherTool()], llm=zhipu_llm, verbose=True)

    @task
    def executor_task(self) -> Task:
        return Task(config=self.tasks_config['executor_task'])

    @crew
    def crew(self) -> Crew:
        return Crew(agents=self.agents, tasks=self.tasks, verbose=True)


@CrewBase
class StepVerifierCrew:
    """StepVerifier 状态专用 Crew"""
    agents_config = 'config/agent.yaml'
    tasks_config = 'config/step_validator_tasks.yaml'

    @agent
    def step_verifier_agent(self) -> Agent:
        return Agent(config=self.agents_config['logic_validator_agent'], tools=[], llm=zhipu_llm, verbose=True)

    @task
    def step_validator_task(self) -> Task:
        return Task(config=self.tasks_config['step_validator_task'])

    @crew
    def crew(self) -> Crew:
        return Crew(agents=self.agents, tasks=self.tasks, verbose=True)


@CrewBase
class PartialReplannerCrew:
    """PartialReplanner 状态专用 Crew"""
    agents_config = 'config/agent.yaml'
    tasks_config = 'config/replan_tasks.yaml'

    @agent
    def partial_replanner_agent(self) -> Agent:
        return Agent(config=self.agents_config['planner_agent'], tools=[], llm=zhipu_llm, verbose=True)

    @task
    def replan_task(self) -> Task:
        return Task(config=self.tasks_config['replan_task'])

    @crew
    def crew(self) -> Crew:
        return Crew(agents=self.agents, tasks=self.tasks, verbose=True)


@CrewBase
class FinalVerifierCrew:
    """FinalVerifier 状态专用 Crew"""
    agents_config = 'config/agent.yaml'
    tasks_config = 'config/final_validator_tasks.yaml'

    @agent
    def final_verifier_agent(self) -> Agent:
        return Agent(config=self.agents_config['logic_validator_agent'], tools=[], llm=zhipu_llm, verbose=True)

    @task
    def final_validator_task(self) -> Task:
        return Task(config=self.tasks_config['final_validator_task'])

    @crew
    def crew(self) -> Crew:
        return Crew(agents=self.agents, tasks=self.tasks, verbose=True)


# ==================== 新 6 状态机框架 ====================
# 状态流转: Planner -> StepPreparer -> StepExecutor -> StepVerifier -> (PartialReplanner) -> FinalVerifier

class TravelWorkflow(Flow[TravelState]):
    def __init__(self, status_callback=None, content_callback=None):
        super().__init__()
        self.max_step_retries = 3
        self.max_replan_attempts = 3
        self.step_retry_counts: dict[int, int] = {}
        self.status_callback = status_callback
        self.content_callback = content_callback

    # ========================================
    # 【 Hooks 】全局状态检查钩子
    # ========================================
    def _check_ask_user_hook(self):
        """检查是否需要向用户提问 - 所有状态执行前都会调用此钩子"""
        if self.state.needs_user_input:
            self.notify(f"🙋 [AskUser] {self.state.user_question}")
            self.state.final_report = self.state.user_question
            return True  # 返回 True 表示需要中断流程
        return False

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
        result = crew_instance.kickoff(inputs=inputs)
        # Crew 输出安全截断
        if hasattr(result, 'raw') and len(result.raw) > 8000:
            print(f"[Truncation] Crew 输出从 {len(result.raw)} 截断至 8000 字符")
            result.raw = result.raw[:8000] + "\n\n[输出过长已截断]"
        return result

    # ========================================
    # 【新状态机】状态 1: Planner - 生成粗粒度步骤列表
    # ========================================
    @start()
    def plan_steps(self):
        # 全局钩子：检查是否需要向用户提问
        if self._check_ask_user_hook():
            return

        print(f"\n{'='*60}")
        print(f"[Planner] 决策官剖析需求中...")
        print(f"{'='*60}")
        self.notify("📋 [Planner] 决策大脑正在建立行程执行策略...")

        # 如果还没有步骤列表，调用 PlannerCrew 生成
        if not self.state.steps:
            inputs = {
                "message": self.state.message,
                "user_id": self.state.user_id,
                "focus": self.state.focus,
                "previous_plan": "无历史计划（首次规划）",
                "current_step": "无当前工单（首次执行）",
                "current_draft": "无进度草案",
            }

            result = self._run_crew_with_callback(PlannerCrew, inputs)
            raw_text = result.raw.strip()
            print(f"[Planner] 原始输出: {raw_text[:500]}")

            # 解析 Planner 输出的 JSON，提取步骤列表
            try:
                json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
                if json_match:
                    plan_data = json.loads(json_match.group(0))
                    self.state.is_complex = plan_data.get("is_complex", True)
                    self.state.simple_answer = plan_data.get("simple_answer", "")
                    self.state.location = plan_data.get("location", "未知")
                    self.state.focus = plan_data.get("focus", "")
                    self.state.assumptions = plan_data.get("assumptions", []) or []

                    # 信息不足时 Planner 可能直接发起结构化提问
                    if plan_data.get("needs_user_input") or plan_data.get("verdict") == "ask_user":
                        question = plan_data.get("user_question") or plan_data.get("question") or "信息不足，请补充。"
                        self._set_ask_user_question(question)
                        return

                    # 提取步骤列表
                    steps = plan_data.get("steps", [])
                    is_complex_val = plan_data.get("is_complex", True)
                    simple_answer_val = plan_data.get("simple_answer", "")
                    print(f"[Planner] 解析结果: is_complex={is_complex_val}, simple_answer='{simple_answer_val[:50]}', steps={len(steps)}")
                    if steps:
                        self.state.steps = [StepPlan(**s) for s in steps]
                        print(f"[Planner] 生成了 {len(self.state.steps)} 个粗粒度步骤")
                else:
                    print(f"[Planner] 警告: 输出中没有 JSON 块")
            except Exception as e:
                print(f"[Planner] 解析失败: {e}")
                self.state.is_complex = True

            # 【兜底】Planner 没产出 steps（解析失败 / 模型直接闲聊）：
            # 把原始输出当成简单回答，直接终止流程，避免整条链路因 not steps 静默死掉。
            if not self.state.steps:
                fallback_answer = (
                    self.state.simple_answer.strip()
                    if self.state.simple_answer
                    else raw_text[:600] if raw_text else "抱歉，我暂时无法理解您的需求，请补充更多信息。"
                )
                print(f"[Planner] 兜底: 未生成 steps，直接返回 simple_answer / raw_text")
                self.state.final_report = fallback_answer
                self.notify("⚠️ [Planner] 未生成多步骤计划，直接返回简要回答")
                return

        # 初始化当前步骤索引
        if self.state.steps:
            self.state.current_step_index = 0

    # ========================================
    # 【新状态机】用户提问辅助方法
    # ========================================
    def _set_ask_user_question(self, question: str = None):
        """设置需要向用户提问的问题，流程会在下一个钩子检查时中断"""
        if question is None:
            question = "信息不足，无法继续规划。"
        self.state.needs_user_input = True
        self.state.user_question = question
        print(f"\n{'='*60}")
        print(f"[AskUser] 信息不足，需要向用户提问...")
        print(f"{'='*60}")
        self.notify(f"🙋 [AskUser] {question}")

    # ========================================
    # 【新状态机】状态 2: StepPreparer - 为当前步骤生成执行计划
    # ========================================
    @listen(plan_steps)
    def step_preparer(self):
        import traceback
        print(f"[StepPreparer] 被调用，检查是否从重规划来...")

        # 全局钩子：检查是否需要向用户提问
        if self._check_ask_user_hook():
            return

        # Planner 已提前给出 final_report（兜底简单回答）：直接结束
        if self.state.final_report and not self.state.steps:
            print(f"[StepPreparer] 跳过: Planner 已给出兜底 final_report")
            return

        if not self.state.steps:
            print(f"[StepPreparer] 跳过: 没有步骤，生成兜底报告")
            self.state.final_report = "抱歉，我没能为您的需求规划出执行步骤，请补充更多信息后再试。"
            return

        step_idx = self.state.current_step_index
        if step_idx >= len(self.state.steps):
            print(f"[StepPreparer] 跳过: 索引超出范围")
            return

        current_step = self.state.steps[step_idx]
        # 跳过已完成的步骤
        if current_step.status == "completed":
            print(f"[StepPreparer] 跳过: 步骤 {step_idx} 已完成")
            return

        # 已有工具：跳过 LLM 规划，但仍然推进到 step_executor
        if current_step.tools:
            print(f"[StepPreparer] 跳过 LLM 规划（已有工具 {current_step.tools}），直接推进到 step_executor")
            self.step_executor()
            return

        print(f"\n{'='*60}")
        print(f"[StepPreparer] 为步骤 {step_idx} 生成执行计划...")
        print(f"{'='*60}")
        self.notify(f"📋 [StepPreparer] 正在为步骤生成执行计划...")

        # 构建上下文信息
        previous_results = {
            str(r.step_index): r.result
            for r in self.state.step_results if r.passed
        }

        inputs = {
            "step_index": step_idx,
            "step_goal": current_step.description,
            "previous_step_results": json.dumps(previous_results, ensure_ascii=False),
            "global_constraints": json.dumps({
                "user_id": self.state.user_id,
                "location": self.state.location,
                "focus": self.state.focus
            }, ensure_ascii=False),
        }

        result = self._run_crew_with_callback(StepPreparerCrew, inputs)
        raw_text = result.raw.strip()

        # 解析执行计划（填充工具调用序列）
        try:
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group(0))
                tools_to_call = plan_data.get("tools_to_call", [])
                if tools_to_call:
                    current_step.tools = [t.get("tool_name", "") for t in tools_to_call]
                    print(f"[StepPreparer] 为步骤 {step_idx} 填充了工具: {current_step.tools}")
                else:
                    print(f"[StepPreparer] 警告: 未找到 tools_to_call")
        except Exception as e:
            print(f"[StepPreparer] 解析执行计划失败: {e}")
            # 默认使用 weather_tool（最常用）
            current_step.tools = ["weather_tool"]

        # 显式调用 step_executor（Flow 的 @listen 不会对直接调用传播）
        self.step_executor()

    # ========================================
    # 【新状态机】状态 3: StepExecutor - 执行工具调用
    # ========================================
    @listen(step_preparer)
    def step_executor(self):
        print(f"[StepExecutor] 监听到 step_preparer 完成")
        if not self.state.steps:
            print(f"[StepExecutor] 跳过: 没有步骤")
            return

        # 全局钩子：检查是否需要向用户提问
        if self._check_ask_user_hook():
            return

        step_idx = self.state.current_step_index
        if step_idx >= len(self.state.steps):
            return

        current_step = self.state.steps[step_idx]
        if current_step.status == "completed":
            return
        # 幂等保护：如果正在执行中则跳过
        if current_step.status == "executing":
            print(f"[StepExecutor] 步骤 {step_idx} 正在执行中，跳过重复调用")
            return
        current_step.status = "executing"

        print(f"\n{'='*60}")
        print(f"[StepExecutor] 执行步骤 {step_idx}: {current_step.description[:50]}...")
        print(f"{'='*60}")
        self.notify(f"🛠️ [StepExecutor] 正在执行工具调用...")

        # 构建执行计划
        execution_plan = {
            "step_index": step_idx,
            "step_goal": current_step.description,
            "tools": current_step.tools
        }

        inputs = {
            "step_index": step_idx,
            "step_goal": current_step.description,
            "execution_plan": json.dumps(execution_plan, ensure_ascii=False),
        }

        result = self._run_crew_with_callback(StepExecutorCrew, inputs)
        raw_text = result.raw.strip()

        # 解析执行结果
        try:
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                exec_data = json.loads(json_match.group(0))
                results = exec_data.get("execution_results", [])

                # 记录执行结果
                step_result = ""
                has_error = False
                for r in results:
                    step_result += f"\n[{r.get('tool_name', 'unknown')}] {str(r.get('output', ''))[:500]}"
                    if r.get('error'):
                        has_error = True

                current_step.result = step_result
                current_step.status = "failed" if has_error else "completed"

                # 记录到历史
                self.state.step_results.append(StepResult(
                    step_index=step_idx,
                    step_description=current_step.description,
                    result=step_result,
                    passed=not has_error,
                    validation_feedback=""
                ))

                print(f"[StepExecutor] 步骤 {step_idx} 执行完成: {'成功' if not has_error else '失败'}")
        except Exception as e:
            print(f"[StepExecutor] 解析结果失败: {e}")
            current_step.result = raw_text[:1000]
            current_step.status = "completed"

        # 显式调用 step_verifier（Flow 的 @listen 不会对直接调用传播）
        self.step_verifier()

    # ========================================
    # 【新状态机】状态 4: StepVerifier - 审核单个步骤结果
    # ========================================
    @listen(step_executor)
    def step_verifier(self):
        if not self.state.steps:
            return

        # 全局钩子：检查是否需要向用户提问
        if self._check_ask_user_hook():
            return

        step_idx = self.state.current_step_index
        if step_idx >= len(self.state.steps):
            return

        current_step = self.state.steps[step_idx]

        # 跳过已处理的步骤（防止 Flow 重复触发）
        if current_step.status in ("completed", "failed") and current_step.validation_feedback:
            print(f"[StepVerifier] 步骤 {step_idx} 已处理过，跳过")
            return

        print(f"\n{'='*60}")
        print(f"[StepVerifier] 开始执行... (current_step_index: {self.state.current_step_index})")
        print(f"{'='*60}")

        if not self.state.steps:
            print(f"[StepVerifier] 跳过: steps={len(self.state.steps) if self.state.steps else 0}")
            return

        # 全局钩子：检查是否需要向用户提问
        if self._check_ask_user_hook():
            return

        step_idx = self.state.current_step_index
        if step_idx >= len(self.state.steps):
            return

        current_step = self.state.steps[step_idx]

        print(f"\n{'='*60}")
        print(f"[StepVerifier] 审核步骤 {step_idx} 结果...")
        print(f"{'='*60}")
        self.notify(f"🔍 [StepVerifier] 正在审核步骤结果...")

        # 【确定性短路】如果步骤已有非空 result 且 status==completed，
        # 直接 pass，不浪费 LLM 调用。这避免 LLM 因"数据不够丰富"挑刺退回 retry。
        if current_step.status == "completed" and current_step.result and current_step.result.strip():
            print(f"[StepVerifier] 短路 pass：步骤 {step_idx} 有非空结果且 StepExecutor 已标记 completed")
            self.notify(f"✅ [StepVerifier] 步骤 {step_idx} 直接通过（有数据）")
            current_step.validation_feedback = "有非空结果，直接通过"
            self.state.current_step_index += 1
            if self.state.current_step_index >= len(self.state.steps):
                self.final_verifier()
            else:
                self.step_preparer()
            return

        # 构建审核输入
        inputs = {
            "step_index": step_idx,
            "step_goal": current_step.description,
            "execution_plan": json.dumps({"tools": current_step.tools}, ensure_ascii=False),
            "execution_results": current_step.result,
        }

        result = self._run_crew_with_callback(StepVerifierCrew, inputs)
        raw_text = result.raw.strip()

        # 解析验证反馈
        feedback = self._parse_step_feedback(raw_text)

        current_step.validation_feedback = feedback.get("reason", "")

        # 处理用户提问
        if feedback.get("verdict") == "ask_user":
            print(f"[StepVerifier] 检测到用户提问指令")
            self.notify(f"🙋 [AskUser] {self.state.user_question}")
            # 设置最终报告为用户提问内容
            self.state.final_report = self.state.user_question
            return

        if feedback.get("verdict") == "pass":
            print(f"[StepVerifier] 步骤 {step_idx} 审核通过")
            self.notify(f"✅ [StepVerifier] 步骤 {step_idx} 审核通过")

            # 标记完成
            current_step.status = "completed"
            current_step.validation_feedback = feedback.get("reason", "通过")

            # 推进到下一个步骤
            self.state.current_step_index += 1
            if self.state.current_step_index >= len(self.state.steps):
                # 所有步骤完成，进入 FinalVerifier
                self.final_verifier()
            else:
                # 还有更多步骤，继续执行下一个步骤
                print(f"[StepVerifier] 准备执行步骤 {self.state.current_step_index}")
                # 调用 step_preparer，Flow 会通过 @listen 自动触发 step_executor
                self.step_preparer()
        elif feedback.get("verdict") == "retry":
            # 重试当前步骤
            retry_count = self.step_retry_counts.get(step_idx, 0)
            if retry_count < self.max_step_retries:
                self.step_retry_counts[step_idx] = retry_count + 1
                print(f"[StepVerifier] 步骤 {step_idx} 重试中 ({retry_count + 1}/{self.max_step_retries})")
                self.notify(f"🔄 [StepVerifier] 步骤 {step_idx} 重试中...")
                # 调用 step_executor 重试，Flow 会通过 @listen 自动触发 step_verifier
                self.step_executor()
            else:
                print(f"[StepVerifier] 步骤 {step_idx} 重试耗尽，标记为失败")
                current_step.status = "failed"
                current_step.validation_feedback = f"重试 {self.max_step_retries} 次后失败"
                self.state.failed_steps_indices.append(step_idx)
                # 推进到下一个步骤
                self.state.current_step_index += 1
                if self.state.current_step_index >= len(self.state.steps):
                    self.final_verifier()
                else:
                    self.step_preparer()
                    self.step_executor()
        else:  # fail - 需要 PartialReplanner
            print(f"[StepVerifier] 步骤 {step_idx} 审核失败，触发 PartialReplanner")
            # 将当前步骤标记为失败
            current_step.status = "failed"
            self.state.failed_steps_indices.append(step_idx)
            self.partial_replanner(feedback)

    # ========================================
    # 【新状态机】状态 5: PartialReplanner - 局部重规划
    # ========================================
    def partial_replanner(self, failure_feedback: dict):
        # 全局钩子：检查是否需要向用户提问
        if self._check_ask_user_hook():
            return

        # 防止无限重规划
        replan_count = getattr(self.state, '_replan_count', 0) + 1
        self.state._replan_count = replan_count
        if replan_count > self.max_replan_attempts:
            print(f"[PartialReplanner] 重规划次数超限 ({replan_count}/{self.max_replan_attempts})，强制结束")
            self.state.final_report = self._generate_final_report()
            self.finalize()
            return

        print(f"\n{'='*60}")
        print(f"[PartialReplanner] 触发局部重规划 (第 {replan_count} 次)...")
        print(f"{'='*60}")
        self.notify(f"🔄 [PartialReplanner] 正在局部重规划...")

        failed_indices = list(set(self.state.failed_steps_indices))
        self.state.failed_steps_indices = []  # 清空，重新开始
        print(f"[PartialReplanner] 失败步骤索引: {failed_indices}")
        if not failed_indices:
            print(f"[PartialReplanner] 没有失败的步骤，无法重规划")
            return

        # 保留已完成的步骤
        preserved_steps = [i for i in range(len(self.state.steps)) if i < min(failed_indices)]
        print(f"[PartialReplanner] 保留步骤: {preserved_steps}")

        # 原始剩余步骤
        original_remaining = [i for i in range(len(self.state.steps)) if i >= min(failed_indices)]
        print(f"[PartialReplanner] 原始剩余步骤: {original_remaining}")

        inputs = {
            "failure_reason": failure_feedback.get("reason", ""),
            "failed_step_indices": json.dumps(failed_indices),
            "suggested_corrections": json.dumps(failure_feedback.get("suggested_corrections", {}), ensure_ascii=False),
            "preserved_steps_results": json.dumps({
                str(i): self.state.steps[i].result
                for i in preserved_steps if i < len(self.state.steps)
            }, ensure_ascii=False),
            "original_remaining_steps": json.dumps(original_remaining),
        }

        result = self._run_crew_with_callback(PartialReplannerCrew, inputs)
        raw_text = result.raw.strip()

        # 解析重规划结果
        try:
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                replan_data = json.loads(json_match.group(0))
                new_steps = replan_data.get("new_coarse_steps", [])

                if new_steps:
                    # 替换失败步骤及其后续为新计划
                    new_step_objects = [StepPlan(**s) for s in new_steps]

                    # 保留已完成步骤
                    preserved = [self.state.steps[i] for i in preserved_steps if i < len(self.state.steps)]

                    # 合并新步骤
                    self.state.steps = preserved + new_step_objects

                    # 更新当前步骤索引为第一个新步骤
                    self.state.current_step_index = len(preserved)

                    print(f"[PartialReplanner] 重规划完成，共 {len(self.state.steps)} 个步骤")
                    print(f"[PartialReplanner] 当前步骤索引: {self.state.current_step_index}")

                    # 调用 step_preparer 开始执行新步骤
                    # Flow 会通过 @listen 自动串联 step_executor → step_verifier
                    self.step_preparer()
                else:
                    print(f"[PartialReplanner] 警告: 重规划未返回新步骤")
            else:
                print(f"[PartialReplanner] 警告: 无法解析重规划结果")
        except Exception as e:
            print(f"[PartialReplanner] 解析重规划结果失败: {e}")

    # ========================================
    # 【新状态机】状态 6: FinalVerifier - 整体审核
    # ========================================
    @listen(step_verifier)
    def final_verifier(self):
        import traceback
        print(f"[FinalVerifier] 被调用，检查是否已执行...")

        # 检查是否已经执行过（避免重复执行）
        if getattr(self.state, '_final_verifier_executed', False):
            print(f"[FinalVerifier] 已执行过，跳过")
            return

        # 全局钩子：检查是否需要向用户提问
        if self._check_ask_user_hook():
            return

        # 只有在所有步骤都完成时才执行最终审核
        if self.state.current_step_index < len(self.state.steps):
            print(f"[FinalVerifier] 跳过: 还有 {len(self.state.steps) - self.state.current_step_index} 个步骤未完成")
            return

        print(f"\n{'='*60}")
        print(f"[FinalVerifier] 开始执行...")
        print(f"{'='*60}")
        self.notify(f"🔍 [FinalVerifier] 正在进行整体审核...")

        # 标记已执行
        self.state._final_verifier_executed = True

        # 收集所有步骤结果
        all_steps_with_results = [
            {
                "index": i,
                "description": s.description,
                "status": s.status,
                "result": s.result
            }
            for i, s in enumerate(self.state.steps)
        ]

        inputs = {
            "all_steps_with_results": json.dumps(all_steps_with_results, ensure_ascii=False),
            "full_plan_document": "\n".join([f"步骤 {i}: {s.description}" for i, s in enumerate(self.state.steps)]),
        }

        result = self._run_crew_with_callback(FinalVerifierCrew, inputs)
        raw_text = result.raw.strip()
        print(f"[FinalVerifier] 原始输出: {raw_text[:500]}")

        # 解析全局验证反馈
        feedback = self._parse_step_feedback(raw_text)
        print(f"[FinalVerifier] 解析反馈: {feedback}")

        if feedback.get("verdict") == "pass":
            print(f"[FinalVerifier] 整体审核通过")
            self.notify(f"🎉 [FinalVerifier] 整体审核通过")

            # 生成最终报告
            self.state.final_report = self._generate_final_report()
            report_len = len(self.state.final_report) if self.state.final_report else 0
            print(f"[FinalVerifier] 生成的最终报告长度: {report_len}")
            print(f"[FinalVerifier] 生成的最终报告内容: {self.state.final_report[:200] if self.state.final_report else 'None'}")
            # 调用 finalize
            self.finalize()
        else:
            print(f"[FinalVerifier] 整体审核不通过，触发局部重规划")
            self.notify(f"⚠️ [FinalVerifier] 整体审核不通过")

            failed_indices = feedback.get("failed_step_ids", [])
            if failed_indices:
                self.state.failed_steps_indices = failed_indices
                self.partial_replanner(feedback.get("global_feedback", {}))
                # 重规划后重新从 StepPreparer 开始
                self.step_preparer()

    # ========================================
    # 辅助函数
    # ========================================

    def _parse_step_feedback(self, raw_text: str) -> dict:
        """
        解析步骤验证反馈。优先识别结构化 JSON：
        - {"verdict": "pass" | "retry" | "fail" | "ask_user", "reason": "...", "question": "..."}
        - FinalVerifier 也支持 global_verdict / failed_step_ids / suggested_corrections

        命中 ask_user 时，会通过 _set_ask_user_question 设置结构化中断标记。
        """
        try:
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                feedback = json.loads(json_match.group(0))

                # FinalVerifier 兼容：global_verdict -> verdict
                if "global_verdict" in feedback and "verdict" not in feedback:
                    feedback["verdict"] = "pass" if feedback["global_verdict"] == "pass" else "fail_with_patches"

                # 结构化用户提问
                if feedback.get("verdict") == "ask_user":
                    question = feedback.get("question") or feedback.get("reason") or "信息不足，请补充。"
                    self._set_ask_user_question(question)
                return feedback
        except Exception:
            pass

        # 没有 JSON：默认通过（保守策略，避免无谓重试）
        print(f"[_parse_step_feedback] 无法解析 JSON，默认通过: {raw_text[:120]}")
        return {"verdict": "pass", "reason": "无法解析反馈，默认通过"}

    def _generate_final_report(self) -> str:
        """
        将全部步骤的执行结果交给 LLM 合成为用户可读的旅游计划。

        步骤的 description 是执行指令（如"查询天气"），result 才是实际数据。
        这里只收集 result，让 LLM 生成最终的用户旅行计划。
        """
        if not self.state.steps:
            return "未能生成行程报告"

        # 收集所有步骤的结果（实际数据，不是执行指令）
        results_text = []
        for s in self.state.steps:
            label = "✅" if s.status == "completed" else "⚠️"
            if s.result:
                results_text.append(f"[{label}] {s.result}")
            elif s.status == "completed":
                results_text.append(f"[{label}] {s.description} - 已完成")
            else:
                results_text.append(f"[{label}] {s.description} - 未完成")

        collected = "\n\n".join(results_text)

        # 把规划阶段做出的假设带给报告生成 LLM，让它在开头明确披露
        assumptions_block = ""
        if self.state.assumptions:
            bullets = "\n".join(f"- {a}" for a in self.state.assumptions)
            assumptions_block = f"\n【系统所做的关键假设（必须在报告开头以 📌 形式向用户披露，并提示用户可调整）】\n{bullets}\n"

        prompt = f"""你是一位资深旅游规划师。请根据以下执行数据，为用户撰写一份完整的旅行计划报告。

【用户需求】
- 目的地: {self.state.location}
- 关注重点: {self.state.focus}
- 用户原话: {self.state.message[:500]}
{assumptions_block}
【已收集的数据】
{collected}

【撰写要求】
1. 绝不要重复原始的执行步骤描述（如"查询天气"、"获取偏好"等）。
2. 把所有数据整合成一份自然流畅的旅行计划，像真人旅游顾问写的那样。
3. 包含以下板块（根据数据情况可省略无数据的板块）：
   - 目的地概况
   - 天气与最佳出行建议
   - 行程安排（按天列出）
   - 预算参考
   - 注意事项

4. 如果某个板块完全没有数据，直接跳过不要硬编。
5. 使用清晰的小标题、emoji 和分段，方便用户在飞书上阅读。
6. 总字数控制在 800 字以内。"""

        try:
            report = zhipu_llm.call([{"role": "user", "content": prompt}]).strip()
            if report and len(report) > 20:
                return report
        except Exception as e:
            print(f"[_generate_final_report] LLM 生成报告失败: {e}")

        # fallback: 返回收集到的原始结果
        if collected:
            return f"📋 行程规划结果\n\n{collected}"
        return "未能生成行程报告"

    def finalize(self):
        """流程结束方法 - 手动调用，不需要 listen 装饰器"""
        # 全局钩子：检查是否需要向用户提问
        if self._check_ask_user_hook():
            return

        print(f"\n{'='*60}")
        print(f"[结束] 流程结束")
        print(f"{'='*60}")
        if not self.state.final_report:
            self.state.final_report = self._generate_final_report() or '未能生成报告'

    # ========================================
    # 【外部统一入口】供 long_conn_bot / main.py / test_agent 调用
    # ========================================
    @classmethod
    def run_for_user(
        cls,
        user_text: str,
        user_id: str,
        session_id: str | None = None,
        memory: "MemoryManager | None" = None,
        status_callback=None,
        content_callback=None,
    ) -> str:
        """
        统一的"对一句用户输入跑完一轮 Flow"入口。

        连接层（飞书、FastAPI、CLI）只需传 user_text + user_id 即可：
        memory / redis 客户端 / prompt 构造 / 状态机调度 / final_report
        生成 / 记忆写回（episodic→working、shortterm→semantic）全部封装在此处。

        Args:
            user_text: 用户输入
            user_id: 用户 id（飞书 open_id / 网页 user_id）
            session_id: 可选会话 id；不传时自动派生
            memory: 可选已有的 MemoryManager（极少用，仅在外部要复用同一会话上下文时）
            status_callback / content_callback: 流式回调

        Returns:
            最终给用户的回复文本（旅行计划 / 用户提问 / 错误信息）
        """
        try:
            sid = session_id or f"sess_{user_id}_{abs(hash(user_text)) % 1000000:06d}"

            # 由 crew 自己构造 MemoryManager（连接层无需关心 redis 客户端）
            if memory is None:
                memory = MemoryManager(sid, user_id, _redis_client, _is_redis_fallback)

            # 1) 写入用户输入到 episodic 记忆
            memory.add_message("user", user_text)

            # 2) 把原始对话蒸馏到 working memory（短期约束提取）
            try:
                memory.convert_episodic_to_working(zhipu_llm)
            except Exception as e:
                print(f"[run_for_user] episodic→working 失败（可忽略）: {e}")

            # 3) 跑 Flow
            flow = cls(status_callback=status_callback, content_callback=content_callback)
            flow.state.message = user_text
            flow.state.user_id = user_id
            flow.state.session_id = sid
            flow.state.focus = memory.get_global_context_prompt(user_text)

            flow.kickoff()

            # 4) 取最终输出，按优先级兜底
            final_report = (flow.state.final_report or "").strip()

            if not final_report:
                if flow.state.needs_user_input and flow.state.user_question:
                    final_report = flow.state.user_question
                else:
                    completed_results = [
                        s.result for s in (flow.state.steps or [])
                        if s.status == "completed" and s.result
                    ]
                    if completed_results:
                        final_report = "\n\n".join(completed_results)
                    elif flow.state.simple_answer:
                        final_report = flow.state.simple_answer
                    else:
                        final_report = "抱歉，本次未能成功生成行程，请稍后重试或补充更多信息。"

            print(f"[run_for_user] final_report 长度: {len(final_report)}, "
                  f"steps: {len(flow.state.steps or [])}, "
                  f"needs_user_input: {flow.state.needs_user_input}")

            # 5) 写助手回复到 episodic
            memory.add_message("assistant", final_report)

            # 6) 异步把短期摘要蒸馏到 semantic（长期偏好）
            try:
                memory.convert_to_semantic(zhipu_llm)
            except Exception as e:
                print(f"[run_for_user] shortterm→semantic 失败（可忽略）: {e}")

            return final_report

        except Exception as e:
            print(f"[TravelWorkflow.run_for_user] 调用失败: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return f"抱歉，处理您的请求时出现了错误：{str(e)}"