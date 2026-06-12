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
    # === 基础元数据 (保留) ===
    message: str = ''
    user_id: str = 'default_user'
    session_id: str = 'default_sess'

    # === 新状态机字段 ===
    steps: list[StepPlan] = []          # 步骤列表（替代 plan_document）
    current_step_index: int = 0         # 当前执行步骤索引
    step_results: list[StepResult] = [] # 步骤执行结果历史
    failed_steps_indices: list[int] = [] # 失败步骤索引列表

    # === 全局控制字段 ===
    ask_user_question: str = ""         # 如果非空，表示需要向用户提问，流程会停止并返回此问题
    skip_remaining_steps: bool = False  # 如果为 True，跳过剩余步骤直接输出结果

    # === 旧字段（临时保留，用于兼容） ===
    is_complex: bool = True
    simple_answer: str = ""
    location: str = "未知地点"
    focus: str = ""
    tools_needed: list[str] = []
    plan_document: str = ""
    draft_report: str = ""
    previous_draft: str = ""
    last_validation_feedback: str = ""
    final_report: str = ""
    current_step_instruction: str = ""

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
            manager_llm=zhipu_llm,        # 必须指定一个模型作为"包工头"来做路由分配
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
        if self.state.ask_user_question:
            self.notify(f"🙋 [AskUser] {self.state.ask_user_question}")
            self.state.final_report = self.state.ask_user_question
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

    def _compact_feedback_history(self):
        """Compaction: 将旧质检记录压缩为摘要，保留最近 3 条原文，防止 Token 爆炸"""
        if len(self.feedback_history) <= self.MAX_FEEDBACK_ENTRIES:
            return

        # 将最早 N-3 条发送给 LLM 压缩
        old_entries = self.feedback_history[:len(self.feedback_history) - 3]
        recent_entries = self.feedback_history[-3:]

        all_old = "\n---\n".join(old_entries)
        prompt = f"""将以下质检反馈历史压缩为一条不超过{self.COMPACT_SUMMARY_CHARS}字的精炼摘要。
只保留【发现的问题类型】、【是否已修复】、【仍存在的争议点】，丢弃过程性细节和重复内容。

【质检历史】:
{all_old}

【压缩摘要（{self.COMPACT_SUMMARY_CHARS}字以内）】:"""

        try:
            summary = zhipu_llm.call([{"role": "user", "content": prompt}]).strip()
            if len(summary) > self.COMPACT_SUMMARY_CHARS:
                summary = summary[:self.COMPACT_SUMMARY_CHARS] + "..."
            self.feedback_history = [f"[历史质检摘要] {summary}"] + recent_entries
            print(f"[Compaction] feedback_history 从 {len(old_entries) + len(recent_entries)} 条压缩至 {len(self.feedback_history)} 条（1摘要 + 3原文）")
        except Exception as e:
            print(f"[Compaction] LLM压缩失败({e}), 回退到最近3条")
            self.feedback_history = recent_entries

    def _error_fingerprint(self, feedback: str) -> str:
        """从反馈中提取粗粒度指纹，用于检测重复错误"""
        cleaned = re.sub(r'^\[.*?\]\s*', '', feedback).strip()
        return cleaned[:80]

    def _check_circuit_breaker(self, feedback_type: str, feedback: str) -> str | None:
        """检测修正死循环，触发时返回干预消息，否则返回 None"""
        if feedback_type not in ("error", "adjust", "incomplete"):
            self._consecutive_same_error = 0
            self._last_error_key = ""
            return None

        fp = self._error_fingerprint(feedback)
        if fp == self._last_error_key:
            self._consecutive_same_error += 1
        else:
            self._consecutive_same_error = 1
            self._last_error_key = fp

        if self._consecutive_same_error >= 3:
            self._circuit_breaker_triggered = True
            return (
                f"[强制干预] 检测到连续 {self._consecutive_same_error} 次相同类型的错误反馈，"
                f"系统判定已陷入修正死循环。请直接接受当前最佳版本并向前推进。"
                f"上一次的错误反馈是: {feedback[:200]}"
            )

        if self.current_adjust_count >= 5 and self._consecutive_same_error >= 2:
            self._circuit_breaker_triggered = True
            return (
                f"[强制干预] 当前步骤已修改 {self.current_adjust_count} 次，"
                f"其中连续 {self._consecutive_same_error} 次收到相似反馈。"
                f"请忽略细枝末节，接受当前草案并输出 final_report。"
            )

        return None

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

        # 尝试从旧字段兼容转换
        if self.state.plan_document and not self.state.steps:
            self._convert_old_plan_to_steps()

        # 如果还没有步骤列表，调用 PlannerCrew 生成
        if not self.state.steps:
            inputs = {
                "message": self.state.message,
                "user_id": self.state.user_id,
                "focus": self.state.focus,
                "previous_plan": self.state.plan_document or "无历史计划（首次规划）",
                "current_step": self.state.current_step_instruction or "无当前工单（首次执行）",
                "current_draft": self.state.draft_report or "无进度草案",
            }

            result = self._run_crew_with_callback(PlannerCrew, inputs)
            raw_text = result.raw.strip()

            # 解析 Planner 输出的 JSON，提取步骤列表
            try:
                json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
                if json_match:
                    plan_data = json.loads(json_match.group(0))
                    self.state.is_complex = plan_data.get("is_complex", True)
                    self.state.simple_answer = plan_data.get("simple_answer", "")
                    self.state.location = plan_data.get("location", "未知")
                    self.state.focus = plan_data.get("focus", "")
                    self.state.tools_needed = plan_data.get("tools_needed", [])

                    # 提取步骤列表
                    steps = plan_data.get("steps", [])
                    if steps:
                        self.state.steps = [StepPlan(**s) for s in steps]
                        print(f"[Planner] 生成了 {len(self.state.steps)} 个粗粒度步骤")
                    else:
                        # 兼容：如果没有 steps 字段，使用旧的 current_step_instruction
                        step_instruction = plan_data.get("current_step_instruction", "")
                        if step_instruction and not step_instruction.startswith("[提问]"):
                            self.state.steps = [StepPlan(
                                index=0,
                                description=step_instruction,
                                status="pending"
                            )]

                    # 提取当前步骤工单（用于旧版兼容）
                    step_instruction = plan_data.get("current_step_instruction", "")
                    if step_instruction:
                        self.state.current_step_instruction = step_instruction
            except Exception as e:
                print(f"[Planner] 解析失败: {e}")
                self.state.is_complex = True

        # 初始化当前步骤索引
        if self.state.steps:
            self.state.current_step_index = 0

        # 检查是否需要向用户提问（信息不足）
        if self.state.current_step_instruction and self.state.current_step_instruction.startswith("[提问]"):
            self._set_ask_user_question()
            return

    # ========================================
    # 【新状态机】用户提问辅助方法
    # ========================================
    def _set_ask_user_question(self, question: str = None):
        """设置需要向用户提问的问题，流程会在下一个钩子检查时中断"""
        if question is None and self.state.current_step_instruction:
            question = self.state.current_step_instruction.replace("[提问]", "").strip()
        self.state.ask_user_question = question or "信息不足，无法继续规划。"
        print(f"\n{'='*60}")
        print(f"[AskUser] 信息不足，需要向用户提问...")
        print(f"{'='*60}")
        self.notify(f"🙋 [AskUser] {self.state.ask_user_question}")

    # ========================================
    # 【新状态机】状态 2: StepPreparer - 为当前步骤生成执行计划
    # ========================================
    @listen(plan_steps)
    def step_preparer(self):
        if not self.state.is_complex or not self.state.steps:
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

        result = self._run_crew_with_callback(TravelExpertCrew, inputs)
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
        except Exception as e:
            print(f"[StepPreparer] 解析执行计划失败: {e}")
            # 默认使用 weather_tool（最常用）
            current_step.tools = ["weather_tool", "read_memory_tool"]

    # ========================================
    # 【新状态机】状态 3: StepExecutor - 执行工具调用
    # ========================================
    @listen(step_preparer)
    def step_executor(self):
        if not self.state.is_complex or not self.state.steps:
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

        result = self._run_crew_with_callback(TravelExpertCrew, inputs)
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

    # ========================================
    # 【新状态机】状态 4: StepVerifier - 审核单个步骤结果
    # ========================================
    @listen(step_executor)
    def step_verifier(self):
        if not self.state.is_complex or not self.state.steps:
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

        # 构建审核输入
        inputs = {
            "step_index": step_idx,
            "step_goal": current_step.description,
            "execution_plan": json.dumps({"tools": current_step.tools}, ensure_ascii=False),
            "execution_results": current_step.result,
        }

        result = self._run_crew_with_callback(ValidatorCrew, inputs)
        raw_text = result.raw.strip()

        # 解析验证反馈
        feedback = self._parse_step_feedback(raw_text)

        current_step.validation_feedback = feedback.get("reason", "")

        # 处理用户提问
        if feedback.get("verdict") == "ask_user":
            print(f"[StepVerifier] 检测到用户提问指令")
            self.notify(f"🙋 [AskUser] {self.state.ask_user_question}")
            return

        if feedback.get("verdict") == "pass":
            print(f"[StepVerifier] 步骤 {step_idx} 审核通过")
            self.notify(f"✅ [StepVerifier] 步骤 {step_idx} 审核通过")

            # 推进到下一个步骤
            self.state.current_step_index += 1
            if self.state.current_step_index >= len(self.state.steps):
                # 所有步骤完成，进入 FinalVerifier
                self.final_verifier()
        elif feedback.get("verdict") == "retry":
            # 重试当前步骤
            retry_count = self.step_retry_counts.get(step_idx, 0)
            if retry_count < self.max_step_retries:
                self.step_retry_counts[step_idx] = retry_count + 1
                print(f"[StepVerifier] 步骤 {step_idx} 重试中 ({retry_count + 1}/{self.max_step_retries})")
                self.notify(f"🔄 [StepVerifier] 步骤 {step_idx} 重试中...")
                self.step_executor()
            else:
                print(f"[StepVerifier] 步骤 {step_idx} 重试耗尽，标记为失败")
                current_step.status = "failed"
                self.state.failed_steps_indices.append(step_idx)
                # 推进到下一个步骤
                self.state.current_step_index += 1
                if self.state.current_step_index >= len(self.state.steps):
                    self.final_verifier()
        else:  # fail - 需要 PartialReplanner
            print(f"[StepVerifier] 步骤 {step_idx} 审核失败，触发 PartialReplanner")
            self.partial_replanner(feedback)

    # ========================================
    # 【新状态机】状态 5: PartialReplanner - 局部重规划
    # ========================================
    def partial_replanner(self, failure_feedback: dict):
        # 全局钩子：检查是否需要向用户提问
        if self._check_ask_user_hook():
            return

        print(f"\n{'='*60}")
        print(f"[PartialReplanner] 触发局部重规划...")
        print(f"{'='*60}")
        self.notify(f"🔄 [PartialReplanner] 正在局部重规划...")

        failed_indices = self.state.failed_steps_indices
        if not failed_indices:
            return

        # 保留已完成的步骤
        preserved_steps = [i for i in range(len(self.state.steps)) if i < min(failed_indices)]

        # 原始剩余步骤
        original_remaining = [i for i in range(len(self.state.steps)) if i >= min(failed_indices)]

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

        result = self._run_crew_with_callback(PlannerCrew, inputs)
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
        except Exception as e:
            print(f"[PartialReplanner] 解析重规划结果失败: {e}")

    # ========================================
    # 【新状态机】状态 6: FinalVerifier - 整体审核
    # ========================================
    def final_verifier(self):
        # 全局钩子：检查是否需要向用户提问
        if self._check_ask_user_hook():
            return

        print(f"\n{'='*60}")
        print(f"[FinalVerifier] 进行整体审核...")
        print(f"{'='*60}")
        self.notify(f"🔍 [FinalVerifier] 正在进行整体审核...")

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

        result = self._run_crew_with_callback(ValidatorCrew, inputs)
        raw_text = result.raw.strip()

        # 解析全局验证反馈
        feedback = self._parse_step_feedback(raw_text)

        if feedback.get("verdict") == "pass":
            print(f"[FinalVerifier] 整体审核通过")
            self.notify(f"🎉 [FinalVerifier] 整体审核通过")

            # 生成最终报告
            self.state.final_report = self._generate_final_report()
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

    def _convert_old_plan_to_steps(self):
        """将旧格式的 plan_document 转换为 steps 列表"""
        # 简单的兼容转换：将 plan_document 按行分割作为步骤
        if not self.state.plan_document:
            return

        lines = self.state.plan_document.split('\n')
        steps = []
        for i, line in enumerate(lines[:8]):  # 最多 8 个步骤
            line = line.strip()
            if line and not line.startswith('```'):
                steps.append(StepPlan(index=i, description=line))

        if steps:
            self.state.steps = steps
            print(f"[Compat] 将旧格式 plan_document 转换为 {len(steps)} 个步骤")

    def _parse_step_feedback(self, raw_text: str) -> dict:
        """解析步骤验证反馈（支持 JSON 格式）"""
        # 检查用户提问标签 [提问]
        if raw_text.startswith("[提问]"):
            question = raw_text.replace("[提问]", "").strip()
            self.state.ask_user_question = question
            return {"verdict": "ask_user", "feedback": {"passed": True, "question": question}}

        # 尝试解析 JSON
        try:
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                feedback = json.loads(json_match.group(0))
                return feedback
        except Exception:
            pass

        # 兼容旧格式解析
        if raw_text.startswith("[通过]"):
            return {"verdict": "pass", "feedback": {"passed": True}}
        elif raw_text.startswith("[重试]"):
            return {"verdict": "retry", "feedback": {"passed": False}}
        elif raw_text.startswith("[失败步骤列表]"):
            return {"verdict": "fail_with_patches", "failed_step_ids": [self.state.current_step_index]}

        return {"verdict": "pass", "feedback": {"passed": True}}

    def _generate_final_report(self) -> str:
        """生成最终报告"""
        if not self.state.steps:
            return "未能生成行程报告"

        report_parts = ["# 旅行行程规划报告\n"]

        for step in self.state.steps:
            status_icon = "✅" if step.status == "completed" else "⚠️"
            report_parts.append(f"\n## 步骤 {step.index}: {step.description}\n")
            report_parts.append(f"状态: {status_icon} {step.status}\n")
            if step.result:
                report_parts.append(f"结果:\n```\n{step.result[:2000]}\n```\n")

        return "\n".join(report_parts)

    @listen(final_verifier)
    def finalize(self):
        # 全局钩子：检查是否需要向用户提问
        if self._check_ask_user_hook():
            return

        print(f"\n{'='*60}")
        print(f"[结束] 流程结束")
        print(f"{'='*60}")
        if not self.state.final_report:
            self.state.final_report = self._generate_final_report() or '未能生成报告'