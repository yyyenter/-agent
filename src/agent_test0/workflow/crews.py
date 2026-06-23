# agent_test0/workflow/crews.py
"""
7 个状态节点对应的 Crew 定义集中放置。

每个 Crew 都是 CrewAI 的"配置容器"，绑定 yaml 中的 agent + task 定义。
Flow 节点通过 `run_crew_with_callback(flow, CrewClass, inputs)` 调用它们。

新加 Crew 的步骤：
  1. 在 config/agent.yaml 里声明 agent
  2. 在 config/<task>.yaml 里声明 task
  3. 在本文件加一个 @CrewBase 类，绑定 agent + task

LLM 共享自 workflow.llm（zhipu_llm）。
"""

from crewai import Agent, Crew, Task
from crewai.project import CrewBase, agent, task, crew

from agent_test0.tools.custom_tool import WeatherTool
from agent_test0.workflow.llm import zhipu_llm, search_tool


@CrewBase
class PlannerCrew:
    """状态 1：复杂度判定 + 步骤生成"""
    agents_config = '../config/agent.yaml'
    tasks_config = '../config/tasks.yaml'

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
class StepPreparerCrew:
    """状态 2：为粗粒度步骤生成工具调用计划"""
    agents_config = '../config/agent.yaml'
    tasks_config = '../config/step_preparer_tasks.yaml'

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
    """状态 3：执行工具调用（绑定 WeatherTool）"""
    agents_config = '../config/agent.yaml'
    tasks_config = '../config/executor_tasks.yaml'

    @agent
    def step_executor_agent(self) -> Agent:
        tools = [WeatherTool()]
        if search_tool is not None:
            tools.append(search_tool)
        return Agent(config=self.agents_config['info_search_agent'], tools=tools, llm=zhipu_llm, verbose=True)

    @task
    def executor_task(self) -> Task:
        return Task(config=self.tasks_config['executor_task'])

    @crew
    def crew(self) -> Crew:
        return Crew(agents=self.agents, tasks=self.tasks, verbose=True)


@CrewBase
class StepVerifierCrew:
    """状态 4：单步骤结果审核"""
    agents_config = '../config/agent.yaml'
    tasks_config = '../config/step_validator_tasks.yaml'

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
    """状态 5：仅修复失败步骤，不重新规划全局"""
    agents_config = '../config/agent.yaml'
    tasks_config = '../config/replan_tasks.yaml'

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
    """状态 6：整体审核（liberal pass）"""
    agents_config = '../config/agent.yaml'
    tasks_config = '../config/final_validator_tasks.yaml'

    @agent
    def final_verifier_agent(self) -> Agent:
        return Agent(config=self.agents_config['logic_validator_agent'], tools=[], llm=zhipu_llm, verbose=True)

    @task
    def final_validator_task(self) -> Task:
        return Task(config=self.tasks_config['final_validator_task'])

    @crew
    def crew(self) -> Crew:
        return Crew(agents=self.agents, tasks=self.tasks, verbose=True)


__all__ = [
    "PlannerCrew",
    "StepPreparerCrew",
    "StepExecutorCrew",
    "StepVerifierCrew",
    "PartialReplannerCrew",
    "FinalVerifierCrew",
]
