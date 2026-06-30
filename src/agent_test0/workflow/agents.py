"""
LangGraph 节点内部使用的 CrewAI Agent 集合。
外层状态机由 LangGraph 控制，每个节点内部用独立的 Agent 执行具体任务。

Agent 配置从 config/agent.yaml 加载，遵循 CrewAI 配置即代码理念。
"""
from pathlib import Path
import yaml

from crewai import Agent

from agent_test0.workflow.llm import zhipu_llm


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def _load_agent_config(name: str) -> dict:
    """从 agent.yaml 加载指定 agent 的配置。"""
    path = CONFIG_DIR / "agent.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data[name]


def create_agent(name: str, **kwargs) -> Agent:
    """工厂函数：从 YAML 配置 + 覆盖参数创建 Agent。"""
    cfg = _load_agent_config(name)
    cfg.update(kwargs)
    return Agent(
        role=cfg["role"],
        goal=cfg["goal"],
        backstory=cfg["backstory"],
        llm=cfg.get("llm", zhipu_llm),
        verbose=cfg.get("verbose", False),
        reasoning=cfg.get("reasoning", kwargs.get("reasoning", False)),
    )


# ============================================================
# 导出 5 个专用 Agent（YAML 配置驱动，不改人设只改 YAML）
# ============================================================
planner_agent = create_agent("planner_agent", reasoning=True)
step_preparer_agent = create_agent("step_preparer_agent")
step_verifier_agent = create_agent("step_verifier_agent")
partial_replanner_agent = create_agent("partial_replanner_agent")
final_verifier_agent = create_agent("final_verifier_agent")
