# Agent 评估体系说明文档

## 概述

本评估体系提供四维量化评估Agent性能:

- **任务完成度**: 准确率、完整性、收敛性
- **智能体协作**: 路由正确率、工具调用有效性、质检反馈质量
- **记忆系统**: 偏好提取准确率、长期记忆利用率、上下文一致性
- **系统健康度**: 响应延迟、错误率、熔断触发率

## 安装依赖

```bash
uv add aiohttp pytest
```

## 使用方法

### 方法 1: Python 代码调用

```python
import asyncio
from agent_test0.eval.evaluator import AgentEvaluator, TestDataset

async def main():
    evaluator = AgentEvaluator(TestDataset(), "http://localhost:8000")
    report = await evaluator.run_evaluation(max_cases=5)
    
    # 打印报告
    print(report)
    
    # 保存报告
    evaluator.save_report(report, "my_eval_report.json")

asyncio.run(main())
```

### 方法 2: CLI 命令

```bash
# 快速评估 (默认3个用例)
uv run python -m agent_test0.eval.cli quick

# 运行完整评估 (按标签过滤)
uv run python -m agent_test0.eval.cli run --tags "planning,shanghai" --count 10

# 运行单个测试
uv run python -m agent_test0.eval.cli test tc_003

# 自定义 API 地址
uv run python -m agent_test0.eval.cli run --url http://localhost:8000
```

### 方法 3: 直接运行脚本

```bash
uv run python src/agent_test0/eval/run_eval.py
```

## 评估指标说明

### 1. 任务完成度

| 指标 | 描述 | 评分范围 |
|------|------|----------|
| accuracy_score | 输出符合用户需求的程度 | 0-10 |
| completeness_score | 行程要素(时间、地点、交通等)完整度 | 0-10 |
| convergence_score | 多少轮次完成(轮次越少越好) | 0-10 |
| overall_score | 综合得分(加权平均) | 0-10 |

### 2. 系统指标

| 指标 | 描述 |
|------|------|
| latency_ms | API 响应延迟 |
| adjustment_count | 质检反馈后的调整轮次 |
| circuit_breaker_triggered | 是否触发熔断器 |

## 测试用例

预设测试用例位于 `src/agent_test0/eval/dataset.py`:

| ID | 名称 | 标签 | 优先级 |
|----|------|------|--------|
| tc_001 | 简单天气查询 | basic, weather | high |
| tc_002 | 简单闲聊 | basic, chitchat | high |
| tc_003 | 基础行程规划 | planning | high |
| tc_004 | 带偏好的行程 | planning, preferences | high |
| tc_005 | 复杂多轮对话 | multi-turn, memory | normal |
| tc_006 | 异常输入 | edge, abnormal | low |
| tc_007 | 预算敏感型规划 | budget | normal |
| tc_008 | 季节性规划 | seasonal | normal |

## 评估报告示例

```json
{
  "timestamp": "2026-06-12T10:30:00",
  "summary": {
    "total": 5,
    "passed": 4,
    "failed": 1,
    "pass_rate": 0.8
  },
  "average_scores": {
    "overall": 7.8,
    "accuracy": 8.2,
    "completeness": 7.5,
    "convergence": 8.0
  },
  "average_latency_ms": 12500,
  "detailed_results": [...]
}
```

## 扩展评估

如需添加自定义评估指标，在 `metrics.py` 中添加新类:

```python
@dataclass
class CustomMetric:
    def calculate(self, output: str, expected: str) -> float:
        # 实现你的评估逻辑
        pass
```

## 在 CrewAI 中集成

评估结果可用于:
- 连续训练: 根据评估结果调整提示词
- A/B 测试: 对比不同模型/配置的性能
- 监控告警: 实时监控 Agent 质量

```python
# 在 main.py 中集成评估
from agent_test0.eval import AgentEvaluator

def evaluate_request(request):
    evaluator = AgentEvaluator()
    result = evaluator.evaluate(request.message)
    return result.metrics
```
