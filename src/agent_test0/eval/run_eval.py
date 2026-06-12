# -*- coding: utf-8 -*-
"""
Agent 评估示例脚本
"""

import asyncio
import json
from pathlib import Path

from agent_test0.eval.evaluator import AgentEvaluator
from agent_test0.eval.dataset import TestDataset


async def main():
    """主函数 - 运行评估"""
    print("🚀 Agent 评估系统")
    print("=" * 50)

    # 1. 创建评估器
    dataset = TestDataset()
    evaluator = AgentEvaluator(dataset, "http://localhost:8000")

    # 2. 运行评估 (最多 3 个测试用例)
    print("\n开始评估...")
    report = await evaluator.run_evaluation(max_cases=3)

    # 3. 保存报告
    output_file = "eval_report.json"
    evaluator.save_report(report, output_file)

    # 4. 打印摘要
    summary = report.get("summary", {})
    scores = report.get("average_scores", {})

    print("\n" + "=" * 50)
    print("评估摘要")
    print("=" * 50)
    print(f"总测试数: {summary.get('total', 0)}")
    print(f"通过: {summary.get('passed', 0)} | 失败: {summary.get('failed', 0)}")
    print(f"通过率: {summary.get('pass_rate', 0):.1%}")
    print(f"\n平均得分:")
    print(f"  综合得分: {scores.get('overall', 0):.2f}/10")
    print(f"  准确率: {scores.get('accuracy', 0):.2f}")
    print(f"  完整性: {scores.get('completeness', 0):.2f}")
    print(f"  收敛性: {scores.get('convergence', 0):.2f}")
    print(f"  平均延迟: {report.get('average_latency_ms', 0):.0f}ms")

    print(f"\n报告已保存到: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
