# -*- coding: utf-8 -*-
"""
Agent 评估系统 - 快速入门示例
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent_test0.eval import AgentEvaluator, TestDataset, load_test_cases


async def example_basic_usage():
    """示例 1: 基础使用"""
    print("=" * 60)
    print("示例 1: 基础使用")
    print("=" * 60)

    # 创建评估器
    dataset = TestDataset()
    evaluator = AgentEvaluator(dataset, "http://localhost:8000")

    # 运行评估
    report = await evaluator.run_evaluation(max_cases=2)

    # 打印结果
    print(f"\n评估完成!")
    print(f"总测试数: {report['summary']['total']}")
    print(f"通过率: {report['summary']['pass_rate']:.1%}")
    print(f"平均得分: {report['average_scores']['overall']:.2f}/10")


async def example_filter_tags():
    """示例 2: 按标签过滤"""
    print("\n" + "=" * 60)
    print("示例 2: 按标签过滤")
    print("=" * 60)

    dataset = TestDataset()
    # 只评估 planning 类型的测试用例
    planning_cases = dataset.filter_test_cases(tags=["planning"])
    print(f"\n找到 {len(planning_cases)} 个 planning 类型的测试用例")


async def example_single_test():
    """示例 3: 单个测试用例"""
    print("\n" + "=" * 60)
    print("示例 3: 单个测试用例")
    print("=" * 60)

    dataset = TestDataset()
    evaluator = AgentEvaluator(dataset, "http://localhost:8000")

    # 获取单个测试用例
    test_case = dataset.test_cases[2]  # tc_003
    print(f"\n测试用例: {test_case.name}")
    print(f"输入: {test_case.input_message}")

    # 运行评估
    result = await evaluator.evaluate_test_case(test_case)

    print(f"\n评估结果:")
    print(f"  输出: {result.actual_output[:200]}...")
    print(f"  得分: {result.metrics.get('overall_score', 0):.2f}/10")
    print(f"  通过: {result.passed}")


async def example_benchmark():
    """示例 4: 运行基准测试"""
    print("\n" + "=" * 60)
    print("示例 4: 基准测试")
    print("=" * 60)

    dataset = TestDataset()
    evaluator = AgentEvaluator(dataset, "http://localhost:8000")

    # 运行基准测试
    report = await evaluator.run_evaluation(tags=["benchmark"])

    print(f"\n基准测试完成!")
    print(f"平均得分: {report['average_scores']['overall']:.2f}/10")


async def main():
    """主函数 - 运行所有示例"""
    print("\n🚀 Agent 评估系统 - 示例集合")
    print("请确保 API 服务正在运行: uv run python src/agent_test0/main.py")

    # 示例 1: 基础使用
    try:
        await example_basic_usage()
    except Exception as e:
        print(f"\n示例 1 失败 (API 连接可能失败): {e}")

    # 示例 2: 按标签过滤
    try:
        await example_filter_tags()
    except Exception as e:
        print(f"\n示例 2 失败: {e}")

    # 示例 3: 单个测试用例
    try:
        await example_single_test()
    except Exception as e:
        print(f"\n示例 3 失败 (API 连接可能失败): {e}")

    # 示例 4: 基准测试
    try:
        await example_benchmark()
    except Exception as e:
        print(f"\n示例 4 失败 (API 连接可能失败): {e}")

    print("\n" + "=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
