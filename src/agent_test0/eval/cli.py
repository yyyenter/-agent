# -*- coding: utf-8 -*-
"""
评估脚本入口
支持 CLI 调用和模块调用
"""

import asyncio
import argparse
import json
from pathlib import Path

from agent_test0.eval.evaluator import AgentEvaluator, EvaluationResult, quick_evaluate
from agent_test0.eval.dataset import TestDataset, load_test_cases


def print_report(report: dict):
    """打印格式化的评估报告"""
    print("\n" + "=" * 60)
    print("AGENT 评估报告")
    print("=" * 60)

    summary = report.get("summary", {})
    scores = report.get("average_scores", {})

    print(f"\n📊 总体 summary")
    print(f"   总测试数: {summary.get('total', 0)}")
    print(f"   通过: {summary.get('passed', 0)}")
    print(f"   失败: {summary.get('failed', 0)}")
    print(f"   通过率: {summary.get('pass_rate', 0):.1%}")

    print(f"\n📈 平均得分 (满分10分)")
    print(f"   综合得分: {scores.get('overall', 0):.2f}")
    print(f"   准确率: {scores.get('accuracy', 0):.2f}")
    print(f"   完整性: {scores.get('completeness', 0):.2f}")
    print(f"   收敛性: {scores.get('convergence', 0):.2f}")

    print(f"\n⚡ 响应速度")
    print(f"   平均延迟: {report.get('average_latency_ms', 0):.0f} ms")

    print("\n" + "=" * 60)


async def cmd_run(args):
    """run 命令"""
    dataset = TestDataset()
    evaluator = AgentEvaluator(dataset, args.url)

    report = await evaluator.run_evaluation(
        tags=args.tags.split(",") if args.tags else None,
        max_cases=args.count
    )

    # 保存报告
    output_file = args.output or f"eval_results_{args.name}.json"
    evaluator.save_report(report, output_file)

    # 打印摘要
    print_report(report)


async def cmd_quick(args):
    """quick 命令 - 快速评估"""
    report = await quick_evaluate(args.url, max_cases=args.count)
    print_report(report)


async def cmd_test(args):
    """test 命令 - 运行单个测试"""
    dataset = TestDataset()
    # 加载测试用例
    try:
        dataset.load_from_file()
    except Exception:
        pass

    # 查找指定测试用例
    test_case = None
    for tc in dataset.test_cases:
        if tc.id == args.test_id or tc.name == args.test_id:
            test_case = tc
            break

    if not test_case:
        print(f"未找到测试用例: {args.test_id}")
        return

    evaluator = AgentEvaluator(dataset, args.url)
    result = await evaluator.evaluate_test_case(test_case)

    print(f"\n🔍 详细结果")
    print(f"输入: {test_case.input_message}")
    print(f"输出: {result.actual_output[:500]}...")
    print(f"得分: {result.metrics.get('overall_score', 0):.2f}")
    print(f"通过: {result.passed}")


def main():
    parser = argparse.ArgumentParser(description="Agent 评估工具")
    subparsers = parser.add_subparsers(dest="command", help="命令")

    # run 命令
    run_parser = subparsers.add_parser("run", help="运行完整评估")
    run_parser.add_argument("--url", default="http://localhost:8000", help="API 地址")
    run_parser.add_argument("--tags", default=None, help="标签过滤, 逗号分隔")
    run_parser.add_argument("--count", type=int, default=None, help="测试用例数量")
    run_parser.add_argument("--name", default="default", help="评估名称")
    run_parser.add_argument("--output", default=None, help="输出文件")

    # quick 命令
    quick_parser = subparsers.add_parser("quick", help="快速评估")
    quick_parser.add_argument("--url", default="http://localhost:8000", help="API 地址")
    quick_parser.add_argument("--count", type=int, default=3, help="测试用例数量")

    # test 命令
    test_parser = subparsers.add_parser("test", help="运行单个测试")
    test_parser.add_argument("test_id", help="测试用例 ID 或名称")
    test_parser.add_argument("--url", default="http://localhost:8000", help="API 地址")

    args = parser.parse_args()

    if args.command == "run":
        asyncio.run(cmd_run(args))
    elif args.command == "quick":
        asyncio.run(cmd_quick(args))
    elif args.command == "test":
        asyncio.run(cmd_test(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
