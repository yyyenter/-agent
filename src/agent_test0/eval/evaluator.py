# -*- coding: utf-8 -*-
"""
自动化评估器
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path

from agent_test0.eval.metrics import TaskCompletionMetric
from agent_test0.eval.dataset import TestDataset, TestCase


@dataclass
class EvaluationResult:
    """评估结果"""
    test_case: TestCase
    actual_output: str
    metrics: Dict[str, Any]
    timestamps: Dict[str, float] = field(default_factory=dict)
    error: str = None
    passed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_case": asdict(self.test_case),
            "actual_output": self.actual_output[:500] + "..." if len(self.actual_output) > 500 else self.actual_output,
            "metrics": self.metrics,
            "timestamps": self.timestamps,
            "error": self.error,
            "passed": self.passed,
        }


class AgentEvaluator:
    """Agent 自动化评估器"""

    def __init__(self, test_dataset: TestDataset = None, api_base_url: str = "http://localhost:8000"):
        """
        初始化评估器

        Args:
            test_dataset: 测试数据集
            api_base_url: API 服务地址
        """
        self.test_dataset = test_dataset or TestDataset()
        self.api_base_url = api_base_url.rstrip('/')
        self.results: List[EvaluationResult] = []

        # 异步客户端
        self._session = None

    async def _call_api(self, test_case: TestCase) -> Tuple[str, float, Dict]:
        """
        调用 Agent API 并返回结果

        Returns:
            (output_text, latency_ms, metadata)
        """
        import aiohttp

        payload = {
            "user_id": "eval_user",
            "session_id": f"eval_{test_case.id}_{int(time.time())}",
            "message": test_case.input_message,
        }

        start_time = time.time()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_base_url}/api/chat_stream",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return "", 0, {"error": f"HTTP {response.status}: {error_text}"}

                    # 读取 SSE 流
                    output_parts = []
                    metadata = {"events": []}

                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if not line:
                            continue

                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                metadata["events"].append(data)

                                if data.get("type") == "finish":
                                    output_parts.append(data.get("content", ""))
                                elif data.get("type") == "status":
                                    # 状态信息用于分析
                                    pass
                            except json.JSONDecodeError:
                                pass

                    output_text = "\n".join(output_parts)
                    latency_ms = (time.time() - start_time) * 1000

                    return output_text, latency_ms, metadata

        except asyncio.TimeoutError:
            return "", 0, {"error": "Request timeout"}
        except Exception as e:
            return "", 0, {"error": str(e)}

    async def evaluate_test_case(self, test_case: TestCase) -> EvaluationResult:
        """评估单个测试用例"""
        result = EvaluationResult(test_case=test_case, actual_output="")

        print(f"\n{'='*50}")
        print(f" evaluating: {test_case.name} (ID: {test_case.id})")
        print(f"{'='*50}")

        try:
            # 调用 API
            output, latency_ms, metadata = await self._call_api(test_case)

            result.actual_output = output
            result.timestamps["request_start"] = 0  # 相对时间不记录
            result.timestamps["request_end"] = latency_ms

            # 计算指标
            metric_calculator = TaskCompletionMetric()

            # 任务完成度分析
            metric_calculator.analyze_response(output, test_case.expected_elements)

            # 分析收敛性 (从 metadata 中提取)
            adjustment_count = sum(1 for e in metadata.get("events", []) if "调整" in str(e))
            circuit_breaker = any("熔断" in str(e) for e in metadata.get("events", []))

            metric_calculator.analyze_convergence(adjustment_count, circuit_breaker)

            result.metrics = metric_calculator.to_dict()
            result.metrics["latency_ms"] = latency_ms

            # 判定是否通过
            result.passed = (
                metric_calculator.overall_score >= 6.0 and
                latency_ms < 30000  # 30秒超时
            )

            print(f"[Result] Score: {result.metrics['overall_score']}/10 | "
                  f"Latency: {latency_ms:.0f}ms | "
                  f"Adjustments: {adjustment_count} | "
                  f"Passed: {result.passed}")

            if not result.passed:
                print(f"[Info] Issues: {json.dumps(metric_calculator.to_dict(), ensure_ascii=False)}")

        except Exception as e:
            result.error = str(e)
            result.passed = False
            print(f"[Error] {e}")

        self.results.append(result)
        return result

    async def run_evaluation(self, tags: List[str] = None, max_cases: int = None) -> Dict[str, Any]:
        """
        运行完整评估

        Args:
            tags: 过滤标签
            max_cases: 最大测试用例数

        Returns:
            评估汇总报告
        """
        # 获取测试用例
        test_cases = self.test_dataset.filter_test_cases(tags=tags)
        if max_cases:
            test_cases = test_cases[:max_cases]

        print(f"\n{'#'*60}")
        print(f"# 开始评估: {len(test_cases)} 个测试用例")
        print(f"# {'#' * 56}")

        # 并发执行评估
        tasks = [self.evaluate_test_case(tc) for tc in test_cases]
        await asyncio.gather(*tasks)

        # 生成汇总报告
        return self._generate_report()

    def _generate_report(self) -> Dict[str, Any]:
        """生成评估报告"""
        if not self.results:
            return {"error": "No evaluation results"}

        # 统计指标
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed

        avg_scores = {
            "overall": sum(r.metrics.get("overall_score", 0) for r in self.results) / total if total > 0 else 0,
            "accuracy": sum(r.metrics.get("accuracy_score", 0) for r in self.results) / total if total > 0 else 0,
            "completeness": sum(r.metrics.get("completeness_score", 0) for r in self.results) / total if total > 0 else 0,
            "convergence": sum(r.metrics.get("convergence_score", 0) for r in self.results) / total if total > 0 else 0,
        }

        avg_latency = sum(r.metrics.get("latency_ms", 0) for r in self.results) / total if total > 0 else 0

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "pass_rate": passed / total if total > 0 else 0,
            },
            "average_scores": avg_scores,
            "average_latency_ms": avg_latency,
            "detailed_results": [r.to_dict() for r in self.results],
        }

        return report

    def save_report(self, report: Dict[str, Any], filepath: str = None):
        """保存评估报告到文件"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"eval_results_{timestamp}.json"

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n报告已保存到: {filepath}")
        return filepath


# 快速评估函数
async def quick_evaluate(api_base_url: str = "http://localhost:8000", max_cases: int = 3) -> Dict:
    """快速评估函数"""
    dataset = TestDataset()
    evaluator = AgentEvaluator(dataset, api_base_url)
    return await evaluator.run_evaluation(max_cases=max_cases)


if __name__ == "__main__":
    # 示例用法
    async def main():
        evaluator = AgentEvaluator()
        report = await evaluator.run_evaluation(max_cases=2)
        evaluator.save_report(report)

    asyncio.run(main())
