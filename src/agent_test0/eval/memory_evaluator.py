# -*- coding: utf-8 -*-
"""
记忆系统评估 - 专门评估记忆相关的功能
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class MemoryMetricType(Enum):
    EXTRACTION = "extraction"      # 偏好提取准确率
    RETENTION = "retention"        # 记忆保持准确率
    CONSISTENCY = "consistency"    # 上下文一致性
    RELEVANCE = "relevance"        # 记忆相关性


@dataclass
class MemoryEvaluationResult:
    """记忆评估结果"""
    metric_type: MemoryMetricType
    score: float  # 0-10
    details: Dict[str, any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)


class MemoryEvaluator:
    """记忆系统评估器"""

    def __init__(self):
        self.results: List[MemoryEvaluationResult] = []

    def evaluate_preference_extraction(
        self,
        user_message: str,
        extracted_preferences: Dict[str, str],
        expected_preferences: Dict[str, str]
    ) -> MemoryEvaluationResult:
        """
        评估偏好提取的准确率

        Args:
            user_message: 用户输入
            extracted_preferences: 提取出的偏好
            expected_preferences: 期望的偏好

        Returns:
            MemoryEvaluationResult
        """
        score = 10.0
        issues = []
        details = {
            "user_message": user_message,
            "extracted": extracted_preferences,
            "expected": expected_preferences,
        }

        # 检查每个期望的偏好
        for key, expected_value in expected_preferences.items():
            if key not in extracted_preferences:
                score -= 2.0
                issues.append(f"缺少偏好: {key}")
            elif extracted_preferences[key].lower() != expected_value.lower():
                score -= 1.0
                issues.append(f"偏好值不匹配: {key} (期望: {expected_value}, 实际: {extracted_preferences[key]})")

        # 检查是否有误提取
        for key in extracted_preferences:
            if key not in expected_preferences:
                score -= 0.5
                issues.append(f"误提取: {key}")

        result = MemoryEvaluationResult(
            metric_type=MemoryMetricType.EXTRACTION,
            score=max(0.0, min(10.0, score)),
            details=details,
            issues=issues
        )
        self.results.append(result)
        return result

    def evaluate_context_consistency(
        self,
        history_messages: List[Dict[str, str]],
        current_response: str
    ) -> MemoryEvaluationResult:
        """
        评估上下文一致性 - 检查是否记得之前的对话

        Args:
            history_messages: 历史消息列表
            current_response: 当前响应

        Returns:
            MemoryEvaluationResult
        """
        score = 10.0
        issues = []
        details = {
            "history_length": len(history_messages),
            "response_length": len(current_response),
        }

        # 检查是否引用了历史信息
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history_messages])

        # 如果历史中有提到特定地点，检查响应是否相关
        location_mentions = re.findall(r'(北京|上海|广州|深圳|成都|杭州|西安|哈尔滨|苏州|杭州)', history_text)
        if location_mentions:
            response_has_location = any(loc in current_response for loc in location_mentions)
            if not response_has_location:
                score -= 1.5
                issues.append("未提及历史对话中提到的地点")

        # 检查是否记得用户偏好
        preference_keywords = ["之前", "之前说", "记得", "你提到", "说过"]
        if any(kw in current_response for kw in preference_keywords):
            score += 0.5  # 奖励记得历史信息
        else:
            # 如果历史有明确偏好，但响应没有体现
            if "喜欢" in history_text or "偏好" in history_text:
                score -= 1.0
                issues.append("未体现对历史偏好的考虑")

        result = MemoryEvaluationResult(
            metric_type=MemoryMetricType.CONSISTENCY,
            score=max(0.0, min(10.0, score)),
            details=details,
            issues=issues
        )
        self.results.append(result)
        return result

    def evaluate_memory_relevance(
        self,
        user_query: str,
        retrieved_memory: List[Dict[str, str]],
        response: str
    ) -> MemoryEvaluationResult:
        """
        评估记忆的相关性

        Args:
            user_query: 用户查询
            retrieved_memory: 检索到的记忆
            response: 生成的响应

        Returns:
            MemoryEvaluationResult
        """
        score = 10.0
        issues = []
        details = {
            "query": user_query,
            "memory_count": len(retrieved_memory),
            "memory_used": retrieved_memory,
        }

        if not retrieved_memory:
            # 检查是否应该检索到记忆
            if any(kw in user_query for kw in ["之前", "之前说", "记得", "喜欢"]):
                score -= 2.0
                issues.append("应该检索到用户记忆但未检索到")

        # 检查响应是否有效利用了记忆
        if retrieved_memory:
            memory_text = "\n".join([f"{m['memory_key']}: {m['memory_value']}" for m in retrieved_memory])
            if memory_text not in response and not any(m['memory_value'] in response for m in retrieved_memory):
                score -= 1.0
                issues.append("检索到记忆但未在响应中体现")

        result = MemoryEvaluationResult(
            metric_type=MemoryMetricType.RELEVANCE,
            score=max(0.0, min(10.0, score)),
            details=details,
            issues=issues
        )
        self.results.append(result)
        return result

    def get_summary(self) -> Dict[str, any]:
        """获取评估摘要"""
        if not self.results:
            return {"error": "No evaluation results"}

        summary = {
            "total_evaluations": len(self.results),
            "average_scores": {},
            "issue_count": 0,
        }

        # 计算平均分
        for metric_type in MemoryMetricType:
            type_results = [r for r in self.results if r.metric_type == metric_type]
            if type_results:
                avg_score = sum(r.score for r in type_results) / len(type_results)
                summary["average_scores"][metric_type.value] = avg_score

        # 统计问题
        summary["issue_count"] = sum(len(r.issues) for r in self.results)
        summary["issues"] = [issue for r in self.results for issue in r.issues]

        return summary

    def get_detailed_report(self) -> List[Dict]:
        """获取详细报告"""
        return [
            {
                "metric_type": r.metric_type.value,
                "score": r.score,
                "issues": r.issues,
                "details": r.details,
            }
            for r in self.results
        ]


# 快速评估函数
def evaluate_memory_quick(
    user_message: str,
    extracted_preferences: Dict[str, str],
    expected_preferences: Dict[str, str]
) -> float:
    """快速评估偏好提取"""
    evaluator = MemoryEvaluator()
    result = evaluator.evaluate_preference_extraction(
        user_message, extracted_preferences, expected_preferences
    )
    return result.score


# 示例
if __name__ == "__main__":
    # 示例 1: 偏好提取评估
    evaluator = MemoryEvaluator()

    result = evaluator.evaluate_preference_extraction(
        user_message="我想去成都玩，要吃辣的，预算5000",
        extracted_preferences={"food": "辣", "budget": "5000"},
        expected_preferences={"food": "辣", "budget": "5000", "location": "成都"}
    )

    print(f"偏好提取评估: {result.score}/10")
    print(f"问题: {result.issues}")

    # 示例 2: 上下文一致性评估
    history = [
        {"role": "user", "content": "我喜欢历史景点"},
        {"role": "assistant", "content": "好的，我会推荐历史景点"},
    ]
    response = "这是北京行程..."

    result = evaluator.evaluate_context_consistency(history, response)
    print(f"\n上下文一致性评估: {result.score}/10")
    print(f"问题: {result.issues}")

    # 打印摘要
    print(f"\n评估摘要: {evaluator.get_summary()}")