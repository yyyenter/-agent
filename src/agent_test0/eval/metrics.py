# -*- coding: utf-8 -*-
"""
任务完成度评估指标
"""

import re
import json
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class CompletionGrade(Enum):
    EXCELLENT = 5  # 优秀
    GOOD = 4         # 良好
    FAIR = 3         # 中等
    POOR = 2         # 及格
    FAILED = 1       # 不及格


@dataclass
class TaskCompletionMetric:
    """任务完成度评估器"""

    # 评估结果
    accuracy_score: float = 0.0        # 准确率 (0-10)
    completeness_score: float = 0.0    # 完整性 (0-10)
    convergence_score: float = 0.0     # 收敛性 (轮次越少越好)
    overall_score: float = 0.0         # 综合得分

    # 详细指标
    elements_found: List[str] = field(default_factory=list)
    elements_required: List[str] = field(default_factory=list)
    adjustment_count: int = 0          # 调整轮次
    circuit_breaker_triggered: bool = False
    error_count: int = 0

    def calculate_overall(self) -> float:
        """计算综合得分 (加权平均)"""
        # 权重配置
        w_accuracy = 0.4
        w_completeness = 0.35
        w_convergence = 0.25

        self.overall_score = (
            self.accuracy_score * w_accuracy +
            self.completeness_score * w_completeness +
            self.convergence_score * w_convergence
        )
        return round(self.overall_score, 2)

    def analyze_response(self, response: str, expected_elements: List[str] = None) -> 'TaskCompletionMetric':
        """
        分析 Agent 输出,计算完成度指标

        Args:
            response: Agent 的最终输出
            expected_elements: 期望包含的要素列表
        """
        if expected_elements is None:
            expected_elements = [
                "时间安排", "地点", "交通方式",
                "餐饮推荐", "预算", "天气建议"
            ]
        self.elements_required = expected_elements

        # 1. 准确率分析
        self._analyze_accuracy(response)

        # 2. 完整性分析
        self._analyze_completeness(response)

        return self

    def _analyze_accuracy(self, response: str):
        """分析准确率 - 基于关键字匹配和语义理解"""
        score = 10.0

        # 检查是否生成了有效内容
        if not response or len(response.strip()) < 20:
            score -= 5.0
        elif "无法回答" in response or "不知道" in response:
            score -= 6.0
        elif "抱歉" in response and "不能" in response:
            score -= 4.0

        # 检查是否有明确的行程规划
        has_itinerary = any(kw in response for kw in ["行程", "安排", "计划", "天", "天"])
        if not has_itinerary:
            score -= 2.0

        # 限制分数范围
        self.accuracy_score = max(0.0, min(10.0, score))

    def _analyze_completeness(self, response: str):
        """分析完整性 - 检查要素覆盖度"""
        found = []
        missing = []

        for element in self.elements_required:
            if self._contains_element(response, element):
                found.append(element)
            else:
                missing.append(element)

        self.elements_found = found

        # 计算完整性分数
        if self.elements_required:
            ratio = len(found) / len(self.elements_required)
            self.completeness_score = ratio * 10.0
        else:
            self.completeness_score = 10.0

    def _contains_element(self, text: str, element: str) -> bool:
        """检查文本是否包含特定要素"""
        keywords_map = {
            "时间安排": ["时间", "安排", "行程", "天", "上午", "下午", "晚上"],
            "地点": ["地点", "地方", "北京", "上海", "景点", "位置"],
            "交通方式": ["交通", "地铁", "公交", "打车", "开车", "航班"],
            "餐饮推荐": ["吃饭", "餐厅", "美食", "早餐", "午餐", "晚餐", "推荐"],
            "预算": ["预算", "花费", "费用", "价格", "多少"],
            "天气建议": ["天气", "温度", "冷热", "穿衣", "建议", "注意"],
        }

        keywords = keywords_map.get(element, [])
        return any(kw in text for kw in keywords)

    def analyze_convergence(self, adjustment_count: int, circuit_breaker_triggered: bool = False):
        """
        分析收敛性 - 调整轮次越少越好

        Args:
            adjustment_count: 质检反馈后的调整轮次
            circuit_breaker_triggered: 熔断器是否触发
        """
        self.adjustment_count = adjustment_count
        self.circuit_breaker_triggered = circuit_breaker_triggered

        # 基础分数
        score = 10.0

        # 每多一轮调整扣分
        if adjustment_count > 0:
            # 前2轮免费,第3轮开始扣分
            extra_adjustments = max(0, adjustment_count - 2)
            score -= extra_adjustments * 1.5

        # 熔断器触发严重扣分
        if circuit_breaker_triggered:
            score -= 3.0

        # 调整次数过多直接不及格
        if adjustment_count >= 8:
            score = 0.0

        self.convergence_score = max(0.0, min(10.0, score))

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "accuracy_score": self.accuracy_score,
            "completeness_score": self.completeness_score,
            "convergence_score": self.convergence_score,
            "overall_score": self.calculate_overall(),
            "elements_found": self.elements_found,
            "elements_required": self.elements_required,
            "adjustment_count": self.adjustment_count,
            "circuit_breaker_triggered": self.circuit_breaker_triggered,
        }


def evaluate_plan_quality(plan_text: str) -> Tuple[float, List[str]]:
    """
    评估行程计划质量的快速工具函数

    Args:
        plan_text: 行程计划文本

    Returns:
        (质量分数, 发现的问题列表)
    """
    issues = []
    score = 100.0

    # 检查长度
    if len(plan_text) < 100:
        issues.append("内容过短")
        score -= 30

    # 检查是否包含必要元素
    if "天" not in plan_text:
        issues.append("缺少天数信息")
        score -= 15

    if any(kw in plan_text for kw in ["早上", "上午", "下午", "晚上"]):
        pass  # 有时间安排
    else:
        issues.append("缺少时间安排")
        score -= 15

    # 检查逻辑连贯性
    sentences = re.split(r'[。！！]', plan_text)
    if len(sentences) < 3:
        issues.append("内容过于简略")
        score -= 10

    return max(0.0, score), issues
