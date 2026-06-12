# -*- coding: utf-8 -*-
"""
测试数据集管理
"""

import json
import os
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class TestCase:
    """测试用例"""
    id: str
    name: str
    input_message: str
    expected_output_contains: List[str] = None
    expected_elements: List[str] = None
    tags: List[str] = None
    priority: str = "normal"  # high, normal, low
    is_complex: bool = True

    def __post_init__(self):
        if self.expected_output_contains is None:
            self.expected_output_contains = []
        if self.expected_elements is None:
            self.expected_elements = [
                "时间安排", "地点", "交通方式", "餐饮推荐", "预算"
            ]
        if self.tags is None:
            self.tags = ["default"]


class TestDataset:
    """测试数据集管理器"""

    def __init__(self, test_dir: str = None):
        """初始化测试数据集"""
        if test_dir is None:
            # 默认路径
            self.test_dir = Path(__file__).parent.parent / "test_cases"
        else:
            self.test_dir = Path(test_dir)

        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.test_cases: List[TestCase] = []

        # 加载预设测试用例
        self._load_default_test_cases()

    def _load_default_test_cases(self):
        """加载默认测试用例"""
        default_cases = [
            TestCase(
                id="tc_001",
                name="简单天气查询",
                input_message="今天北京天气怎么样？",
                expected_output_contains=["北京", "天气", "温度"],
                tags=["basic", "weather"],
                priority="high",
                is_complex=False,
            ),
            TestCase(
                id="tc_002",
                name="简单闲聊",
                input_message="你好！",
                expected_output_contains=["你好", "欢迎"],
                tags=["basic", "chitchat"],
                priority="high",
                is_complex=False,
            ),
            TestCase(
                id="tc_003",
                name="基础行程规划",
                input_message="给我规划一个上海3天行程",
                expected_elements=["时间安排", "地点", "交通方式", "餐饮推荐"],
                tags=["planning", "shanghai"],
                priority="high",
                is_complex=True,
            ),
            TestCase(
                id="tc_004",
                name="带偏好的行程",
                input_message="我和家人想去成都玩，要吃辣的，预算5000",
                expected_elements=["时间安排", "地点", "交通方式", "餐饮推荐", "预算"],
                tags=["planning", "chengdu", "preferences"],
                priority="high",
                is_complex=True,
            ),
            TestCase(
                id="tc_005",
                name="复杂多轮对话",
                input_message="我想去西安旅游，之前说过喜欢历史景点",
                expected_elements=["时间安排", "地点", "历史景点", "交通方式"],
                tags=["multi-turn", "memory", "xi'an"],
                priority="normal",
                is_complex=True,
            ),
            TestCase(
                id="tc_006",
                name="异常输入",
                input_message="asdfghjkl",
                tags=["edge", "abnormal"],
                priority="low",
                is_complex=False,
            ),
            TestCase(
                id="tc_007",
                name="预算敏感型规划",
                input_message="预算有限，2000元以内玩广州3天",
                expected_elements=["预算", "交通方式", "餐饮推荐"],
                tags=["planning", "guangzhou", "budget"],
                priority="normal",
                is_complex=True,
            ),
            TestCase(
                id="tc_008",
                name="季节性规划",
                input_message="冬天去哈尔滨玩，有什么推荐？",
                expected_elements=["时间安排", "地点", "冬季活动", "保暖建议"],
                tags=["planning", "harbin", "seasonal"],
                priority="normal",
                is_complex=True,
            ),
        ]

        self.test_cases.extend(default_cases)

    def add_test_case(self, test_case: TestCase):
        """添加测试用例"""
        self.test_cases.append(test_case)

    def filter_test_cases(self, tags: List[str] = None, priority: str = None) -> List[TestCase]:
        """过滤测试用例"""
        results = self.test_cases

        if tags:
            results = [tc for tc in results if any(tag in tc.tags for tag in tags)]

        if priority:
            results = [tc for tc in results if tc.priority == priority]

        return results

    def save_to_file(self, filepath: str = None):
        """保存测试用例到文件"""
        if filepath is None:
            filepath = self.test_dir / "test_cases.json"

        data = [asdict(tc) for tc in self.test_cases]

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_from_file(self, filepath: str = None):
        """从文件加载测试用例"""
        if filepath is None:
            filepath = self.test_dir / "test_cases.json"

        if not os.path.exists(filepath):
            return []

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.test_cases = [TestCase(**tc_data) for tc_data in data]
        return self.test_cases

    def __len__(self) -> int:
        return len(self.test_cases)

    def __iter__(self):
        return iter(self.test_cases)


def load_test_cases(test_dir: str = None) -> TestDataset:
    """工厂函数: 创建并返回测试数据集"""
    dataset = TestDataset(test_dir)
    # 尝试从文件加载
    try:
        dataset.load_from_file()
    except Exception:
        # 如果文件不存在，使用默认用例
        pass
    return dataset


# 预设的基准测试集
BENCHMARK_TEST_CASES = [
    TestCase(
        id="bm_001",
        name="基准测试: 简单问答",
        input_message="今天上海天气如何？",
        expected_output_contains=["上海", "天气"],
        tags=["benchmark"],
        priority="high",
    ),
    TestCase(
        id="bm_002",
        name="基准测试: 行程规划",
        input_message="北京2日游行程",
        expected_elements=["时间安排", "地点", "交通方式"],
        tags=["benchmark"],
        priority="high",
    ),
    TestCase(
        id="bm_003",
        name="基准测试: 偏好记忆",
        input_message="我想去杭州，之前说过喜欢西湖和龙井茶",
        expected_elements=["西湖", "龙井茶", "杭州"],
        tags=["benchmark", "memory"],
        priority="normal",
    ),
]
