"""评估器提供者的抽象基类（父类）。

本模块定义了评估器提供者的可插拔接口，
通过配置驱动的方式实例化，可以无缝切换不同的评估后端。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseEvaluator(ABC):
    """评估器提供者的抽象基类（父类）。

    所有评估器实现都必须继承本类，并实现 evaluate() 方法。
    这样能保证不同评估后端之间接口一致。

    应用的设计原则：
    - 可插拔：子类可以互相替换，不用改上游调用代码。
    - 可观测：接受可选的 TraceContext，用于接入可观测性。
    - 配置驱动：实例通过工厂根据 settings 创建。
    """

    @abstractmethod
    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Any],
        generated_answer: Optional[str] = None,
        ground_truth: Optional[Any] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """评估检索和生成的质量。

        参数：
            query: 用户的查询字符串。
            retrieved_chunks: 要评估的检索到的切块或记录。
            generated_answer: 可选的生成答案文本。
            ground_truth: 可选的标准答案数据（id 或答案）。
            trace: 可选的 TraceContext，用于可观测性（预留给 Stage F 使用）。
            **kwargs: 提供者特有参数。

        返回：
            指标名到浮点值的字典。

        抛出：
            ValueError: 如果输入不合法。
            RuntimeError: 如果评估意外失败。
        """
        pass

    def validate_query(self, query: str) -> None:
        """校验查询字符串。

        参数：
            query: 要校验的查询字符串。

        抛出：
            ValueError: 如果 query 不合法。
        """
        if not isinstance(query, str):
            raise ValueError(f"Query must be a string, got {type(query).__name__}")
        if not query.strip():
            raise ValueError("Query cannot be empty or whitespace-only")

    def validate_retrieved_chunks(self, retrieved_chunks: List[Any]) -> None:
        """校验检索到的切块结构。

        参数：
            retrieved_chunks: 要校验的检索到的切块列表。

        抛出：
            ValueError: 如果 retrieved_chunks 不合法。
        """
        if not isinstance(retrieved_chunks, list):
            raise ValueError("retrieved_chunks must be a list")
        if not retrieved_chunks:
            raise ValueError("retrieved_chunks cannot be empty")


class NoneEvaluator(BaseEvaluator):
    """空操作的评估器，返回空指标。

    当评估被禁用时，会用到本实现。
    """

    def __init__(self, settings: Any = None, **kwargs: Any) -> None:
        self.settings = settings
        self.kwargs = kwargs

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Any],
        generated_answer: Optional[str] = None,
        ground_truth: Optional[Any] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        self.validate_query(query)
        self.validate_retrieved_chunks(retrieved_chunks)
        return {}