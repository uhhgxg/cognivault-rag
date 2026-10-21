"""重排序器提供者的抽象基类（父类）。

本模块定义了重排序器提供者的可插拔接口，
通过配置驱动的方式实例化，可以无缝切换不同的重排策略
（None 空实现、Cross-Encoder、基于 LLM 的重排）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseReranker(ABC):
    """重排序器提供者的抽象基类（父类）。

    所有重排序器实现都必须继承本类，并实现 rerank() 方法。
    这样能保证不同重排策略之间接口一致。

    应用的设计原则：
    - 可插拔：子类可以互相替换，不用改上游调用代码。
    - 可观测：接受可选的 TraceContext，用于接入可观测性。
    - 配置驱动：实例通过工厂根据 settings 创建。
    - 兜底：实现方应当支持安全降级为原始顺序。
    """
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """针对某个查询，对候选切块进行重排序。

        参数：
            query: 用户的查询字符串。
            candidates: 要重排的候选记录列表。每条至少包含一个标识，
                以及重排序器实现所需的其他字段（例如 text、score、metadata）。
            trace: 可选的 TraceContext，用于可观测性（预留给 Stage F 使用）。
            **kwargs: 提供者特有参数（top_k、timeout 等）。

        返回：
            重排后的候选列表。实现方应当保留原始候选对象，
            只改变顺序，除非另有明确说明。

        抛出：
            ValueError: 如果 query 或 candidates 不合法。
            RuntimeError: 如果重排序器意外失败。
        """
        pass

    def validate_query(self, query: str) -> None:
        """校验查询字符串。

        参数：
            query: 要校验的查询字符串。

        抛出：
            ValueError: 如果 query 不是非空字符串。
        """
        if not isinstance(query, str):
            raise ValueError(f"Query must be a string, got {type(query).__name__}")
        if not query.strip():
            raise ValueError("Query cannot be empty or whitespace-only")

    def validate_candidates(self, candidates: List[Dict[str, Any]]) -> None:
        """校验候选列表结构。

        参数：
            candidates: 要校验的候选记录列表。

        抛出：
            ValueError: 如果候选列表为空或格式不对。
        """
        if not isinstance(candidates, list):
            raise ValueError("Candidates must be a list of dicts")
        if not candidates:
            raise ValueError("Candidates list cannot be empty")
        for i, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                raise ValueError(
                    f"Candidate at index {i} is not a dict (type: {type(candidate).__name__})"
                )


class NoneReranker(BaseReranker):
    """空操作的重排序器，保持原始顺序。

    当重排被禁用、或提供者设为 'none' 时，会用到本实现。
    它会校验输入，然后把候选项原样返回。
    """

    def __init__(self, settings: Any = None, **kwargs: Any) -> None:
        self.settings = settings
        self.kwargs = kwargs

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """按原始顺序返回候选项。

        参数：
            query: 查询字符串。
            candidates: 要返回的候选列表。
            trace: 可选的 TraceContext（不使用）。
            **kwargs: 忽略。

        返回：
            保持顺序的候选项浅拷贝。
        """
        self.validate_query(query)
        self.validate_candidates(candidates)
        return list(candidates)
