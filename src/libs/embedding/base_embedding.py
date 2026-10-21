"""Embedding 服务提供方的抽象基类（父类）。

本模块定义了 Embedding 服务提供方的可插拔接口，
通过配置驱动的方式，可以在不同后端（OpenAI、本地/BGE 等）之间无缝切换。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional


class BaseEmbedding(ABC):
    """Embedding 服务提供方的抽象基类（父类）。

    所有 Embedding 实现都必须继承本类，并实现 embed() 方法。
    这样可以保证不同提供方（OpenAI、本地/BGE、Ollama 等）的接口一致。

    应用的设计原则：
    - 可插拔（Pluggable）：子类可以替换，而不用改动上层代码。
    - 可观测（Observable）：接受可选的 TraceContext，用于接入可观测性。
    - 配置驱动（Config-Driven）：通过工厂根据配置创建实例。
    - 批处理优先（Batch-First）：专为批处理设计，以最大化效率。
    """
    
    @abstractmethod
    def embed(
        self,
        texts: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """为一批文本生成 embedding（向量）。

        Args:
            texts: 要向量化的文本字符串列表，不能为空。
            trace: 可选的 TraceContext，用于可观测性（为 Stage F 预留）。
            **kwargs: 各提供方专属参数（batch_size、dimensions 等）。

        Returns:
            embedding 向量列表，每个向量是一个浮点数列表。
            外层列表长度等于 len(texts)；
            内层列表长度（向量维度）由具体提供方决定。

        Raises:
            ValueError: 当 texts 为空或包含无效条目时。
            RuntimeError: 当 Embedding 提供方调用失败时。

        Example:
            >>> embeddings = embedding.embed(["hello", "world"])
            >>> len(embeddings)  # 2 个向量
            >>> len(embeddings[0])  # 维度（例如 OpenAI 为 1536）
        """
        pass
    
    def validate_texts(self, texts: List[str]) -> None:
        """校验输入的文本列表。

        Args:
            texts: 待校验的文本列表。

        Raises:
            ValueError: 当 texts 为空或包含无效条目时。
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(
                    f"Text at index {i} is not a string (type: {type(text).__name__})"
                )
            if not text.strip():
                raise ValueError(
                    f"Text at index {i} is empty or whitespace-only. "
                    "Embedding providers typically reject empty strings."
                )
    
    def get_dimension(self) -> int:
        """获取本提供方产生的 embedding 维度。

        Returns:
            向量维度（例如 OpenAI text-embedding-3-small 为 1536）。

        Raises:
            NotImplementedError: 当子类没有重写本方法时。

        Note:
            子类应当重写本方法，返回其专属维度。
            这在校验和存储配置时很有用。
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_dimension() method"
        )
