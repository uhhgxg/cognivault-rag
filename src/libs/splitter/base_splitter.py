"""文本切分器的抽象基类（父类）。

本模块定义了切分器提供者的可插拔接口，
通过配置驱动的方式实例化，可以无缝切换不同的切分策略
（递归切分、语义切分、定长切分等）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional


class BaseSplitter(ABC):
    """文本切分器的抽象基类（父类）。

    所有切分器实现都必须继承本类，并实现 split_text() 方法。
    这样能保证不同切分策略之间接口一致。

    应用的设计原则：
    - 可插拔：子类可以互相替换，不用改上游调用代码。
    - 可观测：接受可选的 TraceContext，用于接入可观测性。
    - 配置驱动：实例通过工厂根据 settings 创建。
    """

    @abstractmethod
    def split_text(
        self,
        text: str,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[str]:
        """把输入文本切分成一组切块。

        参数：
            text: 要切分的输入文本，必须是非空字符串。
            trace: 可选的 TraceContext，用于可观测性（预留给 Stage F 使用）。
            **kwargs: 各策略特有的参数（chunk_size、overlap 等）。

        返回：
            文本切块列表。顺序必须与原文顺序一致。

        抛出：
            ValueError: 如果输入文本不合法。
            RuntimeError: 如果切分器意外失败。
        """
        pass

    def validate_text(self, text: str) -> None:
        """校验输入文本。

        参数：
            text: 要校验的输入文本。

        抛出：
            ValueError: 如果 text 不是非空字符串。
        """
        if not isinstance(text, str):
            raise ValueError(f"Input text must be a string, got {type(text).__name__}")
        if not text.strip():
            raise ValueError("Input text cannot be empty or whitespace-only")

    def validate_chunks(self, chunks: List[str]) -> None:
        """校验输出的切块。

        参数：
            chunks: 要校验的切块字符串列表。

        抛出：
            ValueError: 如果切块列表为空，或包含不合法的元素。
        """
        if not isinstance(chunks, list):
            raise ValueError("Chunks must be a list of strings")
        if not chunks:
            raise ValueError("Chunks list cannot be empty")
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, str):
                raise ValueError(
                    f"Chunk at index {i} is not a string (type: {type(chunk).__name__})"
                )
            if not chunk.strip():
                raise ValueError(
                    f"Chunk at index {i} is empty or whitespace-only"
                )