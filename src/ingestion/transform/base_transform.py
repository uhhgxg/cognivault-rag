"""分块变换操作的基类。"""

from abc import ABC, abstractmethod
from typing import List, Optional

from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext


class BaseTransform(ABC):
    """分块变换操作的抽象基类。

    变换操作会处理分块，用于提升质量、补充元数据，
    或为下游处理（向量化、索引）做准备。

    设计原则：
        - 单一职责：每个变换只做一种增强
        - 原子操作：一个分块失败不影响其他分块
        - 可观测：在 TraceContext 中记录处理信息
        - 优雅降级：遇到无法恢复的错误时返回原始分块
    """
    
    @abstractmethod
    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """变换一组分块。

        参数：
            chunks: 要变换的分块列表
            trace: 可选的 trace 上下文，用于可观测性

        返回：
            变换后的分块列表（长度与输入相同）

        可能抛出：
            ValueError: 如果输入校验失败
        """
        pass
