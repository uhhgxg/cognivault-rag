"""文档加载器的抽象基类（父类）。

本模块定义了文档加载器的可插拔接口，
支持以统一的输出结构无缝加载不同文档格式（PDF、Markdown 等）。

设计原则：
- 单一职责（Single Responsibility）：加载器只负责格式统一 + 结构提取
- 类型安全（Type Safety）：返回 core.types 中标准化的 Document 类型
- 不做切分（No Splitting）：加载器不做文档切块，只做解析和归一化
- 优雅降级（Graceful Degradation）：可选功能（如图片提取）失败时不应阻断文本解析
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from src.core.types import Document


class BaseLoader(ABC):
    """文档加载器的抽象基类（父类）。

    所有加载器都必须实现 load() 方法，解析文件并返回标准化的 Document 对象，包含：
    - text：归一化后的内容（尽量使用 Markdown 格式）
    - metadata：至少必须包含 'source_path'

    加载器应当处理：
    - 格式专属的解析逻辑
    - 元数据提取（标题、页数等）
    - 结构归一化（尽量转成 Markdown）
    - 可选：图片提取与占位符插入
    """

    @abstractmethod
    def load(self, file_path: str | Path) -> Document:
        """加载并解析一个文档文件。

        Args:
            file_path: 待加载的文档文件路径。

        Returns:
            包含解析内容和元数据的 Document 对象。
            metadata 至少必须包含 'source_path'。

        Raises:
            FileNotFoundError: 当文件不存在时。
            ValueError: 当文件格式无效或不被支持时。
            RuntimeError: 当解析出现致命错误时。

        Example:
            >>> loader = PdfLoader()
            >>> doc = loader.load("data/documents/report.pdf")
            >>> assert "source_path" in doc.metadata
            >>> assert doc.text  # 非空文本
        """
        pass

    @staticmethod
    def _validate_file(file_path: str | Path) -> Path:
        """校验文件是否存在且可读。

        Args:
            file_path: 待校验的路径。

        Returns:
            解析后的 Path 对象。

        Raises:
            FileNotFoundError: 当文件不存在时。
            PermissionError: 当文件不可读时。
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        return path
