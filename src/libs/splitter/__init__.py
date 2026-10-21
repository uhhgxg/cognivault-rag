"""
文本切分器模块。

本包包含文本切分器的抽象定义和具体实现：
- 基础切分器类（父类）
- 切分器工厂
- 具体实现（递归切分、语义切分、定长切分）
"""

from src.libs.splitter.base_splitter import BaseSplitter
from src.libs.splitter.splitter_factory import SplitterFactory

# 导入具体实现（它们会自动注册到工厂中）
try:
    from src.libs.splitter.recursive_splitter import RecursiveSplitter
except ImportError:
    RecursiveSplitter = None  # type: ignore[assignment, misc]

__all__ = [
    "BaseSplitter",
    "SplitterFactory",
    "RecursiveSplitter",
]
