"""
向量存储模块。

本包包含向量存储的抽象定义和具体实现：
- 基础向量存储类（父类）
- 向量存储工厂
- 具体实现（Chroma 等）
"""

from src.libs.vector_store.base_vector_store import BaseVectorStore
from src.libs.vector_store.vector_store_factory import VectorStoreFactory

# 自动注册 ChromaStore 提供者
try:
    from src.libs.vector_store.chroma_store import ChromaStore
    VectorStoreFactory.register_provider('chroma', ChromaStore)
except ImportError:
    # ChromaDB 没装，跳过注册
    pass

__all__ = [
    'BaseVectorStore',
    'VectorStoreFactory',
    'ChromaStore',
]
