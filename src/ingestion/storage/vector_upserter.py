"""向量写入器，用于把分块写入向量库。

本模块实现 VectorUpserter 组件，负责：
- 根据内容生成确定性的分块 ID
- 把分块和向量转换成存储记录
- 调用 VectorStore 做幂等写入
- 支持保持顺序的批量操作

设计原则：
- 幂等：相同内容产生相同 ID，重复写入也安全
- 可观测：接收 TraceContext 以便未来接入
- 配置驱动：通过 VectorStoreFactory 从 settings 获取
- 确定性：基于稳定哈希的 ID 生成
- 类型安全：完整的类型标注和校验
"""

import hashlib
from typing import List, Dict, Any, Optional

from src.core.types import Chunk
from src.core.settings import Settings
from src.libs.vector_store.vector_store_factory import VectorStoreFactory


class VectorUpserter:
    """把分块和向量写入向量库，保证幂等。

    这个写入器接收来自 DenseEncoder 的分块和稠密向量，
    生成稳定的分块 ID，并写入已配置的向量库。

    分块 ID 格式：
        {source_path_hash}_{chunk_index:04d}_{content_hash}

        其中：
        - source_path_hash = SHA256(source_path) 的前 8 个字符
        - chunk_index = 补零到 4 位的索引
        - content_hash = SHA256(chunk.text) 的前 8 个字符

    这样设计保证：
        - 相同内容 → 相同 ID（幂等）
        - 内容变化 → 不同 ID（版本化）
        - 人可读且可追溯到来源

    示例：
        >>> upserter = VectorUpserter(settings)
        >>> 
        >>> chunks = [
        ...     Chunk(id="temp1", text="Hello world", metadata={"source_path": "doc.pdf", "chunk_index": 0}),
        ...     Chunk(id="temp2", text="Python rocks", metadata={"source_path": "doc.pdf", "chunk_index": 1})
        ... ]
        >>> vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        >>> 
        >>> upserter.upsert(chunks, vectors)
        >>> # Chunks written with stable IDs like: "a1b2c3d4_0000_e5f6g7h8"
    """
    
    def __init__(self, settings: Settings, collection_name: Optional[str] = None):
        """用已配置的向量库初始化 VectorUpserter。

        参数：
            settings: 包含 vector_store 配置的应用配置。
            collection_name: 可选的集合名，覆盖 settings 中的默认值。

        可能抛出：
            ValueError: 如果 settings 无效或向量库无法创建。
        """
        self.settings = settings
        kwargs = {}
        if collection_name:
            kwargs['collection_name'] = collection_name
        self.vector_store = VectorStoreFactory.create(settings, **kwargs)
    
    def upsert(
        self,
        chunks: List[Chunk],
        vectors: List[List[float]],
        trace: Optional[Any] = None,
    ) -> List[str]:
        """把分块及其向量 upsert 到向量库。

        参数：
            chunks: 要存储的 Chunk 对象列表。
            vectors: embedding 向量列表（顺序和长度与 chunks 一致）。
            trace: 可选的 TraceContext，用于可观测性（预留给 Stage F）。

        返回：
            生成的分块 ID 列表（顺序与输入分块一致）。

        可能抛出：
            ValueError: 如果 chunks 和 vectors 长度不匹配，或缺少必需的元数据字段。
            RuntimeError: 如果向量库 upsert 操作失败。

        示例：
            >>> chunks = [Chunk(...), Chunk(...)]
            >>> vectors = [[0.1, 0.2], [0.3, 0.4]]
            >>> chunk_ids = upserter.upsert(chunks, vectors)
            >>> len(chunk_ids) == len(chunks)  # True
        """
        # 校验输入长度是否匹配
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Chunk count ({len(chunks)}) must match vector count ({len(vectors)})"
            )
        
        if not chunks:
            raise ValueError("Cannot upsert empty chunks list")
        
        # 生成稳定的分块 ID 并构建记录
        records = []
        chunk_ids = []
        
        for chunk, vector in zip(chunks, vectors):
            # 生成确定性的分块 ID
            chunk_id = self._generate_chunk_id(chunk)
            chunk_ids.append(chunk_id)
            
            # 构建存储记录
            record = {
                "id": chunk_id,
                "vector": vector,
                "metadata": {
                    **chunk.metadata,  # 保留所有原始元数据
                    "text": chunk.text,  # 存储文本用于检索
                    "chunk_id": chunk_id,  # 冗余但方便查询
                },
            }
            records.append(record)
        
        # 执行幂等 upsert
        try:
            self.vector_store.upsert(records, trace=trace)
        except Exception as e:
            raise RuntimeError(
                f"Vector store upsert failed: {str(e)}"
            ) from e
        
        return chunk_ids
    
    def _generate_chunk_id(self, chunk: Chunk) -> str:
        """根据内容生成确定性的分块 ID。

        参数：
            chunk: 要生成 ID 的 Chunk 对象。

        返回：
            稳定的分块 ID 字符串。

        可能抛出：
            ValueError: 如果缺少必需的元数据字段。
        """
        # 校验必需的元数据
        if "source_path" not in chunk.metadata:
            raise ValueError("Chunk metadata must contain 'source_path'")
        if "chunk_index" not in chunk.metadata:
            raise ValueError("Chunk metadata must contain 'chunk_index'")
        
        source_path = chunk.metadata["source_path"]
        chunk_index = chunk.metadata["chunk_index"]
        
        # 计算稳定哈希
        source_hash = hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:8]
        content_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()[:8]
        
        # 格式：{source_hash}_{index:04d}_{content_hash}
        chunk_id = f"{source_hash}_{chunk_index:04d}_{content_hash}"
        
        return chunk_id
    
    def upsert_batch(
        self,
        batches: List[tuple[List[Chunk], List[List[float]]]],
        trace: Optional[Any] = None,
    ) -> List[str]:
        """upsert 多个批次的分块和向量。

        这是一个便捷方法，用于处理 BatchProcessor 的输出。
        所有批次会被展平，在单次 upsert 操作中处理，
        以保持顺序并减少向量库的往返次数。

        参数：
            batches: 来自批处理的 (chunks, vectors) 元组列表。
            trace: 可选的 TraceContext，用于可观测性。

        返回：
            所有生成的分块 ID 列表（按顺序）。

        示例：
            >>> batch1 = ([chunk1, chunk2], [[0.1, 0.2], [0.3, 0.4]])
            >>> batch2 = ([chunk3], [[0.5, 0.6]])
            >>> chunk_ids = upserter.upsert_batch([batch1, batch2])
            >>> len(chunk_ids)  # 3
        """
        # 展平所有批次
        all_chunks = []
        all_vectors = []
        
        for chunks, vectors in batches:
            all_chunks.extend(chunks)
            all_vectors.extend(vectors)
        
        # 单次 upsert 操作
        return self.upsert(all_chunks, all_vectors, trace=trace)
