"""稠密编码器，用于把文本分块转换成向量。

本模块实现摄取流水线中的稠密编码器组件，
负责用可配置的 embedding 提供方把文本分块转换成稠密向量表示。

设计原则：
- 配置驱动：通过工厂模式从 settings 获取 embedding 提供方
- 批处理：通过分批调用优化 API 请求
- 可观测：接收 TraceContext 以便未来接入可观测性
- 错误处理：单条失败不应让整批崩溃
- 确定性：相同输入产生相同输出
"""

from typing import List, Optional, Any
from src.core.types import Chunk
from src.libs.embedding.base_embedding import BaseEmbedding


class DenseEncoder:
    """用 BaseEmbedding 提供方把文本分块编码成稠密向量。

    这个编码器作为摄取流水线和可插拔 embedding 层之间的桥梁。
    它负责批处理、错误恢复，并保持输入分块和输出向量之间的对齐关系。

    设计：
    - 依赖注入：接收 BaseEmbedding 实例（不直接调用工厂）
    - 批处理优先：按可配置的批次大小处理所有分块
    - 无状态：encode() 调用之间不保存内部状态

    示例：
        >>> from src.libs.embedding.embedding_factory import EmbeddingFactory
        >>> from src.core.settings import load_settings
        >>> 
        >>> settings = load_settings("config/settings.yaml")
        >>> embedding = EmbeddingFactory.create(settings)
        >>> encoder = DenseEncoder(embedding, batch_size=32)
        >>> 
        >>> chunks = [Chunk(id="1", text="Hello world", metadata={})]
        >>> vectors = encoder.encode(chunks)
        >>> print(len(vectors))  # 1
        >>> print(len(vectors[0]))  # dimension (e.g., 1536)
    """
    
    def __init__(
        self,
        embedding: BaseEmbedding,
        batch_size: int = 100,
    ):
        """初始化 DenseEncoder。

        参数：
            embedding: embedding 提供方实例（来自 EmbeddingFactory）
            batch_size: 每次 API 调用处理的分块数量（默认 100）

        可能抛出：
            ValueError: 如果 batch_size <= 0
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        
        self.embedding = embedding
        self.batch_size = batch_size
    
    def encode(
        self,
        chunks: List[Chunk],
        trace: Optional[Any] = None,
    ) -> List[List[float]]:
        """把分块编码成稠密向量。

        本方法流程：
        1. 从每个分块提取文本
        2. 按 batch_size 把文本分批
        3. 对每个批次调用 embedding.embed()
        4. 按分块顺序拼接结果

        参数：
            chunks: 要编码的 Chunk 对象列表
            trace: 可选的 TraceContext，用于可观测性（预留给 Stage F）

        返回：
            稠密向量列表（每个分块一个，顺序与输入一致）。
            每个向量是一个浮点数列表，维度与 embedding 模型一致。

        可能抛出：
            ValueError: 如果分块列表为空
            RuntimeError: 如果所有批次的 embedding 提供方都失败

        示例：
            >>> chunks = [
            ...     Chunk(id="1", text="First chunk", metadata={}),
            ...     Chunk(id="2", text="Second chunk", metadata={})
            ... ]
            >>> vectors = encoder.encode(chunks)
            >>> len(vectors) == len(chunks)  # True
        """
        if not chunks:
            raise ValueError("Cannot encode empty chunks list")
        
        # 从分块中提取文本
        texts = [chunk.text for chunk in chunks]
        
        # 校验所有文本都非空
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(
                    f"Chunk at index {i} (id={chunks[i].id}) has empty or whitespace-only text"
                )
        
        # 分批处理
        all_vectors: List[List[float]] = []
        
        for batch_start in range(0, len(texts), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]
            
            try:
                # 调用 embedding 提供方
                batch_vectors = self.embedding.embed(
                    texts=batch_texts,
                    trace=trace,
                )
                
                # 校验输出形状
                if len(batch_vectors) != len(batch_texts):
                    raise RuntimeError(
                        f"Embedding provider returned {len(batch_vectors)} vectors "
                        f"for {len(batch_texts)} texts in batch {batch_start}-{batch_end}"
                    )
                
                all_vectors.extend(batch_vectors)
                
            except Exception as batch_err:
                # 兜底：逐条编码并重试，应对临时性 API 失败
                import time
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Batch embedding failed for batch {batch_start}-{batch_end} ({len(batch_texts)} chunks). "
                    f"Falling back to single-chunk encoding. Error: {batch_err}"
                )
                
                for idx, text in enumerate(batch_texts):
                    chunk_idx = batch_start + idx
                    for attempt in range(3):
                        try:
                            single_vector = self.embedding.embed(texts=[text], trace=trace)
                            if len(single_vector) != 1:
                                raise RuntimeError(
                                    f"Expected 1 vector for single text, got {len(single_vector)}"
                                )
                            all_vectors.append(single_vector[0])
                            break
                        except Exception as single_err:
                            if attempt < 2:
                                wait = 2 ** attempt
                                logger.warning(
                                    f"Single-chunk embedding failed for chunk {chunk_idx} "
                                    f"(attempt {attempt + 1}/3). Retrying in {wait}s... Error: {single_err}"
                                )
                                time.sleep(wait)
                            else:
                                raise RuntimeError(
                                    f"Failed to encode chunk {chunk_idx} after 3 attempts: {single_err}"
                                ) from single_err
        
        # 最终校验
        if len(all_vectors) != len(chunks):
            raise RuntimeError(
                f"Vector count mismatch: got {len(all_vectors)} vectors "
                f"for {len(chunks)} chunks"
            )
        
        # 校验向量维度是否一致
        if all_vectors:
            expected_dim = len(all_vectors[0])
            for i, vec in enumerate(all_vectors):
                if len(vec) != expected_dim:
                    raise RuntimeError(
                        f"Inconsistent vector dimensions: vector {i} has "
                        f"{len(vec)} dimensions, expected {expected_dim}"
                    )
        
        return all_vectors
    
    def get_batch_count(self, num_chunks: int) -> int:
        """计算给定分块数需要多少个批次。

        工具方法，用于日志或进度跟踪。

        参数：
            num_chunks: 要编码的分块数

        返回：
            所需的批次数
        """
        if num_chunks <= 0:
            return 0
        return (num_chunks + self.batch_size - 1) // self.batch_size
