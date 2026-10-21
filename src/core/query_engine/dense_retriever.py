"""稠密检索器，使用向量嵌入进行语义搜索。

本模块实现 DenseRetriever 组件，通过嵌入查询向量并从向量库中检索相似 chunk
来执行语义搜索。它是混合搜索引擎中的稠密检索路径。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.core.types import RetrievalResult

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.libs.embedding.base_embedding import BaseEmbedding
    from src.libs.vector_store.base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


class DenseRetriever:
    """基于向量嵌入的稠密检索器。
    
    本类通过以下步骤执行语义检索：
    1. 使用配置好的 embedding 客户端对查询进行向量化
    2. 在向量库中查询相似向量
    3. 返回标准化的 RetrievalResult 对象
    
    应用的设计原则：
    - 可插拔：通过依赖注入接收 embedding_client 和 vector_store。
    - 配置驱动：默认 top_k 从 settings.retrieval.dense_top_k 读取。
    - 可观测：接受可选的 TraceContext 以集成可观测性。
    - 快速失败：尽早校验输入并给出清晰的错误信息。
    - 类型安全：返回标准化的 RetrievalResult 对象。
    
    属性：
        embedding_client: 用于查询向量化的 embedding 提供方。
        vector_store: 用于相似度搜索的向量库。
        default_top_k: 默认返回的结果数量。
    
    示例：
        >>> from src.libs.embedding.embedding_factory import EmbeddingFactory
        >>> from src.libs.vector_store.vector_store_factory import VectorStoreFactory
        >>> 
        >>> settings = Settings.load('config/settings.yaml')
        >>> embedding_client = EmbeddingFactory.create(settings)
        >>> vector_store = VectorStoreFactory.create(settings)
        >>> 
        >>> retriever = DenseRetriever(
        ...     settings=settings,
        ...     embedding_client=embedding_client,
        ...     vector_store=vector_store
        ... )
        >>> results = retriever.retrieve("What is RAG?", top_k=5)
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        embedding_client: Optional[BaseEmbedding] = None,
        vector_store: Optional[BaseVectorStore] = None,
        default_top_k: int = 10,
    ) -> None:
        """初始化 DenseRetriever 及其依赖。

        参数：
            settings: 应用设置。用于在未显式提供时提取 default_top_k。
            embedding_client: 用于查询向量化的 embedding 提供方。
                              实际检索操作时必需。
            vector_store: 用于相似度搜索的向量库。
                          实际检索操作时必需。
            default_top_k: 默认返回的结果数量（默认：10）。
                           可被 settings.retrieval.dense_top_k 覆盖。

        抛出：
            ValueError: 当必需的 embedding_client 或 vector_store 为 None 时抛出。

        说明：
            依赖可通过注入方式提供，既可用于测试（配合 mock），也可用于
            生产环境（使用工厂创建的真实实现）。
        """
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        
        # 如果可用则从 settings 提取 default_top_k
        self.default_top_k = default_top_k
        if settings is not None:
            retrieval_config = getattr(settings, 'retrieval', None)
            if retrieval_config is not None:
                self.default_top_k = getattr(
                    retrieval_config, 'dense_top_k', default_top_k
                )
        
        logger.info(
            f"DenseRetriever initialized with default_top_k={self.default_top_k}"
        )
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        """检索与查询在语义上相似的 chunk。

        参数：
            query: 搜索查询字符串，不能为空。
            top_k: 最多返回的结果数量。如果为 None，则使用 default_top_k。
            filters: 可选的元数据过滤条件（例如 {"collection": "api-docs"}）。
            trace: 可选的 TraceContext，用于可观测性（为 Stage F 预留）。

        返回：
            按相似度降序排列的 RetrievalResult 对象列表。
            每个结果包含 chunk_id、score、text 和 metadata。

        抛出：
            ValueError: 查询为空或无效时抛出。
            RuntimeError: embedding_client 或 vector_store 未配置，
                         或检索操作失败时抛出。

        示例：
            >>> results = retriever.retrieve("How to configure Azure OpenAI?")
            >>> for result in results:
            ...     print(f"[{result.score:.2f}] {result.chunk_id}: {result.text[:50]}...")
        """
        # 校验输入
        self._validate_query(query)
        self._validate_dependencies()
        
        # 未指定时使用默认 top_k
        effective_top_k = top_k if top_k is not None else self.default_top_k
        
        logger.debug(f"Retrieving for query='{query[:50]}...', top_k={effective_top_k}")
        
        # 第 1 步：对查询进行向量化
        try:
            query_vectors = self.embedding_client.embed([query], trace=trace)
            query_vector = query_vectors[0]
        except Exception as e:
            raise RuntimeError(
                f"Failed to embed query: {e}. "
                "Check embedding client configuration and connectivity."
            ) from e
        
        # 第 2 步：查询向量库
        try:
            raw_results = self.vector_store.query(
                vector=query_vector,
                top_k=effective_top_k,
                filters=filters,
                trace=trace,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to query vector store: {e}. "
                "Check vector store configuration and data availability."
            ) from e
        
        # 第 3 步：转换为 RetrievalResult 对象
        results = self._transform_results(raw_results)
        
        logger.debug(f"Retrieved {len(results)} results for query")
        return results
    
    def _validate_query(self, query: str) -> None:
        """校验查询字符串。

        参数：
            query: 要校验的查询字符串。

        抛出：
            ValueError: 查询为空或不是字符串时抛出。
        """
        if not isinstance(query, str):
            raise ValueError(
                f"Query must be a string, got {type(query).__name__}"
            )
        if not query.strip():
            raise ValueError("Query cannot be empty or whitespace-only")
    
    def _validate_dependencies(self) -> None:
        """校验必需的依赖是否已配置。

        抛出：
            RuntimeError: embedding_client 或 vector_store 为 None 时抛出。
        """
        if self.embedding_client is None:
            raise RuntimeError(
                "DenseRetriever requires an embedding_client. "
                "Provide one during initialization or via setter."
            )
        if self.vector_store is None:
            raise RuntimeError(
                "DenseRetriever requires a vector_store. "
                "Provide one during initialization or via setter."
            )
    
    def _transform_results(
        self,
        raw_results: List[Dict[str, Any]],
    ) -> List[RetrievalResult]:
        """将向量库的原始结果转换为 RetrievalResult 对象。

        参数：
            raw_results: 向量库查询返回的原始结果。
                         每条结果应包含：id、score、text、metadata。

        返回：
            RetrievalResult 对象列表。
        """
        results = []
        for raw in raw_results:
            try:
                result = RetrievalResult(
                    chunk_id=str(raw.get('id', '')),
                    score=float(raw.get('score', 0.0)),
                    text=str(raw.get('text', '')),
                    metadata=raw.get('metadata', {}),
                )
                results.append(result)
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Failed to transform result {raw.get('id', 'unknown')}: {e}. "
                    "Skipping this result."
                )
                continue
        
        return results


def create_dense_retriever(
    settings: Settings,
    embedding_client: Optional[BaseEmbedding] = None,
    vector_store: Optional[BaseVectorStore] = None,
) -> DenseRetriever:
    """工厂函数：创建带可选依赖注入的 DenseRetriever。

    本函数简化 DenseRetriever 的创建过程，如果未提供依赖则自动从工厂创建。

    参数：
        settings: 应用设置。
        embedding_client: 可选的预配置 embedding 客户端。
                          如果为 None，则由 EmbeddingFactory 创建。
        vector_store: 可选的预配置向量库。
                      如果为 None，则由 VectorStoreFactory 创建。

    返回：
        已配置的 DenseRetriever 实例。

    示例：
        >>> settings = Settings.load('config/settings.yaml')
        >>> retriever = create_dense_retriever(settings)
    """
    # 懒导入以避免循环依赖
    if embedding_client is None:
        from src.libs.embedding.embedding_factory import EmbeddingFactory
        embedding_client = EmbeddingFactory.create(settings)
    
    if vector_store is None:
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory
        vector_store = VectorStoreFactory.create(settings)
    
    return DenseRetriever(
        settings=settings,
        embedding_client=embedding_client,
        vector_store=vector_store,
    )
