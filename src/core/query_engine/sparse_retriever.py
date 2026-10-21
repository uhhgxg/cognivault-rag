"""稀疏检索器，使用 BM25 进行基于关键词的搜索。

本模块实现 SparseRetriever 组件，通过 BM25 倒排索引执行基于关键词的
搜索。它是混合搜索引擎中的稀疏检索路径，与 DenseRetriever 的语义搜索互补。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.core.types import RetrievalResult

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.ingestion.storage.bm25_indexer import BM25Indexer
    from src.libs.vector_store.base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


class SparseRetriever:
    """基于 BM25 关键词搜索的稀疏检索器。
    
    本类通过以下步骤执行基于关键词的检索：
    1. 用关键词查询 BM25 索引，获取匹配的 chunk ID 和分数
    2. 通过 get_by_ids() 从向量库中获取文本和元数据
    3. 返回标准化的 RetrievalResult 对象
    
    应用的设计原则：
    - 可插拔：通过依赖注入接收 bm25_indexer 和 vector_store。
    - 配置驱动：默认 top_k 和 collection 从 settings 读取。
    - 可观测：接受可选的 TraceContext 以集成可观测性。
    - 快速失败：尽早校验输入并给出清晰的错误信息。
    - 类型安全：返回标准化的 RetrievalResult 对象（与 DenseRetriever 相同）。
    
    属性：
        bm25_indexer: 用于关键词搜索的 BM25 索引器。
        vector_store: 用于获取文本和元数据的向量库。
        default_top_k: 默认返回的结果数量。
        default_collection: 默认查询的 BM25 索引集合。
    
    示例：
        >>> from src.ingestion.storage.bm25_indexer import BM25Indexer
        >>> from src.libs.vector_store.vector_store_factory import VectorStoreFactory
        >>> 
        >>> settings = Settings.load('config/settings.yaml')
        >>> bm25_indexer = BM25Indexer(index_dir="data/db/bm25")
        >>> bm25_indexer.load("default")
        >>> vector_store = VectorStoreFactory.create(settings)
        >>> 
        >>> retriever = SparseRetriever(
        ...     settings=settings,
        ...     bm25_indexer=bm25_indexer,
        ...     vector_store=vector_store
        ... )
        >>> results = retriever.retrieve(["RAG", "retrieval"], top_k=5)
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        bm25_indexer: Optional[BM25Indexer] = None,
        vector_store: Optional[BaseVectorStore] = None,
        default_top_k: int = 10,
        default_collection: str = "default",
    ) -> None:
        """初始化 SparseRetriever 及其依赖。

        参数：
            settings: 应用设置。用于在未显式提供时提取 default_top_k。
            bm25_indexer: 用于关键词搜索的 BM25 索引器。
                          实际检索操作时必需。
            vector_store: 用于获取文本和元数据的向量库。
                          实际检索操作时必需。
            default_top_k: 默认返回的结果数量（默认：10）。
                           可被 settings.retrieval.sparse_top_k 覆盖。
            default_collection: 默认的 BM25 索引集合名（默认："default"）。

        说明：
            依赖可通过注入方式提供，既可用于测试（配合 mock），也可用于
            生产环境（使用工厂创建的真实实现）。
        """
        self.bm25_indexer = bm25_indexer
        self.vector_store = vector_store
        self.default_collection = default_collection
        
        # 如果可用则从 settings 提取 default_top_k
        self.default_top_k = default_top_k
        if settings is not None:
            retrieval_config = getattr(settings, 'retrieval', None)
            if retrieval_config is not None:
                self.default_top_k = getattr(
                    retrieval_config, 'sparse_top_k', default_top_k
                )
        
        logger.info(
            f"SparseRetriever initialized with default_top_k={self.default_top_k}, "
            f"default_collection='{self.default_collection}'"
        )
    
    def retrieve(
        self,
        keywords: List[str],
        top_k: Optional[int] = None,
        collection: Optional[str] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        """使用 BM25 检索匹配给定关键词的 chunk。

        参数：
            keywords: 要搜索的关键词列表（通常来自 QueryProcessor）。
            top_k: 最多返回的结果数量。如果为 None，则使用 default_top_k。
            collection: 要查询的 BM25 索引集合。如果为 None，则使用 default_collection。
            trace: 可选的 TraceContext，用于可观测性（为 Stage F 预留）。

        返回：
            按 BM25 分数降序排列的 RetrievalResult 对象列表。
            每个结果包含 chunk_id、score、text 和 metadata。

        抛出：
            ValueError: 关键词列表为空时抛出。
            RuntimeError: bm25_indexer 或 vector_store 未配置，
                         或检索操作失败时抛出。

        示例：
            >>> results = retriever.retrieve(["Azure", "OpenAI", "配置"])
            >>> for result in results:
            ...     print(f"[{result.score:.2f}] {result.chunk_id}: {result.text[:50]}...")
        """
        # 校验输入
        self._validate_keywords(keywords)
        self._validate_dependencies()
        
        # 未指定时使用默认值
        effective_top_k = top_k if top_k is not None else self.default_top_k
        effective_collection = collection if collection is not None else self.default_collection
        
        logger.debug(
            f"Retrieving for keywords={keywords[:5]}{'...' if len(keywords) > 5 else ''}, "
            f"top_k={effective_top_k}, collection='{effective_collection}'"
        )
        
        # 第 1 步：确保索引已加载
        if not self._ensure_index_loaded(effective_collection):
            logger.warning(
                f"BM25 index for collection '{effective_collection}' not available. "
                "Returning empty results."
            )
            return []
        
        # 第 2 步：查询 BM25 索引
        try:
            bm25_results = self.bm25_indexer.query(
                query_terms=keywords,
                top_k=effective_top_k,
                trace=trace,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to query BM25 index: {e}. "
                "Check index availability and query terms."
            ) from e
        
        # 无匹配时提前返回
        if not bm25_results:
            logger.debug("BM25 query returned no results")
            return []
        
        # 第 3 步：从向量库获取文本和元数据
        chunk_ids = [r["chunk_id"] for r in bm25_results]
        try:
            records = self.vector_store.get_by_ids(chunk_ids, trace=trace)
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch records from vector store: {e}. "
                "Check vector store configuration and data availability."
            ) from e
        
        # 第 4 步：把 BM25 分数与文本/元数据合并
        results = self._merge_results(bm25_results, records)
        
        logger.debug(f"Retrieved {len(results)} results for keywords")
        return results
    
    def _validate_keywords(self, keywords: List[str]) -> None:
        """校验关键词列表。

        参数：
            keywords: 要校验的关键词列表。

        抛出：
            ValueError: 当 keywords 为空或不是列表时抛出。
        """
        if not isinstance(keywords, list):
            raise ValueError(
                f"Keywords must be a list, got {type(keywords).__name__}"
            )
        if not keywords:
            raise ValueError("Keywords list cannot be empty")
        # 过滤掉空字符串，但允许调用继续往下走
        # （空字符串只是不会匹配到任何东西）
    
    def _validate_dependencies(self) -> None:
        """校验必需的依赖是否已配置。

        抛出：
            RuntimeError: 当 bm25_indexer 或 vector_store 为 None 时抛出。
        """
        if self.bm25_indexer is None:
            raise RuntimeError(
                "SparseRetriever requires a bm25_indexer. "
                "Provide one during initialization or via setter."
            )
        if self.vector_store is None:
            raise RuntimeError(
                "SparseRetriever requires a vector_store. "
                "Provide one during initialization or via setter."
            )
    
    def _ensure_index_loaded(self, collection: str) -> bool:
        """确保给定集合的 BM25 索引已加载。

        每次都从磁盘重新加载，因为索引可能已被其他进程更新
        （例如 dashboard 的入库操作）。与整个查询相比，加载很快
        （只是读取一个 JSON 文件）。

        参数：
            collection: 要加载的集合名。

        返回：
            索引加载成功并就绪返回 True，否则返回 False。
        """
        try:
            loaded = self.bm25_indexer.load(collection=collection)
            return loaded
        except Exception as e:
            logger.warning(f"Failed to load BM25 index for collection '{collection}': {e}")
            return False
    
    def _merge_results(
        self,
        bm25_results: List[Dict[str, Any]],
        records: List[Dict[str, Any]],
    ) -> List[RetrievalResult]:
        """把 BM25 分数与向量库中的文本和元数据合并起来。

        参数：
            bm25_results: BM25 查询结果，每项包含 'chunk_id' 和 'score'。
            records: 向量库返回的记录，每项包含 'id'、'text'、'metadata'。

        返回：
            包含完整信息的 RetrievalResult 对象列表。
        """
        results = []

        for bm25_result, record in zip(bm25_results, records):
            chunk_id = bm25_result["chunk_id"]
            score = bm25_result["score"]

            # 处理记录未找到的情况
            if not record:
                logger.warning(
                    f"No record found in vector store for chunk_id='{chunk_id}'. "
                    "Skipping this result."
                )
                continue
            
            # 校验记录是否包含预期的字段
            text = record.get('text', '')
            metadata = record.get('metadata', {})
            
            try:
                result = RetrievalResult(
                    chunk_id=chunk_id,
                    score=float(score),
                    text=str(text),
                    metadata=metadata,
                )
                results.append(result)
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Failed to create RetrievalResult for chunk_id='{chunk_id}': {e}. "
                    "Skipping this result."
                )
                continue
        
        return results


def create_sparse_retriever(
    settings: Settings,
    bm25_indexer: Optional[BM25Indexer] = None,
    vector_store: Optional[BaseVectorStore] = None,
    index_dir: str = "data/db/bm25",
) -> SparseRetriever:
    """工厂函数，创建一个可选依赖注入的 SparseRetriever。

    本函数简化了 SparseRetriever 的创建过程，如果调用者未提供依赖，
    会自动从工厂创建。

    参数：
        settings: 应用设置。
        bm25_indexer: 可选的、已配置好的 BM25 索引器。
                      如果为 None，则使用默认 index_dir 创建。
        vector_store: 可选的、已配置好的向量库。
                      如果为 None，则由 VectorStoreFactory 创建。
        index_dir: BM25 索引文件的目录（默认："data/db/bm25"）。

    返回：
        配置好的 SparseRetriever 实例。

    示例：
        >>> settings = Settings.load('config/settings.yaml')
        >>> retriever = create_sparse_retriever(settings)
    """
    # 延迟导入，避免循环依赖
    if bm25_indexer is None:
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        bm25_indexer = BM25Indexer(index_dir=index_dir)
    
    if vector_store is None:
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory
        vector_store = VectorStoreFactory.create(settings)
    
    return SparseRetriever(
        settings=settings,
        bm25_indexer=bm25_indexer,
        vector_store=vector_store,
    )
