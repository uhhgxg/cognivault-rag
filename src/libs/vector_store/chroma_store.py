"""基于 ChromaDB 的向量存储实现。

本模块使用 ChromaDB 提供了 BaseVectorStore 的具体实现。
ChromaDB 是一个轻量、开源的向量数据库，专为本地优先部署而设计。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from src.core.settings import resolve_path
from src.libs.vector_store.base_vector_store import BaseVectorStore

if TYPE_CHECKING:
    from src.core.settings import Settings

logger = logging.getLogger(__name__)


class ChromaStore(BaseVectorStore):
    """基于 ChromaDB 的向量存储实现。

    本类使用 ChromaDB 提供本地优先、可持久化的向量存储，
    支持 upsert（插入或更新）、query（查询）和元数据过滤操作。

    应用的设计原则：
    - 可插拔：实现 BaseVectorStore 接口，可与其他提供者互换。
    - 配置驱动：所有配置（persist_directory、collection_name）都来自 settings.yaml。
    - 幂等：用相同 ID 做 upsert 会覆盖已有记录。
    - 可观测：接受可选的 TraceContext（预留给 Stage F 使用）。
    - 快速失败：初始化时会校验依赖和配置。

    属性：
        client: ChromaDB 客户端实例。
        collection: ChromaDB 中用来存放向量的集合（collection）。
        collection_name: 集合的名字。
        persist_directory: 持久化存储的目录路径。

    示例：
        >>> settings = Settings.load('config/settings.yaml')
        >>> store = ChromaStore(settings=settings)
        >>> records = [
        ...     {
        ...         'id': 'doc1_chunk0',
        ...         'vector': [0.1, 0.2, 0.3],
        ...         'metadata': {'source': 'doc1.pdf'}
        ...     }
        ... ]
        >>> store.upsert(records)
        >>> results = store.query([0.1, 0.2, 0.3], top_k=5)
    """
    
    def __init__(self, settings: Settings, **kwargs: Any) -> None:
        """根据配置初始化 ChromaStore。

        参数：
            settings: 应用配置，里面包含 vector_store 相关配置。
            **kwargs: 可选参数，可覆盖 collection_name 或 persist_directory。

        抛出：
            ImportError: 如果没装 chromadb 包。
            ValueError: 如果缺少必要配置。
            RuntimeError: 如果 ChromaDB 客户端初始化失败。
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "chromadb package is required for ChromaStore. "
                "Install it with: pip install chromadb"
            )

        # 读取配置
        try:
            vector_store_config = settings.vector_store
        except AttributeError as e:
            raise ValueError(
                "Missing required configuration: settings.vector_store. "
                "Please ensure 'vector_store' section exists in settings.yaml"
            ) from e

        # 集合名（允许覆盖）
        self.collection_name = kwargs.get(
            'collection_name',
            getattr(vector_store_config, 'collection_name', 'knowledge_hub')
        )

        # 持久化目录（允许覆盖）
        persist_dir_str = kwargs.get(
            'persist_directory',
            getattr(vector_store_config, 'persist_directory', './data/db/chroma')
        )
        self.persist_directory = resolve_path(persist_dir_str)

        # 确保持久化目录存在
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Initializing ChromaStore: collection='{self.collection_name}', "
            f"persist_directory='{self.persist_directory}'"
        )

        # 初始化 ChromaDB 客户端（带持久化存储）
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                )
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize ChromaDB client at '{self.persist_directory}': {e}"
            ) from e

        # 获取或创建集合
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to get or create collection '{self.collection_name}': {e}"
            ) from e
        
        logger.info(
            f"ChromaStore initialized successfully. "
            f"Collection count: {self.collection.count()}"
        )
    
    def upsert(
        self,
        records: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """向 ChromaDB 插入或更新记录。

        参数：
            records: 要 upsert 的记录列表。每条记录必须包含：
                - 'id': 唯一标识（str）
                - 'vector': 向量（List[float]）
                - 'metadata': 可选的元数据字典
            trace: 可选的 TraceContext，用于可观测性（预留给 Stage F 使用）。
            **kwargs: 提供者特有参数（Chroma 不使用）。

        抛出：
            ValueError: 如果记录列表为空，或包含不合法的记录。
            RuntimeError: 如果 upsert 操作失败。
        """
        # 校验记录
        self.validate_records(records)

        # 准备要传给 ChromaDB 的数据
        ids = []
        embeddings = []
        metadatas = []
        documents = []  # ChromaDB 要求必须有 documents 字段

        for record in records:
            ids.append(str(record['id']))
            embeddings.append(record['vector'])

            # 元数据：取出，没有就当空字典
            metadata = record.get('metadata', {})
            # 确保所有元数据值都能被 JSON 序列化
            # ChromaDB 只接受字符串、整数、浮点数或布尔值
            sanitized_metadata = self._sanitize_metadata(metadata)

            # ChromaDB 要求元数据字典不能为空
            if not sanitized_metadata:
                sanitized_metadata = {'_placeholder': 'true'}

            metadatas.append(sanitized_metadata)

            # 文档内容：优先用 metadata.text，没有就用 id
            document = metadata.get('text', record['id'])
            documents.append(str(document))

        # 执行 upsert（ChromaDB 对相同 ID 是幂等的，会覆盖旧记录）
        try:
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
            )
            logger.debug(f"Successfully upserted {len(records)} records to ChromaDB")
        except Exception as e:
            raise RuntimeError(
                f"Failed to upsert {len(records)} records to ChromaDB: {e}"
            ) from e
    
    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """在 ChromaDB 中查询相似向量。

        参数：
            vector: 要查询的向量（embedding）。
            top_k: 最多返回多少条结果。
            filters: 可选的元数据过滤条件（例如 {'source': 'doc1.pdf'}）。
            trace: 可选的 TraceContext，用于可观测性（预留给 Stage F 使用）。
            **kwargs: 提供者特有参数（Chroma 不使用）。

        返回：
            匹配到的记录列表，按相似度从高到低排序。每条记录包含：
                - 'id': 记录标识
                - 'score': 相似度得分（1.0 = 完全相同，0.0 = 正交）
                - 'metadata': 关联的元数据

        抛出：
            ValueError: 如果向量为空或 top_k 不合法。
            RuntimeError: 如果查询操作失败。
        """
        # 校验查询参数
        self.validate_query_vector(vector, top_k)

        # 根据 filters 构造 ChromaDB 的 where 条件
        where_clause = self._build_where_clause(filters) if filters else None

        # 执行查询
        try:
            results = self.collection.query(
                query_embeddings=[vector],
                n_results=top_k,
                where=where_clause,
                include=["metadatas", "distances", "documents"]
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to query ChromaDB with top_k={top_k}: {e}"
            ) from e

        # 把结果转换成标准格式
        # ChromaDB 返回的是嵌套列表：[[id1, id2, ...]]
        output = []

        if results and results['ids'] and results['ids'][0]:
            ids = results['ids'][0]
            distances = results['distances'][0] if 'distances' in results else [0.0] * len(ids)
            metadatas = results['metadatas'][0] if 'metadatas' in results else [{}] * len(ids)
            documents = results['documents'][0] if 'documents' in results else [''] * len(ids)

            for i, record_id in enumerate(ids):
                # 把距离转换成相似度得分
                # ChromaDB 返回的是余弦距离（0 = 完全相同，2 = 完全相反）
                # 转换成相似度：score = 1 - (distance / 2)
                distance = distances[i]
                score = 1.0 - (distance / 2.0)

                output.append({
                    'id': record_id,
                    'score': max(0.0, score),  # 限制在 [0, 1] 范围内
                    'text': documents[i] if documents[i] else '',  # 把文档内容也带上
                    'metadata': metadatas[i] if metadatas[i] else {}
                })

        logger.debug(f"Query returned {len(output)} results")
        return output
    
    def delete(
        self,
        ids: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """根据 ID 从 ChromaDB 中删除记录。

        参数：
            ids: 要删除的记录 ID 列表。
            trace: 可选的 TraceContext，用于可观测性。
            **kwargs: 提供者特有参数。

        抛出：
            ValueError: 如果 ids 列表为空。
            RuntimeError: 如果删除操作失败。
        """
        if not ids:
            raise ValueError("IDs list cannot be empty")

        try:
            self.collection.delete(ids=[str(id_) for id_ in ids])
            logger.debug(f"Successfully deleted {len(ids)} records from ChromaDB")
        except Exception as e:
            raise RuntimeError(
                f"Failed to delete {len(ids)} records from ChromaDB: {e}"
            ) from e

    def clear(
        self,
        collection_name: Optional[str] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """清空 ChromaDB 集合中的所有记录。

        参数：
            collection_name: 可选，要清空的集合名。如果为 None，就清空当前集合。
            trace: 可选的 TraceContext，用于可观测性。
            **kwargs: 提供者特有参数。

        抛出：
            RuntimeError: 如果清空操作失败。
        """
        try:
            target_collection = collection_name or self.collection_name

            # 删除并重建集合（这是 Chroma 里清空集合最高效的方式）
            self.client.delete_collection(name=target_collection)
            self.collection = self.client.get_or_create_collection(
                name=target_collection,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Successfully cleared collection '{target_collection}'")
        except Exception as e:
            raise RuntimeError(
                f"Failed to clear collection '{collection_name or self.collection_name}': {e}"
            ) from e

    def delete_by_metadata(
        self,
        filter_dict: Dict[str, Any],
        trace: Optional[Any] = None,
    ) -> int:
        """根据元数据过滤条件删除记录。

        参数：
            filter_dict: 要匹配的元数据键值对
                （例如 ``{"source_hash": "abc123"}``）。
            trace: 可选的 TraceContext，用于可观测性。

        返回：
            删除的记录数量。

        抛出：
            ValueError: 如果 *filter_dict* 为空。
            RuntimeError: 如果操作失败。
        """
        if not filter_dict:
            raise ValueError("filter_dict cannot be empty")

        try:
            where = self._build_where_clause(filter_dict)
            # 先查出匹配的 ID
            results = self.collection.get(where=where, include=[])
            matching_ids = results.get("ids", [])

            if not matching_ids:
                logger.debug(f"delete_by_metadata: no records matched {filter_dict}")
                return 0

            self.collection.delete(ids=matching_ids)
            logger.info(
                f"delete_by_metadata: deleted {len(matching_ids)} records "
                f"matching {filter_dict}"
            )
            return len(matching_ids)
        except Exception as e:
            raise RuntimeError(
                f"Failed to delete by metadata {filter_dict}: {e}"
            ) from e
    
    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """清洗元数据，使其能被 ChromaDB 接受。

        ChromaDB 要求元数据的值只能是字符串、整数、浮点数或布尔值。
        本方法会把不兼容的类型转换掉，或者过滤掉。

        参数：
            metadata: 原始的元数据字典。

        返回：
            清洗后的元数据字典。
        """
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif value is None:
                # 跳过 None 值
                continue
            elif isinstance(value, (list, tuple)):
                # 转成逗号分隔的字符串
                sanitized[key] = ",".join(str(v) for v in value)
            else:
                # 兜底：直接转成字符串
                sanitized[key] = str(value)

        return sanitized

    def _build_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """根据 filters 构造 ChromaDB 的 where 条件。

        把标准的过滤字典转换成 ChromaDB 的查询格式。

        参数：
            filters: 标准的过滤字典（例如 {'source': 'doc1.pdf'}）。

        返回：
            ChromaDB 的 where 条件字典。

        说明：
            ChromaDB 支持 $eq、$ne、$gt、$lt、$in 等操作符。
            为了简单起见，目前只支持精确相等匹配。
            未来扩展：支持复杂过滤条件。
        """
        # 简单实现：只做精确相等匹配
        # 如果要支持复杂过滤（例如 {'score': {'$gt': 0.5}}），再扩展本方法
        where = {}
        for key, value in filters.items():
            if isinstance(value, dict):
                # 已经是 ChromaDB 的操作符格式（例如 {'$eq': 'value'}）
                where[key] = value
            else:
                # 简单相等
                where[key] = value

        return where

    def get_collection_stats(self) -> Dict[str, Any]:
        """获取当前集合的统计信息。

        返回：
            包含集合统计信息的字典：
                - count: 集合中的记录数量
                - name: 集合名
                - metadata: 集合的元数据
        """
        return {
            'count': self.collection.count(),
            'name': self.collection_name,
            'metadata': self.collection.metadata
        }
    
    def get_by_ids(
        self,
        ids: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """根据 ID 从 ChromaDB 中取出记录。

        本方法被 SparseRetriever（稀疏检索器）使用，用来取出
        被 BM25 搜索命中的切块对应的文本和元数据。

        参数：
            ids: 要取出的记录 ID 列表。
            trace: 可选的 TraceContext，用于可观测性。
            **kwargs: 提供者特有参数（Chroma 不使用）。

        返回：
            记录列表，顺序与输入的 ids 一致。每条记录包含：
                - 'id': 记录标识
                - 'text': 存储的文本内容
                - 'metadata': 关联的元数据
            如果某个 ID 找不到，对应位置返回空字典。

        抛出：
            ValueError: 如果 ids 列表为空。
            RuntimeError: 如果取记录操作失败。
        """
        if not ids:
            raise ValueError("IDs list cannot be empty")

        # 确保所有 ID 都是字符串
        str_ids = [str(id_) for id_ in ids]

        try:
            # ChromaDB 的 get 方法可以按 ID 取记录
            results = self.collection.get(
                ids=str_ids,
                include=["metadatas", "documents"]
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to get records by IDs from ChromaDB: {e}"
            ) from e

        # 建立 ID 到结果的映射，方便 O(1) 查找
        id_to_result: Dict[str, Dict[str, Any]] = {}

        if results and results.get('ids'):
            result_ids = results['ids']
            documents = results.get('documents', [None] * len(result_ids))
            metadatas = results.get('metadatas', [{}] * len(result_ids))

            for i, record_id in enumerate(result_ids):
                id_to_result[record_id] = {
                    'id': record_id,
                    'text': documents[i] if documents and documents[i] else '',
                    'metadata': metadatas[i] if metadatas and metadatas[i] else {}
                }

        # 按输入 ids 的顺序返回结果
        output = []
        for id_ in str_ids:
            if id_ in id_to_result:
                output.append(id_to_result[id_])
            else:
                # 找不到该 ID，对应位置返回空字典
                output.append({})

        logger.debug(f"Retrieved {len([r for r in output if r])} of {len(ids)} records by IDs")
        return output
