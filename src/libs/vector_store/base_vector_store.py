"""向量存储提供者的抽象基类（父类）。

本模块定义了向量存储提供者的可插拔接口，
通过配置驱动的方式实例化，可以无缝切换不同的后端
（Chroma、Qdrant、Milvus 等）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseVectorStore(ABC):
    """向量存储提供者的抽象基类（父类）。

    所有向量存储实现都必须继承本类，并实现 upsert() 和 query() 方法。
    这样能保证不同提供者（Chroma、Qdrant、Milvus 等）之间接口一致。

    应用的设计原则：
    - 可插拔：子类可以互相替换，不用改上游调用代码。
    - 可观测：接受可选的 TraceContext，用于接入可观测性。
    - 配置驱动：实例通过工厂根据 settings 创建。
    - 幂等：upsert() 操作应当可以安全地重复执行。
    """
    
    @abstractmethod
    def upsert(
        self,
        records: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """向向量存储中插入或更新记录。

        参数：
            records: 要 upsert 的记录列表。每条记录是一个字典，包含：
                - 'id': 唯一标识（str）
                - 'vector': 向量（List[float]）
                - 'metadata': 可选的元数据字典（source、chunk_index 等）
            trace: 可选的 TraceContext，用于可观测性（预留给 Stage F 使用）。
            **kwargs: 提供者特有参数。

        抛出：
            ValueError: 如果记录列表为空，或包含不合法的记录。
            RuntimeError: 如果向量存储操作失败。

        示例：
            >>> records = [
            ...     {
            ...         'id': 'doc1_chunk0',
            ...         'vector': [0.1, 0.2, ..., 0.5],
            ...         'metadata': {'source': 'doc1.pdf', 'page': 1}
            ...     }
            ... ]
            >>> vector_store.upsert(records)

        说明：
            - 本操作应当是幂等的：对同一条记录多次 upsert，
              最终状态应当一致。
            - 实现方应当高效地处理批量操作。
        """
        pass
    
    @abstractmethod
    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """在向量存储中查询相似向量。

        参数：
            vector: 要查询的向量（embedding）。
            top_k: 最多返回多少条结果。
            filters: 可选的元数据过滤条件（例如 {'source': 'doc1.pdf'}）。
            trace: 可选的 TraceContext，用于可观测性（预留给 Stage F 使用）。
            **kwargs: 提供者特有参数。

        返回：
            匹配到的记录列表，按相似度从高到低排序。每条记录是一个字典，包含：
                - 'id': 记录标识
                - 'score': 相似度得分（越高越相似）
                - 'metadata': 关联的元数据
                - 'vector': 可选，存储的向量（视提供者而定）

        抛出：
            ValueError: 如果向量为空或 top_k 不合法。
            RuntimeError: 如果向量存储查询失败。

        示例：
            >>> query_vector = [0.1, 0.2, ..., 0.5]
            >>> results = vector_store.query(query_vector, top_k=5)
            >>> for result in results:
            ...     print(f"ID: {result['id']}, Score: {result['score']}")
        """
        pass
    
    def validate_records(self, records: List[Dict[str, Any]]) -> None:
        """在 upsert 之前校验记录。

        参数：
            records: 要校验的记录列表。

        抛出：
            ValueError: 如果记录列表为空，或包含不合法的记录。
        """
        if not records:
            raise ValueError("Records list cannot be empty")

        for i, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(
                    f"Record at index {i} is not a dict (type: {type(record).__name__})"
                )

            # 校验必填字段
            if 'id' not in record:
                raise ValueError(f"Record at index {i} is missing required field: 'id'")
            if 'vector' not in record:
                raise ValueError(f"Record at index {i} is missing required field: 'vector'")

            # 校验向量格式
            vector = record['vector']
            if not isinstance(vector, (list, tuple)):
                raise ValueError(
                    f"Record at index {i} has invalid vector type: {type(vector).__name__}. "
                    "Expected list or tuple of floats."
                )

            if not vector:
                raise ValueError(f"Record at index {i} has empty vector")

    def validate_query_vector(self, vector: List[float], top_k: int) -> None:
        """校验查询参数。

        参数：
            vector: 要校验的查询向量。
            top_k: 要校验的结果数量。

        抛出：
            ValueError: 如果参数不合法。
        """
        if not isinstance(vector, (list, tuple)):
            raise ValueError(
                f"Query vector must be a list or tuple, got {type(vector).__name__}"
            )
        
        if not vector:
            raise ValueError("Query vector cannot be empty")
        
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")
    
    def delete(
        self,
        ids: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """根据 ID 从向量存储中删除记录。

        参数：
            ids: 要删除的记录 ID 列表。
            trace: 可选的 TraceContext，用于可观测性。
            **kwargs: 提供者特有参数。

        抛出：
            ValueError: 如果 ids 列表为空。
            RuntimeError: 如果删除操作失败。
            NotImplementedError: 如果该提供者不支持删除操作。

        说明：
            这是一个可选操作。不支持删除的提供者应当抛出
            NotImplementedError，并附上清晰的说明。
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement delete() method. "
            "This operation is optional and provider-dependent."
        )

    def clear(
        self,
        collection_name: Optional[str] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """清空向量存储或某个集合中的所有记录。

        参数：
            collection_name: 可选，要清空的集合名。如果为 None，就清空默认集合。
            trace: 可选的 TraceContext，用于可观测性。
            **kwargs: 提供者特有参数。

        抛出：
            RuntimeError: 如果清空操作失败。
            NotImplementedError: 如果该提供者不支持清空操作。

        说明：
            本操作主要用于测试和开发。生产环境请谨慎使用。
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement clear() method. "
            "This operation is optional and primarily for testing."
        )
    
    def get_by_ids(
        self,
        ids: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """根据 ID 取出记录。

        本方法被 SparseRetriever（稀疏检索器）使用，用来取出
        被 BM25 搜索命中的切块对应的文本和元数据
        （BM25 只返回 ID 和得分）。

        参数：
            ids: 要取出的记录 ID 列表。
            trace: 可选的 TraceContext，用于可观测性（预留给 Stage F 使用）。
            **kwargs: 提供者特有参数。

        返回：
            记录列表，顺序与输入的 ids 一致。每条记录是一个字典，包含：
                - 'id': 记录标识
                - 'text': 存储的文本内容
                - 'metadata': 关联的元数据
            如果某个 ID 找不到，对应位置返回空字典。

        抛出：
            ValueError: 如果 ids 列表为空。
            RuntimeError: 如果取记录操作失败。
            NotImplementedError: 如果该提供者不支持本操作。

        示例：
            >>> ids = ["chunk_001", "chunk_002", "chunk_003"]
            >>> records = vector_store.get_by_ids(ids)
            >>> for record in records:
            ...     print(f"ID: {record['id']}, Text: {record['text'][:50]}...")

        说明：
            本操作对混合检索很关键：BM25 返回的是切块 ID，
            需要通过本方法从向量存储中补全文本和元数据。
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_by_ids() method. "
            "This operation is required for SparseRetriever support."
        )
