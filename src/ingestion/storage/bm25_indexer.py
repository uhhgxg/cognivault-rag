"""BM25 索引器，用于构建和查询倒排索引。

本模块实现 BM25 索引组件，负责：
- 计算 IDF（逆文档频率）分数
- 构建倒排索引结构
- 索引的持久化和从磁盘加载
- 支持增量更新

设计原则：
- 幂等：相同输入重新构建会得到相同结果
- 可观测：接收 TraceContext 以便未来接入
- 持久化：索引保存到 data/db/bm25/ 目录
- 确定性：相同语料库产生相同 IDF 分数
"""

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


class BM25Indexer:
    """构建和查询 BM25 倒排索引。

    这个索引器接收来自 SparseEncoder 的词项统计，
    构建一个带 IDF 分数和倒排表（posting list）的可查询 BM25 索引。

    索引结构：
        {
            "metadata": {
                "num_docs": int,
                "avg_doc_length": float,
                "total_terms": int
            },
            "index": {
                "term": {
                    "idf": float,
                    "df": int,  # 文档频率
                    "postings": [
                        {"chunk_id": str, "tf": int, "doc_length": int},
                        ...
                    ]
                },
                ...
            }
        }

    BM25 IDF 公式：
        IDF(term) = log((N - df + 0.5) / (df + 0.5))

        其中：
        - N = 文档总数
        - df = 文档频率（包含该词项的文档数）

    示例：
        >>> indexer = BM25Indexer(index_dir="data/db/bm25")
        >>> 
        >>> # 从 SparseEncoder 输出构建索引
        >>> term_stats = [
        ...     {"chunk_id": "1", "term_frequencies": {"hello": 2, "world": 1}, "doc_length": 3},
        ...     {"chunk_id": "2", "term_frequencies": {"hello": 1, "python": 1}, "doc_length": 2}
        ... ]
        >>> indexer.build(term_stats)
        >>> 
        >>> # 查询索引
        >>> results = indexer.query(["hello"], top_k=2)
        >>> len(results) <= 2  # True
    """
    
    def __init__(
        self,
        index_dir: str = "data/db/bm25",
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """初始化 BM25Indexer。

        参数：
            index_dir: 存放索引文件的目录（默认：data/db/bm25）
            k1: BM25 词频饱和参数（默认：1.5）
            b: BM25 长度归一化参数（默认：0.75）

        可能抛出：
            ValueError: 如果 k1 或 b 超出有效范围
        """
        if k1 <= 0:
            raise ValueError(f"k1 must be > 0, got {k1}")
        if not 0 <= b <= 1:
            raise ValueError(f"b must be in [0, 1], got {b}")
        
        self.index_dir = Path(index_dir)
        self.k1 = k1
        self.b = b
        
        # 内存中的索引结构
        self._index: Dict[str, Dict[str, Any]] = {}
        self._metadata: Dict[str, Any] = {}
        
    def build(
        self,
        term_stats: List[Dict[str, Any]],
        collection: str = "default",
        trace: Optional[Any] = None,
    ) -> None:
        """根据词项统计构建 BM25 索引。

        本方法流程：
        1. 计算语料库级统计（N、avg_doc_length、DF）
        2. 为每个词项计算 IDF
        3. 构建带倒排表的倒排索引
        4. 持久化到磁盘

        参数：
            term_stats: 来自 SparseEncoder.encode() 的统计列表。
                每个元素应包含：chunk_id、term_frequencies、doc_length
            collection: 用于组织索引的集合名（默认："default"）
            trace: 可选的 TraceContext，用于可观测性

        可能抛出：
            ValueError: 如果 term_stats 为空或结构无效

        示例：
            >>> term_stats = [
            ...     {
            ...         "chunk_id": "doc1_chunk0",
            ...         "term_frequencies": {"machine": 2, "learning": 1},
            ...         "doc_length": 3
            ...     }
            ... ]
            >>> indexer.build(term_stats, collection="my_docs")
        """
        if not term_stats:
            raise ValueError("Cannot build index from empty term_stats")
        
        # 校验结构
        self._validate_term_stats(term_stats)
        
        # 第 1 步：计算语料库级统计
        num_docs = len(term_stats)
        total_length = sum(stat["doc_length"] for stat in term_stats)
        avg_doc_length = total_length / num_docs if num_docs > 0 else 0.0
        
        # 计算每个词项的文档频率（DF）
        doc_freq: Dict[str, int] = {}
        for stat in term_stats:
            for term in stat["term_frequencies"].keys():
                doc_freq[term] = doc_freq.get(term, 0) + 1
        
        # 第 2 步：构建带 IDF 的倒排索引
        index: Dict[str, Dict[str, Any]] = {}
        
        for term, df in doc_freq.items():
            # 用 BM25 公式计算 IDF
            idf = self._calculate_idf(num_docs, df)
            
            # 为该词项构建倒排表
            postings = []
            for stat in term_stats:
                tf = stat["term_frequencies"].get(term, 0)
                if tf > 0:  # 只包含含有该词项的文档
                    postings.append({
                        "chunk_id": stat["chunk_id"],
                        "tf": tf,
                        "doc_length": stat["doc_length"]
                    })
            
            index[term] = {
                "idf": idf,
                "df": df,
                "postings": postings
            }
        
        # 第 3 步：保存元数据
        self._metadata = {
            "num_docs": num_docs,
            "avg_doc_length": avg_doc_length,
            "total_terms": len(index),
            "collection": collection,
        }
        
        self._index = index
        
        # 第 4 步：持久化到磁盘
        self._save(collection)
    
    def load(
        self,
        collection: str = "default",
        trace: Optional[Any] = None,
    ) -> bool:
        """从磁盘加载索引。

        参数：
            collection: 要加载的集合名
            trace: 可选的 TraceContext，用于可观测性

        返回：
            索引加载成功返回 True，没找到返回 False

        可能抛出：
            ValueError: 如果索引文件损坏
        """
        index_path = self._get_index_path(collection)
        
        if not index_path.exists():
            return False
        
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 校验结构
            if "metadata" not in data or "index" not in data:
                raise ValueError(f"Invalid index file structure: missing metadata or index")
            
            self._metadata = data["metadata"]
            self._index = data["index"]
            
            return True
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Corrupted index file at {index_path}: {e}")
    
    def query(
        self,
        query_terms: List[str],
        top_k: int = 10,
        trace: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """用 BM25 打分查询索引。

        参数：
            query_terms: 要搜索的词项列表
            top_k: 最多返回的结果数
            trace: 可选的 TraceContext，用于可观测性

        返回：
            按 BM25 分数降序排列的结果列表。
            每个结果：{"chunk_id": str, "score": float}

        可能抛出：
            ValueError: 如果索引未加载或 query_terms 为空

        示例：
            >>> indexer.load("my_docs")
            >>> results = indexer.query(["machine", "learning"], top_k=5)
            >>> results[0]["score"] > 0  # 找到匹配时为 True
        """
        if not self._index:
            raise ValueError("Index not loaded. Call load() or build() first.")
        
        if not query_terms:
            raise ValueError("query_terms cannot be empty")
        
        # 把查询词项转小写以匹配索引（SparseEncoder 在构建时做了小写化）
        query_terms = [t.lower() for t in query_terms]
        
        # 计算所有文档的 BM25 分数
        scores: Dict[str, float] = {}
        
        for term in query_terms:
            if term not in self._index:
                continue  # 词项不在语料库中，跳过
            
            term_data = self._index[term]
            idf = term_data["idf"]
            
            for posting in term_data["postings"]:
                chunk_id = posting["chunk_id"]
                tf = posting["tf"]
                doc_length = posting["doc_length"]
                
                # 该词项的 BM25 分数贡献
                term_score = self._calculate_bm25_score(
                    tf=tf,
                    doc_length=doc_length,
                    avg_doc_length=self._metadata["avg_doc_length"],
                    idf=idf
                )
                
                scores[chunk_id] = scores.get(chunk_id, 0.0) + term_score
        
        # 按分数降序排序并返回 top_k 个结果
        sorted_results = sorted(
            [{"chunk_id": cid, "score": score} for cid, score in scores.items()],
            key=lambda x: x["score"],
            reverse=True
        )
        
        return sorted_results[:top_k]
    
    def rebuild(
        self,
        term_stats: List[Dict[str, Any]],
        collection: str = "default",
        trace: Optional[Any] = None,
    ) -> None:
        """从头重建索引（build 的别名，意图更明确）。

        这是一个便捷方法，在替换已有索引时让意图更清晰。

        参数：
            term_stats: 来自 SparseEncoder 的统计列表
            collection: 集合名
            trace: 可选的 TraceContext，用于可观测性
        """
        self.build(term_stats, collection, trace)

    def add_documents(
        self,
        term_stats: List[Dict[str, Any]],
        collection: str = "default",
        doc_id: Optional[str] = None,
        trace: Optional[Any] = None,
    ) -> None:
        """向 BM25 索引中增量添加文档。

        加载已有索引（如果有），可选地移除给定 *doc_id* 的旧倒排记录
        （用于支持重新摄取），合并新的词项统计，重新计算 IDF 分数，
        然后保存。

        参数：
            term_stats: 来自 SparseEncoder.encode() 的新词项统计。
            collection: 集合名。
            doc_id: 如果提供，则在添加新记录前，先移除 chunk_id 以此
                为前缀的已有记录（实现幂等的重新摄取）。
            trace: 可选的 TraceContext。
        """
        if not term_stats:
            return

        self._validate_term_stats(term_stats)

        # 加载已有索引（没有就从头开始）
        if not self._index:
            self.load(collection)

        # 移除该文档的旧倒排记录（重新摄取的情况）
        if doc_id and self._index:
            self.remove_document(doc_id, collection)

        # 从当前索引的倒排表重建已有的 term_stats
        existing_stats: Dict[str, Dict[str, Any]] = {}  # chunk_id -> 统计信息
        for term, term_data in self._index.items():
            for posting in term_data["postings"]:
                cid = posting["chunk_id"]
                if cid not in existing_stats:
                    existing_stats[cid] = {
                        "chunk_id": cid,
                        "term_frequencies": {},
                        "doc_length": posting["doc_length"],
                    }
                existing_stats[cid]["term_frequencies"][term] = posting["tf"]

        # 合并：已有 + 新增
        combined = list(existing_stats.values()) + list(term_stats)

        # 用合并后的统计重建完整索引
        self.build(combined, collection, trace)

    def remove_document(
        self,
        doc_id: str,
        collection: str = "default",
    ) -> bool:
        """从 BM25 索引中移除某文档的所有倒排记录。

        加载索引（如果尚未加载），移除所有 ``chunk_id`` 以 *doc_id*
        开头的倒排记录，重新计算统计信息，然后重新保存索引。

        参数：
            doc_id: 文档标识（或前缀）。所有 ``chunk_id`` 以此值开头
                的记录都会被移除。
            collection: 集合名。

        返回：
            如果移除了任何记录返回 ``True``，否则返回 ``False``。
        """
        if not self._index:
            if not self.load(collection):
                return False

        removed_any = False
        terms_to_delete: list[str] = []

        for term, term_data in self._index.items():
            original_len = len(term_data["postings"])
            term_data["postings"] = [
                p for p in term_data["postings"]
                if not p["chunk_id"].startswith(doc_id)
            ]
            if len(term_data["postings"]) < original_len:
                removed_any = True
            # 标记空词项以便清理
            if not term_data["postings"]:
                terms_to_delete.append(term)
            else:
                term_data["df"] = len(term_data["postings"])

        # 移除空词项
        for term in terms_to_delete:
            del self._index[term]

        if removed_any:
            # 重新计算全局元数据
            all_chunk_ids: set[str] = set()
            total_length = 0
            for td in self._index.values():
                for p in td["postings"]:
                    all_chunk_ids.add(p["chunk_id"])
                    total_length += p["doc_length"]

            num_docs = len(all_chunk_ids)
            avg_doc_length = total_length / num_docs if num_docs else 0.0

            # 重新计算 IDF 值
            for td in self._index.values():
                td["idf"] = self._calculate_idf(num_docs, td["df"])

            self._metadata = {
                "num_docs": num_docs,
                "avg_doc_length": avg_doc_length,
                "total_terms": len(self._index),
                "collection": collection,
            }
            self._save(collection)

        return removed_any
    
    # ===== 私有辅助方法 =====
    
    def _calculate_idf(self, num_docs: int, df: int) -> float:
        """用 BM25 公式计算 IDF。

        公式：IDF(term) = log((N - df + 0.5) / (df + 0.5))

        参数：
            num_docs: 语料库中的文档总数
            df: 文档频率（包含该词项的文档数）

        返回：
            IDF 分数（对于非常常见的词项可能为负数）
        """
        return math.log((num_docs - df + 0.5) / (df + 0.5))
    
    def _calculate_bm25_score(
        self,
        tf: int,
        doc_length: int,
        avg_doc_length: float,
        idf: float
    ) -> float:
        """计算文档中单个词项的 BM25 分数。

        公式：score = IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avg_doc_length)))

        参数：
            tf: 文档中的词频
            doc_length: 文档长度（词项数）
            avg_doc_length: 语料库的平均文档长度
            idf: 该词项的 IDF 分数

        返回：
            BM25 分数贡献值
        """
        # 避免除零
        if avg_doc_length == 0:
            avg_doc_length = 1.0
        
        # BM25 公式
        numerator = tf * (self.k1 + 1)
        denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / avg_doc_length))
        
        return idf * (numerator / denominator)
    
    def _validate_term_stats(self, term_stats: List[Dict[str, Any]]) -> None:
        """校验 term_stats 结构。

        可能抛出：
            ValueError: 如果结构无效
        """
        for i, stat in enumerate(term_stats):
            if not isinstance(stat, dict):
                raise ValueError(f"term_stats[{i}] must be a dict, got {type(stat)}")
            
            required_fields = ["chunk_id", "term_frequencies", "doc_length"]
            for field in required_fields:
                if field not in stat:
                    raise ValueError(f"term_stats[{i}] missing required field: {field}")
            
            if not isinstance(stat["term_frequencies"], dict):
                raise ValueError(
                    f"term_stats[{i}]['term_frequencies'] must be dict, "
                    f"got {type(stat['term_frequencies'])}"
                )
            
            if not isinstance(stat["doc_length"], int) or stat["doc_length"] < 0:
                raise ValueError(
                    f"term_stats[{i}]['doc_length'] must be non-negative int, "
                    f"got {stat['doc_length']}"
                )
    
    def _get_index_path(self, collection: str) -> Path:
        """获取索引文件的路径。

        参数：
            collection: 集合名

        返回：
            索引文件路径
        """
        return self.index_dir / f"{collection}_bm25.json"
    
    def _save(self, collection: str) -> None:
        """把索引保存到磁盘。

        参数：
            collection: 集合名
        """
        # 确保目录存在
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        index_path = self._get_index_path(collection)
        
        # 准备数据
        data = {
            "metadata": self._metadata,
            "index": self._index
        }
        
        # 原子写入（先写临时文件，再重命名）
        temp_path = index_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # 原子重命名
            temp_path.replace(index_path)
            
        except Exception as e:
            # 写入失败时清理临时文件
            if temp_path.exists():
                temp_path.unlink()
            raise
