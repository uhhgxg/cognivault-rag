"""倒数排名融合（Reciprocal Rank Fusion，RRF），用于合并多路检索结果。

本模块实现 RRF 融合算法，把稠密检索和稀疏检索产出的排序列表合并成
统一排序。RRF 是一种简单但有效的排名聚合方法，不需要做分数归一化。

参考论文：
    Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009).
    "Reciprocal rank fusion outperforms condorcet and individual rank learning methods."
    SIGIR '09.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.core.types import RetrievalResult

logger = logging.getLogger(__name__)


class RRFFusion:
    """倒数排名融合（RRF），用于合并多个排序列表。

    RRF 用以下公式合并来自多个来源的排名：
        RRF_score(d) = Σ 1 / (k + rank(d))

    其中：
        - d 是一个文档（chunk）
        - k 是平滑常数（通常是 60）
        - rank(d) 是文档 d 在某个排序列表中的排名（从 1 开始）

    关键特性：
    - 确定性：相同的输入始终产生相同的输出顺序
    - 不依赖分数：只用排名位置，不用原始分数
    - 不需要归一化：能处理不同量纲的分数
    - 能处理缺失文档：只在某一个列表中出现的文档也能贡献分数

    应用的设计原则：
    - 配置驱动：k 参数可配置（默认：60）
    - 类型安全：返回标准化的 RetrievalResult 对象
    - 确定性：用 chunk_id 做平局打破，排序稳定
    - 可观测：通过日志调试融合过程

    属性：
        k: RRF 公式的平滑常数（默认：60）。
           k 越大，排名靠后的文档权重越高。

    示例：
        >>> fusion = RRFFusion(k=60)
        >>> dense_results = [
        ...     RetrievalResult(chunk_id="a", score=0.9, text="...", metadata={}),
        ...     RetrievalResult(chunk_id="b", score=0.8, text="...", metadata={}),
        ... ]
        >>> sparse_results = [
        ...     RetrievalResult(chunk_id="b", score=5.2, text="...", metadata={}),
        ...     RetrievalResult(chunk_id="c", score=4.1, text="...", metadata={}),
        ... ]
        >>> fused = fusion.fuse([dense_results, sparse_results], top_k=5)
    """

    # 原始 RRF 论文推荐的默认平滑常数
    DEFAULT_K = 60

    def __init__(self, k: int = DEFAULT_K) -> None:
        """初始化 RRF 融合器，可配置平滑常数。

        参数：
            k: RRF 公式的平滑常数（默认：60）。
               - 必须是正整数
               - 值越大，排名差异的影响越小
               - 常用值：60（原论文）、20、100

        抛出：
            ValueError: 当 k 不是正整数时抛出。
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")
        
        self.k = k
        logger.info(f"RRFFusion initialized with k={k}")
    
    def fuse(
        self,
        ranking_lists: List[List[RetrievalResult]],
        top_k: Optional[int] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        """用倒数排名融合（RRF）合并多个排序列表。

        参数：
            ranking_lists: 排序列表的列表，每个列表包含按相关度降序排列的
                           RetrievalResult 对象。
                           通常是 [dense_results, sparse_results]。
            top_k: 最多返回的结果数量。如果为 None，则返回全部。
            trace: 可选的 TraceContext，用于可观测性（为 Stage F 预留）。

        返回：
            按 RRF 融合分数降序排列的 RetrievalResult 对象列表。
            score 字段存的是 RRF 分数，不是原始检索分数。
            text 和 metadata 取自每个 chunk 第一次出现时的数据。

        抛出：
            ValueError: 当 ranking_lists 为空时抛出。

        说明：
            - 出现在多个列表中的文档会累加所有列表的贡献
            - 只出现在一个列表中的文档也会得到 RRF 分数
            - 平局打破：RRF 分数相同时，按 chunk_id 排序以保证稳定性

        示例：
            >>> fusion = RRFFusion(k=60)
            >>> fused = fusion.fuse([dense_results, sparse_results], top_k=10)
            >>> for r in fused:
            ...     print(f"[RRF={r.score:.4f}] {r.chunk_id}")
        """
        if not ranking_lists:
            raise ValueError("ranking_lists cannot be empty")
        
        # 过滤掉空列表
        non_empty_lists = [lst for lst in ranking_lists if lst]
        
        if not non_empty_lists:
            logger.debug("All ranking lists are empty, returning empty result")
            return []
        
        logger.debug(
            f"Fusing {len(non_empty_lists)} ranking lists with "
            f"sizes {[len(lst) for lst in non_empty_lists]}"
        )
        
        # 第 1 步：为每个唯一的 chunk 计算 RRF 分数
        rrf_scores: Dict[str, float] = {}
        chunk_data: Dict[str, RetrievalResult] = {}  # 保留 text/metadata

        for list_idx, ranking_list in enumerate(non_empty_lists):
            for rank, result in enumerate(ranking_list, start=1):
                chunk_id = result.chunk_id

                # 计算 RRF 贡献值：1 / (k + rank)
                rrf_contribution = 1.0 / (self.k + rank)

                # 累加分数
                if chunk_id not in rrf_scores:
                    rrf_scores[chunk_id] = 0.0
                    # 保存第一次出现时的数据（text、metadata）
                    chunk_data[chunk_id] = result

                rrf_scores[chunk_id] += rrf_contribution
        
        logger.debug(f"Computed RRF scores for {len(rrf_scores)} unique chunks")
        
        # 第 2 步：用 RRF 分数创建融合结果
        fused_results = []
        for chunk_id, rrf_score in rrf_scores.items():
            original = chunk_data[chunk_id]
            fused_results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    score=rrf_score,
                    text=original.text,
                    metadata=original.metadata.copy(),
                )
            )
        
        # 第 3 步：按 RRF 分数降序排列，分数相同时按 chunk_id 排序保证稳定
        fused_results.sort(key=lambda r: (-r.score, r.chunk_id))

        # 第 4 步：如果指定了 top_k，则做截断
        if top_k is not None and top_k > 0:
            fused_results = fused_results[:top_k]
        
        logger.debug(
            f"Fusion complete: {len(fused_results)} results "
            f"(top_k={top_k if top_k else 'all'})"
        )
        
        return fused_results
    
    def fuse_with_weights(
        self,
        ranking_lists: List[List[RetrievalResult]],
        weights: Optional[List[float]] = None,
        top_k: Optional[int] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        """带可选权重的多排序列表融合。

        这是 fuse() 的扩展版本，允许给不同排名来源设置权重。例如语义类
        查询可以给稠密检索更高权重，关键词类查询可以给稀疏检索更高权重。

        参数：
            ranking_lists: 排序列表的列表，每个列表包含 RetrievalResult 对象。
            weights: 可选的权重列表，对应每个排序列表（默认：等权）。
                     如果提供，长度必须和 ranking_lists 一致。
                     权重会乘到 RRF 贡献值上。
            top_k: 最多返回的结果数量。如果为 None，则返回全部。
            trace: 可选的 TraceContext，用于可观测性（为 Stage F 预留）。

        返回：
            按加权 RRF 分数降序排列的 RetrievalResult 对象列表。

        抛出：
            ValueError: 当 ranking_lists 为空，或 weights 长度不匹配时抛出。

        示例：
            >>> fusion = RRFFusion(k=60)
            >>> # 给稠密检索 1.5 倍权重
            >>> fused = fusion.fuse_with_weights(
            ...     [dense_results, sparse_results],
            ...     weights=[1.5, 1.0],
            ...     top_k=10
            ... )
        """
        if not ranking_lists:
            raise ValueError("ranking_lists cannot be empty")
        
        # 默认使用等权
        if weights is None:
            weights = [1.0] * len(ranking_lists)
        
        if len(weights) != len(ranking_lists):
            raise ValueError(
                f"weights length ({len(weights)}) must match "
                f"ranking_lists length ({len(ranking_lists)})"
            )
        
        # 校验权重
        for i, w in enumerate(weights):
            if not isinstance(w, (int, float)) or w < 0:
                raise ValueError(f"Weight at index {i} must be non-negative, got {w}")
        
        # 过滤掉空列表（同时保持权重对齐）
        filtered = [
            (lst, w) for lst, w in zip(ranking_lists, weights) if lst
        ]
        
        if not filtered:
            logger.debug("All ranking lists are empty, returning empty result")
            return []
        
        non_empty_lists, filtered_weights = zip(*filtered)
        
        logger.debug(
            f"Fusing {len(non_empty_lists)} ranking lists with "
            f"weights={list(filtered_weights)}"
        )
        
        # 计算加权 RRF 分数
        rrf_scores: Dict[str, float] = {}
        chunk_data: Dict[str, RetrievalResult] = {}

        for list_idx, (ranking_list, weight) in enumerate(zip(non_empty_lists, filtered_weights)):
            for rank, result in enumerate(ranking_list, start=1):
                chunk_id = result.chunk_id

                # 加权 RRF 贡献值
                rrf_contribution = weight * (1.0 / (self.k + rank))
                
                if chunk_id not in rrf_scores:
                    rrf_scores[chunk_id] = 0.0
                    chunk_data[chunk_id] = result
                
                rrf_scores[chunk_id] += rrf_contribution
        
        # 创建结果并排序
        fused_results = [
            RetrievalResult(
                chunk_id=chunk_id,
                score=rrf_score,
                text=chunk_data[chunk_id].text,
                metadata=chunk_data[chunk_id].metadata.copy(),
            )
            for chunk_id, rrf_score in rrf_scores.items()
        ]
        
        fused_results.sort(key=lambda r: (-r.score, r.chunk_id))
        
        if top_k is not None and top_k > 0:
            fused_results = fused_results[:top_k]
        
        return fused_results


def rrf_score(rank: int, k: int = RRFFusion.DEFAULT_K) -> float:
    """计算单个排名位置的 RRF 分数贡献值。

    这是一个工具函数，用于计算单个 RRF 贡献值。

    参数：
        rank: 从 1 开始的排名位置（1 = 排名最高）
        k: 平滑常数（默认：60）

    返回：
        RRF 分数贡献值：1 / (k + rank)

    抛出：
        ValueError: 当 rank 不是正整数或 k 不是正数时抛出。

    示例：
        >>> rrf_score(1, k=60)  # 排名第一的文档
        0.01639344262295082
        >>> rrf_score(10, k=60)  # 排名第十的文档
        0.014285714285714285
    """
    if not isinstance(rank, int) or rank <= 0:
        raise ValueError(f"rank must be a positive integer, got {rank}")
    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")
    
    return 1.0 / (k + rank)
