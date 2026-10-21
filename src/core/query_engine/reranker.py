"""核心层重排器（Reranker），协调 libs.reranker 后端并支持降级。

本模块实现 CoreReranker 类，功能包括：
1. 通过 RerankerFactory 集成 libs.reranker（LLM、CrossEncoder、None 三种后端）
2. 后端失败或超时时提供优雅降级
3. 在 RetrievalResult 与重排器输入/输出格式之间做转换
4. 支持 TraceContext 以便做可观测性记录

设计原则：
- 可插拔：用 RerankerFactory 创建配置好的后端
- 配置驱动：从 settings.yaml 读取重排相关设置
- 优雅降级：后端失败时返回原始顺序
- 可观测：集成 TraceContext，方便调试
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.core.types import RetrievalResult
from src.libs.reranker.base_reranker import BaseReranker, NoneReranker
from src.libs.reranker.reranker_factory import RerankerFactory

if TYPE_CHECKING:
    from src.core.settings import Settings

logger = logging.getLogger(__name__)


class RerankError(RuntimeError):
    """重排失败时抛出。"""


@dataclass
class RerankConfig:
    """CoreReranker 的配置。

    属性：
        enabled: 是否启用重排
        top_k: 重排后返回的结果数量
        timeout: 重排后端的超时时间（秒）
        fallback_on_error: 出错时是否返回原始顺序
    """
    enabled: bool = True
    top_k: int = 5
    timeout: float = 30.0
    fallback_on_error: bool = True


@dataclass
class RerankResult:
    """一次重排操作的结果。

    属性：
        results: 重排后的 RetrievalResults 列表
        used_fallback: 是否因后端失败而使用了降级
        fallback_reason: 降级原因（如适用）
        reranker_type: 使用的重排器类型（'llm'、'cross_encoder'、'none'）
        original_order: 重排前的原始结果（用于调试）
    """
    results: List[RetrievalResult] = field(default_factory=list)
    used_fallback: bool = False
    fallback_reason: Optional[str] = None
    reranker_type: str = "none"
    original_order: Optional[List[RetrievalResult]] = None


class CoreReranker:
    """核心层重排器，支持降级。

    本类封装 libs.reranker 的实现，提供：
    1. RetrievalResult 与重排器 dict 格式之间的类型转换
    2. 后端失败时的优雅降级
    3. 通过配置选择后端
    4. 集成 TraceContext

    应用的设计原则：
    - 可插拔：通过 RerankerFactory 选择后端
    - 配置驱动：所有参数都来自 settings
    - 降级：失败时返回原始顺序
    - 可观测：支持 TraceContext

    示例：
        >>> from src.core.settings import load_settings
        >>> settings = load_settings("config/settings.yaml")
        >>> reranker = CoreReranker(settings)
        >>> results = [RetrievalResult(chunk_id="1", score=0.8, text="...", metadata={})]
        >>> reranked = reranker.rerank("query", results)
        >>> print(reranked.results)
    """
    
    def __init__(
        self,
        settings: Settings,
        reranker: Optional[BaseReranker] = None,
        config: Optional[RerankConfig] = None,
    ) -> None:
        """初始化 CoreReranker。

        参数：
            settings: 应用设置，包含重排配置。
            reranker: 可选的重排后端。如果为 None，则通过 RerankerFactory 创建。
            config: 可选的 RerankConfig。如果为 None，则从 settings 中提取。
        """
        self.settings = settings

        # 从 settings 提取配置，或使用外部传入的配置
        if config is not None:
            self.config = config
        else:
            self.config = self._extract_config(settings)

        # 初始化重排后端
        if reranker is not None:
            self._reranker = reranker
        elif not self.config.enabled:
            self._reranker = NoneReranker(settings=settings)
        else:
            try:
                self._reranker = RerankerFactory.create(settings)
            except Exception as e:
                logger.warning(f"Failed to create reranker, using NoneReranker: {e}")
                self._reranker = NoneReranker(settings=settings)

        # 确定重排器类型，用于结果上报
        self._reranker_type = self._get_reranker_type()
    
    def _extract_config(self, settings: Settings) -> RerankConfig:
        """从 settings 中提取 RerankConfig。

        参数：
            settings: 应用设置。

        返回：
            从 settings 读取值的 RerankConfig。
        """
        try:
            rerank_settings = settings.rerank
            return RerankConfig(
                enabled=bool(rerank_settings.enabled) if rerank_settings else False,
                top_k=int(rerank_settings.top_k) if rerank_settings and hasattr(rerank_settings, 'top_k') else 5,
                timeout=float(getattr(rerank_settings, 'timeout', 30.0)) if rerank_settings else 30.0,
                fallback_on_error=True,
            )
        except AttributeError:
            logger.warning("Missing rerank configuration, using defaults (disabled)")
            return RerankConfig(enabled=False)
    
    def _get_reranker_type(self) -> str:
        """获取当前重排后端的类型名。

        返回：
            重排器类型的字符串标识。
        """
        class_name = self._reranker.__class__.__name__
        if "LLM" in class_name:
            return "llm"
        elif "CrossEncoder" in class_name:
            return "cross_encoder"
        elif "None" in class_name:
            return "none"
        else:
            return class_name.lower()
    
    def _results_to_candidates(self, results: List[RetrievalResult]) -> List[Dict[str, Any]]:
        """把 RetrievalResults 转换成重排器候选格式。

        参数：
            results: RetrievalResult 对象列表。

        返回：
            适合作为重排器输入的 dict 列表。
        """
        candidates = []
        for result in results:
            candidates.append({
                "id": result.chunk_id,
                "text": result.text,
                "score": result.score,
                "metadata": result.metadata.copy(),
            })
        return candidates
    
    def _candidates_to_results(
        self,
        candidates: List[Dict[str, Any]],
        original_results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """把重排后的候选结果转换回 RetrievalResults。

        参数：
            candidates: 重排器返回的候选结果。
            original_results: 原始结果，用于参考。

        返回：
            按重排顺序排列的 RetrievalResult 列表。
        """
        # 用原始结果建一个查找表
        id_to_original = {r.chunk_id: r for r in original_results}

        results = []
        for candidate in candidates:
            chunk_id = candidate["id"]

            # 取原始结果，或者新建一个
            if chunk_id in id_to_original:
                original = id_to_original[chunk_id]
                # 用更新后的分数创建新结果
                rerank_score = candidate.get("rerank_score", candidate.get("score", 0.0))
                results.append(RetrievalResult(
                    chunk_id=original.chunk_id,
                    score=rerank_score,
                    text=original.text,
                    metadata={
                        **original.metadata,
                        "original_score": original.score,
                        "rerank_score": rerank_score,
                        "reranked": True,
                    },
                ))
            else:
                # 候选不在原始结果中——直接用候选数据构建
                results.append(RetrievalResult(
                    chunk_id=chunk_id,
                    score=candidate.get("rerank_score", candidate.get("score", 0.0)),
                    text=candidate.get("text", ""),
                    metadata=candidate.get("metadata", {}),
                ))
        
        return results
    
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> RerankResult:
        """使用配置好的后端对检索结果做重排。

        参数：
            query: 用户查询字符串。
            results: 要重排的 RetrievalResult 对象列表。
            top_k: 返回的结果数量。如果为 None，则使用 config.top_k。
            trace: 可选的 TraceContext，用于可观测性。
            **kwargs: 传给重排后端的额外参数。

        返回：
            包含重排结果和元信息的 RerankResult。
        """
        effective_top_k = top_k if top_k is not None else self.config.top_k

        # 结果为空或只有一条时，直接提前返回
        if not results:
            return RerankResult(
                results=[],
                used_fallback=False,
                reranker_type=self._reranker_type,
            )
        
        if len(results) == 1:
            return RerankResult(
                results=results[:],
                used_fallback=False,
                reranker_type=self._reranker_type,
            )
        
        # 如果重排未启用，按原始顺序返回 top_k 条结果
        if not self.config.enabled or isinstance(self._reranker, NoneReranker):
            return RerankResult(
                results=results[:effective_top_k],
                used_fallback=False,
                reranker_type="none",
                original_order=results[:],
            )
        
        # 转换成重排器输入格式
        candidates = self._results_to_candidates(results)

        # 尝试重排
        try:
            logger.debug(f"Reranking {len(candidates)} candidates with {self._reranker_type}")
            _t0 = time.monotonic()
            reranked_candidates = self._reranker.rerank(
                query=query,
                candidates=candidates,
                trace=trace,
                **kwargs,
            )
            _elapsed = (time.monotonic() - _t0) * 1000.0
            
            # 转换回 RetrievalResult
            reranked_results = self._candidates_to_results(reranked_candidates, results)

            # 应用 top_k 限制
            final_results = reranked_results[:effective_top_k]
            
            logger.info(f"Reranking complete: {len(final_results)} results returned")
            
            if trace is not None:
                trace.record_stage("rerank", {
                    "method": self._reranker_type,
                    "provider": self._reranker_type,
                    "input_count": len(candidates),
                    "output_count": len(final_results),
                    "chunks": [
                        {
                            "chunk_id": r.chunk_id,
                            "score": round(r.score, 4),
                            "text": r.text or "",
                            "source": r.metadata.get("source_path", r.metadata.get("source", "")),
                        }
                        for r in final_results
                    ],
                }, elapsed_ms=_elapsed)
            
            return RerankResult(
                results=final_results,
                used_fallback=False,
                reranker_type=self._reranker_type,
                original_order=results[:],
            )
            
        except Exception as e:
            logger.warning(f"Reranking failed, using fallback: {e}")
            
            if self.config.fallback_on_error:
                # 降级：返回原始顺序
                fallback_results = []
                for result in results[:effective_top_k]:
                    fallback_results.append(RetrievalResult(
                        chunk_id=result.chunk_id,
                        score=result.score,
                        text=result.text,
                        metadata={
                            **result.metadata,
                            "reranked": False,
                            "rerank_fallback": True,
                        },
                    ))
                
                return RerankResult(
                    results=fallback_results,
                    used_fallback=True,
                    fallback_reason=str(e),
                    reranker_type=self._reranker_type,
                    original_order=results[:],
                )
            else:
                raise RerankError(f"Reranking failed and fallback disabled: {e}") from e
    
    @property
    def reranker_type(self) -> str:
        """获取当前重排后端的类型。"""
        return self._reranker_type

    @property
    def is_enabled(self) -> bool:
        """检查重排是否已启用。"""
        return self.config.enabled and not isinstance(self._reranker, NoneReranker)


def create_core_reranker(
    settings: Settings,
    reranker: Optional[BaseReranker] = None,
) -> CoreReranker:
    """工厂函数，创建 CoreReranker 实例。

    参数：
        settings: 应用设置。
        reranker: 可选的重排后端，用于覆盖默认创建的后端。

    返回：
        配置好的 CoreReranker 实例。
    """
    return CoreReranker(settings=settings, reranker=reranker)
