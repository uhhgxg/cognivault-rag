"""基于来源的评估器，使用文档级别的标准答案。

通过检查检索到的切块是否来自期望的来源文档
（测试集里的 expected_sources）来评估检索质量，
避免对 chunk_ids 的循环依赖。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.libs.evaluator.base_evaluator import BaseEvaluator


class SourceEvaluator(BaseEvaluator):
    """基于期望来源文档匹配的评估器。

    指标：
    - source_hit_rate: 命中率（0 或 1），只要检索到的切块里有任意一个来自 expected_sources 就算 1
    - source_mrr: 第一个正确来源切块的倒数排名（MRR）
    - source_coverage: 期望来源中被检索到的占比
    """

    SUPPORTED_METRICS = {"source_hit_rate", "source_mrr", "source_coverage"}

    def __init__(
        self,
        settings: Any = None,
        metrics: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        self.settings = settings
        self.kwargs = kwargs

        if metrics is None:
            metrics = ["source_hit_rate", "source_mrr"]

        normalized = [str(m).strip().lower() for m in metrics]
        unsupported = [m for m in normalized if m not in self.SUPPORTED_METRICS]
        if unsupported:
            raise ValueError(
                f"Unsupported metrics: {', '.join(unsupported)}. "
                f"Supported: {', '.join(sorted(self.SUPPORTED_METRICS))}"
            )
        self.metrics = normalized

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Any],
        generated_answer: Optional[str] = None,
        ground_truth: Optional[Any] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        self.validate_query(query)

        expected_sources = self._extract_expected_sources(ground_truth)
        if not expected_sources:
            return {m: 0.0 for m in self.metrics}

        # 从每个检索到的切块中提取 source_path
        retrieved_sources = []
        for chunk in retrieved_chunks:
            source = self._extract_source(chunk)
            retrieved_sources.append(source)

        results: Dict[str, float] = {}

        if "source_hit_rate" in self.metrics:
            results["source_hit_rate"] = self._compute_hit_rate(
                retrieved_sources, expected_sources
            )

        if "source_mrr" in self.metrics:
            results["source_mrr"] = self._compute_mrr(
                retrieved_sources, expected_sources
            )

        if "source_coverage" in self.metrics:
            results["source_coverage"] = self._compute_coverage(
                retrieved_sources, expected_sources
            )

        return results

    def _extract_expected_sources(self, ground_truth: Optional[Any]) -> List[str]:
        if ground_truth is None:
            return []
        if isinstance(ground_truth, dict):
            sources = ground_truth.get("expected_sources", [])
            if isinstance(sources, list):
                return [str(s) for s in sources if s]
        return []

    def _extract_source(self, chunk: Any) -> Optional[str]:
        if isinstance(chunk, dict):
            return chunk.get("source_path") or chunk.get("source") or chunk.get("metadata", {}).get("source_path")
        if hasattr(chunk, "metadata"):
            meta = getattr(chunk, "metadata", {}) or {}
            if isinstance(meta, dict):
                return meta.get("source_path") or meta.get("source")
        return None

    def _compute_hit_rate(self, retrieved: List[Optional[str]], expected: List[str]) -> float:
        return 1.0 if any(r in expected for r in retrieved if r) else 0.0

    def _compute_mrr(self, retrieved: List[Optional[str]], expected: List[str]) -> float:
        for rank, source in enumerate(retrieved, start=1):
            if source and source in expected:
                return 1.0 / rank
        return 0.0

    def _compute_coverage(self, retrieved: List[Optional[str]], expected: List[str]) -> float:
        if not expected:
            return 0.0
        found = set(r for r in retrieved if r) & set(expected)
        return len(found) / len(expected)
