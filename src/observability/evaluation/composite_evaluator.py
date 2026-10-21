"""组合多个评估器的复合评估器。

本评估器实现了组合模式：它持有一组 BaseEvaluator 实例，
把它们全部运行一遍，并将各自的指标字典合并成一个结果。

设计原则：
- 可插拔：任何 BaseEvaluator 都可以被组合进来。
- 配置驱动：通过 `evaluation.backends: [ragas, custom]` 驱动组合。
- 可观测：会记录各子评估器的成功/失败情况。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from src.libs.evaluator.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class CompositeEvaluator(BaseEvaluator):
    """组合多个评估器并合并指标的评估器。

    每个子评估器都会用相同的参数被调用。结果会被合并进一个
    指标字典。如果两个评估器产生了相同的指标键，后者的值会
    覆盖前者（并打印一条警告）。

    允许部分失败：如果某个子评估器失败，会记录其错误，
    其余子评估器仍会继续执行。

    Example::

        composite = CompositeEvaluator(evaluators=[
            CustomEvaluator(metrics=["hit_rate", "mrr"]),
            RagasEvaluator(metrics=["faithfulness"]),
        ])
        metrics = composite.evaluate(
            query="test", retrieved_chunks=[...],
            generated_answer="...", ground_truth=[...]
        )
        # metrics == {"hit_rate": 1.0, "mrr": 0.5, "faithfulness": 0.92}
    """

    def __init__(
        self,
        evaluators: Optional[Sequence[BaseEvaluator]] = None,
        settings: Any = None,
        **kwargs: Any,
    ) -> None:
        """初始化 CompositeEvaluator。

        Args:
            evaluators: 预先构建好的评估器实例。如果为 None，则从 settings 构建。
            settings: 应用配置（用于配置驱动的组合）。
            **kwargs: 额外参数，会转发给各子评估器。

        Raises:
            ValueError: 如果既没有传入评估器，settings 中也没有
                指定 backends。
        """
        self.settings = settings
        self.kwargs = kwargs

        if evaluators is not None:
            self._evaluators: List[BaseEvaluator] = list(evaluators)
        else:
            self._evaluators = self._build_from_settings(settings, **kwargs)

        if not self._evaluators:
            raise ValueError(
                "CompositeEvaluator requires at least one sub-evaluator. "
                "Provide evaluators directly or configure "
                "'evaluation.backends' in settings.yaml."
            )

        logger.info(
            "CompositeEvaluator initialised with %d evaluator(s): %s",
            len(self._evaluators),
            [type(e).__name__ for e in self._evaluators],
        )

    @property
    def evaluators(self) -> List[BaseEvaluator]:
        """返回已组合的评估器列表。"""
        return list(self._evaluators)

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Any],
        generated_answer: Optional[str] = None,
        ground_truth: Optional[Any] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """运行所有子评估器并合并它们的指标。

        Args:
            query: 用户查询字符串。
            retrieved_chunks: 检索到的 chunk 或记录。
            generated_answer: 可选，生成的回答文本。
            ground_truth: 可选，标准答案数据。
            trace: 可选，用于可观测性的 TraceContext。
            **kwargs: 额外参数。

        Returns:
            合并后的字典，包含所有指标名到浮点数值的映射。

        Raises:
            RuntimeError: 如果所有子评估器都失败了。
        """
        self.validate_query(query)
        self.validate_retrieved_chunks(retrieved_chunks)

        merged: Dict[str, float] = {}
        errors: List[str] = []

        for evaluator in self._evaluators:
            name = type(evaluator).__name__
            try:
                metrics = evaluator.evaluate(
                    query=query,
                    retrieved_chunks=retrieved_chunks,
                    generated_answer=generated_answer,
                    ground_truth=ground_truth,
                    trace=trace,
                    **kwargs,
                )
                for key, value in metrics.items():
                    if key in merged:
                        logger.warning(
                            "Metric '%s' produced by multiple evaluators; "
                            "overwriting with value from %s",
                            key,
                            name,
                        )
                    merged[key] = value

                logger.debug(
                    "%s produced %d metric(s): %s",
                    name,
                    len(metrics),
                    list(metrics.keys()),
                )

            except Exception as exc:
                msg = f"{name} failed: {exc}"
                logger.warning(msg)
                errors.append(msg)

        if not merged and errors:
            raise RuntimeError(
                "All sub-evaluators failed:\n" + "\n".join(errors)
            )

        return merged

    # ── 配置驱动的构建器 ────────────────────────────────────

    @staticmethod
    def _build_from_settings(
        settings: Any,
        **kwargs: Any,
    ) -> List[BaseEvaluator]:
        """根据 settings.evaluation.backends 构建子评估器。

        期望的配置格式::

            evaluation:
              enabled: true
              provider: composite
              backends:
                - ragas
                - custom
              metrics:
                - faithfulness
                - hit_rate
                - mrr

        Args:
            settings: 应用配置。
            **kwargs: 转发给每个子评估器的构造函数。

        Returns:
            BaseEvaluator 实例列表。
        """
        if settings is None:
            return []

        evaluation = getattr(settings, "evaluation", None)
        if evaluation is None:
            return []

        backends = getattr(evaluation, "backends", None)
        if not backends:
            return []

        from src.libs.evaluator.evaluator_factory import EvaluatorFactory

        evaluators: List[BaseEvaluator] = []
        for backend_name in backends:
            backend_name = str(backend_name).strip().lower()
            if backend_name in {"composite", "none", "disabled"}:
                continue  # 避免无限递归 / 空操作

            try:
                # 创建一个 mock settings，覆盖其中的 provider
                from unittest.mock import MagicMock

                sub_settings = MagicMock(wraps=settings)
                sub_eval = MagicMock()
                sub_eval.enabled = True
                sub_eval.provider = backend_name
                sub_eval.metrics = getattr(evaluation, "metrics", [])
                sub_eval.backends = []  # 防止递归
                sub_settings.evaluation = sub_eval

                evaluator = EvaluatorFactory.create(sub_settings, **kwargs)
                evaluators.append(evaluator)
                logger.info("CompositeEvaluator: loaded backend '%s'", backend_name)
            except Exception as exc:
                logger.warning(
                    "CompositeEvaluator: failed to load backend '%s': %s",
                    backend_name,
                    exc,
                )

        return evaluators
