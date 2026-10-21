"""基于 Cross-Encoder 的重排序器实现。

本模块使用 Cross-Encoder 模型对（查询, 段落）对直接打分，从而实现重排序。
既支持通过 sentence-transformers 加载本地模型，也支持基于 API 的接口。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.libs.reranker.base_reranker import BaseReranker

logger = logging.getLogger(__name__)


class CrossEncoderRerankError(RuntimeError):
    """当 Cross-Encoder 重排序失败时抛出。"""


class CrossEncoderReranker(BaseReranker):
    """基于 Cross-Encoder 的重排序器，用来给（查询, 段落）对打分。

    本实现使用 Cross-Encoder 模型（例如 ms-marco-MiniLM），
    直接对（查询, 段落）对进行编码和打分。相比双塔（bi-encoder）方案，
    它能给出更准确的相关性得分，但计算开销更大。

    应用的设计原则：
    - 可插拔：可以通过工厂与其他重排序器实现互换。
    - 配置驱动：模型名和参数都来自 settings.yaml。
    - 可观测：支持 TraceContext，用于监控（Stage F 接入）。
    - 兜底感知：提供超时/失败信号，方便上游做兜底。
    - 可测试：支持注入 mock 打分器，方便测试。
    """
    
    def __init__(
        self,
        settings: Any,
        model: Optional[Any] = None,
        timeout: float = 10.0,
        **kwargs: Any
    ) -> None:
        """初始化 Cross-Encoder 重排序器。

        参数：
            settings: 应用配置，里面包含重排相关配置。
            model: 可选，已初始化好的 CrossEncoder 模型。如果为 None，
                就用 settings.rerank.model 来加载。测试时可用来注入 mock 模型。
            timeout: 重排序最多等待的秒数。默认 10 秒。
                当重排耗时太长时，可借此启用兜底策略。
            **kwargs: 其他提供者特有参数。
        """
        self.settings = settings
        self.timeout = timeout
        self.kwargs = kwargs

        # 初始化或注入模型
        if model is not None:
            self.model = model
        else:
            try:
                model_name = self._get_model_name_from_settings(settings)
                self.model = self._load_cross_encoder_model(model_name)
            except Exception as e:
                raise CrossEncoderRerankError(
                    f"Failed to initialize Cross-Encoder model: {e}"
                ) from e

    def _get_model_name_from_settings(self, settings: Any) -> str:
        """从 settings 中提取模型名。

        参数：
            settings: 应用配置。

        返回：
            模型名字符串。

        抛出：
            AttributeError: 如果没配置 rerank.model。
        """
        try:
            model_name = settings.rerank.model
            if not model_name or not isinstance(model_name, str):
                raise ValueError("Model name must be a non-empty string")
            return model_name
        except AttributeError as e:
            raise AttributeError(
                "Missing configuration: settings.rerank.model. "
                "Please specify 'rerank.model' in settings.yaml"
            ) from e
    
    def _load_cross_encoder_model(self, model_name: str) -> Any:
        """加载 Cross-Encoder 模型。

        参数：
            model_name: Cross-Encoder 模型的名字或路径。

        返回：
            初始化好的 CrossEncoder 实例。

        抛出：
            ImportError: 如果没装 sentence-transformers。
            RuntimeError: 如果模型加载失败。
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for Cross-Encoder reranking. "
                "Install it with: pip install sentence-transformers"
            ) from e
        
        try:
            logger.info(f"Loading Cross-Encoder model: {model_name}")
            model = CrossEncoder(model_name)
            logger.info(f"Cross-Encoder model loaded successfully: {model_name}")
            return model
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Cross-Encoder model '{model_name}': {e}"
            ) from e
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """用 Cross-Encoder 打分来重排候选项。

        参数：
            query: 用户的查询字符串。
            candidates: 要重排的候选记录列表。每条必须包含 'text' 或
                'content' 字段，用来打分。
            trace: 可选的 TraceContext，用于可观测性（Stage F 接入）。
            **kwargs: 额外参数（top_k 用来限制输出数量等）。

        返回：
            重排后的候选列表，按相关性得分从高到低排序。
            每个候选项会带上 'rerank_score' 字段，记录模型给出的得分。

        抛出：
            ValueError: 如果 query 或 candidates 不合法。
            CrossEncoderRerankError: 如果打分失败或超时。
        """
        # 校验输入
        self.validate_query(query)
        self.validate_candidates(candidates)

        # 取出 top_k 参数
        top_k = kwargs.get("top_k", len(candidates))
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        try:
            # 准备要打分的（查询, 段落）对
            pairs = self._prepare_pairs(query, candidates)

            # 用模型给这些对打分
            scores = self._score_pairs(pairs, trace=trace)

            # 把得分贴到候选项上并排序
            reranked = self._attach_scores_and_sort(candidates, scores, top_k)

            if trace:
                self._log_trace(trace, query, len(candidates), len(reranked))

            return reranked

        except Exception as e:
            logger.error(f"Cross-Encoder reranking failed: {e}", exc_info=True)
            # 给上游兜底逻辑发失败信号
            raise CrossEncoderRerankError(
                f"Cross-Encoder reranking failed: {e}"
            ) from e

    def _prepare_pairs(
        self,
        query: str,
        candidates: List[Dict[str, Any]]
    ) -> List[tuple[str, str]]:
        """准备要打分的（查询, 段落）对。

        参数：
            query: 用户的查询。
            candidates: 候选记录列表。

        返回：
            （查询, 段落文本）元组列表。
        """
        pairs = []
        for candidate in candidates:
            # 从候选项中提取文本（同时支持 'text' 和 'content' 两种键）
            text = candidate.get("text") or candidate.get("content", "")
            if not isinstance(text, str):
                text = str(text)
            pairs.append((query, text))
        return pairs

    def _score_pairs(
        self,
        pairs: List[tuple[str, str]],
        trace: Optional[Any] = None
    ) -> List[float]:
        """用 Cross-Encoder 模型给（查询, 段落）对打分。

        参数：
            pairs: （查询, 段落）元组列表。
            trace: 可选的 TraceContext，用于可观测性。

        返回：
            相关性得分列表（每对一个得分）。

        抛出：
            CrossEncoderRerankError: 如果打分失败或超时。
        """
        try:
            # 用 model.predict() 批量给所有对打分
            scores = self.model.predict(pairs)

            # 如果是 numpy 数组，就转成 list
            if hasattr(scores, 'tolist'):
                scores = scores.tolist()

            return scores

        except Exception as e:
            raise CrossEncoderRerankError(
                f"Failed to score pairs with Cross-Encoder: {e}"
            ) from e

    def _attach_scores_and_sort(
        self,
        candidates: List[Dict[str, Any]],
        scores: List[float],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """把得分贴到候选项上，并按相关性排序。

        参数：
            candidates: 原始的候选列表。
            scores: 模型给出的相关性得分。
            top_k: 要返回前多少个候选项。

        返回：
            排好序的前 top_k 个候选项，带 'rerank_score' 字段。
        """
        # 把得分贴到候选项上
        scored_candidates = []
        for candidate, score in zip(candidates, scores):
            # 复制一份，避免改到原始对象
            candidate_copy = candidate.copy()
            candidate_copy["rerank_score"] = float(score)
            scored_candidates.append(candidate_copy)

        # 按得分从高到低排序，取前 top_k 个
        sorted_candidates = sorted(
            scored_candidates,
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return sorted_candidates[:top_k]

    def _log_trace(
        self,
        trace: Any,
        query: str,
        input_count: int,
        output_count: int
    ) -> None:
        """把重排操作记录到 trace 上下文。

        参数：
            trace: TraceContext 实例。
            query: 查询字符串。
            input_count: 输入的候选项数量。
            output_count: 输出的候选项数量。
        """
        # Stage F 接入的占位代码
        # 未来：trace.log_rerank_step(...)
        logger.debug(
            f"Cross-Encoder rerank: query='{query[:50]}...', "
            f"input={input_count}, output={output_count}"
        )
