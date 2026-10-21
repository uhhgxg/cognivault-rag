"""基于 DashScope（阿里云百炼）的重排序器实现。

本模块使用 DashScope API 实现重排序，
支持 qwen3-rerank、gte-rerank-v2 等模型。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from src.libs.reranker.base_reranker import BaseReranker

logger = logging.getLogger(__name__)


class DashScopeRerankError(RuntimeError):
    """当 DashScope 重排序失败时抛出。"""


class DashScopeReranker(BaseReranker):
    """基于 DashScope API 的重排序器，用于云端重排。

    使用 DashScope 兼容的 rerank 接口：
        POST https://dashscope.aliyuncs.com/compatible-api/v1/reranks

    支持纯文本重排模型（例如 qwen3-rerank）。
    """

    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"

    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        **kwargs: Any,
    ) -> None:
        """初始化 DashScope 重排序器。

        参数：
            settings: 应用配置。
            api_key: DashScope 的 API key。如果为 None，就从 OPENAI_API_KEY
                或 DASHSCOPE_API_KEY 环境变量读取。
            model: 模型名（例如 "qwen3-rerank"）。如果为 None，就从
                settings.rerank.model 读取。
            base_url: 自定义的 API 接口地址。
            timeout: HTTP 请求超时时间（秒）。
            **kwargs: 其他参数。
        """
        self.settings = settings
        self.timeout = timeout
        self.kwargs = kwargs

        # API key
        self.api_key = api_key or self._resolve_api_key()

        # 模型名
        self.model = model or self._resolve_model_name(settings)

        # 接口地址
        self.base_url = base_url or self.DEFAULT_BASE_URL

    def _resolve_api_key(self) -> str:
        """从环境变量解析 API key。"""
        key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise DashScopeRerankError(
                "DashScope API key not found. Set DASHSCOPE_API_KEY or OPENAI_API_KEY environment variable."
            )
        return key

    def _resolve_model_name(self, settings: Any) -> str:
        """从 settings 中提取模型名。"""
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

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """使用 DashScope API 重排候选项。

        参数：
            query: 用户的查询字符串。
            candidates: 候选记录列表。每条必须包含 'text' 或 'content'。
            trace: 可选的 TraceContext，用于可观测性。
            **kwargs: 额外参数（top_k 等）。

        返回：
            重排后的候选列表，按相关性得分从高到低排序。
            每个候选项会带上 'rerank_score' 字段。

        抛出：
            ValueError: 如果 query 或 candidates 不合法。
            DashScopeRerankError: 如果 API 调用失败。
        """
        self.validate_query(query)
        self.validate_candidates(candidates)

        if len(candidates) == 1:
            return candidates

        top_k = kwargs.get("top_k", len(candidates))

        # 抽取文档文本
        documents = []
        for candidate in candidates:
            text = candidate.get("text") or candidate.get("content", "")
            if not isinstance(text, str):
                text = str(text)
            documents.append(text)

        # 调用 API
        response_data = self._call_api(query, documents, top_k)

        # 把结果映射回候选项
        reranked = self._map_results(response_data, candidates, top_k)

        if trace:
            logger.debug(
                "DashScope rerank: query='%s...', input=%d, output=%d",
                query[:50],
                len(candidates),
                len(reranked),
            )

        return reranked

    def _call_api(self, query: str, documents: List[str], top_n: int) -> Dict[str, Any]:
        """调用 DashScope 的 rerank 接口。

        参数：
            query: 查询字符串。
            documents: 文档文本列表。
            top_n: 要返回前多少条结果。

        返回：
            解析后的 JSON 响应。

        抛出：
            DashScopeRerankError: 如果 API 调用失败。
        """
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(documents)),
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            import httpx
        except ImportError:
            raise DashScopeRerankError(
                "httpx is required for DashScope reranking. "
                "Install it with: pip install httpx"
            )

        # 针对临时性网络错误，用指数退避策略重试
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = httpx.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                break
            except httpx.HTTPStatusError as e:
                raise DashScopeRerankError(
                    f"DashScope API returned {e.response.status_code}: {e.response.text}"
                ) from e
            except (httpx.NetworkError, httpx.RemoteProtocolError) as e:
                if attempt < max_retries - 1:
                    import time
                    wait = 2 ** attempt
                    logger.warning(f"DashScope API transient error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                raise DashScopeRerankError(f"DashScope API request failed after {max_retries} retries: {e}") from e
            except Exception as e:
                raise DashScopeRerankError(f"DashScope API request failed: {e}") from e

        return data

    def _map_results(
        self,
        response_data: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """把 API 返回的结果映射回原始候选项。

        参数：
            response_data: 从 DashScope API 解析出的 JSON。
            candidates: 原始的候选列表。
            top_k: 限制输出数量。

        返回：
            重排后的候选项，带 'rerank_score' 字段。
        """
        results = response_data.get("results", [])
        if not results:
            logger.warning("DashScope rerank returned empty results")
            return []

        reranked = []
        for result in results[:top_k]:
            index = result.get("index")
            score = result.get("relevance_score", 0.0)

            if index is None or not (0 <= index < len(candidates)):
                logger.warning("Invalid index in rerank result: %s", result)
                continue

            candidate_copy = candidates[index].copy()
            candidate_copy["rerank_score"] = float(score)
            reranked.append(candidate_copy)

        # 按得分从高到低排序（API 通常已排序，但保险起见再排一次）
        reranked.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)

        return reranked
