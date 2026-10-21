"""Ollama Embedding implementation for local embedding models.

This module provides the Ollama Embedding implementation that works with
locally running Ollama instances. Ollama enables running embedding models like
nomic-embed-text, mxbai-embed-large, etc. on local hardware.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from src.libs.embedding.base_embedding import BaseEmbedding


class OllamaEmbeddingError(RuntimeError):
    """当 Ollama Embeddings API 调用失败时抛出。

    本异常提供清晰的错误信息，同时不暴露
    内部 URL 等敏感配置细节。
    """


class OllamaEmbedding(BaseEmbedding):
    """用于本地 Embedding 的 Ollama Embedding 提供方实现。

    本类为 Ollama 的 embeddings API 实现 BaseEmbedding 接口，
    可以在本地生成 Embedding，无需依赖云服务。

    Attributes:
        base_url: Ollama 服务器的基础 URL（默认：http://localhost:11434）。
        model: 使用的模型标识符（例如 'nomic-embed-text'、'mxbai-embed-large'）。
        timeout: 请求超时时间，单位为秒。
        dimension: 本模型产生的 embedding 维度。

    Example:
        >>> from src.core.settings import load_settings
        >>> settings = load_settings('config/settings.yaml')
        >>> embedding = OllamaEmbedding(settings)
        >>> vectors = embedding.embed(["hello world", "test"])
    """

    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_TIMEOUT = 120.0  # 本地推理，超时设得长一些
    DEFAULT_DIMENSION = 768  # 本地 embedding 模型的常见维度
    
    def __init__(
        self,
        settings: Any,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """初始化 Ollama Embedding 提供方。

        Args:
            settings: 应用配置，包含 Embedding 相关配置。
            base_url: 可选的基础 URL 覆盖值（找不到时回退到环境变量 OLLAMA_BASE_URL）。
            timeout: 可选的请求超时覆盖值。
            **kwargs: 其他配置覆盖项。

        Raises:
            ValueError: 当缺少必需配置时。
        """
        self.model = settings.embedding.model

        # 基础 URL 优先级：显式传参 > 环境变量 > 默认值
        self.base_url = (
            base_url
            or os.environ.get("OLLAMA_BASE_URL")
            or self.DEFAULT_BASE_URL
        )

        # 超时优先级：显式传参 > 默认值
        self.timeout = timeout or self.DEFAULT_TIMEOUT

        # 维度优先级：settings > 默认值
        self.dimension = getattr(settings.embedding, 'dimensions', self.DEFAULT_DIMENSION)

        # 保存额外的 kwargs，留作日后使用
        self._extra_config = kwargs
    
    def embed(
        self,
        texts: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """使用 Ollama API 为一批文本生成 embedding（向量）。

        Args:
            texts: 要向量化的文本字符串列表，不能为空。
            trace: 可选的 TraceContext，用于可观测性（为 Stage F 预留）。
            **kwargs: 额外参数（当前未使用，为未来预留）。

        Returns:
            embedding 向量列表，每个向量是一个浮点数列表。

        Raises:
            ValueError: 当 texts 为空或包含无效条目时。
            OllamaEmbeddingError: 当 API 调用失败时。

        Example:
            >>> embeddings = embedding.embed(["hello", "world"])
            >>> len(embeddings)  # 2 个向量
            >>> len(embeddings[0])  # 维度（例如 768）
        """
        # 校验输入
        self.validate_texts(texts)

        try:
            import httpx
        except ImportError as e:
            raise OllamaEmbeddingError(
                "httpx library is required for Ollama Embedding. "
                "Install with: pip install httpx"
            ) from e

        # 准备 API 请求
        url = f"{self.base_url}/api/embeddings"

        embeddings: List[List[float]] = []

        # 逐条处理文本（Ollama API 一次只接受一个 prompt）
        for text in texts:
            payload = {
                "model": self.model,
                "prompt": text,
            }

            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(url, json=payload)
                    response.raise_for_status()

                    result = response.json()

                    # 从响应中提取 embedding
                    if "embedding" not in result:
                        raise OllamaEmbeddingError(
                            f"Unexpected response format from Ollama API. "
                            f"Expected 'embedding' field but got: {list(result.keys())}"
                        )

                    embeddings.append(result["embedding"])

            except httpx.HTTPStatusError as e:
                # HTTP 错误（4xx、5xx）
                raise OllamaEmbeddingError(
                    f"Ollama API request failed with status {e.response.status_code}. "
                    f"Ensure Ollama is running and model '{self.model}' is available."
                ) from e
            except httpx.ConnectError as e:
                # 连接错误（服务器不可达）
                raise OllamaEmbeddingError(
                    f"Failed to connect to Ollama server at {self.base_url}. "
                    f"Ensure Ollama is running (try: ollama serve)"
                ) from e
            except httpx.TimeoutException as e:
                # 请求超时
                raise OllamaEmbeddingError(
                    f"Ollama API request timed out after {self.timeout}s. "
                    f"The model may be loading or the request is too large."
                ) from e
            except httpx.RequestError as e:
                # 其他请求错误
                raise OllamaEmbeddingError(
                    f"Ollama API request failed: {str(e)}"
                ) from e
            except (KeyError, ValueError, TypeError) as e:
                # JSON 解析或数据提取错误
                raise OllamaEmbeddingError(
                    f"Failed to parse Ollama API response: {str(e)}"
                ) from e

        return embeddings

    def get_dimension(self) -> int:
        """获取本提供方产生的 embedding 维度。

        Returns:
            当前实例配置的向量维度。

        Note:
            实际维度会因模型而异。常见维度：
            - nomic-embed-text: 768
            - mxbai-embed-large: 1024
            可通过 settings.embedding.dimensions 配置，否则使用默认值 768。
        """
        return self.dimension
