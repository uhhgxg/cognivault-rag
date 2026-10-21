"""OpenAI Embedding 实现。

本模块提供与标准 OpenAI Embeddings API 对接的 OpenAI Embedding 实现。
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

from src.libs.embedding.base_embedding import BaseEmbedding


class OpenAIEmbeddingError(RuntimeError):
    """当 OpenAI Embeddings API 调用失败时抛出。"""


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI Embedding 提供方实现。

    本类为 OpenAI 的 Embeddings API 实现 BaseEmbedding 接口。
    支持 text-embedding-3-small、text-embedding-3-large，
    以及老一些的 text-embedding-ada-002 等模型。

    Attributes:
        api_key: 用于身份认证的 API 密钥。
        model: 使用的模型标识符。
        dimensions: 可选的维度缩减（仅 text-embedding-3-* 系列支持）。
        base_url: API 的基础 URL（默认为 OpenAI 的端点）。

    Example:
        >>> from src.core.settings import load_settings
        >>> settings = load_settings('config/settings.yaml')
        >>> embedding = OpenAIEmbedding(settings)
        >>> vectors = embedding.embed(["hello world", "test"])
    """
    
    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    
    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """初始化 OpenAI Embedding 提供方。

        Args:
            settings: 应用配置，包含 Embedding 相关配置。
            api_key: 可选的 API 密钥覆盖值（找不到时回退到 settings.embedding.api_key 或环境变量）。
            base_url: 可选的基础 URL 覆盖值。
            **kwargs: 其他配置覆盖项。

        Raises:
            ValueError: 当未提供 API 密钥，且环境变量中也找不到时。

        Note:
            当 settings 中存在 azure_endpoint 时，提供方会自动构造
            Azure 兼容的 OpenAI URL，并使用 api-key 认证。
        """
        self.model = settings.embedding.model

        # 提取可选的 dimensions 设置
        self.dimensions = getattr(settings.embedding, 'dimensions', None)

        # API 密钥优先级：显式传参 > settings > 环境变量
        self.api_key = (
            api_key
            or getattr(settings.embedding, 'api_key', None)
            or os.environ.get("OPENAI_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set in settings.yaml (embedding.api_key), "
                "OPENAI_API_KEY environment variable, or pass api_key parameter."
            )
        
        # 检测是否走 Azure 兼容模式
        azure_endpoint = getattr(settings.embedding, 'azure_endpoint', None)
        self.api_version = getattr(settings.embedding, 'api_version', None)
        self._use_azure_auth = False

        if base_url:
            self.base_url = base_url
        elif azure_endpoint:
            # Azure 兼容模式：构造基于 deployment 的 URL
            deployment = getattr(settings.embedding, 'deployment_name', None) or self.model
            self.base_url = f"{azure_endpoint.rstrip('/')}/openai/deployments/{deployment}"
            self._use_azure_auth = True
            if not self.api_version:
                self.api_version = "2024-02-15-preview"
        else:
            settings_base_url = getattr(settings.embedding, 'base_url', None)
            self.base_url = settings_base_url if settings_base_url else self.DEFAULT_BASE_URL

        # 保存额外的 kwargs，留作日后使用
        self._extra_config = kwargs
    
    def embed(
        self,
        texts: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """使用 OpenAI API 为一批文本生成 embedding（向量）。

        Args:
            texts: 要向量化的文本字符串列表，不能为空。
            trace: 可选的 TraceContext，用于可观测性（为 Stage F 预留）。
            **kwargs: 覆盖参数（dimensions 等）。

        Returns:
            embedding 向量列表，每个向量是一个浮点数列表。
            外层列表长度等于 len(texts)。

        Raises:
            ValueError: 当 texts 为空或包含无效条目时。
            OpenAIEmbeddingError: 当 API 调用失败时。
        """
        # 校验输入
        self.validate_texts(texts)

        # 导入 OpenAI 客户端（延迟导入，避免在模块层就引入依赖）
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "OpenAI Python package not installed. "
                "Install with: pip install openai"
            ) from e

        # 初始化 OpenAI 客户端
        client_kwargs = {
            "api_key": self.api_key,
            "base_url": self.base_url,
        }
        # Azure 兼容模式：附加 api-version 查询参数和 api-key 请求头
        if self._use_azure_auth and self.api_version:
            client_kwargs["default_query"] = {"api-version": self.api_version}
            client_kwargs["default_headers"] = {"api-key": self.api_key}

        client = OpenAI(**client_kwargs)

        # 准备 API 调用参数
        api_params = {
            "input": texts,
            "model": self.model,
        }

        # 仅 text-embedding-3-* 系列才加 dimensions 参数
        # text-embedding-ada-002 不支持 dimensions 参数
        dimensions = kwargs.get("dimensions", self.dimensions)
        if dimensions is not None and self.model.startswith("text-embedding-3"):
            api_params["dimensions"] = dimensions

        # 调用 OpenAI API，并带重试
        last_error = None
        for attempt in range(3):
            try:
                response = client.embeddings.create(**api_params)
                break
            except Exception as e:
                last_error = e
                if attempt < 2:
                    import time
                    time.sleep(1.0 * (2 ** attempt))
                continue
        else:
            raise OpenAIEmbeddingError(
                f"OpenAI Embeddings API call failed after 3 retries: {last_error}"
            ) from last_error
        
        # 从响应中提取 embedding
        # 响应格式：response.data 是一个对象列表，每个对象带 .embedding 属性
        try:
            embeddings = [item.embedding for item in response.data]
        except (AttributeError, KeyError) as e:
            raise OpenAIEmbeddingError(
                f"Failed to parse OpenAI Embeddings API response: {e}"
            ) from e

        # 校验输出长度与输入一致
        if len(embeddings) != len(texts):
            raise OpenAIEmbeddingError(
                f"Output length mismatch: expected {len(texts)}, got {len(embeddings)}"
            )
        
        return embeddings
    
    def get_dimension(self) -> Optional[int]:
        """获取当前所配置模型的 embedding 维度。

        Returns:
            embedding 维度；若无法确定则返回 None。

        Note:
            对于配置了自定义维度的 text-embedding-3-* 系列模型，
            返回配置的维度；其它模型返回其默认维度。
        """
        # 若显式配置了 dimensions，则直接返回
        if self.dimensions is not None:
            return self.dimensions

        # 各模型默认维度
        model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        
        return model_dimensions.get(self.model)
