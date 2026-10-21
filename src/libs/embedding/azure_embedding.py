"""Azure OpenAI Embedding 实现。

本模块提供 Azure OpenAI Embedding 实现，
负责处理 Azure 专属配置（endpoint、api-version、deployment 名称），
同时复用 OpenAI embedding 的核心逻辑。
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

from src.libs.embedding.base_embedding import BaseEmbedding


class AzureEmbeddingError(RuntimeError):
    """当 Azure OpenAI Embeddings API 调用失败时抛出。"""


class AzureEmbedding(BaseEmbedding):
    """Azure OpenAI Embedding 提供方实现。

    本类为 Azure OpenAI 的 Embeddings API 实现 BaseEmbedding 接口。
    负责处理 endpoint、api-version、deployment 名称等 Azure 专属配置。

    Attributes:
        api_key: 用于身份认证的 Azure API 密钥。
        azure_endpoint: Azure OpenAI 的端点 URL。
        api_version: 使用的 API 版本。
        deployment_name: 部署名称（在标准 OpenAI 中替代 'model'）。
        dimensions: 可选的维度缩减（仅 text-embedding-3-* 系列支持）。

    Example:
        >>> from src.core.settings import load_settings
        >>> settings = load_settings('config/settings.yaml')
        >>> embedding = AzureEmbedding(settings)
        >>> vectors = embedding.embed(["hello world", "test"])
    """
    
    DEFAULT_API_VERSION = "2024-02-01"
    
    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """初始化 Azure OpenAI Embedding 提供方。

        Args:
            settings: 应用配置，包含 Embedding 相关配置。
            api_key: 可选的 API 密钥覆盖值（找不到时回退到环境变量 AZURE_OPENAI_API_KEY）。
            azure_endpoint: 可选的端点覆盖值（找不到时回退到环境变量 AZURE_OPENAI_ENDPOINT）。
            api_version: 可选的 API 版本覆盖值。
            **kwargs: 其他配置覆盖项。

        Raises:
            ValueError: 当缺少 Azure 专属的必需配置时。
        """
        # Azure 用 'deployment_name' 代替 'model'
        # 先尝试 settings.embedding.deployment_name，找不到再回退到 model
        self.deployment_name = (
            getattr(settings.embedding, 'deployment_name', None) or
            settings.embedding.model
        )

        # 提取可选的 dimensions 设置
        self.dimensions = getattr(settings.embedding, 'dimensions', None)

        # API 密钥优先级：显式传参 > settings.yaml > 环境变量（为向后兼容而设的回退）
        self.api_key = (
            api_key or
            getattr(settings.embedding, 'api_key', None) or
            os.environ.get("AZURE_OPENAI_API_KEY") or
            os.environ.get("OPENAI_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "Azure OpenAI API key not provided. Configure 'api_key' in settings.yaml, "
                "set AZURE_OPENAI_API_KEY environment variable, or pass api_key parameter."
            )
        
        # Azure 端点优先级：显式传参 > settings.yaml > 环境变量（回退）
        self.azure_endpoint = (
            azure_endpoint or
            getattr(settings.embedding, 'azure_endpoint', None) or
            os.environ.get("AZURE_OPENAI_ENDPOINT")
        )
        if not self.azure_endpoint:
            raise ValueError(
                "Azure OpenAI endpoint not provided. Configure 'azure_endpoint' in settings.yaml, "
                "set AZURE_OPENAI_ENDPOINT environment variable, or pass azure_endpoint parameter."
            )

        # API 版本优先级：显式传参 > settings > 默认值
        self.api_version = (
            api_version or
            getattr(settings.embedding, 'api_version', None) or
            self.DEFAULT_API_VERSION
        )

        # 保存额外的 kwargs，留作日后使用
        self._extra_config = kwargs
    
    def embed(
        self,
        texts: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """使用 Azure OpenAI API 为一批文本生成 embedding（向量）。

        Args:
            texts: 要向量化的文本字符串列表，不能为空。
            trace: 可选的 TraceContext，用于可观测性（为 Stage F 预留）。
            **kwargs: 覆盖参数（dimensions 等）。

        Returns:
            embedding 向量列表，每个向量是一个浮点数列表。
            外层列表长度等于 len(texts)。

        Raises:
            ValueError: 当 texts 为空或包含无效条目时。
            AzureEmbeddingError: 当 API 调用失败时。
        """
        # 校验输入
        self.validate_texts(texts)

        # 导入 Azure OpenAI 客户端（延迟导入，避免在模块层就引入依赖）
        try:
            from openai import AzureOpenAI
        except ImportError as e:
            raise RuntimeError(
                "OpenAI Python package not installed. "
                "Install with: pip install openai"
            ) from e

        # 初始化 Azure OpenAI 客户端
        client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version,
        )

        # 准备 API 调用参数
        # Azure 使用 'model' 参数，但实际期望的是 deployment 名称
        api_params = {
            "input": texts,
            "model": self.deployment_name,
        }

        # 仅 text-embedding-3-* 系列才加 dimensions 参数
        # text-embedding-ada-002 不支持 dimensions 参数
        dimensions = kwargs.get("dimensions", self.dimensions)
        if dimensions is not None and "text-embedding-3" in self.deployment_name.lower():
            api_params["dimensions"] = dimensions

        # 调用 Azure OpenAI API
        try:
            response = client.embeddings.create(**api_params)
        except Exception as e:
            raise AzureEmbeddingError(
                f"Azure OpenAI Embeddings API call failed: {e}"
            ) from e

        # 从响应中提取 embedding
        # 响应格式与 OpenAI 完全一致
        try:
            embeddings = [item.embedding for item in response.data]
        except (AttributeError, KeyError) as e:
            raise AzureEmbeddingError(
                f"Failed to parse Azure OpenAI Embeddings API response: {e}"
            ) from e

        # 校验输出长度与输入一致
        if len(embeddings) != len(texts):
            raise AzureEmbeddingError(
                f"Output length mismatch: expected {len(texts)}, got {len(embeddings)}"
            )

        return embeddings

    def get_dimension(self) -> Optional[int]:
        """获取当前所配置部署的 embedding 维度。

        Returns:
            embedding 维度；若无法确定则返回 None。

        Note:
            对于配置了自定义维度的 text-embedding-3-* 部署，
            返回配置的维度；其它部署返回其默认维度。
        """
        # 若显式配置了 dimensions，则直接返回
        if self.dimensions is not None:
            return self.dimensions

        # Azure 部署的常见默认维度
        # 注意：部署名是用户自定义的，但通常遵循这些命名模式
        deployment_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

        # 先尝试精确匹配
        if self.deployment_name in deployment_dimensions:
            return deployment_dimensions[self.deployment_name]

        # 再尝试部分匹配（例如 "my-embedding-3-large-prod" 包含 "embedding-3-large"）
        # 先匹配较长的模式，避免误匹配
        for model_key in sorted(deployment_dimensions.keys(), key=len, reverse=True):
            if model_key in self.deployment_name:
                return deployment_dimensions[model_key]

        # 无法确定维度
        return None
