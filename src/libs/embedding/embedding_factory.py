"""用于创建 Embedding 提供方实例的工厂。

本模块实现工厂模式，根据配置实例化对应的 Embedding 提供方，
从而通过配置即可切换不同后端，无需改动代码。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.libs.embedding.base_embedding import BaseEmbedding

if TYPE_CHECKING:
    from src.core.settings import Settings


class EmbeddingFactory:
    """用于创建 Embedding 提供方实例的工厂。

    本工厂从 settings 读取提供方配置，并实例化对应的 Embedding 实现。
    已支持的提供方：OpenAI、Azure，后续任务还会继续增加。

    应用的设计原则：
    - 工厂模式（Factory Pattern）：集中对象创建逻辑。
    - 配置驱动（Config-Driven）：根据 settings.yaml 选择提供方。
    - 快速失败（Fail-Fast）：对未知提供方抛出清晰错误。
    """

    # 已支持提供方的注册表
    _PROVIDERS: dict[str, type[BaseEmbedding]] = {}

    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseEmbedding]) -> None:
        """注册一个新的 Embedding 提供方实现。

        本方法允许各提供方实现将自己注册到工厂中，便于扩展。

        Args:
            name: 提供方标识符（例如 'openai'、'azure'、'local'）。
            provider_class: 实现该提供方的 BaseEmbedding 子类。

        Raises:
            ValueError: 当 provider_class 没有继承 BaseEmbedding 时。
        """
        if not issubclass(provider_class, BaseEmbedding):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit from BaseEmbedding"
            )
        cls._PROVIDERS[name.lower()] = provider_class
    
    @classmethod
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseEmbedding:
        """根据配置创建一个 Embedding 实例。

        Args:
            settings: 应用配置，包含 Embedding 相关配置。
            **override_kwargs: 可选参数，用于覆盖配置值。

        Returns:
            所配置的 Embedding 提供方实例。

        Raises:
            ValueError: 当所配置的提供方不被支持时。
            AttributeError: 当缺少必需的配置字段时。

        Example:
            >>> settings = Settings.load('config/settings.yaml')
            >>> embedding = EmbeddingFactory.create(settings)
            >>> vectors = embedding.embed(["hello world", "test"])
        """
        # 从 settings 中提取提供方名称
        try:
            provider_name = settings.embedding.provider.lower()
        except AttributeError as e:
            raise ValueError(
                "Missing required configuration: settings.embedding.provider. "
                "Please ensure 'embedding.provider' is specified in settings.yaml"
            ) from e

        # 在注册表中查找提供方类
        provider_class = cls._PROVIDERS.get(provider_name)
        
        if provider_class is None:
            available = ", ".join(sorted(cls._PROVIDERS.keys())) if cls._PROVIDERS else "none"
            raise ValueError(
                f"Unsupported Embedding provider: '{provider_name}'. "
                f"Available providers: {available}"
            )
        
        # 实例化提供方
        # 提供方类应接受 settings 以及可选的 kwargs
        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate Embedding provider '{provider_name}': {e}"
            ) from e

    @classmethod
    def list_providers(cls) -> list[str]:
        """列出所有已注册的提供方名称。

        Returns:
            按字母排序的可用提供方标识符列表。
        """
        return sorted(cls._PROVIDERS.keys())


# 模块导入时自动注册提供方
def _register_builtin_providers() -> None:
    """将内置的 Embedding 提供方注册到工厂。"""
    try:
        from src.libs.embedding.openai_embedding import OpenAIEmbedding
        EmbeddingFactory.register_provider("openai", OpenAIEmbedding)
    except ImportError:
        pass  # OpenAI 提供方不可用

    try:
        from src.libs.embedding.azure_embedding import AzureEmbedding
        EmbeddingFactory.register_provider("azure", AzureEmbedding)
    except ImportError:
        pass  # Azure 提供方不可用

    try:
        from src.libs.embedding.ollama_embedding import OllamaEmbedding
        EmbeddingFactory.register_provider("ollama", OllamaEmbedding)
    except ImportError:
        pass  # Ollama 提供方不可用


# 模块被导入时注册提供方
_register_builtin_providers()
