"""创建向量存储（VectorStore）提供者实例的工厂。

本模块实现了工厂模式，根据配置来创建对应的向量存储实现，
这样就能通过改配置来切换不同的后端，而不用改代码。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.libs.vector_store.base_vector_store import BaseVectorStore

if TYPE_CHECKING:
    from src.core.settings import Settings


class VectorStoreFactory:
    """创建向量存储提供者实例的工厂。

    本工厂从 settings 读取提供者配置，然后创建对应的向量存储实现。
    支持的提供者会在后续任务（B7.6 及之后）中陆续添加。

    应用的设计原则：
    - 工厂模式：把对象创建的逻辑集中在一处管理。
    - 配置驱动：根据 settings.yaml 来选择提供者。
    - 快速失败：遇到未知的提供者时，直接抛出清晰的错误。
    """

    # 已支持的提供者注册表（在 B7.x 任务中填充）
    _PROVIDERS: dict[str, type[BaseVectorStore]] = {}

    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseVectorStore]) -> None:
        """注册一个新的向量存储提供者实现。

        本方法允许提供者实现把自己注册到工厂中，从而支持扩展。

        参数：
            name: 提供者的标识名（例如 'chroma'、'qdrant'、'milvus'）。
            provider_class: 实现该提供者的 BaseVectorStore 子类。

        抛出：
            ValueError: 如果 provider_class 不是 BaseVectorStore 的子类。
        """
        if not issubclass(provider_class, BaseVectorStore):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit from BaseVectorStore"
            )
        cls._PROVIDERS[name.lower()] = provider_class
    
    @classmethod
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseVectorStore:
        """根据配置创建一个向量存储实例。

        参数：
            settings: 应用配置，里面包含向量存储相关配置。
            **override_kwargs: 可选参数，用来覆盖配置里的默认值。

        返回：
            配置好的向量存储提供者实例。

        抛出：
            ValueError: 如果配置的提供者不支持。
            AttributeError: 如果缺少必要的配置字段。

        示例：
            >>> settings = Settings.load('config/settings.yaml')
            >>> vector_store = VectorStoreFactory.create(settings)
            >>> vector_store.upsert([{'id': 'doc1', 'vector': [0.1, 0.2]}])
        """
        # 从 settings 中取出提供者名字
        try:
            provider_name = settings.vector_store.provider.lower()
        except AttributeError as e:
            raise ValueError(
                "Missing required configuration: settings.vector_store.provider. "
                "Please ensure 'vector_store.provider' is specified in settings.yaml"
            ) from e

        # 在注册表里查找提供者类
        provider_class = cls._PROVIDERS.get(provider_name)

        if provider_class is None:
            available = ", ".join(sorted(cls._PROVIDERS.keys())) if cls._PROVIDERS else "none"
            raise ValueError(
                f"Unsupported VectorStore provider: '{provider_name}'. "
                f"Available providers: {available}. "
                f"Provider implementations will be added in task B7.6 and beyond."
            )

        # 实例化提供者
        # 提供者类应当接受 settings 和可选的 kwargs
        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate VectorStore provider '{provider_name}': {e}"
            ) from e
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """列出所有已注册的提供者名字。

        返回：
            已排序的提供者名字列表。

        示例：
            >>> VectorStoreFactory.list_providers()
            ['chroma', 'milvus', 'qdrant']
        """
        return sorted(cls._PROVIDERS.keys())
