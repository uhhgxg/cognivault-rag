"""创建切分器（Splitter）实例的工厂。

本模块实现了工厂模式，根据配置来创建对应的切分器实现，
这样就能通过改配置来切换不同的切分策略，而不用改代码。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.libs.splitter.base_splitter import BaseSplitter

if TYPE_CHECKING:
    from src.core.settings import Settings


def _register_builtin_providers() -> None:
    """注册内置的切分器提供者。

    本函数在模块被导入时会自动调用，
    把所有可用的切分器实现注册到工厂中。
    """
    # 在这里导入是为了避免循环导入，同时优雅地处理依赖缺失的情况
    try:
        from src.libs.splitter.recursive_splitter import RecursiveSplitter
        SplitterFactory.register_provider("recursive", RecursiveSplitter)
    except ImportError:
        pass  # RecursiveSplitter 不可用（缺少 langchain 依赖）


class SplitterFactory:
    """创建切分器提供者实例的工厂。

    本工厂从配置（settings）中读取切分器配置，然后创建对应的切分器实现。
    支持的提供者会在后续任务（B7.5）中陆续添加。

    应用的设计原则：
    - 工厂模式：把对象创建的逻辑集中在一处管理。
    - 配置驱动：根据 settings.yaml 来选择提供者。
    - 快速失败：遇到未知的提供者时，直接抛出清晰的错误。
    """
    
    _PROVIDERS: dict[str, type[BaseSplitter]] = {}
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseSplitter]) -> None:
        """注册一个新的切分器提供者实现。

        参数：
            name: 提供者的标识名（例如 'recursive'、'semantic'、'fixed'）。
            provider_class: 实现该提供者的 BaseSplitter 子类。

        抛出：
            ValueError: 如果 provider_class 不是 BaseSplitter 的子类。
        """
        if not issubclass(provider_class, BaseSplitter):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit from BaseSplitter"
            )
        cls._PROVIDERS[name.lower()] = provider_class
    
    @classmethod
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseSplitter:
        """根据配置创建一个切分器实例。

        参数：
            settings: 应用配置，里面包含文档摄入（ingestion）相关配置。
            **override_kwargs: 可选参数，用来覆盖配置里的默认值。

        返回：
            配置好的切分器提供者实例。

        抛出：
            ValueError: 如果配置的提供者不支持或者缺失。
        """
        try:
            ingestion_settings = settings.ingestion
            if ingestion_settings is None:
                raise AttributeError("settings.ingestion is None")
            provider_name = ingestion_settings.splitter.lower()
        except AttributeError as e:
            raise ValueError(
                "Missing required configuration: settings.ingestion.splitter. "
                "Please ensure 'ingestion.splitter' is specified in settings.yaml"
            ) from e
        
        provider_class = cls._PROVIDERS.get(provider_name)
        if provider_class is None:
            available = ", ".join(sorted(cls._PROVIDERS.keys())) if cls._PROVIDERS else "none"
            raise ValueError(
                f"Unsupported Splitter provider: '{provider_name}'. "
                f"Available providers: {available}. "
                "Provider implementations will be added in task B7.5."
            )
        
        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate Splitter provider '{provider_name}': {e}"
            ) from e
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """列出所有已注册的提供者名字。

        返回：
            已排序的可用提供者标识列表。
        """
        return sorted(cls._PROVIDERS.keys())


# 模块被导入时自动注册内置提供者
_register_builtin_providers()