"""创建重排序器（Reranker）提供者实例的工厂。

本模块实现了工厂模式，根据配置来创建对应的重排序器实现，
这样就能通过改配置来切换不同的后端，而不用改代码。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.libs.reranker.base_reranker import BaseReranker, NoneReranker

if TYPE_CHECKING:
    from src.core.settings import Settings


def _lazy_import_llm_reranker():
    """延迟导入，避免循环依赖。"""
    from src.libs.reranker.llm_reranker import LLMReranker
    return LLMReranker


def _lazy_import_cross_encoder_reranker():
    """延迟导入，避免循环依赖。"""
    from src.libs.reranker.cross_encoder_reranker import CrossEncoderReranker
    return CrossEncoderReranker


def _lazy_import_dashscope_reranker():
    """延迟导入，避免循环依赖。"""
    from src.libs.reranker.dashscope_reranker import DashScopeReranker
    return DashScopeReranker


class RerankerFactory:
    """创建重排序器提供者实例的工厂。

    本工厂从 settings 中读取重排配置，然后创建对应的重排序器实现。
    提供者实现会在后续任务（B7.7、B7.8）中陆续添加。

    应用的设计原则：
    - 工厂模式：把对象创建的逻辑集中在一处管理。
    - 配置驱动：根据 settings.yaml 来选择提供者。
    - 兜底：如果禁用或设为 'none'，就返回 NoneReranker。
    - 快速失败：遇到未知的提供者时，直接抛出清晰的错误。
    """
    
    _PROVIDERS: dict[str, type[BaseReranker]] = {}
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseReranker]) -> None:
        """注册一个新的重排序器提供者实现。

        参数：
            name: 提供者的标识名（例如 'cross_encoder'、'llm'）。
            provider_class: 实现该提供者的 BaseReranker 子类。

        抛出：
            ValueError: 如果 provider_class 不是 BaseReranker 的子类。
        """
        if not issubclass(provider_class, BaseReranker):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit from BaseReranker"
            )
        cls._PROVIDERS[name.lower()] = provider_class
    
    @classmethod
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseReranker:
        """根据配置创建一个重排序器实例。

        参数：
            settings: 应用配置，里面包含重排相关配置。
            **override_kwargs: 可选参数，用来覆盖配置里的默认值。

        返回：
            配置好的重排序器提供者实例。

        抛出：
            ValueError: 如果配置的提供者不支持或缺失。
            RuntimeError: 如果提供者初始化失败。
        """
        # 如果还没注册 LLM 重排序器，就延迟注册
        if "llm" not in cls._PROVIDERS:
            LLMReranker = _lazy_import_llm_reranker()
            cls.register_provider("llm", LLMReranker)

        # 如果还没注册 Cross-Encoder 重排序器，就延迟注册
        if "cross_encoder" not in cls._PROVIDERS:
            CrossEncoderReranker = _lazy_import_cross_encoder_reranker()
            cls.register_provider("cross_encoder", CrossEncoderReranker)

        # 如果还没注册 DashScope 重排序器，就延迟注册
        if "dashscope" not in cls._PROVIDERS:
            DashScopeReranker = _lazy_import_dashscope_reranker()
            cls.register_provider("dashscope", DashScopeReranker)
        
        try:
            rerank_settings = settings.rerank
            if rerank_settings is None:
                raise AttributeError("settings.rerank is None")
            provider_name = rerank_settings.provider.lower()
            enabled = bool(rerank_settings.enabled)
        except AttributeError as e:
            raise ValueError(
                "Missing required configuration: settings.rerank.provider. "
                "Please ensure 'rerank.provider' is specified in settings.yaml"
            ) from e
        
        if not enabled or provider_name == "none":
            return NoneReranker(settings=settings, **override_kwargs)
        
        provider_class = cls._PROVIDERS.get(provider_name)
        if provider_class is None:
            available = ", ".join(sorted(cls._PROVIDERS.keys())) if cls._PROVIDERS else "none"
            raise ValueError(
                f"Unsupported Reranker provider: '{provider_name}'. "
                f"Available providers: {available}."
            )
        
        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate Reranker provider '{provider_name}': {e}"
            ) from e
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """列出所有已注册的提供者名字。

        返回：
            已排序的可用提供者标识列表。
        """
        return sorted(cls._PROVIDERS.keys())
