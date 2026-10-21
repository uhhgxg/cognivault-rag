"""创建评估器（Evaluator）提供者实例的工厂。

本模块实现了工厂模式，根据配置来创建对应的评估器实现。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.libs.evaluator.base_evaluator import BaseEvaluator, NoneEvaluator
from src.libs.evaluator.custom_evaluator import CustomEvaluator

if TYPE_CHECKING:
    from src.core.settings import Settings


def _get_ragas_evaluator() -> type[BaseEvaluator]:
    """延迟导入 RagasEvaluator，避免对 ragas 产生硬依赖。"""
    from src.observability.evaluation.ragas_evaluator import RagasEvaluator
    return RagasEvaluator


def _get_composite_evaluator() -> type[BaseEvaluator]:
    """延迟导入 CompositeEvaluator。"""
    from src.observability.evaluation.composite_evaluator import CompositeEvaluator
    return CompositeEvaluator


class EvaluatorFactory:
    """创建评估器提供者实例的工厂。

    应用的设计原则：
    - 工厂模式：把对象创建的逻辑集中在一处管理。
    - 配置驱动：根据 settings.yaml 来选择提供者。
    - 兜底：禁用评估时返回 NoneEvaluator。
    - 快速失败：遇到未知的提供者时，直接抛出清晰的错误。
    """

    _PROVIDERS: dict[str, type[BaseEvaluator]] = {
        "custom": CustomEvaluator,
    }

    # 延迟加载的提供者（按需导入，避免硬依赖）
    _LAZY_PROVIDERS: dict[str, Any] = {
        "ragas": _get_ragas_evaluator,
        "composite": _get_composite_evaluator,
    }

    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseEvaluator]) -> None:
        """注册一个新的评估器提供者实现。

        参数：
            name: 提供者的标识名（例如 'ragas'、'custom'）。
            provider_class: 实现该提供者的 BaseEvaluator 子类。

        抛出：
            ValueError: 如果 provider_class 不是 BaseEvaluator 的子类。
        """
        if not issubclass(provider_class, BaseEvaluator):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit from BaseEvaluator"
            )
        cls._PROVIDERS[name.lower()] = provider_class

    @classmethod
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseEvaluator:
        """根据配置创建一个评估器实例。

        参数：
            settings: 应用配置，里面包含评估相关配置。
            **override_kwargs: 可选参数，用来覆盖配置里的默认值。

        返回：
            配置好的评估器提供者实例。

        抛出：
            ValueError: 如果配置的提供者不支持或缺失。
            RuntimeError: 如果提供者初始化失败。
        """
        try:
            # 既接受完整的 Settings（带 .evaluation 属性），
            # 也接受单独的 EvaluationSettings 对象
            if hasattr(settings, "evaluation"):
                evaluation_settings = settings.evaluation
            elif hasattr(settings, "provider") and hasattr(settings, "enabled"):
                evaluation_settings = settings
            else:
                raise AttributeError("settings has no 'evaluation' attribute")
            if evaluation_settings is None:
                raise AttributeError("settings.evaluation is None")
            provider_name = evaluation_settings.provider.lower()
            enabled = bool(evaluation_settings.enabled)
        except AttributeError as e:
            raise ValueError(
                "Missing required configuration: settings.evaluation.provider. "
                "Please ensure 'evaluation.provider' is specified in settings.yaml"
            ) from e

        if not enabled or provider_name in {"none", "disabled"}:
            return NoneEvaluator(settings=settings, **override_kwargs)

        provider_class = cls._PROVIDERS.get(provider_name)
        if provider_class is None and provider_name in cls._LAZY_PROVIDERS:
            try:
                provider_class = cls._LAZY_PROVIDERS[provider_name]()
                cls._PROVIDERS[provider_name] = provider_class  # 缓存起来，下次直接用
            except ImportError as e:
                raise ValueError(
                    f"Provider '{provider_name}' requires additional dependencies: {e}"
                ) from e
        if provider_class is None:
            all_providers = sorted(set(cls._PROVIDERS.keys()) | set(cls._LAZY_PROVIDERS.keys()))
            available = ", ".join(all_providers) if all_providers else "none"
            raise ValueError(
                f"Unsupported Evaluator provider: '{provider_name}'. "
                f"Available providers: {available}."
            )

        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate Evaluator provider '{provider_name}': {e}"
            ) from e

    @classmethod
    def list_providers(cls) -> list[str]:
        """列出所有已注册的提供者名字。

        返回：
            已排序的可用提供者标识列表。
        """
        return sorted(cls._PROVIDERS.keys())