"""用于创建 LLM 提供方实例的工厂。

本模块实现工厂模式，根据配置实例化对应的 LLM 提供方，
从而通过配置即可切换不同后端，无需改动代码。

同时支持仅处理文本的 LLM 和多模态 Vision LLM。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.libs.llm.base_llm import BaseLLM
from src.libs.llm.base_vision_llm import BaseVisionLLM

if TYPE_CHECKING:
    from src.core.settings import Settings


# 模块加载时导入并注册 Vision LLM 提供方
def _register_vision_providers() -> None:
    """注册所有 Vision LLM 提供方实现。

    本函数在模块导入时被调用，用于填充 Vision LLM 提供方注册表。
    新增的提供方在实现后在此处加入即可。
    """
    try:
        from src.libs.llm.azure_vision_llm import AzureVisionLLM
        from src.libs.llm.llm_factory import LLMFactory
        LLMFactory.register_vision_provider("azure", AzureVisionLLM)
    except ImportError:
        # 提供方尚未实现，跳过注册
        pass

    try:
        from src.libs.llm.openai_vision_llm import OpenAIVisionLLM
        from src.libs.llm.llm_factory import LLMFactory
        LLMFactory.register_vision_provider("openai", OpenAIVisionLLM)
    except ImportError:
        pass


class LLMFactory:
    """用于创建 LLM 提供方实例的工厂。

    本工厂从 settings 读取提供方配置，并实例化对应的 LLM 实现。
    同时支持仅处理文本的 LLM 和多模态 Vision LLM。

    应用的设计原则：
    - 工厂模式（Factory Pattern）：集中对象创建逻辑。
    - 配置驱动（Config-Driven）：根据 settings.yaml 选择提供方。
    - 快速失败（Fail-Fast）：对未知提供方抛出清晰错误。
    - 分离（Separation）：文本 LLM 与 Vision LLM 注册表相互独立。
    """

    # 已支持的仅文本 LLM 提供方注册表（在 B7.x 任务中填充）
    _PROVIDERS: dict[str, type[BaseLLM]] = {}

    # 已支持的 Vision LLM 提供方注册表（在 B9+ 任务中填充）
    _VISION_PROVIDERS: dict[str, type[BaseVisionLLM]] = {}

    @classmethod
    def register_provider(cls, name: str, provider_class: type[BaseLLM]) -> None:
        """注册一个新的 LLM 提供方实现。

        本方法允许各提供方实现将自己注册到工厂中，便于扩展。

        Args:
            name: 提供方标识符（例如 'openai'、'azure'、'ollama'）。
            provider_class: 实现该提供方的 BaseLLM 子类。

        Raises:
            ValueError: 当 provider_class 没有继承 BaseLLM 时。
        """
        if not issubclass(provider_class, BaseLLM):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit from BaseLLM"
            )
        cls._PROVIDERS[name.lower()] = provider_class
    
    @classmethod
    def create(cls, settings: Settings, **override_kwargs: Any) -> BaseLLM:
        """根据配置创建一个 LLM 实例。

        Args:
            settings: 应用配置，包含 LLM 相关配置。
            **override_kwargs: 可选参数，用于覆盖配置值。

        Returns:
            所配置的 LLM 提供方实例。

        Raises:
            ValueError: 当所配置的提供方不被支持时。
            AttributeError: 当缺少必需的配置字段时。

        Example:
            >>> settings = Settings.load('config/settings.yaml')
            >>> llm = LLMFactory.create(settings)
            >>> response = llm.chat([Message(role='user', content='Hello')])
        """
        # 从 settings 中提取提供方名称
        try:
            provider_name = settings.llm.provider.lower()
        except AttributeError as e:
            raise ValueError(
                "Missing required configuration: settings.llm.provider. "
                "Please ensure 'llm.provider' is specified in settings.yaml"
            ) from e

        # 在注册表中查找提供方类
        provider_class = cls._PROVIDERS.get(provider_name)
        
        if provider_class is None:
            available = ", ".join(sorted(cls._PROVIDERS.keys())) if cls._PROVIDERS else "none"
            raise ValueError(
                f"Unsupported LLM provider: '{provider_name}'. "
                f"Available providers: {available}. "
                f"Provider implementations will be added in tasks B7.1-B7.2."
            )
        
        # 实例化提供方
        # 提供方类应接受 settings 以及可选的 kwargs
        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate LLM provider '{provider_name}': {e}"
            ) from e

    @classmethod
    def list_providers(cls) -> list[str]:
        """列出所有已注册的提供方名称。

        Returns:
            按字母排序的可用提供方标识符列表。
        """
        return sorted(cls._PROVIDERS.keys())

    @classmethod
    def register_vision_provider(
        cls,
        name: str,
        provider_class: type[BaseVisionLLM]
    ) -> None:
        """注册一个新的 Vision LLM 提供方实现。

        本方法允许各 Vision LLM 提供方实现将自己注册到工厂中，便于扩展。

        Args:
            name: 提供方标识符（例如 'azure'、'ollama'）。
            provider_class: 实现该提供方的 BaseVisionLLM 子类。

        Raises:
            ValueError: 当 provider_class 没有继承 BaseVisionLLM 时。
        """
        if not issubclass(provider_class, BaseVisionLLM):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit from BaseVisionLLM"
            )
        cls._VISION_PROVIDERS[name.lower()] = provider_class
    
    @classmethod
    def create_vision_llm(
        cls,
        settings: Settings,
        **override_kwargs: Any
    ) -> BaseVisionLLM:
        """根据配置创建一个 Vision LLM 实例。

        Vision LLM 支持多模态输入（文本 + 图片），可用于图像描述、
        视觉问答、含内嵌图片的文档理解等任务。

        Args:
            settings: 应用配置，包含 Vision LLM 相关配置。
            **override_kwargs: 可选参数，用于覆盖配置值。

        Returns:
            所配置的 Vision LLM 提供方实例。

        Raises:
            ValueError: 当所配置的提供方不被支持，或缺少配置时。
            RuntimeError: 当提供方实例化失败时。

        Example:
            >>> settings = Settings.load('config/settings.yaml')
            >>> vision_llm = LLMFactory.create_vision_llm(settings)
            >>> image = ImageInput(path="diagram.png")
            >>> response = vision_llm.chat_with_image("Describe this", image)
        """
        # 从 settings 中提取提供方名称
        # Vision LLM 配置可能放在 settings.vision_llm，也可能放在 settings.llm 下
        try:
            # 先尝试 vision_llm 段
            if hasattr(settings, 'vision_llm') and hasattr(settings.vision_llm, 'provider'):
                provider_name = settings.vision_llm.provider.lower()
            # 回退到 llm.provider（部分提供方同时支持文本和视觉）
            elif hasattr(settings, 'llm') and hasattr(settings.llm, 'provider'):
                provider_name = settings.llm.provider.lower()
            else:
                raise AttributeError("No vision_llm or llm provider configuration found")
        except AttributeError as e:
            raise ValueError(
                "Missing required configuration: settings.vision_llm.provider or settings.llm.provider. "
                "Please ensure 'vision_llm.provider' or 'llm.provider' is specified in settings.yaml"
            ) from e

        # 在 vision 注册表中查找提供方类
        provider_class = cls._VISION_PROVIDERS.get(provider_name)
        
        if provider_class is None:
            available = ", ".join(sorted(cls._VISION_PROVIDERS.keys())) if cls._VISION_PROVIDERS else "none"
            raise ValueError(
                f"Unsupported Vision LLM provider: '{provider_name}'. "
                f"Available Vision LLM providers: {available}. "
                f"Vision LLM implementations will be added in tasks B9+."
            )
        
        # 实例化提供方
        try:
            return provider_class(settings=settings, **override_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate Vision LLM provider '{provider_name}': {e}"
            ) from e

    @classmethod
    def list_vision_providers(cls) -> list[str]:
        """列出所有已注册的 Vision LLM 提供方名称。

        Returns:
            按字母排序的可用 Vision LLM 提供方标识符列表。
        """
        return sorted(cls._VISION_PROVIDERS.keys())


# 模块加载时注册 Vision LLM 提供方
_register_vision_providers()
