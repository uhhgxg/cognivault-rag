"""Vision LLM 服务提供方的抽象基类（父类）。

本模块定义了 Vision Language Model 服务提供方的可插拔接口，
支持多模态交互（文本 + 图片），通过配置即可在不同后端
（Azure Vision、Ollama Vision 等）之间无缝切换。

Vision LLM 在标准 LLM 基础上扩展，可同时接收图片输入和文本提示，
从而支持图像描述（caption）、视觉问答、含内嵌图片的文档理解等任务。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from src.libs.llm.base_llm import ChatResponse, Message


@dataclass
class ImageInput:
    """表示 Vision LLM 的图片输入。

    支持多种输入格式：
    - 文件路径：本地图片文件，会被读取并编码
    - 字节：原始图片字节（已加载）
    - Base64：已编码好的图片字符串

    Attributes:
        path: 图片文件路径（从磁盘加载时使用）。
        data: 原始图片字节（已加载时使用）。
        base64: Base64 编码的图片字符串（已编码时使用）。
        mime_type: 图片的 MIME 类型（例如 'image/png'、'image/jpeg'）。
    """
    path: Optional[Union[str, Path]] = None
    data: Optional[bytes] = None
    base64: Optional[str] = None
    mime_type: str = "image/png"

    def __post_init__(self) -> None:
        """校验是否恰好提供了一种输入格式。"""
        provided_inputs = sum([
            self.path is not None,
            self.data is not None,
            self.base64 is not None,
        ])
        if provided_inputs == 0:
            raise ValueError("Must provide one of: path, data, or base64")
        if provided_inputs > 1:
            raise ValueError("Must provide exactly one of: path, data, or base64")


class BaseVisionLLM(ABC):
    """Vision LLM 服务提供方的抽象基类（父类）。

    Vision LLM 可同时接收文本和图片输入，从而支持图像描述、
    视觉问答、含内嵌图片的文档分析等多模态理解任务。

    所有 Vision LLM 实现都必须继承本类，并实现 chat_with_image() 方法。
    这样可以保证不同提供方（Azure Vision、Ollama Vision 等）的接口一致。

    应用的设计原则：
    - 可插拔（Pluggable）：子类可以替换，而不用改动上层代码。
    - 可观测（Observable）：接受可选的 TraceContext，用于接入可观测性。
    - 配置驱动（Config-Driven）：通过工厂根据配置创建实例。
    - 接口隔离（Interface Segregation）：仅聚焦多模态输入的最小接口。
    - 扩展点（Extension Point）：图片预处理（压缩、格式转换）可在子类中
      添加，无需改动基类接口。
    """
    
    @abstractmethod
    def chat_with_image(
        self,
        text: str,
        image: ImageInput,
        messages: Optional[list[Message]] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """根据文本提示和图片输入生成响应。

        本方法支持多模态交互，模型可以"看"图片，
        并针对图片回答问题或生成描述。

        Args:
            text: 关于图片的文本提示或问题。
            image: 图片输入（路径、字节或 base64）。
            messages: 可选的对话历史，用于提供上下文。若提供，
                文本 + 图片会作为最新的一条 user 消息追加。
            trace: 可选的 TraceContext，用于可观测性（为 Stage F 预留）。
            **kwargs: 各提供方专属参数（temperature、max_tokens 等）。

        Returns:
            包含生成文本和元数据的 ChatResponse。

        Raises:
            ValueError: 当 text 为空或图片输入无效时。
            RuntimeError: 当 Vision LLM 提供方调用失败时。

        Example:
            >>> image = ImageInput(path="diagram.png")
            >>> response = vision_llm.chat_with_image(
            ...     text="Describe this diagram",
            ...     image=image
            ... )
            >>> print(response.content)
            "This diagram shows a system architecture with..."
        """
        pass

    def validate_text(self, text: str) -> None:
        """校验文本提示。

        Args:
            text: 待校验的文本提示。

        Raises:
            ValueError: 当 text 为空或不是字符串时。
        """
        if not isinstance(text, str):
            raise ValueError(f"Text must be a string, got {type(text).__name__}")
        if not text or not text.strip():
            raise ValueError("Text prompt cannot be empty")

    def validate_image(self, image: ImageInput) -> None:
        """校验图片输入。

        Args:
            image: 待校验的图片输入。

        Raises:
            ValueError: 当 image 不是 ImageInput 实例时。
        """
        if not isinstance(image, ImageInput):
            raise ValueError(
                f"Image must be an ImageInput instance, got {type(image).__name__}"
            )

    def preprocess_image(
        self,
        image: ImageInput,
        max_size: Optional[tuple[int, int]] = None,
    ) -> ImageInput:
        """在发送给 Vision LLM 之前对图片做预处理。

        本方法提供图片预处理的扩展点，例如：
        - 调整尺寸以满足提供方的尺寸限制
        - 格式转换（例如 PNG 转 JPEG）
        - 压缩以减小负载大小

        默认实现直接返回原图，不做任何处理。子类可以重写，
        添加提供方专属的预处理逻辑。

        Args:
            image: 待预处理的输入图片。
            max_size: 可选的最大尺寸（宽、高），单位为像素。

        Returns:
            预处理后的 ImageInput（若无需改动，可能是同一实例）。

        Note:
            预处理应当是幂等的——对同一输入多次调用，结果应保持一致。
        """
        # 默认：不做预处理
        return image
