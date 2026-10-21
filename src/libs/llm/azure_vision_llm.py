"""Azure OpenAI Vision LLM 实现。

本模块提供 Azure OpenAI Vision LLM 实现，支持多模态交互（文本 + 图片）。
支持 GPT-4o 和 GPT-4-Vision-Preview 模型，可用于图像描述、
视觉问答、文档分析等图像理解任务。
"""

from __future__ import annotations

import base64
import io
import os
import requests
from pathlib import Path
from typing import Any, Optional

from src.libs.llm.base_llm import ChatResponse, Message
from src.libs.llm.base_vision_llm import BaseVisionLLM, ImageInput


class AzureVisionLLMError(RuntimeError):
    """当 Azure Vision API 调用失败时抛出。"""


class AzureVisionLLM(BaseVisionLLM):
    """Azure OpenAI Vision LLM 提供方实现。

    本类为 Azure 的 OpenAI Vision 服务实现 BaseVisionLLM 接口，
    支持 GPT-4o 和 GPT-4-Vision-Preview 模型。负责处理 Azure 专属的
    身份认证、端点配置和图片预处理。

    应用的设计原则：
    - 可插拔（Pluggable）：实现 BaseVisionLLM，可无缝切换提供方
    - 配置驱动（Config-Driven）：所有配置均来自 settings.yaml
    - 可观测（Observable）：接受 TraceContext 参数（为 Stage F 预留）
    - 友好错误（Graceful Errors）：提供清晰、可操作的错误信息
    - 图片预处理（Image Preprocessing）：超过 max_image_size 的图片自动压缩

    Attributes:
        api_key: 用于身份认证的 Azure API 密钥。
        endpoint: Azure OpenAI 的端点 URL。
        deployment_name: Vision 模型的部署名称。
        api_version: 使用的 API 版本。
        max_image_size: 图片最大边长，单位为像素（默认 2048）。
        default_temperature: 生成时使用的默认温度。
        default_max_tokens: 生成时使用的默认最大 token 数。

    Example:
        >>> from src.core.settings import load_settings
        >>> settings = load_settings('config/settings.yaml')
        >>> vision_llm = AzureVisionLLM(
        ...     settings,
        ...     endpoint='https://my-resource.openai.azure.com',
        ...     deployment_name='gpt-4o'
        ... )
        >>> image = ImageInput(path="diagram.png")
        >>> response = vision_llm.chat_with_image(
        ...     text="Describe this diagram",
        ...     image=image
        ... )
    """

    DEFAULT_API_VERSION = "2024-02-15-preview"
    DEFAULT_MAX_IMAGE_SIZE = 2048  # 像素
    
    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment_name: Optional[str] = None,
        api_version: Optional[str] = None,
        max_image_size: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """初始化 Azure OpenAI Vision LLM 提供方。

        Args:
            settings: 应用配置，包含 LLM 相关配置。
            api_key: 可选的 API 密钥覆盖值（找不到时回退到环境变量）。
            endpoint: 可选的端点覆盖值（找不到时回退到环境变量）。
            deployment_name: 可选的部署名称（默认为 settings.llm.model）。
            api_version: 可选的 API 版本覆盖值。
            max_image_size: 用于自动压缩的图片最大边长，单位为像素。
            **kwargs: 其他配置覆盖项。

        Raises:
            ValueError: 当缺少必需配置时。
        """
        self.deployment_name = deployment_name or settings.llm.model
        self.default_temperature = settings.llm.temperature
        self.default_max_tokens = settings.llm.max_tokens

        # 检查 vision settings 的来源
        vision_settings = getattr(settings, "vision_llm", None)

        # 解析部署名称
        # 优先级：参数 > vision_settings.deployment > vision_settings.model > settings.llm.model
        vision_dep = getattr(vision_settings, "deployment_name", None)
        vision_model = getattr(vision_settings, "model", None)
        self.deployment_name = deployment_name or vision_dep or vision_model or settings.llm.model

        # 解析图片最大边长
        # 优先级：参数 > vision_settings.max_image_size > 默认值
        vision_max_size = getattr(vision_settings, "max_image_size", None)
        self.max_image_size = max_image_size or vision_max_size or self.DEFAULT_MAX_IMAGE_SIZE


        # API 密钥解析顺序：
        # 1. 构造函数参数
        # 2. Vision settings（settings.vision_llm.api_key）
        # 3. 环境变量（AZURE_OPENAI_API_KEY）
        # 4. LLM settings 回退？（settings 对象里的 API key 一般不走这里，但检查一下也无妨）

        self.api_key = api_key
        if not self.api_key and vision_settings and vision_settings.api_key:
             self.api_key = vision_settings.api_key
        if not self.api_key:
             self.api_key = os.environ.get("AZURE_OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Azure OpenAI API key not provided. Set AZURE_OPENAI_API_KEY "
                "environment variable or pass api_key parameter."
            )

        # 端点解析顺序
        self.endpoint = endpoint
        if not self.endpoint and vision_settings and vision_settings.azure_endpoint:
             self.endpoint = vision_settings.azure_endpoint
        if not self.endpoint:
             self.endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT") or getattr(settings.llm, "azure_endpoint", None)

        if not self.endpoint:
            raise ValueError("Azure OpenAI endpoint not provided.")

        # API 版本解析顺序
        self.api_version = api_version
        if not self.api_version and vision_settings and vision_settings.api_version:
             self.api_version = vision_settings.api_version
        if not self.api_version:
             self.api_version = getattr(settings.llm, "api_version", None) or self.DEFAULT_API_VERSION

        # 校验已初始化的客户端
        # 这里不初始化客户端对象，因为每次调用都用 request/httpx 或自定义逻辑
        # 但要确保密钥存在。


        # 保存额外的 kwargs，留作日后使用
        self._extra_config = kwargs
    
    def chat_with_image(
        self,
        text: str,
        image: ImageInput,
        messages: Optional[list[Message]] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """根据文本提示和图片输入生成响应。

        本方法将文本和图片发送给 Azure OpenAI Vision API
        （GPT-4o 或 GPT-4-Vision-Preview），并返回生成的响应。

        Args:
            text: 关于图片的文本提示或问题。
            image: 图片输入（路径、字节或 base64）。
            messages: 可选的对话历史，用于提供上下文。
            trace: 可选的 TraceContext，用于可观测性（为 Stage F 预留）。
            **kwargs: 覆盖参数（temperature、max_tokens 等）。

        Returns:
            包含生成文本和元数据的 ChatResponse。

        Raises:
            ValueError: 当 text 或图片输入无效时。
            AzureVisionLLMError: 当 API 调用失败时。

        Example:
            >>> image = ImageInput(path="chart.png")
            >>> response = vision_llm.chat_with_image(
            ...     text="What does this chart show?",
            ...     image=image
            ... )
        """
        # 校验输入
        self.validate_text(text)
        self.validate_image(image)

        # 预处理图片（如有必要则压缩）
        processed_image = self.preprocess_image(
            image,
            max_size=(self.max_image_size, self.max_image_size)
        )

        # 如有必要将图片转为 base64
        image_base64 = self._get_image_base64(processed_image)

        # 准备请求参数
        temperature = kwargs.get("temperature", self.default_temperature)
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)
        deployment = kwargs.get("deployment_name", self.deployment_name)

        # 构造消息列表
        api_messages = []
        if messages:
            # 追加对话历史
            api_messages.extend([{"role": m.role, "content": m.content} for m in messages])

        # 追加当前 文本 + 图片 消息
        current_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{processed_image.mime_type};base64,{image_base64}"
                    }
                }
            ]
        }
        api_messages.append(current_message)

        # 调用 API
        try:
            response_data = self._call_api(
                messages=api_messages,
                deployment=deployment,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # 解析响应
            content = response_data["choices"][0]["message"]["content"]
            usage = response_data.get("usage")
            
            return ChatResponse(
                content=content,
                model=response_data.get("model", deployment),
                usage=usage,
                raw_response=response_data,
            )
        except KeyError as e:
            raise AzureVisionLLMError(
                f"[Azure Vision] Unexpected response format: missing key {e}"
            ) from e
        except Exception as e:
            if isinstance(e, AzureVisionLLMError):
                raise
            raise AzureVisionLLMError(
                f"[Azure Vision] API call failed: {type(e).__name__}: {e}"
            ) from e
    
    def preprocess_image(
        self,
        image: ImageInput,
        max_size: Optional[tuple[int, int]] = None,
    ) -> ImageInput:
        """在发送给 Azure Vision API 之前对图片做预处理。

        若图片超过 max_size，则压缩以减小 API 负载大小，
        并满足 Azure 的尺寸限制。使用 PIL 进行图片操作。

        本操作是幂等的——对同一输入多次调用，结果保持一致。

        Args:
            image: 待预处理的输入图片。
            max_size: 最大尺寸（宽、高），单位为像素。

        Returns:
            预处理后的 ImageInput（如需压缩则包含压缩后的数据）。

        Note:
            若图片已在尺寸限制内，则原样返回。
        """
        if not max_size:
            return image

        try:
            from PIL import Image
        except ImportError:
            # 若 PIL 不可用，则跳过预处理
            return image

        # 获取图片字节
        if image.data:
            image_bytes = image.data
        elif image.path:
            image_bytes = Path(image.path).read_bytes()
        elif image.base64:
            # 已编码，跳过预处理
            return image
        else:
            return image

        # 加载图片并检查尺寸
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size

        # 检查是否需要压缩
        max_width, max_height = max_size
        if width <= max_width and height <= max_height:
            # 无需压缩
            return image

        # 按比例计算新尺寸（保持宽高比）
        ratio = min(max_width / width, max_height / height)
        new_size = (int(width * ratio), int(height * ratio))

        # 调整图片尺寸
        img_resized = img.resize(new_size, Image.Resampling.LANCZOS)

        # 转回字节
        buffer = io.BytesIO()
        # 尽量保留原格式
        img_format = img.format or "PNG"
        img_resized.save(buffer, format=img_format)
        compressed_bytes = buffer.getvalue()

        # 返回包含压缩数据的新 ImageInput
        return ImageInput(
            data=compressed_bytes,
            mime_type=image.mime_type
        )

    def _get_image_base64(self, image: ImageInput) -> str:
        """将 ImageInput 转为 base64 字符串。

        Args:
            image: 待转换的图片。

        Returns:
            Base64 编码的图片字符串。

        Raises:
            AzureVisionLLMError: 当图片无法编码时。
        """
        try:
            if image.base64:
                return image.base64
            elif image.data:
                return base64.b64encode(image.data).decode("utf-8")
            elif image.path:
                image_bytes = Path(image.path).read_bytes()
                return base64.b64encode(image_bytes).decode("utf-8")
            else:
                raise ValueError("ImageInput has no valid data source")
        except Exception as e:
            raise AzureVisionLLMError(
                f"[Azure Vision] Failed to encode image: {e}"
            ) from e
    
    def _call_api(
        self,
        messages: list[dict],
        deployment: str,
        temperature: float,
        max_tokens: int,
    ) -> dict:
        """向 Azure OpenAI Vision API 发起 HTTP 请求。

        Args:
            messages: API 格式的消息列表。
            deployment: 使用的部署名称。
            temperature: 生成温度。
            max_tokens: 最大生成 token 数。

        Returns:
            API 响应（字典形式）。

        Raises:
            AzureVisionLLMError: 当 API 调用失败时。
        """
        # 去掉 endpoint 末尾的斜杠
        endpoint = self.endpoint.rstrip("/")

        # 构造 API URL
        url = (
            f"{endpoint}/openai/deployments/{deployment}/chat/completions"
            f"?api-version={self.api_version}"
        )

        # 准备请求负载
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
             error_details = e.response.text if e.response else str(e)
             raise AzureVisionLLMError(
                 f"[Azure Vision] HTTP Error: {e} - Response: {error_details}"
             ) from e
        except requests.exceptions.RequestException as e:
             raise AzureVisionLLMError(
                 f"[Azure Vision] Request failed: {e}"
             ) from e
