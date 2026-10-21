"""兼容 OpenAI 的 Vision LLM 实现。

本模块提供兼容 OpenAI 协议的 Vision LLM 实现，支持多模态交互（文本 + 图片）。
支持 GPT-4o 等具备视觉能力的模型。

当 settings 中存在 azure_endpoint 时，提供方会自动构造 Azure 兼容的 URL，
并使用 api-key 认证请求头，使同一套 OpenAI 标准协议可以直接对接 Azure OpenAI 端点。
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any, Optional

from src.libs.llm.base_llm import ChatResponse, Message
from src.libs.llm.base_vision_llm import BaseVisionLLM, ImageInput


class OpenAIVisionLLMError(RuntimeError):
    """当 OpenAI Vision API 调用失败时抛出。"""


class OpenAIVisionLLM(BaseVisionLLM):
    """兼容 OpenAI 的 Vision LLM 提供方实现。

    本类使用 OpenAI 标准协议实现 BaseVisionLLM 接口。
    同时支持标准 OpenAI 端点，以及兼容模式下的 Azure OpenAI 端点。

    当检测到 settings.vision_llm 中存在 azure_endpoint 时，会自动：
    - 构造基于 deployment 的 URL
    - 使用 api-key 请求头做身份认证
    - 追加 api-version 查询参数

    Attributes:
        api_key: 用于身份认证的 API 密钥。
        base_url: API 的基础 URL。
        model: 模型标识符 / 部署名称。
        api_version: 可选的 API 版本（用于 Azure 兼容）。
        max_image_size: 图片最大边长，单位为像素（默认 2048）。
        default_temperature: 生成时使用的默认温度。
        default_max_tokens: 生成时使用的默认最大 token 数。

    Example:
        >>> from src.core.settings import load_settings
        >>> settings = load_settings('config/settings.yaml')
        >>> vision_llm = OpenAIVisionLLM(settings)
        >>> image = ImageInput(path="diagram.png")
        >>> response = vision_llm.chat_with_image("Describe this", image)
    """

    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_MAX_IMAGE_SIZE = 2048  # 像素
    
    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_image_size: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """初始化 OpenAI Vision LLM 提供方。

        Args:
            settings: 应用配置，包含 vision_llm 相关配置。
            api_key: 可选的 API 密钥覆盖值。
            base_url: 可选的基础 URL 覆盖值。
            max_image_size: 用于自动压缩的图片最大边长，单位为像素。
            **kwargs: 其他配置覆盖项。

        Raises:
            ValueError: 当缺少必需配置时。
        """
        # 读取 vision 配置段
        vision_settings = getattr(settings, "vision_llm", None)

        # temperature / max_tokens 优先级：vision_llm 段 > llm 段默认值
        self.default_temperature = getattr(settings.llm, 'temperature', 0.0)
        self.default_max_tokens = getattr(settings.llm, 'max_tokens', 4096)

        # 模型 / 部署名称
        vision_model = getattr(vision_settings, 'model', None) if vision_settings else None
        vision_dep = getattr(vision_settings, 'deployment_name', None) if vision_settings else None
        self.model = vision_dep or vision_model or settings.llm.model

        # 图片最大边长
        vision_max_size = getattr(vision_settings, 'max_image_size', None) if vision_settings else None
        self.max_image_size = max_image_size or vision_max_size or self.DEFAULT_MAX_IMAGE_SIZE

        # API 密钥优先级：显式传参 > vision_settings > llm settings > 环境变量
        self.api_key = api_key
        if not self.api_key and vision_settings:
            self.api_key = getattr(vision_settings, 'api_key', None)
        if not self.api_key:
            self.api_key = getattr(settings.llm, 'api_key', None)
        if not self.api_key:
            self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set in settings.yaml (vision_llm.api_key), "
                "OPENAI_API_KEY environment variable, or pass api_key parameter."
            )

        # 检测是否走 Azure 兼容模式
        azure_endpoint = None
        if vision_settings:
            azure_endpoint = getattr(vision_settings, 'azure_endpoint', None)
        if not azure_endpoint:
            azure_endpoint = getattr(settings.llm, 'azure_endpoint', None)

        self.api_version = None
        if vision_settings:
            self.api_version = getattr(vision_settings, 'api_version', None)
        if not self.api_version:
            self.api_version = getattr(settings.llm, 'api_version', None)

        self._use_azure_auth = False

        # 检查 vision_llm 或 llm settings 中是否配置了 base_url
        config_base_url = None
        if vision_settings:
            config_base_url = getattr(vision_settings, 'base_url', None)
        if not config_base_url:
            config_base_url = getattr(settings.llm, 'base_url', None)

        if base_url:
            self.base_url = base_url
        elif azure_endpoint:
            # Azure 兼容模式
            self.base_url = (
                f"{azure_endpoint.rstrip('/')}/openai/deployments/{self.model}"
            )
            self._use_azure_auth = True
            if not self.api_version:
                self.api_version = "2024-02-15-preview"
        elif config_base_url:
            self.base_url = config_base_url
        else:
            self.base_url = self.DEFAULT_BASE_URL

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

        Args:
            text: 关于图片的文本提示或问题。
            image: 图片输入（路径、字节或 base64）。
            messages: 可选的对话历史，用于提供上下文。
            trace: 可选的 TraceContext，用于可观测性。
            **kwargs: 覆盖参数（temperature、max_tokens 等）。

        Returns:
            包含生成文本和元数据的 ChatResponse。

        Raises:
            ValueError: 当 text 或图片输入无效时。
            OpenAIVisionLLMError: 当 API 调用失败时。
        """
        # 校验输入
        self.validate_text(text)
        self.validate_image(image)

        # 预处理图片（如有必要则压缩）
        processed_image = self.preprocess_image(
            image,
            max_size=(self.max_image_size, self.max_image_size)
        )

        # 将图片转为 base64
        image_base64 = self._get_image_base64(processed_image)

        # 准备请求参数
        temperature = kwargs.get("temperature", self.default_temperature)
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)

        # 构造消息列表
        api_messages = []
        if messages:
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
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            content = response_data["choices"][0]["message"]["content"]
            usage = response_data.get("usage")
            
            return ChatResponse(
                content=content,
                model=response_data.get("model", self.model),
                usage=usage,
                raw_response=response_data,
            )
        except KeyError as e:
            raise OpenAIVisionLLMError(
                f"[OpenAI Vision] Unexpected response format: missing key {e}"
            ) from e
        except Exception as e:
            if isinstance(e, OpenAIVisionLLMError):
                raise
            raise OpenAIVisionLLMError(
                f"[OpenAI Vision] API call failed: {type(e).__name__}: {e}"
            ) from e
    
    def preprocess_image(
        self,
        image: ImageInput,
        max_size: Optional[tuple[int, int]] = None,
    ) -> ImageInput:
        """在发送给 Vision API 之前对图片做预处理。

        若图片超过 max_size，则压缩以减小负载大小。

        Args:
            image: 待预处理的输入图片。
            max_size: 最大尺寸（宽、高），单位为像素。

        Returns:
            预处理后的 ImageInput（如需压缩则包含压缩后的数据）。
        """
        if not max_size:
            return image

        try:
            from PIL import Image
        except ImportError:
            return image

        # 获取图片字节
        if image.data:
            image_bytes = image.data
        elif image.path:
            image_bytes = Path(image.path).read_bytes()
        elif image.base64:
            return image
        else:
            return image

        # 加载图片并检查尺寸
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size

        max_width, max_height = max_size
        if width <= max_width and height <= max_height:
            return image

        # 按比例计算新尺寸（保持宽高比）
        ratio = min(max_width / width, max_height / height)
        new_size = (int(width * ratio), int(height * ratio))

        # 调整图片尺寸
        img_resized = img.resize(new_size, Image.Resampling.LANCZOS)

        # 转回字节
        buffer = io.BytesIO()
        img_format = img.format or "PNG"
        img_resized.save(buffer, format=img_format)
        compressed_bytes = buffer.getvalue()

        return ImageInput(
            data=compressed_bytes,
            mime_type=image.mime_type
        )

    def _get_image_base64(self, image: ImageInput) -> str:
        """将 ImageInput 转为 base64 字符串。"""
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
            raise OpenAIVisionLLMError(
                f"[OpenAI Vision] Failed to encode image: {e}"
            ) from e
    
    def _call_api(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> dict:
        """向 Vision API 发起 HTTP 请求。

        Args:
            messages: API 格式的消息列表。
            temperature: 生成温度。
            max_tokens: 最大生成 token 数。

        Returns:
            API 响应（字典形式）。

        Raises:
            OpenAIVisionLLMError: 当 API 调用失败时。
        """
        import httpx

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        if self.api_version:
            url += f"?api-version={self.api_version}"

        if self._use_azure_auth:
            headers = {
                "api-key": self.api_key,
                "Content-Type": "application/json",
            }
        else:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        last_error = None
        for attempt in range(3):
            try:
                with httpx.Client(timeout=60.0) as client:
                    response = client.post(url, json=payload, headers=headers)
                    
                    if response.status_code != 200:
                        error_detail = self._parse_error_response(response)
                        raise OpenAIVisionLLMError(
                            f"[OpenAI Vision] API error (HTTP {response.status_code}): {error_detail}"
                        )
                    
                    return response.json()
            except (httpx.TimeoutException, httpx.RequestError) as e:
                last_error = e
                if attempt < 2:
                    import time
                    time.sleep(1.0 * (2 ** attempt))
                continue
            except OpenAIVisionLLMError:
                raise
            except Exception as e:
                last_error = e
                if attempt < 2:
                    import time
                    time.sleep(1.0 * (2 ** attempt))
                continue
        
        if isinstance(last_error, httpx.TimeoutException):
            raise OpenAIVisionLLMError(
                "[OpenAI Vision] Request timed out after 60 seconds (3 retries)"
            ) from last_error
        elif isinstance(last_error, httpx.RequestError):
            raise OpenAIVisionLLMError(
                f"[OpenAI Vision] Connection failed after 3 retries: {type(last_error).__name__}: {last_error}"
            ) from last_error
        else:
            raise OpenAIVisionLLMError(
                f"[OpenAI Vision] API call failed after 3 retries: {type(last_error).__name__}: {last_error}"
            ) from last_error
    
    def _parse_error_response(self, response: Any) -> str:
        """从 API 响应中解析错误详情。"""
        try:
            error_data = response.json()
            if "error" in error_data:
                error = error_data["error"]
                if isinstance(error, dict):
                    return error.get("message", str(error))
                return str(error)
            return response.text
        except Exception:
            return response.text or "Unknown error"
