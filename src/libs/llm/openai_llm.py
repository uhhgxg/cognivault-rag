"""兼容 OpenAI 的 LLM 实现。

本模块提供与标准 OpenAI API 对接的 OpenAI LLM 实现。
通过配置 base_url，也可以对接其他兼容 OpenAI 协议的端点。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message


class OpenAILLMError(RuntimeError):
    """当 OpenAI API 调用失败时抛出。"""


class OpenAILLM(BaseLLM):
    """OpenAI LLM 提供方实现。

    本类为 OpenAI 的聊天补全 API 实现 BaseLLM 接口。
    支持标准 OpenAI API，也支持任何兼容 OpenAI 协议的端点。

    Attributes:
        api_key: 用于身份认证的 API 密钥。
        base_url: API 的基础 URL（默认：OpenAI 的端点）。
        model: 使用的模型标识符。
        default_temperature: 生成时使用的默认温度。
        default_max_tokens: 生成时使用的默认最大 token 数。

    Example:
        >>> from src.core.settings import load_settings
        >>> settings = load_settings('config/settings.yaml')
        >>> llm = OpenAILLM(settings)
        >>> response = llm.chat([Message(role='user', content='Hello')])
    """
    
    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    
    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """初始化 OpenAI LLM 提供方。

        Args:
            settings: 应用配置，包含 LLM 相关配置。
            api_key: 可选的 API 密钥覆盖值（找不到时回退到 settings.llm.api_key 或环境变量）。
            base_url: 可选的基础 URL 覆盖值。
            **kwargs: 其他配置覆盖项。

        Raises:
            ValueError: 当未提供 API 密钥，且环境变量中也找不到时。

        Note:
            当 settings 中存在 azure_endpoint 时，提供方会自动构造
            Azure 兼容的 OpenAI URL，并使用 api-key 认证请求头。
        """
        self.model = settings.llm.model
        self.default_temperature = settings.llm.temperature
        self.default_max_tokens = settings.llm.max_tokens

        # API 密钥优先级：显式传参 > settings > 环境变量
        self.api_key = (
            api_key
            or getattr(settings.llm, 'api_key', None)
            or os.environ.get("OPENAI_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set in settings.yaml (llm.api_key), "
                "OPENAI_API_KEY environment variable, or pass api_key parameter."
            )

        # 检测是否走 Azure 兼容模式
        azure_endpoint = getattr(settings.llm, 'azure_endpoint', None)
        config_base_url = getattr(settings.llm, 'base_url', None)
        self.api_version = getattr(settings.llm, 'api_version', None)

        if base_url:
            self.base_url = base_url
            self._use_azure_auth = False
        elif azure_endpoint:
            # Azure 兼容模式：构造基于 deployment 的 URL
            deployment = getattr(settings.llm, 'deployment_name', None) or self.model
            self.base_url = f"{azure_endpoint.rstrip('/')}/openai/deployments/{deployment}"
            self._use_azure_auth = True
            if not self.api_version:
                self.api_version = "2024-02-15-preview"
        elif config_base_url:
            self.base_url = config_base_url
            self._use_azure_auth = False
        else:
            self.base_url = self.DEFAULT_BASE_URL
            self._use_azure_auth = False

        # 保存额外的 kwargs，留作日后使用
        self._extra_config = kwargs
    
    def chat(
        self,
        messages: List[Message],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """使用 OpenAI API 生成一条聊天补全。

        Args:
            messages: 对话消息列表。
            trace: 可选的 TraceContext，用于可观测性（为 Stage F 预留）。
            **kwargs: 覆盖参数（temperature、max_tokens 等）。

        Returns:
            包含生成内容和元数据的 ChatResponse。

        Raises:
            ValueError: 当 messages 无效时。
            OpenAILLMError: 当 API 调用失败时。
        """
        # 校验输入
        self.validate_messages(messages)

        # 准备请求参数
        temperature = kwargs.get("temperature", self.default_temperature)
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)
        model = kwargs.get("model", self.model)

        # 将消息转换为 API 格式
        api_messages = [{"role": m.role, "content": m.content} for m in messages]

        # 调用 API
        try:
            response_data = self._call_api(
                messages=api_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # 解析响应
            content = response_data["choices"][0]["message"]["content"]
            usage = response_data.get("usage")
            
            return ChatResponse(
                content=content,
                model=response_data.get("model", model),
                usage=usage,
                raw_response=response_data,
            )
        except KeyError as e:
            raise OpenAILLMError(
                f"[OpenAI] Unexpected response format: missing key {e}"
            ) from e
        except Exception as e:
            if isinstance(e, OpenAILLMError):
                raise
            raise OpenAILLMError(
                f"[OpenAI] API call failed: {type(e).__name__}: {e}"
            ) from e
    
    def _call_api(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """实际向 OpenAI 发起 API 调用。

        本方法单独抽出，便于在测试中方便地 mock。

        Args:
            messages: API 格式的消息列表。
            model: 模型标识符。
            temperature: 生成温度。
            max_tokens: 最大生成 token 数。

        Returns:
            原始 API 响应（字典形式）。

        Raises:
            OpenAILLMError: 当 API 调用失败时。
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
            "model": model,
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
                        raise OpenAILLMError(
                            f"[OpenAI] API error (HTTP {response.status_code}): {error_detail}"
                        )
                    
                    return response.json()
            except (httpx.TimeoutException, httpx.RequestError) as e:
                last_error = e
                if attempt < 2:
                    import time
                    time.sleep(1.0 * (2 ** attempt))
                continue
            except OpenAILLMError:
                raise
            except Exception as e:
                last_error = e
                if attempt < 2:
                    import time
                    time.sleep(1.0 * (2 ** attempt))
                continue
        
        if isinstance(last_error, httpx.TimeoutException):
            raise OpenAILLMError(
                f"[OpenAI] Request timed out after 60 seconds (3 retries)"
            ) from last_error
        elif isinstance(last_error, httpx.RequestError):
            raise OpenAILLMError(
                f"[OpenAI] Connection failed after 3 retries: {type(last_error).__name__}: {last_error}"
            ) from last_error
        else:
            raise OpenAILLMError(
                f"[OpenAI] API call failed after 3 retries: {type(last_error).__name__}: {last_error}"
            ) from last_error
    
    def _parse_error_response(self, response: Any) -> str:
        """从 API 响应中解析错误详情。

        Args:
            response: HTTP 响应对象。

        Returns:
            人类可读的错误信息。
        """
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
