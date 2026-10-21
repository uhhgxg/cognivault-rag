"""Azure OpenAI LLM 实现。

本模块提供与 Azure 的 OpenAI 服务 API 对接的 Azure OpenAI LLM 实现。
负责处理 Azure 专属的身份认证和端点配置。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message


class AzureLLMError(RuntimeError):
    """当 Azure OpenAI API 调用失败时抛出。"""


class AzureLLM(BaseLLM):
    """Azure OpenAI LLM 提供方实现。

    本类为 Azure 的 OpenAI 服务实现 BaseLLM 接口。
    Azure 采用与标准 OpenAI 不同的身份认证方式（请求头里带 API 密钥）
    和端点结构。

    Attributes:
        api_key: 用于身份认证的 Azure API 密钥。
        endpoint: Azure OpenAI 的端点 URL。
        deployment_name: 模型的部署名称。
        api_version: 使用的 API 版本。
        default_temperature: 生成时使用的默认温度。
        default_max_tokens: 生成时使用的默认最大 token 数。

    Example:
        >>> from src.core.settings import load_settings
        >>> settings = load_settings('config/settings.yaml')
        >>> llm = AzureLLM(settings, endpoint='https://my-resource.openai.azure.com')
        >>> response = llm.chat([Message(role='user', content='Hello')])
    """
    
    DEFAULT_API_VERSION = "2024-02-15-preview"
    
    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment_name: Optional[str] = None,
        api_version: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """初始化 Azure OpenAI LLM 提供方。

        Args:
            settings: 应用配置，包含 LLM 相关配置。
            api_key: 可选的 API 密钥覆盖值（找不到时回退到 settings.llm.api_key 或环境变量）。
            endpoint: 可选的端点覆盖值（找不到时回退到 settings.llm.azure_endpoint 或环境变量）。
            deployment_name: 可选的部署名称（默认为 settings.llm.deployment_name 或 model）。
            api_version: 可选的 API 版本覆盖值。
            **kwargs: 其他配置覆盖项。

        Raises:
            ValueError: 当缺少必需配置时。
        """
        # 部署名称优先级：显式传参 > settings.deployment_name > settings.model
        self.deployment_name = (
            deployment_name
            or getattr(settings.llm, 'deployment_name', None)
            or settings.llm.model
        )
        self.default_temperature = settings.llm.temperature
        self.default_max_tokens = settings.llm.max_tokens

        # API 密钥优先级：显式传参 > settings > 环境变量
        self.api_key = (
            api_key
            or getattr(settings.llm, 'api_key', None)
            or os.environ.get("AZURE_OPENAI_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "Azure OpenAI API key not provided. Set in settings.yaml (llm.api_key), "
                "AZURE_OPENAI_API_KEY environment variable, or pass api_key parameter."
            )

        # 端点优先级：显式传参 > settings > 环境变量
        self.endpoint = (
            endpoint
            or getattr(settings.llm, 'azure_endpoint', None)
            or os.environ.get("AZURE_OPENAI_ENDPOINT")
        )
        if not self.endpoint:
            raise ValueError(
                "Azure OpenAI endpoint not provided. Set in settings.yaml (llm.azure_endpoint), "
                "AZURE_OPENAI_ENDPOINT environment variable, or pass endpoint parameter."
            )

        # API 版本优先级：显式传参 > settings > 默认值
        self.api_version = (
            api_version
            or getattr(settings.llm, 'api_version', None)
            or self.DEFAULT_API_VERSION
        )

        # 保存额外的 kwargs，留作日后使用
        self._extra_config = kwargs
    
    def chat(
        self,
        messages: List[Message],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """使用 Azure OpenAI API 生成一条聊天补全。

        Args:
            messages: 对话消息列表。
            trace: 可选的 TraceContext，用于可观测性（为 Stage F 预留）。
            **kwargs: 覆盖参数（temperature、max_tokens 等）。

        Returns:
            包含生成内容和元数据的 ChatResponse。

        Raises:
            ValueError: 当 messages 无效时。
            AzureLLMError: 当 API 调用失败时。
        """
        # 校验输入
        self.validate_messages(messages)

        # 准备请求参数
        temperature = kwargs.get("temperature", self.default_temperature)
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)
        deployment = kwargs.get("deployment_name", self.deployment_name)

        # 将消息转换为 API 格式
        api_messages = [{"role": m.role, "content": m.content} for m in messages]

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
            raise AzureLLMError(
                f"[Azure] Unexpected response format: missing key {e}"
            ) from e
        except Exception as e:
            if isinstance(e, AzureLLMError):
                raise
            raise AzureLLMError(
                f"[Azure] API call failed: {type(e).__name__}: {e}"
            ) from e
    
    def _call_api(
        self,
        messages: List[Dict[str, str]],
        deployment: str,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """实际向 Azure OpenAI 发起 API 调用。

        本方法单独抽出，便于在测试中方便地 mock。

        Args:
            messages: API 格式的消息列表。
            deployment: 部署名称。
            temperature: 生成温度。
            max_tokens: 最大生成 token 数。

        Returns:
            原始 API 响应（字典形式）。

        Raises:
            AzureLLMError: 当 API 调用失败时。
        """
        import httpx

        # Azure 端点格式：
        # {endpoint}/openai/deployments/{deployment}/chat/completions?api-version={version}
        url = (
            f"{self.endpoint.rstrip('/')}/openai/deployments/{deployment}/"
            f"chat/completions?api-version={self.api_version}"
        )
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload, headers=headers)
                
                if response.status_code != 200:
                    error_detail = self._parse_error_response(response)
                    raise AzureLLMError(
                        f"[Azure] API error (HTTP {response.status_code}): {error_detail}"
                    )
                
                return response.json()
        except httpx.TimeoutException as e:
            raise AzureLLMError(
                f"[Azure] Request timed out after 60 seconds"
            ) from e
        except httpx.RequestError as e:
            raise AzureLLMError(
                f"[Azure] Connection failed: {type(e).__name__}: {e}"
            ) from e
    
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
