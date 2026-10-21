"""用于本地模型推理的 Ollama LLM 实现。

本模块提供与本地运行的 Ollama 实例对接的 Ollama LLM 实现。
Ollama 可在本地硬件上运行 Llama、Mistral、CodeLlama 等 LLM。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message


class OllamaLLMError(RuntimeError):
    """当 Ollama API 调用失败时抛出。

    本异常提供清晰的错误信息，同时不暴露
    内部 URL、凭证等敏感配置细节。
    """


class OllamaLLM(BaseLLM):
    """用于本地推理的 Ollama LLM 提供方实现。

    本类为 Ollama 的聊天 API 实现 BaseLLM 接口，
    可以在本地进行 LLM 推理，无需依赖云服务。

    Attributes:
        base_url: Ollama 服务器的基础 URL（默认：http://localhost:11434）。
        model: 使用的模型标识符（例如 'llama3'、'mistral'）。
        default_temperature: 生成时使用的默认温度。
        default_max_tokens: 生成时使用的默认最大 token 数（Ollama 中对应 num_predict）。
        timeout: 请求超时时间，单位为秒。

    Example:
        >>> from src.core.settings import load_settings
        >>> settings = load_settings('config/settings.yaml')
        >>> llm = OllamaLLM(settings)
        >>> response = llm.chat([Message(role='user', content='Hello')])
    """

    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_TIMEOUT = 120.0  # 本地推理，超时设得长一些
    
    def __init__(
        self,
        settings: Any,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """初始化 Ollama LLM 提供方。

        Args:
            settings: 应用配置，包含 LLM 相关配置。
            base_url: 可选的基础 URL 覆盖值（找不到时回退到环境变量 OLLAMA_BASE_URL）。
            timeout: 可选的请求超时覆盖值。
            **kwargs: 其他配置覆盖项。

        Raises:
            ValueError: 当缺少必需配置时。
        """
        self.model = settings.llm.model
        self.default_temperature = settings.llm.temperature
        self.default_max_tokens = settings.llm.max_tokens

        # 基础 URL 优先级：显式传参 > 环境变量 > 默认值
        self.base_url = (
            base_url
            or os.environ.get("OLLAMA_BASE_URL")
            or self.DEFAULT_BASE_URL
        )

        # 超时优先级：显式传参 > 默认值
        self.timeout = timeout or self.DEFAULT_TIMEOUT

        # 保存额外的 kwargs，留作日后使用
        self._extra_config = kwargs
    
    def chat(
        self,
        messages: List[Message],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """使用 Ollama API 生成一条聊天补全。

        Args:
            messages: 对话消息列表。
            trace: 可选的 TraceContext，用于可观测性（为 Stage F 预留）。
            **kwargs: 覆盖参数（temperature、max_tokens 等）。

        Returns:
            包含生成内容和元数据的 ChatResponse。

        Raises:
            ValueError: 当 messages 无效时。
            OllamaLLMError: 当 API 调用失败时。
        """
        # 校验输入
        self.validate_messages(messages)

        # 准备请求参数
        temperature = kwargs.get("temperature", self.default_temperature)
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)
        model = kwargs.get("model", self.model)

        # 将消息转换为 Ollama API 格式
        api_messages = [{"role": m.role, "content": m.content} for m in messages]

        # 调用 API
        try:
            response_data = self._call_api(
                messages=api_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # 解析响应 —— Ollama 返回的格式和 OpenAI 不同
            # 同时兼容 /api/chat（关闭流式）的响应格式
            if "message" in response_data:
                # 标准聊天响应
                content = response_data["message"]["content"]
            elif "response" in response_data:
                # 旧版 generate 端点的响应（回退方案）
                content = response_data["response"]
            else:
                raise OllamaLLMError(
                    "[Ollama] Unexpected response format: missing 'message' or 'response' key"
                )

            # 如有可用信息，则构造 token 使用统计
            usage = None
            if "eval_count" in response_data or "prompt_eval_count" in response_data:
                usage = {
                    "prompt_tokens": response_data.get("prompt_eval_count", 0),
                    "completion_tokens": response_data.get("eval_count", 0),
                    "total_tokens": (
                        response_data.get("prompt_eval_count", 0) +
                        response_data.get("eval_count", 0)
                    ),
                }
            
            return ChatResponse(
                content=content,
                model=response_data.get("model", model),
                usage=usage,
                raw_response=response_data,
            )
        except KeyError as e:
            raise OllamaLLMError(
                f"[Ollama] Unexpected response format: missing key {e}"
            ) from e
        except Exception as e:
            if isinstance(e, OllamaLLMError):
                raise
            raise OllamaLLMError(
                f"[Ollama] API call failed: {type(e).__name__}: {e}"
            ) from e
    
    def _call_api(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """实际向 Ollama 发起 API 调用。

        本方法单独抽出，便于在测试中方便地 mock。

        Args:
            messages: API 格式的消息列表。
            model: 模型标识符。
            temperature: 生成温度。
            max_tokens: 最大生成 token 数（Ollama 中对应 num_predict）。

        Returns:
            原始 API 响应（字典形式）。

        Raises:
            OllamaLLMError: 当 API 调用失败时。
        """
        import httpx

        url = f"{self.base_url.rstrip('/')}/api/chat"
        headers = {
            "Content-Type": "application/json",
        }

        # Ollama 用 'num_predict' 代替 'max_tokens'
        # 模型参数统一放在 'options' 对象里
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,  # 关闭流式，返回同步响应
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload, headers=headers)
                
                if response.status_code != 200:
                    error_detail = self._parse_error_response(response)
                    raise OllamaLLMError(
                        f"[Ollama] API error (HTTP {response.status_code}): {error_detail}"
                    )
                
                return response.json()
        except httpx.TimeoutException as e:
            raise OllamaLLMError(
                f"[Ollama] Request timed out after {self.timeout} seconds. "
                "Consider increasing timeout for larger models or longer responses."
            ) from e
        except httpx.ConnectError as e:
            raise OllamaLLMError(
                "[Ollama] Connection failed. Ensure Ollama is running locally. "
                "Start it with 'ollama serve' command."
            ) from e
        except httpx.RequestError as e:
            raise OllamaLLMError(
                f"[Ollama] Request failed: {type(e).__name__}"
            ) from e
    
    def _parse_error_response(self, response: Any) -> str:
        """从 API 响应中解析错误详情。

        Args:
            response: HTTP 响应对象。

        Returns:
            人类可读的错误信息，且不暴露敏感细节。
        """
        try:
            error_data = response.json()
            if "error" in error_data:
                return str(error_data["error"])
            return response.text[:200] if response.text else "Unknown error"
        except Exception:
            return response.text[:200] if response.text else "Unknown error"
