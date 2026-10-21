"""LLM 服务提供方的抽象基类（父类）。

本模块定义了 Language Model 服务提供方的可插拔接口，
通过配置驱动的方式，可以在不同后端（OpenAI、Azure、Ollama 等）之间无缝切换。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    """表示对话中的一条消息。

    Attributes:
        role: 消息发送者的角色（'system'、'user' 或 'assistant'）。
        content: 消息的文本内容。
    """
    role: str
    content: str


@dataclass
class ChatResponse:
    """LLM 聊天补全的响应结果。

    Attributes:
        content: 生成的文本响应。
        model: 生成该响应的模型标识符。
        usage: 可选的 token 使用统计（prompt_tokens、completion_tokens、total_tokens）。
        raw_response: 可选的提供方原始响应，用于调试。
    """
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Any] = None


class BaseLLM(ABC):
    """LLM 服务提供方的抽象基类（父类）。

    所有 LLM 实现都必须继承本类，并实现 chat() 方法。
    这样可以保证不同提供方（OpenAI、Azure、DeepSeek、Ollama 等）的接口一致。

    应用的设计原则：
    - 可插拔（Pluggable）：子类可以替换，而不用改动上层代码。
    - 可观测（Observable）：接受可选的 TraceContext，用于接入可观测性。
    - 配置驱动（Config-Driven）：通过工厂根据配置创建实例。
    """
    
    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """生成一条聊天补全响应。

        Args:
            messages: 对话消息列表（包含角色 + 内容）。
            trace: 可选的 TraceContext，用于可观测性（为 Stage F 预留）。
            **kwargs: 各提供方专属参数（temperature、max_tokens 等）。

        Returns:
            包含生成文本和元数据的 ChatResponse。

        Raises:
            ValueError: 当 messages 为空或格式不对时。
            RuntimeError: 当 LLM 提供方调用失败时。
        """
        pass

    def validate_messages(self, messages: List[Message]) -> None:
        """校验消息列表结构。

        Args:
            messages: 待校验的消息列表。

        Raises:
            ValueError: 当 messages 为空或包含无效角色时。
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        valid_roles = {"system", "user", "assistant"}
        for i, msg in enumerate(messages):
            if not isinstance(msg, Message):
                raise ValueError(f"Message at index {i} is not a Message instance")
            if msg.role not in valid_roles:
                raise ValueError(
                    f"Message at index {i} has invalid role '{msg.role}'. "
                    f"Must be one of: {valid_roles}"
                )
            if not msg.content or not msg.content.strip():
                raise ValueError(f"Message at index {i} has empty content")
