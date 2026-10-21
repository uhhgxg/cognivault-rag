"""响应模块。

本包包含响应构建相关组件：
- 响应构建器（Response builder）
- 引用生成器（Citation generator）
- 多模态装配器（Multimodal assembler）
"""

from src.core.response.citation_generator import Citation, CitationGenerator
from src.core.response.multimodal_assembler import (
    ImageContent,
    ImageReference,
    MultimodalAssembler,
)
from src.core.response.response_builder import MCPToolResponse, ResponseBuilder

__all__ = [
    "Citation",
    "CitationGenerator",
    "ImageContent",
    "ImageReference",
    "MCPToolResponse",
    "MultimodalAssembler",
    "ResponseBuilder",
]
