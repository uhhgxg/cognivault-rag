"""响应构建器，用于构造 MCP 格式的响应。

本模块为 MCP 工具构建结构化响应，整合以下内容：
- 带引用标记的可读 Markdown 内容
- 供机器消费的结构化引用数据
- 多模态内容（文本 + 图片）支持
- 对空结果和错误情况的正确处理
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from mcp import types

from src.core.response.citation_generator import Citation, CitationGenerator
from src.core.types import RetrievalResult


@dataclass
class MCPToolResponse:
    """MCP 工具的结构化响应。
    
    属性：
        content: 带引用标记 [1]、[2] 等的可读 Markdown 内容。
        citations: 结构化引用列表，供参考。
        metadata: 额外的响应元数据（query、result_count 等）。
        is_empty: 搜索是否未返回任何结果。
        image_contents: 多模态响应用的 MCP ImageContent 块列表。
    """
    content: str
    citations: List[Citation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_empty: bool = False
    image_contents: List[types.ImageContent] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为 MCP 协议使用的字典。
        
        返回：
            包含 'content' 和 'structuredContent' 字段的字典。
        """
        return {
            "content": self.content,
            "structuredContent": {
                "citations": [c.to_dict() for c in self.citations],
                "metadata": self.metadata,
                "isEmpty": self.is_empty,
            }
        }
    
    def to_mcp_content(self) -> List[Union[types.TextContent, types.ImageContent]]:
        """转换为 MCP content blocks 格式。
        
        返回：
            用于 MCP CallToolResult 的 content block 列表。
            包含 TextContent，可选包含 ImageContent 块。
        """
        blocks: List[Union[types.TextContent, types.ImageContent]] = [
            types.TextContent(
                type="text",
                text=self.content,
            )
        ]
        
        # 如果存在图片块则添加（多模态响应）
        if self.image_contents:
            blocks.extend(self.image_contents)
        
        # 将结构化数据作为单独的文本块添加（JSON 格式）
        if self.citations or self.metadata:
            import json
            structured = {
                "citations": [c.to_dict() for c in self.citations],
                "metadata": self.metadata,
                "has_images": len(self.image_contents) > 0,
                "image_count": len(self.image_contents),
            }
            blocks.append(
                types.TextContent(
                    type="text",
                    text=f"\n---\n**References (JSON):**\n```json\n{json.dumps(structured, ensure_ascii=False, indent=2)}\n```",
                )
            )
        
        return blocks
    
    @property
    def has_images(self) -> bool:
        """检查响应是否包含图片。
        
        返回：
            如果响应包含图片内容则返回 True，否则返回 False。
        """
        return len(self.image_contents) > 0


class ResponseBuilder:
    """从检索结果构建 MCP 格式的响应。
    
    本类将检索结果转换为结构化的 MCP 响应，
    包括带内联引用的可读 Markdown 和供机器消费的结构化引用数据。
    
    当结果元数据中包含图片引用时，支持带图片的多模态响应。
    
    示例：
        >>> builder = ResponseBuilder()
        >>> results = [RetrievalResult(chunk_id="doc1_001", score=0.95, ...)]
        >>> response = builder.build(results, "What is Azure OpenAI?")
        >>> print(response.content)  # 带 [1]、[2] 标记的 Markdown
        >>> print(response.citations[0].source)  # "docs/guide.pdf"
        >>> print(response.has_images)  # 找到图片则为 True
    """
    
    def __init__(
        self,
        citation_generator: Optional[CitationGenerator] = None,
        multimodal_assembler: Optional["MultimodalAssembler"] = None,
        max_results_in_content: int = 5,
        snippet_max_length: int = 300,
        enable_multimodal: bool = True,
    ) -> None:
        """初始化 ResponseBuilder。
        
        参数：
            citation_generator: 可选的 CitationGenerator 实例。
                如果为 None，则创建默认实例。
            multimodal_assembler: 可选的 MultimodalAssembler，用于处理图片。
                如果为 None 且 enable_multimodal=True，则创建默认实例。
            max_results_in_content: Markdown 内容中最多展示的结果数。
            snippet_max_length: 内容中每条结果摘要的最大字符数。
            enable_multimodal: 是否在响应中包含图片（默认：True）。
        """
        self.citation_generator = citation_generator or CitationGenerator()
        self.max_results_in_content = max_results_in_content
        self.snippet_max_length = snippet_max_length
        self.enable_multimodal = enable_multimodal
        
        # 懒加载多模态装配器，避免循环导入
        self._multimodal_assembler = multimodal_assembler
    
    @property
    def multimodal_assembler(self) -> "MultimodalAssembler":
        """获取或创建 MultimodalAssembler 实例。"""
        if self._multimodal_assembler is None:
            from src.core.response.multimodal_assembler import MultimodalAssembler
            self._multimodal_assembler = MultimodalAssembler()
        return self._multimodal_assembler
    
    def build(
        self,
        results: List[RetrievalResult],
        query: str,
        collection: Optional[str] = None,
        include_images: bool = True,
    ) -> MCPToolResponse:
        """从检索结果构建 MCP 响应。
        
        参数：
            results: 搜索返回的 RetrievalResult 列表。
            query: 原始用户查询。
            collection: 可选的集合名称。
            include_images: 是否在响应中包含图片（默认：True）。
            
        返回：
            MCPToolResponse，包含格式化内容、引用和可选图片。
        """
        # 处理空结果
        if not results:
            return self._build_empty_response(query, collection)
        
        # 生成引用
        citations = self.citation_generator.generate(results)
        
        # 构建 Markdown 内容
        content = self._build_markdown_content(results, citations, query)
        
        # Build metadata
        metadata = self._build_metadata(query, collection, len(results))
        
        # 如启用则装配图片内容
        image_contents: List[types.ImageContent] = []
        if self.enable_multimodal and include_images:
            image_blocks = self.multimodal_assembler.assemble(results, collection)
            # Filter to only ImageContent blocks
            image_contents = [
                block for block in image_blocks
                if isinstance(block, types.ImageContent)
            ]
            if image_contents:
                metadata["has_images"] = True
                metadata["image_count"] = len(image_contents)
        
        return MCPToolResponse(
            content=content,
            citations=citations,
            metadata=metadata,
            is_empty=False,
            image_contents=image_contents,
        )
    
    def _build_empty_response(
        self,
        query: str,
        collection: Optional[str] = None,
    ) -> MCPToolResponse:
        """Build response for empty results.
        
        Args:
            query: Original user query.
            collection: Optional collection name.
            
        Returns:
            MCPToolResponse indicating no results found.
        """
        content = f"## 未找到相关结果\n\n"
        content += f"查询: **{query}**\n\n"
        
        if collection:
            content += f"在集合 `{collection}` 中未找到与查询相关的文档。\n\n"
        else:
            content += "未找到与查询相关的文档。\n\n"
        
        content += "**建议:**\n"
        content += "- 尝试使用不同的关键词\n"
        content += "- 检查是否已摄取相关文档\n"
        content += "- 扩大搜索范围（如不指定 collection）\n"
        
        metadata = self._build_metadata(query, collection, 0)
        
        return MCPToolResponse(
            content=content,
            citations=[],
            metadata=metadata,
            is_empty=True,
        )
    
    def _build_markdown_content(
        self,
        results: List[RetrievalResult],
        citations: List[Citation],
        query: str,
    ) -> str:
        """构建带内联引用的 Markdown 内容。
        
        参数：
            results: RetrievalResult 列表。
            citations: Citation 对象列表。
            query: 原始查询字符串。
            
        返回：
            格式化后的 Markdown 字符串。
        """
        lines = []
        
        # 标题
        lines.append(f"## 检索结果\n")
        lines.append(f"针对查询 **\"{query}\"** 找到 {len(results)} 条相关结果:\n")
        
        # 结果区
        display_count = min(len(results), self.max_results_in_content)
        
        for i, (result, citation) in enumerate(zip(results[:display_count], citations[:display_count])):
            marker = self.citation_generator.format_citation_marker(citation.index)
            
            # 格式化单条结果
            lines.append(f"### {marker} 结果 {citation.index}")
            lines.append(f"**相关度:** {citation.score:.2%}")
            lines.append(f"**来源:** `{citation.source}`")
            
            if citation.page is not None:
                lines.append(f"**页码:** {citation.page}")
            
            # 内容摘要
            snippet = self._truncate_text(result.text, self.snippet_max_length)
            lines.append(f"\n> {snippet}\n")
        
        # 额外结果提示
        if len(results) > display_count:
            remaining = len(results) - display_count
            lines.append(f"\n*...还有 {remaining} 条结果未显示*\n")
        
        # 引用来源区
        lines.append("\n---\n")
        lines.append("## 引用来源\n")
        
        for citation in citations:
            source_info = f"`{citation.source}`"
            if citation.page is not None:
                source_info += f" (p.{citation.page})"
            lines.append(f"- [{citation.index}] {source_info}")
        
        return "\n".join(lines)
    
    def _build_metadata(
        self,
        query: str,
        collection: Optional[str],
        result_count: int,
    ) -> Dict[str, Any]:
        """构建响应元数据。
        
        参数：
            query: 原始查询。
            collection: 集合名称。
            result_count: 结果数量。
            
        返回：
            元数据字典。
        """
        metadata = {
            "query": query,
            "result_count": result_count,
        }
        if collection:
            metadata["collection"] = collection
        return metadata
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """将文本截断到最大长度。
        
        参数：
            text: 要截断的文本。
            max_length: 最大字符数。
            
        返回：
            截断后的文本，必要时末尾加省略号。
        """
        if not text:
            return ""
        
        # 清理空白
        cleaned = " ".join(text.split())
        
        if len(cleaned) <= max_length:
            return cleaned
        
        # 在单词边界处截断
        truncated = cleaned[:max_length].rsplit(" ", 1)[0]
        return truncated + "..."
