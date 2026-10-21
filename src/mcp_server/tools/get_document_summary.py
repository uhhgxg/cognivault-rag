"""MCP 工具：get_document_summary

本工具通过 MCP 协议提供文档摘要检索能力。
它根据 doc_id 返回指定文档的标题、摘要和标签。

通过 MCP 使用：
    工具名：get_document_summary
    输入 schema：
        - doc_id（字符串，必填）：要检索摘要的文档 ID
        - collection（字符串，可选）：要搜索的集合名
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from mcp import types

if TYPE_CHECKING:
    from src.mcp_server.protocol_handler import ProtocolHandler
    from src.core.settings import Settings

logger = logging.getLogger(__name__)


# 工具元数据
TOOL_NAME = "get_document_summary"
TOOL_DESCRIPTION = """Get summary and metadata for a specific document.

Returns structured information about a document including:
- Title (extracted or inferred from content)
- Summary (first chunk preview or metadata summary)
- Tags (document-level tags/categories)
- Source path
- Chunk count

Use this tool after list_collections to get details about specific documents.
"""

TOOL_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "doc_id": {
            "type": "string",
            "description": "The document ID to retrieve summary for. Can be full doc_id (e.g., 'doc_abc123') or the hash portion.",
        },
        "collection": {
            "type": "string",
            "description": "Collection name to search in. If not specified, searches the default collection.",
        },
    },
    "required": ["doc_id"],
}


@dataclass
class DocumentSummary:
    """文档的摘要信息。

    属性：
        doc_id: 文档标识符
        title: 文档标题（来自元数据或推断得到）
        summary: 文档内容的简要摘要或预览
        tags: 文档关联的标签/分类列表
        source_path: 原始文件路径
        chunk_count: 该文档的分块数量
        metadata: 额外的文档元数据
    """
    doc_id: str
    title: str
    summary: str
    tags: List[str] = field(default_factory=list)
    source_path: Optional[str] = None
    chunk_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转成字典表示。"""
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "summary": self.summary,
            "tags": self.tags,
            "source_path": self.source_path,
            "chunk_count": self.chunk_count,
            "metadata": self.metadata,
        }


@dataclass
class GetDocumentSummaryConfig:
    """get_document_summary 工具的配置。

    属性：
        persist_directory: ChromaDB 存储目录路径
        default_collection: 未指定时的默认集合名
        summary_max_length: 摘要预览的最大字符数
    """
    persist_directory: str = "./data/db/chroma"
    default_collection: str = "knowledge_hub"
    summary_max_length: int = 500


class DocumentNotFoundError(Exception):
    """找不到指定 ID 的文档时抛出。"""
    
    def __init__(self, doc_id: str, collection: Optional[str] = None):
        self.doc_id = doc_id
        self.collection = collection
        message = f"Document '{doc_id}' not found"
        if collection:
            message += f" in collection '{collection}'"
        super().__init__(message)


class GetDocumentSummaryTool:
    """用于检索文档摘要的 MCP 工具。

    这个类封装了 get_document_summary 工具的逻辑，
    查询向量库来获取文档元数据和内容预览。

    设计原则：
    - 配置驱动：路径来自 settings.yaml
    - 容错性：文档缺失时给出清晰的错误信息
    - 可观测：记日志方便调试
    - 懒加载：ChromaDB 客户端在第一次使用时才创建

    示例：
        >>> tool = GetDocumentSummaryTool(settings)
        >>> result = await tool.execute(doc_id="doc_abc123")
        >>> print(result)
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        config: Optional[GetDocumentSummaryConfig] = None,
    ) -> None:
        """初始化 GetDocumentSummaryTool。

        参数：
            settings: 应用配置。为 None 时从默认路径加载。
            config: 工具配置。为 None 时从 settings 派生。
        """
        self._settings = settings
        self._config = config
        self._chroma_client = None

    @property
    def settings(self) -> Settings:
        """获取配置，必要时加载。"""
        if self._settings is None:
            from src.core.settings import load_settings
            self._settings = load_settings()
        return self._settings

    @property
    def config(self) -> GetDocumentSummaryConfig:
        """获取配置，必要时从 settings 派生。"""
        if self._config is None:
            try:
                persist_dir = getattr(
                    self.settings.vector_store,
                    'persist_directory',
                    './data/db/chroma'
                )
                default_collection = getattr(
                    self.settings.vector_store,
                    'collection_name',
                    'knowledge_hub'
                )
            except AttributeError:
                persist_dir = './data/db/chroma'
                default_collection = 'knowledge_hub'
            
            self._config = GetDocumentSummaryConfig(
                persist_directory=persist_dir,
                default_collection=default_collection,
            )
        return self._config
    
    def _get_chroma_client(self) -> Any:
        """获取或创建 ChromaDB 客户端。

        返回：
            ChromaDB PersistentClient 实例。

        可能抛出：
            ImportError: chromadb 未安装时抛出。
            RuntimeError: 创建客户端失败时抛出。
        """
        if self._chroma_client is not None:
            return self._chroma_client
        
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError:
            raise ImportError(
                "chromadb package is required for get_document_summary. "
                "Install it with: pip install chromadb"
            )
        
        persist_path = Path(self.config.persist_directory).resolve()
        
        if not persist_path.exists():
            logger.warning(f"ChromaDB directory does not exist: {persist_path}")
            persist_path.mkdir(parents=True, exist_ok=True)
        
        try:
            self._chroma_client = chromadb.PersistentClient(
                path=str(persist_path),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                )
            )
            return self._chroma_client
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize ChromaDB client at '{persist_path}': {e}"
            ) from e
    
    def _get_collection(self, collection_name: Optional[str] = None) -> Any:
        """获取 ChromaDB 集合。

        参数：
            collection_name: 集合名。未指定时用默认值。

        返回：
            ChromaDB 集合实例。

        可能抛出：
            ValueError: 集合不存在时抛出。
        """
        client = self._get_chroma_client()
        name = collection_name or self.config.default_collection

        try:
            # 尝试获取已存在的集合
            collection = client.get_collection(name=name)
            return collection
        except Exception as e:
            raise ValueError(
                f"Collection '{name}' does not exist: {e}"
            ) from e
    
    def _find_document_chunks(
        self,
        doc_id: str,
        collection_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """查找属于某个文档的所有分块。

        搜索 source_ref 与 doc_id 匹配的分块。
        如果 source_ref 不可用，则退而用分块 ID 的部分匹配。

        参数：
            doc_id: 要搜索的文档 ID。
            collection_name: 要搜索的集合。

        返回：
            带元数据的分块数据列表。
        """
        collection = self._get_collection(collection_name)

        # 策略 1：按 source_ref 元数据搜索
        # 分块应该有 source_ref 指向父文档
        try:
            results = collection.get(
                where={"source_ref": doc_id},
                include=["metadatas", "documents"]
            )
            
            if results and results.get('ids'):
                chunks = []
                for i, chunk_id in enumerate(results['ids']):
                    chunks.append({
                        'id': chunk_id,
                        'text': results['documents'][i] if results.get('documents') else '',
                        'metadata': results['metadatas'][i] if results.get('metadatas') else {}
                    })
                if chunks:
                    return chunks
        except Exception as e:
            logger.debug(f"source_ref search failed: {e}")
        
        # 策略 2：按分块 ID 前缀中的 doc_id 搜索
        # 分块 ID 格式：{doc_id}_{index:04d}_{hash}
        try:
            # 获取所有分块，按 ID 前缀过滤
            all_results = collection.get(include=["metadatas", "documents"])
            
            if all_results and all_results.get('ids'):
                chunks = []
                for i, chunk_id in enumerate(all_results['ids']):
                    # 检查 chunk_id 是否以 doc_id 开头
                    if chunk_id.startswith(doc_id) or doc_id in chunk_id:
                        chunks.append({
                            'id': chunk_id,
                            'text': all_results['documents'][i] if all_results.get('documents') else '',
                            'metadata': all_results['metadatas'][i] if all_results.get('metadatas') else {}
                        })
                if chunks:
                    return chunks
        except Exception as e:
            logger.debug(f"ID prefix search failed: {e}")
        
        # 没找到分块
        return []
    
    def get_document_summary(
        self,
        doc_id: str,
        collection: Optional[str] = None,
    ) -> DocumentSummary:
        """获取指定文档的摘要。

        参数：
            doc_id: 要检索的文档 ID。
            collection: 要搜索的集合名。

        返回：
            带标题、摘要、标签等的 DocumentSummary。

        可能抛出：
            DocumentNotFoundError: 文档不存在时抛出。
        """
        chunks = self._find_document_chunks(doc_id, collection)

        if not chunks:
            raise DocumentNotFoundError(doc_id, collection)

        # 如果有 chunk_index 就按它排序
        chunks.sort(key=lambda c: c.get('metadata', {}).get('chunk_index', 0))

        # 从第一个分块的元数据提取文档级信息
        first_chunk = chunks[0]
        metadata = first_chunk.get('metadata', {})

        # 提取标题
        title = self._extract_title(metadata, first_chunk.get('text', ''))

        # 提取或生成摘要
        summary = self._extract_summary(chunks)

        # 提取标签
        tags = self._extract_tags(metadata)

        # 提取源文件路径
        source_path = metadata.get('source_path', metadata.get('source', None))

        # 收集额外的元数据（排除内部字段）
        additional_metadata = self._filter_metadata(metadata)
        
        return DocumentSummary(
            doc_id=doc_id,
            title=title,
            summary=summary,
            tags=tags,
            source_path=source_path,
            chunk_count=len(chunks),
            metadata=additional_metadata,
        )
    
    def _extract_title(self, metadata: Dict[str, Any], first_text: str) -> str:
        """从元数据或内容中提取文档标题。

        优先级：
        1. metadata['title']
        2. 内容中的第一个标题
        3. source_path 里的文件名
        4. "Untitled Document"

        参数：
            metadata: 分块元数据。
            first_text: 第一个分块的文本内容。

        返回：
            提取或推断出的标题。
        """
        # 优先级 1：元数据里有显式标题
        if metadata.get('title'):
            return str(metadata['title'])

        # 优先级 2：第一个 Markdown 标题
        if first_text:
            lines = first_text.split('\n')
            for line in lines[:10]:  # 检查前 10 行
                line = line.strip()
                if line.startswith('# '):
                    return line[2:].strip()

        # 优先级 3：source_path 里的文件名
        source_path = metadata.get('source_path', metadata.get('source'))
        if source_path:
            filename = Path(source_path).stem
            # 把 snake_case/kebab-case 转成 Title Case
            title = filename.replace('_', ' ').replace('-', ' ').title()
            return title

        # 优先级 4：默认值
        return "Untitled Document"
    
    def _extract_summary(self, chunks: List[Dict[str, Any]]) -> str:
        """提取或生成文档摘要。

        优先级：
        1. 任一分块的 metadata['summary']
        2. 第一个分块文本的前 N 个字符

        参数：
            chunks: 文档分块列表。

        返回：
            摘要文本。
        """
        # 优先级 1：元数据里有显式摘要
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            if metadata.get('summary'):
                return str(metadata['summary'])

        # 优先级 2：第一个分块的预览
        first_text = chunks[0].get('text', '') if chunks else ''
        if first_text:
            # 清理并截断
            summary = first_text.strip()

            # 预览时跳过 Markdown 标题
            lines = summary.split('\n')
            content_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    content_lines.append(line)
                    if len(' '.join(content_lines)) > self.config.summary_max_length:
                        break
            
            summary = ' '.join(content_lines)
            
            if len(summary) > self.config.summary_max_length:
                summary = summary[:self.config.summary_max_length - 3] + "..."
            
            return summary if summary else "No content preview available."
        
        return "No summary available."
    
    def _extract_tags(self, metadata: Dict[str, Any]) -> List[str]:
        """从元数据中提取标签。

        参数：
            metadata: 分块元数据。

        返回：
            标签列表。
        """
        tags = []

        # 检查是否有显式 tags 字段
        if 'tags' in metadata:
            tag_value = metadata['tags']
            if isinstance(tag_value, list):
                tags.extend(str(t) for t in tag_value)
            elif isinstance(tag_value, str):
                # 可能是逗号分隔的
                tags.extend(t.strip() for t in tag_value.split(',') if t.strip())

        # 如果有 doc_type 就把它也加成标签
        if metadata.get('doc_type'):
            doc_type = str(metadata['doc_type']).upper()
            if doc_type not in tags:
                tags.append(doc_type)
        
        return tags
    
    def _filter_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """过滤元数据，排除内部字段。

        参数：
            metadata: 原始元数据字典。

        返回：
            只含用户相关字段的过滤后元数据。
        """
        # 要从额外元数据中排除的字段
        exclude_fields = {
            'source_ref', 'chunk_index', 'start_offset', 'end_offset',
            '_placeholder', 'text', 'title', 'summary', 'tags',
            'source_path', 'source'
        }
        
        return {
            k: v for k, v in metadata.items()
            if k not in exclude_fields and not k.startswith('_')
        }
    
    def format_response(self, summary: DocumentSummary) -> str:
        """把文档摘要格式化成可读的字符串。

        参数：
            summary: DocumentSummary 对象。

        返回：
            适合 MCP 响应的格式化字符串。
        """
        lines = [
            f"## Document: {summary.title}",
            "",
            f"**Document ID:** `{summary.doc_id}`",
        ]
        
        if summary.source_path:
            lines.append(f"**Source:** {summary.source_path}")
        
        lines.append(f"**Chunks:** {summary.chunk_count}")
        
        if summary.tags:
            tags_str = ", ".join(f"`{tag}`" for tag in summary.tags)
            lines.append(f"**Tags:** {tags_str}")
        
        lines.extend([
            "",
            "### Summary",
            "",
            summary.summary,
        ])
        
        if summary.metadata:
            lines.extend([
                "",
                "### Additional Metadata",
                "",
            ])
            for key, value in summary.metadata.items():
                lines.append(f"- **{key}:** {value}")
        
        return "\n".join(lines)
    
    def format_error(self, error: Exception) -> str:
        """把错误格式化成可读的字符串。

        参数：
            error: 发生的异常。

        返回：
            格式化后的错误信息。
        """
        if isinstance(error, DocumentNotFoundError):
            return f"## Document Not Found\n\n{str(error)}\n\nPlease verify the document ID and collection name."
        elif isinstance(error, ValueError):
            return f"## Invalid Request\n\n{str(error)}"
        else:
            return f"## Error\n\nAn error occurred: {str(error)}"
    
    async def execute(
        self,
        doc_id: str,
        collection: Optional[str] = None,
    ) -> types.CallToolResult:
        """执行 get_document_summary 工具。

        参数：
            doc_id: 要检索摘要的文档 ID。
            collection: 可选的集合名。

        返回：
            带格式化文档摘要或错误的 CallToolResult。
        """
        logger.info(f"Executing get_document_summary (doc_id={doc_id}, collection={collection})")

        try:
            # 把阻塞式 ChromaDB I/O 放到线程里跑，避免阻塞
            # 异步事件循环 / MCP stdio 传输
            summary = await asyncio.to_thread(
                self.get_document_summary, doc_id, collection,
            )
            response_text = self.format_response(summary)
            
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=response_text,
                    )
                ],
                isError=False,
            )
            
        except DocumentNotFoundError as e:
            logger.warning(f"Document not found: {e}")
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=self.format_error(e),
                    )
                ],
                isError=True,
            )
            
        except Exception as e:
            logger.exception("Error executing get_document_summary")
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=self.format_error(e),
                    )
                ],
                isError=True,
            )


def register_tool(protocol_handler: ProtocolHandler) -> None:
    """向协议处理器注册 get_document_summary 工具。

    这个函数由 protocol_handler.py 里的 _register_default_tools() 调用，
    在 MCP 服务启动时注册本工具。

    参数：
        protocol_handler: 要注册到的 ProtocolHandler 实例。
    """
    tool = GetDocumentSummaryTool()
    
    async def handler(
        doc_id: str,
        collection: Optional[str] = None,
    ) -> types.CallToolResult:
        """MCP 工具调用的 handler 函数。"""
        return await tool.execute(doc_id=doc_id, collection=collection)
    
    protocol_handler.register_tool(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        input_schema=TOOL_INPUT_SCHEMA,
        handler=handler,
    )
    
    logger.info(f"Registered MCP tool: {TOOL_NAME}")
