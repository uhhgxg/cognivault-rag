"""MCP 工具：list_collections

本工具通过 MCP 协议提供集合列表能力。
它会列出向量库中所有可用集合及其统计信息。

通过 MCP 使用：
    工具名：list_collections
    输入 schema：
        - include_stats（布尔，可选）：是否包含每个集合的统计信息
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from mcp import types

if TYPE_CHECKING:
    from src.mcp_server.protocol_handler import ProtocolHandler
    from src.core.settings import Settings

logger = logging.getLogger(__name__)


# 工具元数据
TOOL_NAME = "list_collections"
TOOL_DESCRIPTION = """List all available document collections in the knowledge base.

Returns information about each collection including:
- Collection name
- Document count (if include_stats=true)
- Collection metadata

Use this tool to discover available collections before querying.
"""

TOOL_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "include_stats": {
            "type": "boolean",
            "description": "Whether to include statistics (document count) for each collection.",
            "default": True,
        },
    },
    "required": [],
}


@dataclass
class CollectionInfo:
    """单个集合的信息。

    属性：
        name: 集合名
        count: 集合中的文档/分块数量（可选）
        metadata: 集合的元数据字典
    """
    name: str
    count: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转成字典表示。"""
        result: Dict[str, Any] = {"name": self.name}
        if self.count is not None:
            result["count"] = self.count
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class ListCollectionsConfig:
    """list_collections 工具的配置。

    属性：
        persist_directory: ChromaDB 存储目录路径
        include_stats_default: include_stats 参数的默认值
    """
    persist_directory: str = "./data/db/chroma"
    include_stats_default: bool = True


class ListCollectionsTool:
    """用于列出知识库集合的 MCP 工具。

    这个类封装了 list_collections 工具的逻辑，
    查询向量库来枚举可用的集合。

    设计原则：
    - 配置驱动：路径来自 settings.yaml
    - 容错性：目录缺失时优雅处理
    - 可观测：记日志方便调试

    示例：
        >>> tool = ListCollectionsTool(settings)
        >>> result = await tool.execute(include_stats=True)
        >>> print(result)
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        config: Optional[ListCollectionsConfig] = None,
    ) -> None:
        """初始化 ListCollectionsTool。

        参数：
            settings: 应用配置。为 None 时从默认路径加载。
            config: 工具配置。为 None 时从 settings 派生。
        """
        self._settings = settings
        self._config = config

    @property
    def settings(self) -> Settings:
        """获取配置，必要时加载。"""
        if self._settings is None:
            from src.core.settings import load_settings
            self._settings = load_settings()
        return self._settings

    @property
    def config(self) -> ListCollectionsConfig:
        """获取配置，必要时从 settings 派生。"""
        if self._config is None:
            try:
                persist_dir = getattr(
                    self.settings.vector_store,
                    'persist_directory',
                    './data/db/chroma'
                )
            except AttributeError:
                persist_dir = './data/db/chroma'
            
            self._config = ListCollectionsConfig(
                persist_directory=persist_dir
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
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError:
            raise ImportError(
                "chromadb package is required for list_collections. "
                "Install it with: pip install chromadb"
            )
        
        persist_path = Path(self.config.persist_directory).resolve()
        
        if not persist_path.exists():
            logger.warning(f"ChromaDB directory does not exist: {persist_path}")
            # 照样返回客户端 —— 它只是不会有任何集合
            persist_path.mkdir(parents=True, exist_ok=True)
        
        try:
            client = chromadb.PersistentClient(
                path=str(persist_path),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                )
            )
            return client
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize ChromaDB client at '{persist_path}': {e}"
            ) from e
    
    def list_collections(
        self,
        include_stats: bool = True
    ) -> List[CollectionInfo]:
        """列出所有可用集合。

        参数：
            include_stats: 是否包含文档数量。

        返回：
            CollectionInfo 对象列表。
        """
        try:
            client = self._get_chroma_client()
        except (ImportError, RuntimeError) as e:
            logger.error(f"Failed to get ChromaDB client: {e}")
            return []
        
        collections_info: List[CollectionInfo] = []
        
        try:
            # 从 ChromaDB 获取所有集合
            collections = client.list_collections()
            
            for collection in collections:
                info = CollectionInfo(
                    name=collection.name,
                    metadata=collection.metadata
                )
                
                if include_stats:
                    try:
                        info.count = collection.count()
                    except Exception as e:
                        logger.warning(
                            f"Failed to get count for collection '{collection.name}': {e}"
                        )
                        info.count = None
                
                collections_info.append(info)
                
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
        
        logger.info(f"Found {len(collections_info)} collections")
        return collections_info
    
    def format_response(
        self,
        collections: List[CollectionInfo]
    ) -> str:
        """把集合列表格式化成可读的字符串。

        参数：
            collections: CollectionInfo 对象列表。

        返回：
            适合 MCP 响应的格式化字符串。
        """
        if not collections:
            return "No collections found in the knowledge base."
        
        lines = [
            f"## Available Collections ({len(collections)} total)\n"
        ]
        
        for i, coll in enumerate(collections, 1):
            line = f"{i}. **{coll.name}**"
            
            if coll.count is not None:
                line += f" - {coll.count} documents"
            
            if coll.metadata:
                # 过滤掉内部元数据
                user_metadata = {
                    k: v for k, v in coll.metadata.items()
                    if not k.startswith('_') and not k.startswith('hnsw:')
                }
                if user_metadata:
                    meta_str = ", ".join(f"{k}={v}" for k, v in user_metadata.items())
                    line += f" ({meta_str})"
            
            lines.append(line)
        
        return "\n".join(lines)
    
    async def execute(
        self,
        include_stats: bool = True,
    ) -> types.CallToolResult:
        """执行 list_collections 工具。

        参数：
            include_stats: 是否包含每个集合的统计信息。

        返回：
            带格式化集合列表的 CallToolResult。
        """
        logger.info(f"Executing list_collections (include_stats={include_stats})")
        
        try:
            # 把阻塞式 ChromaDB I/O 放到线程里跑，避免阻塞
            # 异步事件循环 / MCP stdio 传输
            collections = await asyncio.to_thread(
                self.list_collections, include_stats,
            )
            response_text = self.format_response(collections)
            
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=response_text,
                    )
                ],
                isError=False,
            )
            
        except Exception as e:
            logger.exception("Error executing list_collections")
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=f"Error listing collections: {str(e)}",
                    )
                ],
                isError=True,
            )


def register_tool(protocol_handler: ProtocolHandler) -> None:
    """向协议处理器注册 list_collections 工具。

    这个函数由 protocol_handler.py 里的 _register_default_tools() 调用，
    在 MCP 服务启动时注册本工具。

    参数：
        protocol_handler: 要注册到的 ProtocolHandler 实例。
    """
    tool = ListCollectionsTool()
    
    async def handler(
        include_stats: bool = True,
    ) -> types.CallToolResult:
        """MCP 工具调用的 handler 函数。"""
        return await tool.execute(include_stats=include_stats)
    
    protocol_handler.register_tool(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        input_schema=TOOL_INPUT_SCHEMA,
        handler=handler,
    )
    
    logger.info(f"Registered MCP tool: {TOOL_NAME}")
