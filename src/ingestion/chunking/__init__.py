"""切块模块 - 文档切分的适配层。

本模块提供文本切分的业务适配器，把
Document 对象转换成带有完整元数据和可追溯信息的 Chunk 对象。
"""

from src.ingestion.chunking.document_chunker import DocumentChunker

__all__ = ["DocumentChunker"]
