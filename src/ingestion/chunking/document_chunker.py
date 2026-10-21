"""文档切块模块 - 为业务层适配 libs.splitter。

本模块作为 libs.splitter（纯文本切分）和摄取流水线（业务对象转换）之间的适配层。
它把 Document 对象转换成 Chunk 对象，同时完成 ID 生成、元数据继承和可追溯信息处理。

核心价值（相比 libs.splitter 的增强）：
1. Chunk ID 生成：为每个分块生成确定性且唯一的 ID
2. 元数据继承：把 Document 的元数据传递给所有分块
3. chunk_index：记录分块在文档中的顺序位置
4. source_ref：建立父子文档的可追溯关系
5. 类型转换：str → Chunk 对象（遵循 core.types 约定）

设计原则：
- 适配器模式：把文本切分工具和业务对象连接起来
- 配置驱动：通过 SplitterFactory 基于配置选择切分策略
- 确定性：同一个 Document 重复切分会得到相同的 Chunk ID
- 类型安全：强制遵守 core.types.Chunk 约定
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, List

from src.core.types import Chunk, Document
from src.libs.splitter.splitter_factory import SplitterFactory

if TYPE_CHECKING:
    from src.core.settings import Settings


class DocumentChunker:
    """把 Document 转换成 Chunk，并加上业务级别的增强信息。

    这个类封装了 libs 层的文本切分器，并加入业务逻辑：
    - 生成稳定的分块 ID
    - 继承并扩展元数据
    - 维护文档的可追溯关系

    属性：
        _splitter: 底层来自 libs 层的文本切分器
        _settings: 控制切分行为的配置项

    示例：
        >>> from src.core.settings import load_settings
        >>> from src.core.types import Document
        >>> settings = load_settings("config/settings.yaml")
        >>> chunker = DocumentChunker(settings)
        >>> document = Document(
        ...     id="doc_123",
        ...     text="Long document content...",
        ...     metadata={"source_path": "data/report.pdf"}
        ... )
        >>> chunks = chunker.split_document(document)
        >>> print(f"Generated {len(chunks)} chunks")
        >>> print(f"First chunk ID: {chunks[0].id}")
        >>> print(f"First chunk index: {chunks[0].metadata['chunk_index']}")
    """
    
    def __init__(self, settings: Settings):
        """用配置初始化 DocumentChunker。

        参数：
            settings: 包含切分器配置的配置项。
                     切分器配置应在 settings.splitter.* 下。

        可能抛出：
            ValueError: 如果切分器配置无效或 provider 未知
        """
        self._settings = settings
        self._splitter = SplitterFactory.create(settings)
    
    def split_document(self, document: Document) -> List[Chunk]:
        """把 Document 切分成 Chunk，并完成完整的业务增强。

        这是主入口，负责编排整个转换流程：
        1. 用底层切分器得到文本片段
        2. 为每个分块生成确定性 ID
        3. 从文档继承并扩展元数据
        4. 创建符合 core.types 约定的 Chunk 对象

        参数：
            document: 要切分的源文档

        返回：
            Chunk 对象列表，每个分块包含：
            - 唯一、确定性的 ID
            - 继承的元数据 + chunk_index + source_ref
            - 正确的类型约定（core.types.Chunk）

        可能抛出：
            ValueError: 如果文档没有文本或结构无效

        示例：
            >>> doc = Document(
            ...     id="doc_abc",
            ...     text="Section 1 content.\\n\\nSection 2 content.",
            ...     metadata={"source_path": "file.pdf", "title": "Report"}
            ... )
            >>> chunker = DocumentChunker(settings)
            >>> chunks = chunker.split_document(doc)
            >>> len(chunks) >= 1
            True
            >>> chunks[0].metadata["source_path"]
            'file.pdf'
            >>> chunks[0].metadata["chunk_index"]
            0
            >>> chunks[0].metadata["source_ref"]
            'doc_abc'
        """
        if not document.text or not document.text.strip():
            raise ValueError(f"Document {document.id} has no text content to split")
        
        # 第 1 步：用底层切分器得到文本片段
        # 先切割成文本块儿
        text_fragments = self._splitter.split_text(document.text)
        
        if not text_fragments:
            raise ValueError(
                f"Splitter returned no chunks for document {document.id}. "
                f"Text length: {len(document.text)}"
            )
        
        # 第 2 步：把文本片段转换成带增强信息的 Chunk 对象
        chunks: List[Chunk] = []
        for index, text in enumerate(text_fragments):
            chunk_id = self._generate_chunk_id(document.id, index, text)
            chunk_metadata = self._inherit_metadata(document, index, text)
            
            chunk = Chunk(
                id=chunk_id,
                text=text,
                metadata=chunk_metadata
            )
            chunks.append(chunk)
        
        return chunks
    
    def _generate_chunk_id(self, doc_id: str, index: int, text: str) -> str:
        """生成唯一且确定性的分块 ID。

        ID 格式：{doc_id}_{index:04d}_{content_hash}
        - doc_id：父文档标识
        - index：顺序位置（补零到 4 位）
        - content_hash：文本 SHA256 哈希的前 8 个字符

        这样设计保证：
        - 唯一性：doc_id + index + content_hash 组合确保唯一
        - 确定性：相同输入总是产生相同 ID
        - 可调试：结构对人友好可读

        参数：
            doc_id：父文档 ID
            index：分块的顺序位置（从 0 开始）
            text：分块文本内容

        返回：
            唯一的分块 ID 字符串

        示例：
            >>> chunker._generate_chunk_id("doc_123", 0, "Hello world")
            'doc_123_0000_c0535e4b'
        """
        # 计算内容哈希以保证唯一性
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        
        # 格式：{doc_id}_{index:04d}_{hash_8chars}
        return f"{doc_id}_{index:04d}_{content_hash}"
    
    def _inherit_metadata(self, document: Document, chunk_index: int, chunk_text: str = "") -> dict:
        """从文档继承元数据，并加上分块专属字段。

        这里会创建一个新的元数据字典，包含：
        - document.metadata 的所有字段（复制，不是引用）
        - chunk_index：顺序位置（从 0 开始）
        - source_ref：指向父文档 ID 的引用
        - image_refs：本分块中引用的图片 ID 列表（从占位符中提取）

        注意：文档级的 'images' 字段不会放进分块元数据，因为会重复。
        取而代之的是根据分块文本中的 [IMAGE: xxx] 占位符来填充分块专属的 'image_refs'。

        参数：
            document: 要继承元数据的源文档
            chunk_index: 本分块的顺序位置
            chunk_text: 本分块的文本内容（用于提取 image_refs）

        返回：
            包含继承字段和分块专属字段的元数据字典

        示例：
            >>> doc = Document(
            ...     id="doc_123",
            ...     text="Content",
            ...     metadata={"source_path": "file.pdf", "title": "Report"}
            ... )
            >>> metadata = chunker._inherit_metadata(doc, 2, "See [IMAGE: img_001]")
            >>> metadata["source_path"]
            'file.pdf'
            >>> metadata["chunk_index"]
            2
            >>> metadata["source_ref"]
            'doc_123'
            >>> metadata["image_refs"]
            ['img_001']
        """
        import re
        
        # 复制文档的所有元数据（对于基本类型，浅拷贝就够了）
        chunk_metadata = document.metadata.copy()
        
        # 取出文档级的图片列表，用于后续查找
        doc_images = document.metadata.get("images", [])
        
        # 移除文档级的 'images' 字段 - 下面会加上分块专属的图片信息
        chunk_metadata.pop("images", None)
        
        # 添加分块专属字段
        chunk_metadata["chunk_index"] = chunk_index
        chunk_metadata["source_ref"] = document.id
        
        # 从分块文本中查找 [IMAGE: xxx] 占位符，提取 image_refs
        image_refs = []
        if chunk_text:
            # 匹配 [IMAGE: image_id] 占位符的正则
            pattern = r'\[IMAGE:\s*([^\]]+)\]'
            matches = re.findall(pattern, chunk_text)
            image_refs = [m.strip() for m in matches]
        
        chunk_metadata["image_refs"] = image_refs
        
        # 为被引用的图片构建带完整元数据的分块专属 'images' 列表
        # ImageCaptioner 需要通过它获取图片路径来调用 Vision API
        chunk_images = []
        if image_refs and doc_images:
            image_lookup = {img.get("id"): img for img in doc_images}
            for img_id in image_refs:
                if img_id in image_lookup:
                    chunk_images.append(image_lookup[img_id])
        
        if chunk_images:
            chunk_metadata["images"] = chunk_images
        
        # 尝试从第一个被引用的图片推断 page_num
        if chunk_images:
            chunk_metadata["page_num"] = chunk_images[0].get("page")
        
        return chunk_metadata
