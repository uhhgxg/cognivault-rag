"""多模态装配器，用于组合文本和图片内容。

本模块通过以下方式装配多模态 MCP 响应：
- 检测 chunk 元数据中的图片引用
- 读取图片文件并编码为 base64
- 构建 MCP 兼容的 ImageContent 块

设计原则：
- 懒加载：仅在明确请求时才加载图片
- 容错性：图片缺失不会破坏文本响应
- 格式检测：根据文件内容自动推断 MIME 类型
"""

from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from mcp import types

from src.core.types import RetrievalResult


logger = logging.getLogger(__name__)


# 支持的图片格式及其 MIME 类型
MIME_TYPE_MAP: Dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".svg": "image/svg+xml",
}

# 用于格式检测的魔数字节（当扩展名不可靠时的兜底方案）
MAGIC_BYTES: Dict[bytes, str] = {
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
    b"GIF87a": "image/gif",
    b"GIF89a": "image/gif",
    b"RIFF": "image/webp",  # WebP 以 RIFF 开头
    b"BM": "image/bmp",
}

# 文本中的图片占位符模式：[IMAGE: image_id]
IMAGE_PLACEHOLDER_PATTERN = re.compile(r"\[IMAGE:\s*([^\]]+)\]")


@dataclass
class ImageReference:
    """指向 chunk 中某张图片的引用。
    
    属性：
        image_id: 图片的唯一标识
        file_path: 图片文件的文件系统路径
        page: 源文档中的页码（可选）
        text_offset: 占位符在 chunk 文本中的字符偏移量
        text_length: 占位符字符串的长度
        caption: 由 ImageCaptioner 生成的可选图片说明
    """
    image_id: str
    file_path: Optional[str] = None
    page: Optional[int] = None
    text_offset: Optional[int] = None
    text_length: Optional[int] = None
    caption: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "image_id": self.image_id,
            "file_path": self.file_path,
            "page": self.page,
            "text_offset": self.text_offset,
            "text_length": self.text_length,
            "caption": self.caption,
        }


@dataclass
class ImageContent:
    """已加载、可用于 MCP 响应的图片内容。
    
    属性：
        image_id: 图片的唯一标识
        data: base64 编码的图片数据
        mime_type: MIME 类型（例如 "image/png"）
        caption: 图片的可选说明
    """
    image_id: str
    data: str  # base64 编码
    mime_type: str
    caption: Optional[str] = None
    
    def to_mcp_content(self) -> types.ImageContent:
        """转换为 MCP ImageContent 块。
        
        返回：
            用于协议响应的 MCP ImageContent 对象。
        """
        return types.ImageContent(
            type="image",
            data=self.data,
            mimeType=self.mime_type,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "image_id": self.image_id,
            "data": self.data,
            "mime_type": self.mime_type,
            "caption": self.caption,
        }


class MultimodalAssembler:
    """从检索结果装配多模态内容。
    
    本类从 chunk 元数据中提取图片引用，
    加载图片文件，并构建组合文本和图片的 MCP 兼容内容块。
    
    示例：
        >>> assembler = MultimodalAssembler()
        >>> results = [RetrievalResult(chunk_id="doc1_001", ...)]
        >>> content_blocks = assembler.assemble(results)
        >>> # 返回 TextContent 和 ImageContent 块的列表
    
    参数：
        image_storage: 可选的 ImageStorage 实例，用于路径查找。
            如果为 None，则直接使用元数据中的文件路径。
        max_images_per_result: 每条结果最多包含的图片数（默认：5）。
        include_captions: 是否将图片说明作为文本包含（默认：True）。
    """
    
    def __init__(
        self,
        image_storage: Optional[Any] = None,
        max_images_per_result: int = 5,
        include_captions: bool = True,
    ) -> None:
        """初始化 MultimodalAssembler。
        
        参数：
            image_storage: 可选的 ImageStorage，用于解析图片路径。
            max_images_per_result: 每条检索结果最多包含的图片数。
            include_captions: 是否添加说明文本块。
        """
        self._image_storage = image_storage
        self.max_images_per_result = max_images_per_result
        self.include_captions = include_captions
    
    def extract_image_refs(
        self,
        result: RetrievalResult,
    ) -> List[ImageReference]:
        """从检索结果中提取图片引用。
        
        在以下位置查找图片引用：
        1. metadata.images 列表（结构化的图片信息）
        2. 文本中的 [IMAGE: id] 占位符（兜底方案）
        
        参数：
            result: 包含 chunk 数据的 RetrievalResult。
            
        返回：
            在结果中找到的 ImageReference 对象列表。
        """
        refs: List[ImageReference] = []
        metadata = result.metadata or {}
        
        # 主要来源：元数据中的结构化图片列表
        images_list = metadata.get("images", [])
        if isinstance(images_list, list):
            for img_info in images_list[:self.max_images_per_result]:
                if isinstance(img_info, dict) and "id" in img_info:
                    ref = ImageReference(
                        image_id=img_info["id"],
                        file_path=img_info.get("path"),
                        page=img_info.get("page"),
                        text_offset=img_info.get("text_offset"),
                        text_length=img_info.get("text_length"),
                    )
                    refs.append(ref)
        
        # 如果有说明则添加
        captions = metadata.get("image_captions", {})
        if isinstance(captions, dict):
            for ref in refs:
                if ref.image_id in captions:
                    ref.caption = captions[ref.image_id]
        
        # 兜底：如果没有结构化引用则从文本解析占位符
        if not refs and result.text:
            placeholders = IMAGE_PLACEHOLDER_PATTERN.findall(result.text)
            for image_id in placeholders[:self.max_images_per_result]:
                image_id = image_id.strip()
                ref = ImageReference(
                    image_id=image_id,
                    caption=captions.get(image_id) if isinstance(captions, dict) else None,
                )
                refs.append(ref)
        
        return refs
    
    def resolve_image_path(
        self,
        ref: ImageReference,
        collection: Optional[str] = None,
    ) -> Optional[str]:
        """为图片引用解析文件系统路径。
        
        参数：
            ref: 要解析的 ImageReference。
            collection: 可选的集合名称，用于构造路径。
            
        返回：
            如果找到则返回绝对文件路径，否则返回 None。
        """
        # 如果有显式路径则使用
        if ref.file_path:
            path = Path(ref.file_path)
            if path.exists():
                return str(path.resolve())
        
        # 尝试 ImageStorage 查找
        if self._image_storage is not None:
            try:
                path = self._image_storage.get_image_path(ref.image_id)
                if path and Path(path).exists():
                    return path
            except Exception as e:
                logger.warning(f"ImageStorage lookup failed for {ref.image_id}: {e}")
        
        # 基于约定的路径：data/images/{collection}/{image_id}.png
        if collection:
            from src.core.settings import resolve_path
            for ext in [".png", ".jpg", ".jpeg", ".webp"]:
                candidate = resolve_path(f"data/images/{collection}/{ref.image_id}{ext}")
                if candidate.exists():
                    return str(candidate.resolve())
        
        return None
    
    def load_image(
        self,
        file_path: str,
    ) -> Optional[ImageContent]:
        """Load and encode an image file.
        
        Args:
            file_path: Path to the image file.
            
        Returns:
            ImageContent with base64 data and MIME type, or None on failure.
        """
        try:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"Image file not found: {file_path}")
                return None
            
            # Read file content
            data = path.read_bytes()
            if not data:
                logger.warning(f"Empty image file: {file_path}")
                return None
            
            # 检测 MIME 类型
            mime_type = self._detect_mime_type(path, data)
            
            # 编码为 base64
            base64_data = base64.b64encode(data).decode("utf-8")
            
            return ImageContent(
                image_id=path.stem,
                data=base64_data,
                mime_type=mime_type,
            )
            
        except Exception as e:
            logger.error(f"Failed to load image {file_path}: {e}")
            return None
    
    def _detect_mime_type(
        self,
        path: Path,
        data: bytes,
    ) -> str:
        """根据文件扩展名或魔数字节检测 MIME 类型。
        
        参数：
            path: 用于扩展名检测的文件路径。
            data: 用于魔数字节检测的文件内容。
            
        返回：
            MIME 类型字符串（未知时默认返回 "image/png"）。
        """
        # 先尝试扩展名
        suffix = path.suffix.lower()
        if suffix in MIME_TYPE_MAP:
            return MIME_TYPE_MAP[suffix]
        
        # 兜底使用魔数字节
        for magic, mime_type in MAGIC_BYTES.items():
            if data.startswith(magic):
                return mime_type
        
        # 默认使用 PNG
        logger.debug(f"Unknown image format for {path}, defaulting to image/png")
        return "image/png"
    
    def assemble_for_result(
        self,
        result: RetrievalResult,
        collection: Optional[str] = None,
    ) -> List[Union[types.TextContent, types.ImageContent]]:
        """为单条结果装配多模态内容块。
        
        参数：
            result: 要处理的 RetrievalResult。
            collection: 可选的集合名称，用于路径解析。
            
        返回：
            MCP 内容块列表（TextContent 和 ImageContent）。
        """
        blocks: List[Union[types.TextContent, types.ImageContent]] = []
        
        # 提取图片引用
        refs = self.extract_image_refs(result)
        
        # 加载并添加图片
        for ref in refs:
            # 解析路径
            file_path = self.resolve_image_path(ref, collection)
            if not file_path:
                logger.debug(f"Could not resolve path for image: {ref.image_id}")
                continue
            
            # 加载图片
            image_content = self.load_image(file_path)
            if image_content is None:
                continue
            
            # 用引用信息更新
            image_content.image_id = ref.image_id
            image_content.caption = ref.caption
            
            # 添加图片块
            blocks.append(image_content.to_mcp_content())
            
            # 如启用则将说明作为文本添加
            if self.include_captions and ref.caption:
                caption_text = f"**Image Caption ({ref.image_id}):** {ref.caption}"
                blocks.append(types.TextContent(type="text", text=caption_text))
        
        return blocks
    
    def assemble(
        self,
        results: List[RetrievalResult],
        collection: Optional[str] = None,
    ) -> List[Union[types.TextContent, types.ImageContent]]:
        """为多条结果装配多模态内容块。
        
        参数：
            results: 要处理的 RetrievalResult 列表。
            collection: 可选的集合名称，用于路径解析。
            
        返回：
            来自所有结果的全部 MCP 内容块列表。
        """
        all_blocks: List[Union[types.TextContent, types.ImageContent]] = []
        seen_image_ids: set = set()
        
        for result in results:
            blocks = self.assemble_for_result(result, collection)
            
            # 跨结果对图片去重
            for block in blocks:
                if isinstance(block, types.ImageContent):
                    # 检查是否已见过此图片
                    # ImageContent 没有 image_id，因此对数据做哈希
                    data_hash = hash(block.data[:100])  # 使用前缀以提高效率
                    if data_hash in seen_image_ids:
                        continue
                    seen_image_ids.add(data_hash)
                
                all_blocks.append(block)
        
        return all_blocks
    
    def has_images(self, result: RetrievalResult) -> bool:
        """检查结果是否包含图片引用。
        
        参数：
            result: 要检查的 RetrievalResult。
            
        返回：
            如果结果包含图片引用则返回 True，否则返回 False。
        """
        refs = self.extract_image_refs(result)
        return len(refs) > 0
    
    def count_images(self, results: List[RetrievalResult]) -> int:
        """统计所有结果中的图片总数。
        
        参数：
            results: 要统计的 RetrievalResult 列表。
            
        返回：
            找到的图片引用总数。
        """
        total = 0
        for result in results:
            refs = self.extract_image_refs(result)
            total += len(refs)
        return total
