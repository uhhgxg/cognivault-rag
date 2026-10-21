"""基于 MarkItDown 的 PDF 加载器实现。

本模块实现 PDF 解析并支持图片提取，
把 PDF 转换为标准化的 Markdown 文本，并在其中插入图片占位符。

功能特点：
- 通过 MarkItDown 提取文本并转换为 Markdown
- 提取图片并保存到本地
- 在文本中插入图片占位符，并记录图片元数据
- 图片提取失败时优雅降级（仅保留文本，不中断流程）
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

from PIL import Image
import io

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader

logger = logging.getLogger(__name__)


class PdfLoader(BaseLoader):
    """基于 MarkItDown 提取文本并转 Markdown 的 PDF 加载器。
    
    该加载器的工作流程：
    1. 从 PDF 中提取文本并转换为 Markdown
    2. 提取图片并保存到 data/images/{doc_hash}/ 目录
    3. 在文本中插入形如 [IMAGE: {image_id}] 的图片占位符
    4. 把图片元数据记录到 Document.metadata.images
    
    配置项：
        extract_images: 是否启用图片提取（默认：True）
        image_storage_dir: 图片存储根目录（默认：data/images）
    
    优雅降级：
        如果图片提取失败，会记录警告日志，并继续以纯文本方式解析。
    """
    
    def __init__(
        self,
        extract_images: bool = True,
        image_storage_dir: str | Path = "data/images"
    ):
        """初始化 PDF 加载器。
        
        Args:
            extract_images: 是否从 PDF 中提取图片。
            image_storage_dir: 存放提取出的图片的根目录。
        """
        if not MARKITDOWN_AVAILABLE:
            raise ImportError(
                "MarkItDown is required for PdfLoader. "
                "Install with: pip install markitdown"
            )
        
        self.extract_images = extract_images
        self.image_storage_dir = Path(image_storage_dir)
        self._markitdown = MarkItDown()
    
    def load(self, file_path: str | Path) -> Document:
        """加载并解析一个 PDF 文件。
        
        Args:
            file_path: PDF 文件路径。
            
        Returns:
            包含 Markdown 文本和元数据的 Document。
            
        Raises:
            FileNotFoundError: 当 PDF 文件不存在时。
            ValueError: 当文件不是有效的 PDF 时。
            RuntimeError: 当解析出现致命错误时。
        """
        # 校验文件
        path = self._validate_file(file_path)
        if path.suffix.lower() != '.pdf':
            raise ValueError(f"File is not a PDF: {path}")
        
        # 计算文档哈希，用于生成唯一 ID 和图片存放目录
        doc_hash = self._compute_file_hash(path)
        doc_id = f"doc_{doc_hash[:16]}"
        
        # 用 MarkItDown 解析 PDF
        try:
            result = self._markitdown.convert(str(path))
            text_content = result.text_content if hasattr(result, 'text_content') else str(result)
        except Exception as e:
            logger.error(f"Failed to parse PDF {path}: {e}")
            raise RuntimeError(f"PDF parsing failed: {e}") from e
        
        # 初始化元数据
        metadata: Dict[str, Any] = {
            "source_path": str(path),
            "doc_type": "pdf",
            "doc_hash": doc_hash,
        }
        
        # 如果能提取到标题就放进元数据
        title = self._extract_title(text_content)
        if title:
            metadata["title"] = title
        
        # 处理图片提取（带优雅降级）
        if self.extract_images:
            try:
                text_content, images_metadata = self._extract_and_process_images(
                    path, text_content, doc_hash
                )
                if images_metadata:
                    metadata["images"] = images_metadata
            except Exception as e:
                logger.warning(
                    f"Image extraction failed for {path}, continuing with text-only: {e}"
                )
        
        return Document(
            id=doc_id,
            text=text_content,
            metadata=metadata
        )
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """计算文件内容的 SHA256 哈希。
        
        Args:
            file_path: 文件路径。
            
        Returns:
            SHA256 哈希的十六进制字符串。
        """
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _extract_title(self, text: str) -> Optional[str]:
        """从首个 Markdown 标题或首个非空行提取标题。
        
        Args:
            text: Markdown 文本内容。
            
        Returns:
            找到标题则返回标题字符串，否则返回 None。
        """
        lines = text.split('\n')
        
        # 先尝试找一个 Markdown 标题
        for line in lines[:20]:  # 只看前 20 行
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
        
        # 兜底：用第一个非空行作为标题
        for line in lines[:10]:
            line = line.strip()
            if line and len(line) > 0:
                return line
        
        return None
    
    def _extract_and_process_images(
        self,
        pdf_path: Path,
        text_content: str,
        doc_hash: str
    ) -> tuple[str, List[Dict[str, Any]]]:
        """从 PDF 中提取图片并插入占位符。
        
        使用 PyMuPDF 提取图片，保存到磁盘，并在文本内容中
        插入占位符。
        
        Args:
            pdf_path: PDF 文件路径。
            text_content: 已提取的文本内容。
            doc_hash: 文档哈希，用于决定图片存放目录。
            
        Returns:
            元组 (修改后的文本, 图片元数据列表)
        """
        if not self.extract_images:
            logger.debug(f"Image extraction disabled for {pdf_path}")
            return text_content, []
        
        if not PYMUPDF_AVAILABLE:
            logger.warning(f"PyMuPDF not available, skipping image extraction for {pdf_path}")
            return text_content, []
        
        images_metadata = []
        modified_text = text_content
        
        try:
            # 创建图片存放目录
            image_dir = self.image_storage_dir / doc_hash
            image_dir.mkdir(parents=True, exist_ok=True)
            
            # 用 PyMuPDF 打开 PDF
            doc = fitz.open(pdf_path)
            
            # 记录文本偏移，用于决定占位符插入位置
            text_offset = 0
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images(full=True)
                
                for img_index, img_info in enumerate(image_list):
                    try:
                        # 提取图片
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # 生成图片 ID 和文件名
                        image_id = self._generate_image_id(doc_hash, page_num + 1, img_index + 1)
                        image_filename = f"{image_id}.{image_ext}"
                        image_path = image_dir / image_filename
                        
                        # 保存图片
                        with open(image_path, "wb") as img_file:
                            img_file.write(image_bytes)
                        
                        # 读取图片宽高
                        try:
                            img = Image.open(io.BytesIO(image_bytes))
                            width, height = img.size
                        except Exception:
                            width, height = 0, 0
                        
                        # 生成占位符
                        placeholder = f"[IMAGE: {image_id}]"
                        
                        # 在当前页内容末尾插入占位符
                        # （此处做了简化处理；生产环境中应解析页面边界再插入）
                        insert_position = len(modified_text)
                        modified_text += f"\n{placeholder}\n"
                        
                        # 把路径转成相对项目根目录的路径，不行就用绝对路径
                        try:
                            relative_path = image_path.relative_to(Path.cwd())
                        except ValueError:
                            # 不在当前工作目录下，就用绝对路径
                            relative_path = image_path.absolute()
                        
                        # 记录元数据
                        image_metadata = {
                            "id": image_id,
                            "path": str(relative_path),
                            "page": page_num + 1,
                            "text_offset": insert_position + 1,  # 加 1 是为了跳过换行符
                            "text_length": len(placeholder),
                            "position": {
                                "width": width,
                                "height": height,
                                "page": page_num + 1,
                                "index": img_index
                            }
                        }
                        images_metadata.append(image_metadata)
                        
                        logger.debug(f"Extracted image {image_id} from page {page_num + 1}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to extract image {img_index} from page {page_num + 1}: {e}")
                        continue
            
            doc.close()
            
            if images_metadata:
                logger.info(f"Extracted {len(images_metadata)} images from {pdf_path}")
            else:
                logger.debug(f"No images found in {pdf_path}")
            
            return modified_text, images_metadata
            
        except Exception as e:
            logger.warning(f"Image extraction failed for {pdf_path}: {e}")
            # 优雅降级：返回原始文本，不带图片
            return text_content, []
    
    @staticmethod
    def _generate_image_id(doc_hash: str, page: int, sequence: int) -> str:
        """生成唯一的图片 ID。
        
        Args:
            doc_hash: 文档哈希。
            page: 页码（从 0 开始）。
            sequence: 该页中图片的序号（从 0 开始）。
            
        Returns:
            唯一的图片 ID 字符串。
        """
        return f"{doc_hash[:8]}_{page}_{sequence}"
