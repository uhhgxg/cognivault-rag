"""用于简单文本文档的文本文件加载器。"""

import hashlib
from pathlib import Path
from typing import Union

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader


class TextLoader(BaseLoader):
    """简单的文本文件加载器。

    支持 .txt、.md、.csv 文件。
    """
    
    SUPPORTED_EXTENSIONS = {'.txt', '.md', '.csv', '.json', '.yaml', '.yml'}
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """计算文件内容的 SHA-256 哈希。"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def load(self, file_path: Union[str, Path]) -> Document:
        """加载一个文本文件。

        Args:
            file_path: 文本文件路径。

        Returns:
            包含文本内容的 Document。
        """
        path = self._validate_file(file_path)

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        # 读取文本文件
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # 计算哈希
        doc_hash = self._compute_file_hash(path)
        doc_id = f"doc_{doc_hash[:16]}"
        
        return Document(
            id=doc_id,
            text=content,
            metadata={
                "source_path": str(path),
                "filename": path.name,
                "file_type": path.suffix.lower(),
            }
        )
