"""带 SQLite 索引的图片存储，用于多模态 RAG。

本模块提供持久化的图片存储，使用 SQLite 做索引，
可以高效地跨集合检索和管理图片。

设计原则：
- 持久化：图片存文件系统，元数据存 SQLite
- 并发：WAL 模式支持并发读写
- 幂等：重复保存同一个 image_id 会安全更新元数据
- 有组织：图片按集合分组，实现命名空间隔离
"""

import sqlite3
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Union


class ImageStorage:
    """基于 SQLite 的图片存储管理器。

    把图片文件存到有组织的目录结构中，并用 SQLite 索引来高效查找和查询。

    目录结构：
        data/images/{collection}/{image_id}.png

    数据库表结构：
        image_index (
            image_id TEXT PRIMARY KEY,
            file_path TEXT NOT NULL,
            collection TEXT,
            doc_hash TEXT,
            page_num INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )

        INDEX idx_collection ON image_index(collection)
        INDEX idx_doc_hash ON image_index(doc_hash)

    参数：
        db_path: SQLite 数据库文件路径（默认 data/db/image_index.db）。
        images_root: 图片存储的根目录（默认 data/images）。

    示例：
        >>> storage = ImageStorage()
        >>> 
        >>> # Save an image
        >>> with open("sample.png", "rb") as f:
        >>>     image_data = f.read()
        >>> path = storage.save_image(
        ...     image_id="doc123_p1_img0",
        ...     image_data=image_data,
        ...     collection="contracts",
        ...     doc_hash="abc123",
        ...     page_num=1
        ... )
        >>> print(path)  # data/images/contracts/doc123_p1_img0.png
        >>> 
        >>> # Retrieve image path
        >>> path = storage.get_image_path("doc123_p1_img0")
        >>> print(path)  # data/images/contracts/doc123_p1_img0.png
        >>> 
        >>> # List images in collection
        >>> images = storage.list_images("contracts")
        >>> print(len(images))  # 1
    """
    
    def __init__(
        self,
        db_path: str = "data/db/image_index.db",
        images_root: str = "data/images"
    ):
        """初始化图片存储，必要时创建数据库。

        参数：
            db_path: SQLite 数据库文件路径。
            images_root: 存放图片文件的根目录。
        """
        self.db_path = db_path
        self.images_root = Path(images_root)
        self._conn = None
        self._ensure_database()
    
    def close(self) -> None:
        """如果数据库连接已打开，则关闭它。"""
        if self._conn:
            self._conn.close()
            self._conn = None
    
    def __del__(self):
        """清理：对象销毁时关闭连接。"""
        self.close()
    
    def _ensure_database(self) -> None:
        """如果数据库文件和表结构不存在，则创建它们。"""
        # 创建父目录
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 创建图片根目录
        self.images_root.mkdir(parents=True, exist_ok=True)
        
        # 连接数据库并初始化表结构
        conn = sqlite3.connect(self.db_path)
        try:
            # 启用 WAL 模式，支持并发读写
            conn.execute("PRAGMA journal_mode=WAL")
            
            # 表不存在则创建
            conn.execute("""
                CREATE TABLE IF NOT EXISTS image_index (
                    image_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    collection TEXT,
                    doc_hash TEXT,
                    page_num INTEGER,
                    created_at TEXT NOT NULL
                )
            """)
            
            # 创建索引以提升查询效率
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_collection 
                ON image_index(collection)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_doc_hash 
                ON image_index(doc_hash)
            """)
            
            conn.commit()
        finally:
            conn.close()
    
    def save_image(
        self,
        image_id: str,
        image_data: Union[bytes, Path, str],
        collection: Optional[str] = None,
        doc_hash: Optional[str] = None,
        page_num: Optional[int] = None,
        extension: str = "png"
    ) -> str:
        """把图片保存到文件系统并在数据库中登记。

        本操作是幂等的 - 用同一个 image_id 重新保存会更新元数据并覆盖文件。

        参数：
            image_id: 图片的唯一标识。
            image_data: 图片的字节数据，或源图片文件的路径。
            collection: 可选的集合/命名空间，用于组织。
            doc_hash: 可选的文档哈希，用于可追溯。
            page_num: 可选的页码（来自分页文档时）。
            extension: 不含点的文件扩展名（默认 "png"）。

        返回：
            图片保存的相对路径。

        可能抛出：
            ValueError: 如果 image_id 为空或无效。
            IOError: 如果图片文件无法保存。
            RuntimeError: 如果数据库操作失败。

        示例：
            >>> # Save from bytes
            >>> path = storage.save_image("img1", b"PNG_DATA", "docs")
            >>> 
            >>> # Save from file
            >>> path = storage.save_image("img2", Path("source.png"), "docs")
        """
        if not image_id or not image_id.strip():
            raise ValueError("image_id cannot be empty")
        
        # 确定集合目录
        if collection:
            collection_dir = self.images_root / collection
        else:
            collection_dir = self.images_root / "default"
        
        collection_dir.mkdir(parents=True, exist_ok=True)
        
        # 构建图片文件路径
        image_filename = f"{image_id}.{extension}"
        image_path = collection_dir / image_filename
        
        # 保存图片文件
        try:
            if isinstance(image_data, bytes):
                # 直接写入字节数据
                image_path.write_bytes(image_data)
            elif isinstance(image_data, (Path, str)):
                # 从源文件复制
                source_path = Path(image_data)
                if not source_path.exists():
                    raise FileNotFoundError(f"Source image not found: {source_path}")
                shutil.copy2(source_path, image_path)
            else:
                raise ValueError(f"Unsupported image_data type: {type(image_data)}")
        except Exception as e:
            raise IOError(f"Failed to save image {image_id}: {e}")
        
        # 存绝对路径，确保能可靠取回
        # （相对路径在测试的临时目录里会出问题）
        stored_path = str(image_path.resolve())
        
        # 在数据库中登记
        now = datetime.now(timezone.utc).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        try:
            # 用 INSERT OR REPLACE 实现幂等写入
            conn.execute("""
                INSERT OR REPLACE INTO image_index 
                (image_id, file_path, collection, doc_hash, page_num, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (image_id, stored_path, collection, doc_hash, page_num, now))
            
            conn.commit()
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to register image {image_id}: {e}")
        finally:
            conn.close()
        
        return stored_path
    
    def register_image(
        self,
        image_id: str,
        file_path: Union[Path, str],
        collection: Optional[str] = None,
        doc_hash: Optional[str] = None,
        page_num: Optional[int] = None
    ) -> str:
        """在数据库索引中登记一个已存在的图片文件。

        和 save_image() 不同，这个方法不会复制或移动文件。
        它只创建一条指向已存在文件的数据库记录。
        当图片已经被其他组件（比如 PdfLoader）保存好、只需登记索引时使用。

        参数：
            image_id: 图片的唯一标识。
            file_path: 已存在的图片文件路径。
            collection: 可选的集合/命名空间，用于组织。
            doc_hash: 可选的文档哈希，用于可追溯。
            page_num: 可选的页码（来自分页文档时）。

        返回：
            已登记图片的绝对路径。

        可能抛出：
            ValueError: 如果 image_id 为空或无效。
            FileNotFoundError: 如果图片文件不存在。
            RuntimeError: 如果数据库操作失败。

        示例：
            >>> # Register an image that was saved by PdfLoader
            >>> path = storage.register_image(
            ...     image_id="doc123_p1_img0",
            ...     file_path="data/images/tech_docs/abc123/doc123_p1_img0.png",
            ...     collection="tech_docs",
            ...     doc_hash="abc123",
            ...     page_num=1
            ... )
        """
        if not image_id or not image_id.strip():
            raise ValueError("image_id cannot be empty")
        
        # 校验文件是否存在
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        # 存绝对路径，确保能可靠取回
        stored_path = str(path.resolve())
        
        # 在数据库中登记
        now = datetime.now(timezone.utc).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        try:
            # 用 INSERT OR REPLACE 实现幂等写入
            conn.execute("""
                INSERT OR REPLACE INTO image_index 
                (image_id, file_path, collection, doc_hash, page_num, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (image_id, stored_path, collection, doc_hash, page_num, now))
            
            conn.commit()
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to register image {image_id}: {e}")
        finally:
            conn.close()
        
        return stored_path
    
    def get_image_path(self, image_id: str) -> Optional[str]:
        """根据 ID 获取图片的文件系统路径。

        参数：
            image_id: 图片的唯一标识。

        返回：
            如果图片存在则返回相对文件路径，否则返回 None。

        示例：
            >>> path = storage.get_image_path("img1")
            >>> if path:
            ...     with open(path, "rb") as f:
            ...         image_data = f.read()
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT file_path FROM image_index WHERE image_id = ?",
                (image_id,)
            )
            result = cursor.fetchone()
            return result[0] if result else None
        finally:
            conn.close()
    
    def image_exists(self, image_id: str) -> bool:
        """检查图片是否在数据库中已存在。

        参数：
            image_id: 图片的唯一标识。

        返回：
            如果图片已登记则返回 True，否则返回 False。
        """
        return self.get_image_path(image_id) is not None
    
    def list_images(
        self,
        collection: Optional[str] = None,
        doc_hash: Optional[str] = None
    ) -> List[Dict[str, any]]:
        """列出图片，支持可选的过滤条件。

        参数：
            collection: 可选的集合过滤条件。
            doc_hash: 可选的文档哈希过滤条件。

        返回：
            图片元数据字典列表，每个字典包含以下键：
            - image_id: 图片唯一标识
            - file_path: 文件系统路径
            - collection: 集合名
            - doc_hash: 文档哈希
            - page_num: 页码（如果有）
            - created_at: 创建时间戳

        示例：
            >>> # 列出某个集合中的所有图片
            >>> images = storage.list_images(collection="contracts")
            >>> for img in images:
            ...     print(img["image_id"], img["file_path"])
            
            >>> # 列出来自某篇文档的所有图片
            >>> images = storage.list_images(doc_hash="abc123")
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 启用字典式访问
        
        try:
            # 构建带可选过滤条件的查询
            query = "SELECT * FROM image_index WHERE 1=1"
            params = []
            
            if collection is not None:
                query += " AND collection = ?"
                params.append(collection)
            
            if doc_hash is not None:
                query += " AND doc_hash = ?"
                params.append(doc_hash)
            
            query += " ORDER BY created_at ASC"
            
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            # 把行数据转成字典
            return [dict(row) for row in rows]
        finally:
            conn.close()
    
    def delete_image(self, image_id: str, remove_file: bool = True) -> bool:
        """从数据库中删除图片，可选是否同时删除文件。

        参数：
            image_id: 图片的唯一标识。
            remove_file: 为 True 时同时删除图片文件（默认 True）。

        返回：
            如果图片被删除则返回 True，没找到则返回 False。

        示例：
            >>> # 同时删除数据库记录和文件
            >>> deleted = storage.delete_image("img1")
            >>> 
            >>> # 只删数据库记录，保留文件
            >>> deleted = storage.delete_image("img2", remove_file=False)
        """
        # 从数据库删除前先拿到文件路径
        file_path = self.get_image_path(image_id)
        
        if file_path is None:
            return False
        
        # 从数据库中删除
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "DELETE FROM image_index WHERE image_id = ?",
                (image_id,)
            )
            conn.commit()
            deleted = cursor.rowcount > 0
        except sqlite3.Error:
            return False
        finally:
            conn.close()
        
        # 按需删除文件
        if remove_file and deleted:
            try:
                Path(file_path).unlink(missing_ok=True)
            except Exception:
                # 记日志但不报错（文件删失败就算了）
                pass
        
        return deleted
    
    def get_collection_stats(self, collection: str) -> Dict[str, any]:
        """获取某个集合的统计信息。

        参数：
            collection: 集合名。

        返回：
            包含统计信息的字典：
            - total_images: 集合中的图片数量
            - total_size_bytes: 文件总大小（如果文件存在）

        示例：
            >>> stats = storage.get_collection_stats("contracts")
            >>> print(f"Total images: {stats['total_images']}")
        """
        images = self.list_images(collection=collection)
        
        total_size = 0
        for img in images:
            try:
                file_path = Path(img["file_path"])
                if file_path.exists():
                    total_size += file_path.stat().st_size
            except Exception:
                pass
        
        return {
            "total_images": len(images),
            "total_size_bytes": total_size
        }
