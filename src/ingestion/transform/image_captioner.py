"""图片描述（caption）变换：用图片描述来丰富分块内容。

性能优化：
1. 只处理分块文本中实际引用到的图片（通过 [IMAGE: id] 占位符识别）
2. 用描述缓存避免对同一张图片重复调用 Vision API
3. 完全跳过不包含图片引用的分块
4. 对不同的图片做并行处理，并配合线程安全的缓存
"""

import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Dict

from src.core.settings import Settings
from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.base_transform import BaseTransform
from src.libs.llm.base_vision_llm import BaseVisionLLM, ImageInput
from src.libs.llm.llm_factory import LLMFactory
from src.observability.logger import get_logger

logger = get_logger(__name__)

# 用于查找图片占位符的正则表达式：[IMAGE: 某个id]
IMAGE_PLACEHOLDER_PATTERN = re.compile(r'\[IMAGE:\s*([^\]]+)\]')

# Vision API 调用的默认最大并行线程数
DEFAULT_MAX_WORKERS = 3  # 比文本 LLM 低，因为成本更高、延迟更大


class ImageCaptioner(BaseTransform):
    """用 Vision LLM 给分块中引用的图片生成描述（caption）。

    这个变换会找出包含图片引用的分块，用 Vision LLM 生成描述性文字，
    再把这些描述填充分块的文本/元数据，从而提升视觉内容的检索效果。

    主要特性：
    - 只处理分块文本中实际引用到的图片（不处理元数据里的所有图片）
    - 缓存描述，避免对同一张图片重复调用 Vision API
    - 线程安全的描述缓存，便于将来并行化扩展
    """
    
    def __init__(
        self, 
        settings: Settings, 
        llm: Optional[BaseVisionLLM] = None,
        max_workers: Optional[int] = None,
    ):
        self.settings = settings
        self.llm = None
        # 描述缓存：图片id -> 描述文本（用锁保证线程安全）
        self._caption_cache: Dict[str, str] = {}
        self._cache_lock = threading.Lock()
        self.max_workers = max_workers or DEFAULT_MAX_WORKERS

        # 检查配置中是否启用了 vision LLM
        if self.settings.vision_llm and self.settings.vision_llm.enabled:
             try:
                 self.llm = llm or LLMFactory.create_vision_llm(settings)
             except Exception as e:
                 logger.error(f"Failed to initialize Vision LLM: {e}")
                 # 这里不抛异常，让流水线在没有图片描述的情况下也能继续
                 # 实际效果是这个变换变成空操作（什么都不做）
        else:
             logger.warning("Vision LLM is disabled or not configured. ImageCaptioner will skip processing.")
        
        self.prompt = self._load_prompt()
        
    def _load_prompt(self) -> str:
        """从配置中加载图片描述提示词。"""
        # 假设使用标准相对路径。生产环境中逻辑可能更健壮。
        from src.core.settings import resolve_path
        prompt_path = resolve_path("config/prompts/image_captioning.txt")
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8").strip()
        return "Describe this image in detail for indexing purposes."

    def _find_referenced_image_ids(self, text: str) -> List[str]:
        """提取分块文本中实际引用到的图片 ID。

        参数：
            text: 分块文本内容

        返回：
            在 [IMAGE: id] 占位符中找到的图片 ID 列表
        """
        matches = IMAGE_PLACEHOLDER_PATTERN.findall(text)
        return [m.strip() for m in matches]

    def _get_caption(
        self, 
        img_id: str, 
        img_path: str, 
        trace: Optional[TraceContext] = None
    ) -> Optional[str]:
        """获取某张图片的描述，有缓存则用缓存。线程安全。

        参数：
            img_id: 图片标识符
            img_path: 图片文件路径
            trace: 可选的链路追踪上下文

        返回：
            描述字符串，失败时返回 None
        """
        # 先查缓存（线程安全读取）
        with self._cache_lock:
            if img_id in self._caption_cache:
                logger.debug(f"Caption cache hit for image {img_id}")
                return self._caption_cache[img_id]

        # 校验路径
        if not img_path or not Path(img_path).exists():
            logger.warning(f"Image path not found: {img_path}")
            return None

        try:
            image_input = ImageInput(path=img_path)
            response = self.llm.chat_with_image(
                text=self.prompt,
                image=image_input,
                trace=trace
            )
            caption = response.content

            # 把结果写入缓存（线程安全写入）
            with self._cache_lock:
                self._caption_cache[img_id] = caption
            logger.debug(f"Generated and cached caption for image {img_id}")
            
            return caption
            
        except Exception as e:
            logger.error(f"Failed to caption image {img_path}: {e}")
            return None

    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """处理分块，给被引用的图片加上描述。

        只处理分块文本中通过 [IMAGE: id] 占位符实际引用到的图片。
        用缓存避免重复调用 API。对不同的图片做并行处理。
        """
        if not self.llm:
            return chunks

        # 从所有分块的元数据构建图片查找表
        image_lookup: Dict[str, dict] = {}
        for chunk in chunks:
            if chunk.metadata and "images" in chunk.metadata:
                for img_meta in chunk.metadata.get("images", []):
                    img_id = img_meta.get("id")
                    if img_id and img_id not in image_lookup:
                        image_lookup[img_id] = img_meta

        logger.info(f"Found {len(image_lookup)} unique images in document")

        # 清空缓存，准备处理新文档
        with self._cache_lock:
            self._caption_cache.clear()

        # 第一遍：收集所有需要生成描述的不同图片 ID
        images_to_caption: Dict[str, str] = {}  # 图片id -> 图片路径
        for chunk in chunks:
            referenced_ids = self._find_referenced_image_ids(chunk.text)
            for img_id in referenced_ids:
                if img_id not in images_to_caption:
                    img_meta = image_lookup.get(img_id)
                    if img_meta and img_meta.get("path"):
                        images_to_caption[img_id] = img_meta.get("path")

        # 为所有不同的图片并行生成描述
        if images_to_caption:
            self._generate_captions_parallel(images_to_caption, trace)

        # 第二遍：把描述应用到分块上
        processed_chunks = []
        total_captions_added = 0
        
        for chunk in chunks:
            referenced_ids = self._find_referenced_image_ids(chunk.text)
            
            if not referenced_ids:
                processed_chunks.append(chunk)
                continue
            
            new_text = chunk.text
            captions = []
            
            for img_id in referenced_ids:
                img_id_stripped = img_id.strip()
                
                # 从缓存取描述（并行处理阶段已经填好缓存了）
                with self._cache_lock:
                    caption = self._caption_cache.get(img_id_stripped)
                
                if caption:
                    captions.append({"id": img_id_stripped, "caption": caption})
                    
                    placeholder = f"[IMAGE: {img_id}]"
                    replacement = f"[IMAGE: {img_id}]\n(Description: {caption})"
                    new_text = new_text.replace(placeholder, replacement)
                    total_captions_added += 1
                    
            chunk.text = new_text
            
            if captions:
                if "image_captions" not in chunk.metadata:
                    chunk.metadata["image_captions"] = []
                chunk.metadata["image_captions"].extend(captions)
            
            processed_chunks.append(chunk)
        
        with self._cache_lock:
            api_calls = len(self._caption_cache)
        logger.info(f"Added {total_captions_added} captions, API calls: {api_calls}")
            
        return processed_chunks
    
    def _generate_captions_parallel(
        self, 
        images_to_caption: Dict[str, str],
        trace: Optional[TraceContext] = None
    ) -> None:
        """为多张图片并行生成描述。

        参数：
            images_to_caption: 图片id -> 图片路径 的字典
            trace: 可选的链路追踪上下文
        """
        if not images_to_caption:
            return
        
        max_workers = min(self.max_workers, len(images_to_caption))
        logger.debug(f"Generating captions for {len(images_to_caption)} images (max_workers={max_workers})")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._get_caption, img_id, img_path, trace): img_id
                for img_id, img_path in images_to_caption.items()
            }
            
            for future in as_completed(futures):
                img_id = futures[future]
                try:
                    caption = future.result()
                    if caption:
                        logger.debug(f"Caption generated for {img_id}")
                except Exception as e:
                    logger.error(f"Failed to generate caption for {img_id}: {e}")
