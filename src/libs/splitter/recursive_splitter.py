"""基于 LangChain 的递归切分器实现。

本模块提供一种基于字符的递归切分策略，会尊重文档结构
（标题、代码块），并按层级递进地切分文本，以保持语义连贯。
"""

from __future__ import annotations

from typing import Any, List, Optional

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None  # type: ignore[misc, assignment]

from src.libs.splitter.base_splitter import BaseSplitter


class RecursiveSplitter(BaseSplitter):
    """基于字符的递归文本切分器。

    本切分器使用 LangChain 的 RecursiveCharacterTextSplitter，按顺序尝试不同的
    分隔符（段落、句子、单词）来切分文本，同时尊重 Markdown 中的标题、代码块
    等结构元素。

    应用的设计原则：
    - 可插拔：实现 BaseSplitter 接口，方便工厂统一创建。
    - 配置驱动：从 settings 读取 chunk_size 和 chunk_overlap。
    - 快速失败：如果没装 langchain-text-splitters，直接抛出 ImportError。
    - 优雅降级：会校验输入，并给出清晰的错误信息。

    属性：
        chunk_size: 每个切块的最大字符数。
        chunk_overlap: 相邻切块之间重叠的字符数。
        separators: 按顺序尝试的分隔符列表（默认是兼容 Markdown 的分隔符）。

    抛出：
        ImportError: 如果没有安装 langchain-text-splitters 包。
    """

    DEFAULT_SEPARATORS = [
        "\n\n",  # 双换行（段落分隔）
        "\n",    # 单换行
        ". ",    # 句子结尾
        "! ",
        "? ",
        "; ",
        ", ",
        " ",     # 空格
        "",      # 单字符
    ]
    
    def __init__(
        self,
        settings: Any,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """初始化 RecursiveSplitter。

        参数：
            settings: 应用配置，里面包含文档摄入（ingestion）相关配置。
            chunk_size: 可选，覆盖切块大小（默认用 settings.ingestion.chunk_size）。
            chunk_overlap: 可选，覆盖重叠字符数（默认用 settings.ingestion.chunk_overlap）。
            separators: 可选，分隔符字符串列表（默认用兼容 Markdown 的分隔符）。
            **kwargs: 其他传给 LangChain 切分器的额外参数。

        抛出：
            ImportError: 如果没装 langchain-text-splitters。
            ValueError: 如果 chunk_size 或 chunk_overlap 不合法。
        """
        if RecursiveCharacterTextSplitter is None:
            raise ImportError(
                "langchain-text-splitters is not installed. "
                "Install it with: pip install langchain-text-splitters"
            )
        
        self.settings = settings

        # 从 settings 读取配置，并允许用参数覆盖
        try:
            ingestion_config = settings.ingestion
            self.chunk_size = chunk_size if chunk_size is not None else ingestion_config.chunk_size
            self.chunk_overlap = chunk_overlap if chunk_overlap is not None else ingestion_config.chunk_overlap
        except AttributeError as e:
            raise ValueError(
                "Missing ingestion configuration in settings. "
                "Expected settings.ingestion.chunk_size and settings.ingestion.chunk_overlap"
            ) from e
        
        # 校验配置
        if not isinstance(self.chunk_size, int) or self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be a positive integer, got: {self.chunk_size}")

        if not isinstance(self.chunk_overlap, int) or self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be a non-negative integer, got: {self.chunk_overlap}")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )

        self.separators = separators if separators is not None else self.DEFAULT_SEPARATORS

        # 初始化 LangChain 切分器
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
            **kwargs,
        )
    
    def split_text(
        self,
        text: str,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[str]:
        """递归地把文本切分成多个切块。

        本方法按层级依次尝试不同的分隔符来切分文本，
        同时保留 Markdown 标题、代码块等文档结构。

        参数：
            text: 要切分的输入文本，必须是非空字符串。
            trace: 可选的 TraceContext，用于可观测性（预留给 Stage F 使用）。
            **kwargs: 额外参数（目前未使用，预留给未来扩展）。

        返回：
            文本切块列表。每个切块都遵循配置的 chunk_size 和 chunk_overlap。
            顺序与原文顺序一致。

        抛出：
            ValueError: 如果输入文本不合法（为空、类型错误等）。
            RuntimeError: 如果切分过程意外失败。

        示例：
            >>> splitter = RecursiveSplitter(settings)
            >>> chunks = splitter.split_text("# Header\\n\\nParagraph 1.\\n\\nParagraph 2.")
            >>> len(chunks)
            1  # 如果文本能放进一个 chunk_size，就只切出 1 个块
        """
        # 校验输入
        self.validate_text(text)

        try:
            # 执行切分
            chunks = self._splitter.split_text(text)

            # 处理边界情况：文本很短时 LangChain 可能返回空列表
            if not chunks:
                chunks = [text]

            # 校验输出
            self.validate_chunks(chunks)

            return chunks

        except Exception as e:
            # 捕获 LangChain 抛出的任何错误，并补充上下文信息
            raise RuntimeError(
                f"RecursiveSplitter failed to split text: {e}. "
                f"Text length: {len(text)}, chunk_size: {self.chunk_size}, "
                f"chunk_overlap: {self.chunk_overlap}"
            ) from e
