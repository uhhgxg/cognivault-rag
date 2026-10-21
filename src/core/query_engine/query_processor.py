"""查询预处理器（Query Processor）。

本模块提供对用户查询的预处理功能，包括：
- 基于规则的关键词提取（分词）
- 中英文停用词过滤
- 从查询语法中解析过滤器（例如 "collection:docs"）
- 查询规范化与清洗

设计原则：
- 以规则为主：优先使用简单、确定性的规则以保证稳定性
- 语言感知：支持中文与英文查询
- 可扩展：方便后续添加同义词扩展或基于 LLM 的处理
- 配置驱动：停用词和模式可通过配置调整
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Pattern, Set

import jieba

from src.core.types import ProcessedQuery


# 中文默认停用词
CHINESE_STOPWORDS: Set[str] = {
    # 疑问词
    "如何", "怎么", "怎样", "什么", "哪个", "哪些", "为什么", "为何",
    "谁", "多少", "几", "是否", "能否", "可否",
    # 助词
    "的", "地", "得", "了", "着", "过", "吗", "呢", "吧", "啊", "呀",
    # 介词/连词
    "在", "于", "和", "与", "或", "及", "并", "而", "但", "但是",
    "因为", "所以", "如果", "那么", "虽然", "然而",
    # 代词
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "这", "那",
    "这个", "那个", "这些", "那些", "这里", "那里",
    # 副词
    "很", "非常", "特别", "更", "最", "都", "也", "还", "又", "再",
    "已", "已经", "正在", "将", "会", "能", "可以", "应该", "必须",
    # 动词(通用)
    "是", "有", "做", "进行", "使用", "通过",
    # 量词
    "个", "种", "类",
    # 标点等
    "？", "。", "！", "，", "、",
}

# 英文默认停用词
ENGLISH_STOPWORDS: Set[str] = {
    # 冠词
    "a", "an", "the",
    # 介词
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "into", "about", "through", "between", "after", "before",
    # 连词
    "and", "or", "but", "if", "then", "because", "while", "although",
    # 代词
    "i", "you", "he", "she", "it", "we", "they", "this", "that",
    "these", "those", "what", "which", "who", "whom", "whose",
    # 助动词
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "can",
    # 常用动词
    "get", "use", "make",
    # 疑问词
    "how", "why", "when", "where",
    # 其他
    "not", "no", "yes", "so", "very", "just", "also", "too",
}

# 合并后的默认停用词
DEFAULT_STOPWORDS: Set[str] = CHINESE_STOPWORDS | ENGLISH_STOPWORDS

# 过滤器语法模式：key:value
FILTER_PATTERN: Pattern = re.compile(r'(\w+):([^\s]+)')


@dataclass
class QueryProcessorConfig:
    """`QueryProcessor` 的配置。

    属性：
        stopwords: 需要过滤的停用词集合
        min_keyword_length: 包含为关键词的最小长度
        max_keywords: 最多提取多少个关键词
        enable_filter_parsing: 是否解析查询中的过滤器语法
    """
    stopwords: Set[str] = field(default_factory=lambda: DEFAULT_STOPWORDS.copy())
    min_keyword_length: int = 1
    max_keywords: int = 20
    enable_filter_parsing: bool = True


class QueryProcessor:
    """用于检索前对用户查询进行预处理的类。

    功能：提取关键词、过滤停用词、解析查询中的过滤器语法，
    以便为稠密检索（Dense）和稀疏检索（Sparse）准备输入。

    示例：
        >>> processor = QueryProcessor()
        >>> result = processor.process("如何配置 Azure OpenAI？")
        >>> print(result.keywords)
        ['配置', 'Azure', 'OpenAI']
    """
    
    def __init__(self, config: Optional[QueryProcessorConfig] = None):
        """初始化 QueryProcessor。

        参数：
            config: 可选的配置；如果未提供则使用默认配置。
        """
        self.config = config or QueryProcessorConfig()
    
    def process(self, query: str) -> ProcessedQuery:
        """将原始查询处理为结构化格式。

        参数：
            query: 原始用户查询字符串

        返回：
            包含提取关键词与解析到的过滤器的 `ProcessedQuery`。
        """
        if not query or not query.strip():
            return ProcessedQuery(
                original_query=query or "",
                keywords=[],
                filters={}
            )
        
        # 去除空白,规范字符串
        normalized = self._normalize(query)
        
        # 分割过滤字典和纯净query
        filters, query_without_filters = self._extract_filters(normalized)
        
        # 用jieba库分词,token为干净的无标点符号的列表
        tokens = self._tokenize(query_without_filters)
        
        # 过滤停用词并应用约束
        keywords = self._filter_keywords(tokens)
        
        return ProcessedQuery(
            original_query=query,
            keywords=keywords,
            filters=filters
        )
    
    def _normalize(self, query: str) -> str:
        """规范化查询字符串。

        - 去除多余空白
        - 统一空白和格式

        返回规范化后的查询字符串。
        """
        # 去除首尾空白并规范化空白字符
        normalized = " ".join(query.split())
        return normalized
    
    def _extract_filters(self, query: str) -> tuple[Dict[str, Any], str]:
        """从查询中提取过滤器语法（如 `collection:api-docs`）。

        返回一个元组：(解析出的过滤器字典, 去除过滤器后的查询文本)。
        """
        if not self.config.enable_filter_parsing:
            return {}, query
        
        filters: Dict[str, Any] = {}
        
        # 找出所有过滤器模式
        matches = FILTER_PATTERN.findall(query)
        for key, value in matches:
            # 支持常见的过滤器键
            key_lower = key.lower()
            if key_lower in ("collection", "col", "c"):
                filters["collection"] = value
            elif key_lower in ("type", "doc_type", "t"):
                filters["doc_type"] = value
            elif key_lower in ("source", "src", "s"):
                filters["source_path"] = value
            elif key_lower in ("tag", "tags"):
                # 标签可以用逗号分隔
                if "tags" not in filters:
                    filters["tags"] = []
                filters["tags"].extend(value.split(","))
            else:
                # 通用过滤器
                filters[key] = value

        # 从查询中移除过滤器模式
        query_without_filters = FILTER_PATTERN.sub("", query).strip()
        query_without_filters = " ".join(query_without_filters.split())
        
        return filters, query_without_filters
    
    def _tokenize(self, text: str) -> List[str]:
        """将文本切分为词/项。

        使用 `jieba` 做中文分词，这样与索引端（例如 SparseEncoder）保持一致，
        有利于 BM25 等稀疏检索的匹配。英文会被保留为整体词项。

        返回分词后的 token 列表。
        """
        tokens: List[str] = []

        # 用 jieba 分词（处理中文 + 保留英文完整）
        raw_tokens = jieba.lcut(text)

        for token in raw_tokens:
            token = token.strip()
            if not token:
                continue
            # 跳过纯标点 / 空白
            if re.fullmatch(r'[\s\W]+', token, re.UNICODE):
                continue
            tokens.append(token)
        
        return tokens
    
    def _filter_keywords(self, tokens: List[str]) -> List[str]:
        """从 token 中过滤并提取有意义的关键词。

        处理流程：
        - 删除停用词
        - 应用最小长度约束
        - 去重（保留顺序）
        - 限制结果数量

        返回最终的关键词列表。
        """
        seen: Set[str] = set()
        keywords: List[str] = []
        
        for token in tokens:
            # 规范化用于比较
            token_lower = token.lower()

            # 已见过则跳过（不区分大小写去重）
            if token_lower in seen:
                continue

            # 跳过停用词（同时检查原始和大小写）
            if token in self.config.stopwords or token_lower in self.config.stopwords:
                continue

            # 太短则跳过
            if len(token) < self.config.min_keyword_length:
                continue

            # 添加关键词（保留原始大小写）
            seen.add(token_lower)
            keywords.append(token)

            # 已足够则停止
            if len(keywords) >= self.config.max_keywords:
                break
        
        return keywords
    
    def add_stopwords(self, words: Set[str]) -> None:
        """向停用词集合中添加词项。"""
        self.config.stopwords.update(words)
    
    def remove_stopwords(self, words: Set[str]) -> None:
        """从停用词集合中移除词项。"""
        self.config.stopwords -= words


def create_query_processor(
    stopwords: Optional[Set[str]] = None,
    min_keyword_length: int = 1,
    max_keywords: int = 20,
    enable_filter_parsing: bool = True
) -> QueryProcessor:
    """创建并返回配置好的 `QueryProcessor`。

    参数：
        stopwords: 自定义停用词集合；若为 `None` 则使用默认停用词
        min_keyword_length: 最小关键词长度
        max_keywords: 最大关键词数量
        enable_filter_parsing: 是否解析过滤器语法

    返回：配置好的 `QueryProcessor` 实例。
    """
    config = QueryProcessorConfig(
        stopwords=stopwords if stopwords is not None else DEFAULT_STOPWORDS.copy(),
        min_keyword_length=min_keyword_length,
        max_keywords=max_keywords,
        enable_filter_parsing=enable_filter_parsing
    )
    return QueryProcessor(config)
