"""稀疏编码器，用于从文本分块生成 BM25 词项统计。

本模块实现摄取流水线中的稀疏编码器组件，
负责提取 BM25 索引所需的词项统计信息。

设计原则：
- 无状态处理：encode() 调用之间不保存内部状态
- 可观测：接收 TraceContext 以便未来接入可观测性
- 确定性：相同输入产生相同的词项统计
- 清晰约定：为下游 BM25Indexer 提供结构明确的输出
"""

from typing import List, Dict, Optional, Any
from collections import Counter
import re

import jieba

from src.core.types import Chunk


class SparseEncoder:
    """把文本分块编码成 BM25 词项统计。

    这个编码器准备 BM25 索引所需的词项级统计。
    实际的索引构建由 BM25Indexer（C12）完成。

    输出结构：
        每个分块产出：
        {
            "chunk_id": str,
            "term_frequencies": Dict[str, int],  # 词项 -> 在本分块中的出现次数
            "doc_length": int,                    # 分块中的词项总数
            "unique_terms": int                   # 分块中的词汇量
        }

    设计：
    - 分词：简单空格 + 小写化（后续可增强）
    - 停用词：默认无（未来迭代可加入）
    - 确定性：相同分块文本总是产生相同统计

    示例：
        >>> from src.core.types import Chunk
        >>> encoder = SparseEncoder()
        >>> 
        >>> chunks = [Chunk(id="1", text="Hello world hello", metadata={})]
        >>> stats = encoder.encode(chunks)
        >>> stats[0]["term_frequencies"]["hello"]  # 2
        >>> stats[0]["doc_length"]  # 3
    """
    
    def __init__(
        self,
        min_term_length: int = 2,
        lowercase: bool = True,
    ):
        """初始化 SparseEncoder。

        参数：
            min_term_length: 词项的最小字符长度（默认 2）
            lowercase: 是否把词项转成小写（默认 True）

        可能抛出：
            ValueError: 如果 min_term_length < 1
        """
        if min_term_length < 1:
            raise ValueError(f"min_term_length must be >= 1, got {min_term_length}")
        
        self.min_term_length = min_term_length
        self.lowercase = lowercase
    
    def encode(
        self,
        chunks: List[Chunk],
        trace: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """把分块编码成 BM25 词项统计。

        对每个分块提取：
        - 词频（词项 -> 出现次数）
        - 文档长度（词项总数）
        - 唯一词项数

        参数：
            chunks: 要编码的 Chunk 对象列表
            trace: 可选的 TraceContext，用于可观测性（预留给 Stage F）

        返回：
            统计信息字典列表（每个分块一个，顺序与输入一致）。
            每个字典包含：chunk_id、term_frequencies、doc_length、unique_terms

        可能抛出：
            ValueError: 如果分块列表为空
            ValueError: 如果某个分块文本为空

        示例：
            >>> chunks = [
            ...     Chunk(id="1", text="machine learning", metadata={}),
            ...     Chunk(id="2", text="deep learning networks", metadata={})
            ... ]
            >>> stats = encoder.encode(chunks)
            >>> len(stats) == len(chunks)  # True
            >>> stats[0]["term_frequencies"]["machine"]  # 1
            >>> stats[1]["doc_length"]  # 3
        """
        if not chunks:
            raise ValueError("Cannot encode empty chunks list")
        
        results = []
        
        for i, chunk in enumerate(chunks):
            # 校验分块文本
            if not chunk.text or not chunk.text.strip():
                raise ValueError(
                    f"Chunk at index {i} (id={chunk.id}) has empty or whitespace-only text"
                )
            
            # 分词并统计词项
            terms = self._tokenize(chunk.text)
            term_frequencies = Counter(terms)
            
            # 构建统计字典
            stat_dict = {
                "chunk_id": chunk.id,
                "term_frequencies": dict(term_frequencies),  # 把 Counter 转成 dict
                "doc_length": len(terms),
                "unique_terms": len(term_frequencies),
            }
            
            results.append(stat_dict)
        
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """把文本分词成词项。

        用 jieba 做中文分词，用正则处理英文。
        这样可以保证和查询侧（QueryProcessor）的分词方式一致，
        这是 BM25 匹配的前提。

        参数：
            text: 要分词的输入文本

        返回：
            有效的词项列表
        """
        tokens: List[str] = []

        # 用 jieba 分词（同时处理中文和英文）
        raw_tokens = jieba.lcut(text)

        # 清洗 token：只保留字母数字和中文
        for token in raw_tokens:
            token = token.strip()
            if not token:
                continue
            # 跳过纯标点 / 空白
            if re.fullmatch(r'[\s\W]+', token, re.UNICODE):
                continue
            tokens.append(token)
        
        # 如果配置了小写，则转换
        if self.lowercase:
            tokens = [t.lower() for t in tokens]
        
        # 按最小长度过滤
        terms = [t for t in tokens if len(t) >= self.min_term_length]
        
        return terms
    
    def get_corpus_stats(
        self,
        encoded_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """根据已编码的分块计算语料库级统计。

        工具方法，供 BM25Indexer 计算：
        - 平均文档长度
        - 文档频率（每个词项出现在多少篇文档中）
        - 文档总数

        参数：
            encoded_chunks: 来自 encode() 的统计字典列表

        返回：
            包含语料库级统计的字典：
            {
                "num_docs": int,
                "avg_doc_length": float,
                "document_frequency": Dict[str, int]  # 词项 -> 包含该词项的文档数
            }
        """
        if not encoded_chunks:
            return {
                "num_docs": 0,
                "avg_doc_length": 0.0,
                "document_frequency": {}
            }
        
        num_docs = len(encoded_chunks)
        total_length = sum(chunk["doc_length"] for chunk in encoded_chunks)
        avg_doc_length = total_length / num_docs if num_docs > 0 else 0.0
        
        # 计算每个词项的文档频率（DF）
        doc_freq: Dict[str, int] = {}
        for chunk_stats in encoded_chunks:
            # 本分块中的每个唯一词项给 DF 贡献 1
            for term in chunk_stats["term_frequencies"].keys():
                doc_freq[term] = doc_freq.get(term, 0) + 1
        
        return {
            "num_docs": num_docs,
            "avg_doc_length": avg_doc_length,
            "document_frequency": doc_freq,
        }
