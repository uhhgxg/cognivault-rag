"""
Embedding（向量化）模块。

本包包含 Embedding 服务的抽象接口与具体实现：
- Embedding 基类
- Embedding 工厂
- 各服务商实现（OpenAI、Azure、Ollama）
"""

from src.libs.embedding.azure_embedding import AzureEmbedding
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.embedding.ollama_embedding import OllamaEmbedding
from src.libs.embedding.openai_embedding import OpenAIEmbedding

__all__ = [
    "BaseEmbedding",
    "EmbeddingFactory",
    "OpenAIEmbedding",
    "AzureEmbedding",
    "OllamaEmbedding",
]
