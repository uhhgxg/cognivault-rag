"""基于 LLM 的重排序器实现。

本模块使用大语言模型（LLM）来评估候选段落与查询的相关性，从而实现重排序。
它从 config/prompts/rerank.txt 读取提示词，并把 LLM 的输出整理成
下游可处理的结构化格式。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.settings import resolve_path
from src.libs.llm.base_llm import BaseLLM, Message
from src.libs.llm.llm_factory import LLMFactory
from src.libs.reranker.base_reranker import BaseReranker


class LLMRerankError(RuntimeError):
    """当 LLM 重排序失败时抛出。"""


class LLMReranker(BaseReranker):
    """使用结构化提示词的、基于 LLM 的重排序器。

    本实现借助 LLM，根据候选段落与查询的相关性来打分和重排。
    它从可配置的文件中读取重排提示词，并期望 LLM 输出结构化的 JSON。

    应用的设计原则：
    - 可插拔：可以通过工厂与其他重排序器实现互换。
    - 配置驱动：提示词文件路径和 LLM 设置都来自配置。
    - 可观测：支持 TraceContext，用于监控（Stage F 接入）。
    - 兜底感知：失败时返回信号，方便上游做兜底处理。
    - 结构化输出：会校验 LLM 输出是否符合预期格式。
    """
    
    def __init__(
        self,
        settings: Any,
        prompt_path: Optional[str] = None,
        llm: Optional[BaseLLM] = None,
        **kwargs: Any
    ) -> None:
        """初始化 LLM 重排序器。

        参数：
            settings: 应用配置，里面包含 LLM 和重排相关配置。
            prompt_path: 可选，重排提示词文件路径。如果为 None，
                就用默认的 'config/prompts/rerank.txt'。测试时可用来注入自定义提示词。
            llm: 可选，LLM 实例。如果为 None，就用 LLMFactory 根据 settings 创建。
                测试时可用来注入 mock LLM。
            **kwargs: 其他提供者特有参数。
        """
        self.settings = settings
        self.prompt_path = prompt_path or str(resolve_path("config/prompts/rerank.txt"))
        self.llm = llm or LLMFactory.create(settings)
        self.kwargs = kwargs

        # 加载提示词模板
        try:
            self.prompt_template = self._load_prompt_template(self.prompt_path)
        except Exception as e:
            raise LLMRerankError(f"Failed to load rerank prompt from {self.prompt_path}: {e}") from e

    def _load_prompt_template(self, path: str) -> str:
        """从文件加载重排提示词模板。

        参数：
            path: 提示词模板文件的路径。

        返回：
            提示词模板字符串。

        抛出：
            FileNotFoundError: 如果提示词文件不存在。
            IOError: 如果文件读不出来。
        """
        prompt_file = Path(path)
        if not prompt_file.exists():
            raise FileNotFoundError(f"Rerank prompt file not found: {path}")
        
        return prompt_file.read_text(encoding="utf-8")
    
    def _build_rerank_prompt(self, query: str, candidates: List[Dict[str, Any]]) -> str:
        """根据查询和候选项构造重排提示词。

        参数：
            query: 用户的查询字符串。
            candidates: 要重排的候选记录列表。

        返回：
            拼好的提示词字符串，可以直接发给 LLM。
        """
        # 把候选项格式化进提示词
        candidates_text = []
        for i, candidate in enumerate(candidates):
            passage_id = candidate.get("id", f"passage_{i}")
            text = candidate.get("text", candidate.get("content", ""))
            candidates_text.append(f"Passage ID: {passage_id}\nText: {text}\n")

        candidates_str = "\n".join(candidates_text)

        # 拼出完整的提示词
        full_prompt = f"{self.prompt_template}\n\nQuery: {query}\n\nPassages:\n{candidates_str}\n\nOutput your response as a JSON array of objects, one per passage."

        return full_prompt

    def _parse_llm_response(self, response_text: str) -> List[Dict[str, Any]]:
        """解析并校验 LLM 的返回。

        参数：
            response_text: LLM 返回的原始文本。

        返回：
            解析后的排序记录列表，每条包含 passage_id 和 score。

        抛出：
            LLMRerankError: 如果返回不符合预期格式。
        """
        # 尝试从返回里提取 JSON（LLM 可能会用 markdown 包裹）
        text = response_text.strip()

        # 去掉 markdown 代码块（如果有的话）
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # 解析 JSON
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMRerankError(
                f"LLM response is not valid JSON: {e}\nResponse: {response_text[:200]}"
            ) from e

        # 校验结构
        if not isinstance(parsed, list):
            raise LLMRerankError(
                f"Expected JSON array, got {type(parsed).__name__}\nResponse: {response_text[:200]}"
            )

        # 校验每一项
        for i, item in enumerate(parsed):
            if not isinstance(item, dict):
                raise LLMRerankError(
                    f"Item {i} is not a dict (type: {type(item).__name__})"
                )
            if "passage_id" not in item:
                raise LLMRerankError(f"Item {i} missing required field 'passage_id'")
            if "score" not in item:
                raise LLMRerankError(f"Item {i} missing required field 'score'")

            # 校验 score 是数字
            score = item["score"]
            if not isinstance(score, (int, float)):
                raise LLMRerankError(
                    f"Item {i} score must be numeric, got {type(score).__name__}: {score}"
                )

        return parsed
    
    def _map_results_to_candidates(
        self,
        parsed_results: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """把 LLM 的排序结果映射回原始的候选对象。

        参数：
            parsed_results: 解析后的 LLM 输出，含 passage_id 和 score。
            candidates: 原始的候选列表。

        返回：
            重排后的候选对象列表，按得分从高到低排序。
        """
        # 建立 id 到候选对象的映射
        id_to_candidate = {}
        for i, candidate in enumerate(candidates):
            candidate_id = candidate.get("id", f"passage_{i}")
            id_to_candidate[candidate_id] = candidate

        # 构造重排后的列表
        reranked = []
        for result in parsed_results:
            passage_id = result["passage_id"]
            score = result["score"]

            if passage_id in id_to_candidate:
                candidate = id_to_candidate[passage_id].copy()
                # 用 LLM 给出的重排得分覆盖
                candidate["rerank_score"] = float(score)
                reranked.append(candidate)

        # 按重排得分从高到低排序
        reranked.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        return reranked
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """用 LLM 基于相关性的打分来重排候选项。

        参数：
            query: 用户的查询字符串。
            candidates: 要重排的候选记录列表。
            trace: 可选的 TraceContext，用于可观测性。
            **kwargs: 额外参数（timeout、temperature 等）。

        返回：
            重排后的候选列表，按 LLM 给出的相关性得分排序。
            成功时，每个候选的 metadata 里会带上 'rerank_score'。

        抛出：
            ValueError: 如果 query 或 candidates 不合法。
            LLMRerankError: 如果 LLM 调用失败或返回格式不对。
        """
        # 校验输入
        self.validate_query(query)
        self.validate_candidates(candidates)

        # 如果只有一个候选项，就没必要重排了
        if len(candidates) == 1:
            return candidates

        # 构造提示词
        try:
            prompt = self._build_rerank_prompt(query, candidates)
        except Exception as e:
            raise LLMRerankError(f"Failed to build rerank prompt: {e}") from e

        # 调用 LLM
        try:
            messages = [Message(role="user", content=prompt)]
            response = self.llm.chat(messages, trace=trace, **kwargs)
            response_text = response.content
        except Exception as e:
            # 返回兜底信号——让上游决定怎么处理
            raise LLMRerankError(f"LLM call failed during reranking: {e}") from e

        # 解析返回
        try:
            parsed_results = self._parse_llm_response(response_text)
        except LLMRerankError:
            raise
        except Exception as e:
            raise LLMRerankError(f"Failed to parse LLM rerank response: {e}") from e

        # 映射回候选项并排序
        try:
            reranked = self._map_results_to_candidates(parsed_results, candidates)
        except Exception as e:
            raise LLMRerankError(f"Failed to map LLM results to candidates: {e}") from e

        return reranked
