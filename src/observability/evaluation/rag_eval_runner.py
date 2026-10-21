"""端到端 RAG 评估运行器。

与 EvalRunner 不同，RAGEvalRunner 会调用完整的 RAGWorkflow（包含
LangGraph 的 query rewrite、intent detect、memory management、generation），
然后使用 Ragas 对最终输出进行质量评估。

设计原则：
- 端到端：评估的是完整系统，而非仅检索模块
- 可观测：输出结构化报告，支持按 tag 分类统计
- 可对比：支持与 reference answer 的语义相似度计算
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.observability.evaluation.eval_runner import (
    EvalReport,
    GoldenTestCase,
    QueryResult,
    load_test_set,
)
from src.ragent_backend.workflow import RAGWorkflow

logger = logging.getLogger(__name__)


class RAGEvalRunner:
    """端到端 RAG 评估编排器。

    Example::

        runner = RAGEvalRunner(
            workflow=rag_workflow,
            ragas_evaluator=RagasEvaluator(settings),
        )
        report = await runner.run("tests/fixtures/golden_test_set_v2.json")
        print(report.aggregate_metrics)
    """

    def __init__(
        self,
        workflow: RAGWorkflow,
        ragas_evaluator: Any,
        custom_evaluator: Any = None,
    ) -> None:
        """初始化 RAGEvalRunner。

        Args:
            workflow: 用于端到端执行的 RAGWorkflow 实例。
            ragas_evaluator: 用于 LLM 评分指标的评估器（例如 Ragas）。
            custom_evaluator: 可选，用于 IR 指标的评估器（例如 hit_rate/MRR）。
        """
        self.workflow = workflow
        self.ragas_evaluator = ragas_evaluator
        self.custom_evaluator = custom_evaluator

    async def run(
        self,
        test_set_path: str | Path,
        top_k: int = 10,
    ) -> EvalReport:
        """对黄金测试集运行端到端评估。

        Args:
            test_set_path: golden_test_set_v2.json 的路径。
            top_k: 每条查询要检索的 chunk 数量。

        Returns:
            包含每条查询和汇总指标的 EvalReport。
        """
        test_cases = load_test_set(test_set_path)
        if not test_cases:
            raise ValueError("Golden test set is empty.")

        logger.info(
            "Starting end-to-end RAG evaluation: %d test cases",
            len(test_cases),
        )

        report = EvalReport(
            evaluator_name=type(self.ragas_evaluator).__name__,
            test_set_path=str(test_set_path),
        )

        t0 = time.monotonic()

        for idx, tc in enumerate(test_cases):
            logger.info("Evaluating [%d/%d]: %s", idx + 1, len(test_cases), tc.query[:60])
            qr = await self._evaluate_single(tc, top_k=top_k)
            report.query_results.append(qr)

        report.total_elapsed_ms = (time.monotonic() - t0) * 1000.0
        report.aggregate_metrics = self._aggregate_metrics(report.query_results)

        logger.info(
            "Evaluation complete: %d queries, aggregate=%s",
            len(report.query_results),
            report.aggregate_metrics,
        )

        return report

    async def _evaluate_single(
        self,
        test_case: GoldenTestCase,
        top_k: int = 10,
    ) -> QueryResult:
        """端到端地评估单个测试用例。"""
        t0 = time.monotonic()
        qr = QueryResult(query=test_case.query)

        # 第 1 步：运行完整的 RAGWorkflow
        thread_id = f"eval_{uuid.uuid4().hex[:8]}"
        initial_state = {
            "query": test_case.query,
            "conversation_id": thread_id,
            "task_id": f"eval_task_{uuid.uuid4().hex[:8]}",
            "top_k": top_k,
            # 如有 history，可注入（当前版本预留）
        }

        try:
            final_state = await self.workflow.run(initial_state, thread_id=thread_id)
        except Exception as exc:
            logger.warning("Workflow failed for '%s': %s", test_case.query[:40], exc)
            qr.elapsed_ms = (time.monotonic() - t0) * 1000.0
            return qr

        # 第 2 步：提取回答和检索到的上下文
        answer = final_state.get("final_answer", "")
        contexts = final_state.get("retrieval_contexts", [])
        if not contexts and final_state.get("retrieval_context"):
            contexts = [final_state["retrieval_context"]]

        qr.generated_answer = answer
        # 尝试从上下文中提取 chunk ids（当前以字符串形式存储，id 提取为 best effort）
        qr.retrieved_chunk_ids = self._extract_chunk_ids(contexts)

        # 第 3 步：Ragas 评估
        metrics: Dict[str, float] = {}
        if answer and self.ragas_evaluator is not None:
            try:
                ragas_metrics = self.ragas_evaluator.evaluate(
                    query=test_case.query,
                    retrieved_chunks=contexts,
                    generated_answer=answer,
                )
                metrics.update(ragas_metrics)
            except Exception as exc:
                logger.warning("Ragas evaluation failed for '%s': %s", test_case.query[:40], exc)

        # 第 4 步：自定义评估（IR 指标）
        if self.custom_evaluator is not None:
            try:
                ground_truth = (
                    {"ids": test_case.expected_chunk_ids}
                    if test_case.expected_chunk_ids
                    else None
                )
                custom_metrics = self.custom_evaluator.evaluate(
                    query=test_case.query,
                    retrieved_chunks=contexts,
                    generated_answer=answer,
                    ground_truth=ground_truth,
                )
                metrics.update(custom_metrics)
            except Exception as exc:
                logger.warning("Custom evaluation failed for '%s': %s", test_case.query[:40], exc)

        qr.metrics = metrics
        qr.elapsed_ms = (time.monotonic() - t0) * 1000.0
        return qr

    @staticmethod
    def _extract_chunk_ids(contexts: List[Any]) -> List[str]:
        """尽力从检索上下文中提取 chunk id。"""
        ids = []
        for ctx in contexts:
            if isinstance(ctx, dict):
                for key in ("chunk_id", "id"):
                    if key in ctx:
                        ids.append(str(ctx[key]))
                        break
            elif hasattr(ctx, "chunk_id"):
                ids.append(str(getattr(ctx, "chunk_id")))
        return ids

    @staticmethod
    def _aggregate_metrics(results: List[QueryResult]) -> Dict[str, float]:
        """计算所有查询结果的平均指标。"""
        if not results:
            return {}

        all_keys: set[str] = set()
        for qr in results:
            all_keys.update(qr.metrics.keys())

        averages: Dict[str, float] = {}
        for key in sorted(all_keys):
            values = [qr.metrics[key] for qr in results if key in qr.metrics]
            averages[key] = sum(values) / len(values) if values else 0.0

        return averages


def analyze_by_tags(report: EvalReport, test_cases: List[GoldenTestCase]) -> Dict[str, Dict[str, float]]:
    """按测试用例标签分组的指标汇总。

    Args:
        report: 由 RAGEvalRunner.run() 产生的 EvalReport。
        test_cases: 原始的黄金测试用例（必须与 report.query_results 一一对应）。

    Returns:
        映射：tag -> {metric_name: 平均值}。
    """
    tag_metrics: Dict[str, List[Dict[str, float]]] = {}

    for qr, tc in zip(report.query_results, test_cases):
        for tag in getattr(tc, "tags", []):
            tag_metrics.setdefault(tag, []).append(qr.metrics)

    result: Dict[str, Dict[str, float]] = {}
    for tag, metrics_list in tag_metrics.items():
        all_keys = set()
        for m in metrics_list:
            all_keys.update(m.keys())
        result[tag] = {}
        for key in sorted(all_keys):
            values = [m[key] for m in metrics_list if key in m]
            result[tag][key] = sum(values) / len(values) if values else 0.0

    return result
