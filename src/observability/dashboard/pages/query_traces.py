"""Query Traces 页面 – 浏览查询 trace 历史，并展示各阶段耗时的瀑布图。

布局：
1. 可选的关键词搜索过滤
2. trace 列表（按时间倒序，过滤为 trace_type=="query"）
3. 详情视图：阶段耗时瀑布图 + Dense 与 Sparse 对比 + Rerank 变化
4. 每条 trace 的 Ragas 评估按钮（LLM 评分）
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import streamlit as st

from src.observability.dashboard.services.trace_service import TraceService

logger = logging.getLogger(__name__)


def render() -> None:
    """渲染 Query Traces 页面。"""
    st.header("🔎 Query Traces")

    svc = TraceService()
    traces = svc.list_traces(trace_type="query")

    if not traces:
        st.info("No query traces recorded yet. Run a query first!")
        return

    # ── 关键词过滤 ─────────────────────────────────────────────
    keyword = st.text_input(
        "Search by query keyword",
        value="",
        key="qt_keyword",
    )
    if keyword.strip():
        kw = keyword.strip().lower()
        traces = [
            t
            for t in traces
            if kw in str(t.get("metadata", {})).lower()
            or kw in str(t.get("stages", [])).lower()
        ]

    st.subheader(f"📋 Query History ({len(traces)})")

    for idx, trace in enumerate(traces):
        trace_id = trace.get("trace_id", "unknown")
        started = trace.get("started_at", "—")
        total_ms = trace.get("elapsed_ms")
        total_label = f"{total_ms:.0f} ms" if total_ms is not None else "—"
        meta = trace.get("metadata", {})
        query_text = meta.get("query", "")
        source = meta.get("source", "unknown")

        # ── 展开器标题：显示查询文本 ────────────────────
        query_preview = (
            query_text[:40] + "…" if len(query_text) > 40 else query_text
        ) if query_text else "—"
        expander_title = (
            f"🔍 \"{query_preview}\"  ·  {total_label}  ·  {started[:19]}"
        )

        with st.expander(expander_title, expanded=(idx == 0)):
            # ── 1. 查询概览 ──────────────────────────────
            st.markdown("#### 💬 Query")
            col_q, col_meta = st.columns([3, 1])
            with col_q:
                st.markdown(f"> {query_text}")
            with col_meta:
                source_emoji = "🤖" if source == "mcp" else "📡"
                st.markdown(f"**Source:** {source_emoji} `{source}`")
                st.markdown(f"**Top-K:** `{meta.get('top_k', '—')}`")
                st.markdown(f"**Collection:** `{meta.get('collection', '—')}`")

            st.divider()

            # ── 2. 概览指标 ────────────────────────────
            timings = svc.get_stage_timings(trace)
            stages_by_name = {t["stage_name"]: t for t in timings}

            dense_d = (stages_by_name.get("dense_retrieval", {}).get("data") or {})
            sparse_d = (stages_by_name.get("sparse_retrieval", {}).get("data") or {})
            fusion_d = (stages_by_name.get("fusion", {}).get("data") or {})
            rerank_d = (stages_by_name.get("rerank", {}).get("data") or {})

            dense_count = dense_d.get("result_count", 0)
            sparse_count = sparse_d.get("result_count", 0)
            fusion_count = fusion_d.get("result_count", 0)
            rerank_count = rerank_d.get("output_count", 0)

            rc1, rc2, rc3, rc4, rc5 = st.columns(5)
            with rc1:
                st.metric("Dense Hits", dense_count)
            with rc2:
                st.metric("Sparse Hits", sparse_count)
            with rc3:
                st.metric("Fused", fusion_count or (dense_count + sparse_count))
            with rc4:
                st.metric("After Rerank", rerank_count if rerank_d else "—")
            with rc5:
                st.metric("Total Time", total_label)

            # ── 诊断提示 ───────────────────────────────
            _render_diagnostics(
                stages_by_name, dense_d, sparse_d, fusion_d, rerank_d,
                dense_count, sparse_count,
            )

            st.divider()

            # ── 3. 阶段耗时瀑布图 ──────────────────────
            main_stage_names = ("query_processing", "dense_retrieval", "sparse_retrieval", "fusion", "rerank")
            main_timings = [t for t in timings if t["stage_name"] in main_stage_names]
            if main_timings:
                st.markdown("#### ⏱️ Stage Timings")
                chart_data = {t["stage_name"]: t["elapsed_ms"] for t in main_timings}
                st.bar_chart(chart_data, horizontal=True)
                st.table([
                    {
                        "Stage": t["stage_name"],
                        "Elapsed (ms)": round(t["elapsed_ms"], 2),
                    }
                    for t in main_timings
                ])

            st.divider()

            # ── 4. 各阶段详情标签页 ───────────────────────
            st.markdown("#### 🔍 Stage Details")

            tab_defs = []
            if "query_processing" in stages_by_name:
                tab_defs.append(("🔤 Query Processing", "query_processing"))
            if "dense_retrieval" in stages_by_name:
                tab_defs.append(("🟦 Dense Retrieval", "dense_retrieval"))
            if "sparse_retrieval" in stages_by_name:
                tab_defs.append(("🟨 Sparse Retrieval", "sparse_retrieval"))
            if "fusion" in stages_by_name:
                tab_defs.append(("🟩 Fusion (RRF)", "fusion"))
            if "rerank" in stages_by_name:
                tab_defs.append(("🟪 Rerank", "rerank"))

            if tab_defs:
                tabs = st.tabs([label for label, _ in tab_defs])
                for tab, (label, key) in zip(tabs, tab_defs):
                    with tab:
                        stage = stages_by_name[key]
                        data = stage.get("data", {})
                        elapsed = stage.get("elapsed_ms")
                        if elapsed is not None:
                            st.caption(f"⏱️ {elapsed:.1f} ms")

                        if key == "query_processing":
                            _render_query_processing_stage(data)
                        elif key == "dense_retrieval":
                            _render_retrieval_stage(data, "Dense", trace_idx=idx)
                        elif key == "sparse_retrieval":
                            _render_retrieval_stage(data, "Sparse", trace_idx=idx)
                        elif key == "fusion":
                            _render_fusion_stage(data, trace_idx=idx)
                        elif key == "rerank":
                            _render_rerank_stage(data, trace_idx=idx)
            else:
                st.info("No stage details available.")

            # ── 5. Ragas 评估按钮 ───────────────────────
            _render_evaluate_button(trace, idx)


def _render_diagnostics(
    stages_by_name: Dict[str, Any],
    dense_d: Dict[str, Any],
    sparse_d: Dict[str, Any],
    fusion_d: Dict[str, Any],
    rerank_d: Dict[str, Any],
    dense_count: int,
    sparse_count: int,
) -> None:
    """渲染关于缺失或出错流水线阶段的诊断提示。"""
    hints: list = []

    # Dense 检索的错误
    dense_err = dense_d.get("error", "")
    if dense_err:
        hints.append(("error", f"**Dense Retrieval failed:** {dense_err}"))
    elif dense_count == 0 and "dense_retrieval" in stages_by_name:
        hints.append(("warning", "Dense Retrieval returned **0 results**. Check if the collection has indexed data."))

    # Sparse 检索的错误 / 为空
    sparse_err = sparse_d.get("error", "")
    if sparse_err:
        hints.append(("error", f"**Sparse Retrieval failed:** {sparse_err}"))
    elif sparse_count == 0 and "sparse_retrieval" in stages_by_name:
        hints.append((
            "warning",
            "Sparse (BM25) Retrieval returned **0 results**. "
            "BM25 index may be empty or not yet built for this collection.",
        ))

    # 缺少 Fusion 阶段
    if "fusion" not in stages_by_name:
        if dense_count > 0 and sparse_count > 0:
            hints.append(("info", "Fusion stage was not recorded even though both retrievers returned results."))
        elif dense_count == 0 or sparse_count == 0:
            only_source = "Dense" if dense_count > 0 else ("Sparse" if sparse_count > 0 else "neither")
            hints.append((
                "info",
                f"**Fusion (RRF) skipped:** only {only_source} retrieval returned results. "
                "Fusion requires both Dense and Sparse results to merge.",
            ))

    # 缺少 Rerank 阶段
    if "rerank" not in stages_by_name:
        if dense_count > 0 or sparse_count > 0:
            hints.append((
                "info",
                "**Rerank skipped:** reranker is not enabled or not configured. "
                "Enable `reranker` in settings.yaml to apply LLM-based reranking.",
            ))

    # 所有结果都为空
    if dense_count == 0 and sparse_count == 0:
        hints.append((
            "warning",
            "**No results found.** The collection may be empty, or the query "
            "doesn't match any indexed content. Try ingesting data first.",
        ))

    # 渲染提示
    for level, msg in hints:
        if level == "error":
            st.error(msg)
        elif level == "warning":
            st.warning(msg)
        else:
            st.info(msg)


def _render_evaluate_button(trace: Dict[str, Any], idx: int) -> None:
    """为单条查询 trace 渲染一个 Ragas 评估按钮。

    会针对存储的查询重新运行检索，并用
    RagasEvaluator（LLM 评分）进行评估。仅当 trace 元数据中
    可用查询文本时才有效。
    """
    meta = trace.get("metadata", {})
    query = meta.get("query", "")
    if not query:
        return

    st.divider()
    st.markdown("#### 📏 Ragas Evaluation")
    st.caption(
        "RAGAS 需要 **Query + Retrieved Context + Answer** 三要素来评估。"
        "日志中仅包含 Query 和检索到的上下文，请在下方输入实际回答后再运行评估。"
    )

    # 回答输入框 —— 用户提供实际生成的回答
    answer_key = f"eval_answer_{idx}"
    user_answer = st.text_area(
        "✏️ Generated Answer (回答)",
        value=st.session_state.get(answer_key, ""),
        height=120,
        key=answer_key,
        placeholder="请输入系统生成的回答，或粘贴 LLM 的实际输出…",
        help=(
            "Ragas 使用 LLM-as-Judge 评估回答质量。"
            "faithfulness 衡量回答是否忠于检索到的上下文，"
            "answer_relevancy 衡量回答与问题的相关性。"
            "如果不填写回答，将无法获得有意义的评估结果。"
        ),
    )

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        clicked = st.button(
            "📏 Ragas Evaluate",
            key=f"eval_trace_{idx}",
            help="Re-run this query and score with Ragas (LLM-as-Judge)",
            disabled=not user_answer.strip(),
        )
    with col_info:
        if not user_answer.strip():
            st.warning("⚠️ 请先在上方输入回答内容，再运行 Ragas 评估。")
        else:
            st.caption(
                "Uses Ragas to score faithfulness, answer relevancy, "
                "and context precision. Calls LLM — may take a few seconds."
            )

    # 显示 session state 中保存的上次结果
    result_key = f"eval_result_{idx}"
    if result_key in st.session_state and not clicked:
        _display_eval_metrics(st.session_state[result_key])

    if clicked:
        with st.spinner("Running Ragas evaluation…"):
            result = _evaluate_single_trace(query, meta, user_answer=user_answer.strip())
        st.session_state[result_key] = result
        _display_eval_metrics(result)


def _evaluate_single_trace(
    query: str,
    meta: Dict[str, Any],
    user_answer: Optional[str] = None,
) -> Dict[str, Any]:
    """重新运行检索并用 Ragas 评估单条查询。

    返回的字典含 'metrics'（分数字典）或 'error'（字符串）。
    """
    try:
        from dataclasses import replace as dc_replace

        from src.core.settings import load_settings, EvaluationSettings
        from src.libs.evaluator.evaluator_factory import EvaluatorFactory

        settings = load_settings()

        # 覆盖评估配置以强制使用 Ragas（frozen dataclass，用 replace）
        ragas_eval = EvaluationSettings(
            enabled=True,
            provider="ragas",
            metrics=["faithfulness", "answer_relevancy", "context_precision"],
        )
        settings = dc_replace(settings, evaluation=ragas_eval)
        evaluator = EvaluatorFactory.create(settings)

        # 重新运行检索
        collection = meta.get("collection", "default")
        top_k = meta.get("top_k", 10)
        chunks = _retrieve_chunks(settings, query, top_k, collection)

        if not chunks:
            return {"error": "No chunks retrieved — is data indexed?"}

        # 使用用户提供的回答；仅作为最后兜底时才拼接 chunk 文本
        # （这样得到的 RAGAS 分数意义不大）。
        if user_answer:
            answer = user_answer
        else:
            _MAX_ANSWER_CHARS = 1500
            texts = []
            for c in chunks:
                if hasattr(c, "text"):
                    texts.append(c.text)
                elif isinstance(c, dict):
                    texts.append(c.get("text", str(c)))
                else:
                    texts.append(str(c))
            answer = " ".join(texts[:3])
            if len(answer) > _MAX_ANSWER_CHARS:
                answer = answer[:_MAX_ANSWER_CHARS]

        # 评估
        metrics = evaluator.evaluate(
            query=query,
            retrieved_chunks=chunks,
            generated_answer=answer,
        )
        return {"metrics": metrics, "answer_used": answer}

    except ImportError as exc:
        return {"error": f"Ragas not installed: {exc}"}
    except Exception as exc:
        logger.exception("Ragas evaluation failed")
        return {"error": str(exc)}


def _retrieve_chunks(
    settings: Any,
    query: str,
    top_k: int,
    collection: str,
) -> list:
    """重新运行 HybridSearch + Rerank，检索用于评估的 chunk。"""
    try:
        from src.core.query_engine.hybrid_search import create_hybrid_search
        from src.core.query_engine.query_processor import QueryProcessor
        from src.core.query_engine.dense_retriever import create_dense_retriever
        from src.core.query_engine.sparse_retriever import create_sparse_retriever
        from src.core.query_engine.reranker import create_core_reranker
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        from src.libs.embedding.embedding_factory import EmbeddingFactory
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory

        vector_store = VectorStoreFactory.create(
            settings, collection_name=collection,
        )
        embedding_client = EmbeddingFactory.create(settings)
        dense_retriever = create_dense_retriever(
            settings=settings,
            embedding_client=embedding_client,
            vector_store=vector_store,
        )
        from src.core.settings import resolve_path
        bm25_indexer = BM25Indexer(index_dir=str(resolve_path(f"data/db/bm25/{collection}")))
        sparse_retriever = create_sparse_retriever(
            settings=settings,
            bm25_indexer=bm25_indexer,
            vector_store=vector_store,
        )
        sparse_retriever.default_collection = collection
        query_processor = QueryProcessor()
        hybrid_search = create_hybrid_search(
            settings=settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
        )

        # 如果启用了重排，则检索更多候选结果
        reranker = create_core_reranker(settings=settings)
        initial_top_k = top_k * 2 if reranker.is_enabled else top_k

        results = hybrid_search.search(query=query, top_k=initial_top_k)
        results = results if isinstance(results, list) else results.results

        # 如果启用了重排，则应用重排
        if reranker.is_enabled and results:
            rerank_result = reranker.rerank(query=query, results=results, top_k=top_k)
            results = rerank_result.results

        return results
    except Exception as exc:
        logger.warning("Retrieval for evaluation failed: %s", exc)
        return []


def _display_eval_metrics(result: Dict[str, Any]) -> None:
    """显示评估结果（指标或错误）。"""
    if "error" in result:
        st.error(f"❌ Evaluation failed: {result['error']}")
        return

    metrics = result.get("metrics", {})
    if not metrics:
        st.warning("No metrics returned.")
        return

    st.markdown("**📏 Ragas Scores**")
    cols = st.columns(min(len(metrics), 4))
    for i, (name, value) in enumerate(sorted(metrics.items())):
        with cols[i % len(cols)]:
            st.metric(
                label=name.replace("_", " ").title(),
                value=f"{value:.4f}",
            )


def _extract_pipeline_chunks(
    timings: List[Dict[str, Any]],
    meta: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    """从每个流水线阶段中提取 chunk 列表。"""
    result: Dict[str, List[Dict[str, Any]]] = {}
    for stage in timings:
        name = stage.get("stage_name", "")
        data = stage.get("data") or {}
        chunks = data.get("chunks")
        if chunks and isinstance(chunks, list):
            result[name] = chunks
    final = meta.get("final_results") or meta.get("results")
    if final and isinstance(final, list):
        result["final"] = final
    return result


# ═══════════════════════════════════════════════════════════════
# 各阶段的渲染函数
# ═══════════════════════════════════════════════════════════════

def _render_query_processing_stage(data: Dict[str, Any]) -> None:
    """渲染 Query Processing 阶段：原始查询 → 关键词。"""
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Original Query**")
        st.info(data.get("original_query", "—"))
    with c2:
        st.markdown("**Method**")
        st.code(data.get("method", "—"))

    keywords = data.get("keywords", [])
    if keywords:
        st.markdown("**Extracted Keywords**")
        st.markdown(" · ".join(f"`{kw}`" for kw in keywords))
    else:
        st.warning("No keywords extracted.")


def _render_retrieval_stage(data: Dict[str, Any], label: str, *, trace_idx: int = 0) -> None:
    """渲染 Dense 或 Sparse 检索阶段：方法、数量、chunk 列表。"""
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Method", data.get("method", "—"))
    with c2:
        extra = data.get("provider", data.get("keyword_count", "—"))
        extra_label = "Provider" if "provider" in data else "Keywords"
        st.metric(extra_label, extra)
    with c3:
        st.metric("Results", data.get("result_count", 0))

    st.markdown(f"**Top-K requested:** `{data.get('top_k', '—')}`")

    chunks = data.get("chunks", [])
    if chunks:
        _render_chunk_list(chunks, prefix=f"{label.lower().replace(' ', '_')}_chunk_{trace_idx}")
    else:
        st.info(f"No {label.lower()} results returned.")


def _render_fusion_stage(data: Dict[str, Any], *, trace_idx: int = 0) -> None:
    """渲染 Fusion（RRF）阶段：输入列表、融合后的结果数量、chunk 列表。"""
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Method", data.get("method", "rrf"))
    with c2:
        st.metric("Input Lists", data.get("input_lists", "—"))
    with c3:
        st.metric("Fused Results", data.get("result_count", 0))

    st.markdown(f"**Top-K:** `{data.get('top_k', '—')}`")

    chunks = data.get("chunks", [])
    if chunks:
        _render_chunk_list(chunks, prefix=f"fusion_chunk_{trace_idx}")
    else:
        st.info("No fusion results.")


def _render_rerank_stage(data: Dict[str, Any], *, trace_idx: int = 0) -> None:
    """渲染 Rerank 阶段：方法、输入/输出数量、重排后的 chunk 列表。"""
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Method", data.get("method", "—"))
    with c2:
        st.metric("Provider", data.get("provider", "—"))
    with c3:
        st.metric("Input", data.get("input_count", "—"))
    with c4:
        st.metric("Output", data.get("output_count", "—"))

    chunks = data.get("chunks", [])
    if chunks:
        _render_chunk_list(chunks, prefix=f"rerank_chunk_{trace_idx}")
    else:
        st.info("No reranked results.")


def _render_chunk_list(chunks: List[Dict[str, Any]], prefix: str = "chunk") -> None:
    """把 chunk 字典列表渲染成紧凑、可读的表格，文本可展开。"""
    for ci, chunk in enumerate(chunks):
        score = chunk.get("score", 0)
        text = chunk.get("text", "")
        chunk_id = chunk.get("chunk_id", "")
        source = chunk.get("source", "")
        title = chunk.get("title", "")

        # 按分数着色的指示器
        if score >= 0.8:
            score_bar = "🟢"
        elif score >= 0.5:
            score_bar = "🟡"
        else:
            score_bar = "🔴"

        header = f"{score_bar} **#{ci + 1}** — Score: `{score:.4f}`"
        if title:
            header += f" — {title}"

        with st.expander(header, expanded=False):
            cols = st.columns([2, 3])
            with cols[0]:
                st.caption(f"Chunk ID: `{chunk_id}`")
            with cols[1]:
                if source:
                    st.caption(f"Source: `{source}`")
            # 显示 chunk 文本（可滚动）
            if text:
                st.text_area(
                    f"{prefix}_{ci}",
                    value=text,
                    height=max(80, min(len(text) // 2, 400)),
                    disabled=True,
                    label_visibility="collapsed",
                )
            else:
                st.caption("_No text available_")


def _find_stage(timings, name):
    """按名称查找一个 stage 字典，找不到则返回 None。"""
    for t in timings:
        if t["stage_name"] == name:
            return t
    return None
