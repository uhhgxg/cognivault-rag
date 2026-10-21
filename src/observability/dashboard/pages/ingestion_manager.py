"""Ingestion Manager 页面 – 上传文件、触发入库、删除文档。

布局：
1. 文件上传器 + 集合选择器
2. 入库按钮 → 进度条（使用 on_progress 回调）
3. 文档列表，带删除按钮
"""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st

from src.observability.dashboard.services.data_service import DataService


def _run_ingestion(
    uploaded_file: "st.runtime.uploaded_file_manager.UploadedFile",
    collection: str,
    progress_bar: "st.delta_generator.DeltaGenerator",
    status_text: "st.delta_generator.DeltaGenerator",
) -> None:
    """把上传的文件保存到临时位置并运行流水线。"""
    from src.core.settings import load_settings
    from src.core.trace import TraceContext, TraceCollector
    from src.ingestion.pipeline import IngestionPipeline

    settings = load_settings()

    # 把上传的文件写到临时位置
    suffix = Path(uploaded_file.name).suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    _STAGE_LABELS = {
        "integrity": "🔍 Checking file integrity…",
        "load": "📄 Loading document…",
        "split": "✂️ Chunking document…",
        "transform": "🔄 Transforming chunks (LLM refine + enrich)…",
        "embed": "🔢 Encoding vectors…",
        "upsert": "💾 Storing to database…",
    }

    def on_progress(stage: str, current: int, total: int) -> None:
        frac = (current - 1) / total  # stage just started, show partial progress
        label = _STAGE_LABELS.get(stage, stage)
        progress_bar.progress(frac, text=f"[{current}/{total}] {label}")
        status_text.caption(label)

    trace = TraceContext(trace_type="ingestion")
    trace.metadata["source_path"] = uploaded_file.name
    trace.metadata["collection"] = collection
    trace.metadata["source"] = "dashboard"

    try:
        pipeline = IngestionPipeline(settings, collection=collection)
        pipeline.run(
            file_path=tmp_path,
            trace=trace,
            on_progress=on_progress,
        )
        progress_bar.progress(1.0, text="✅ Complete")
        status_text.success(f"Successfully ingested **{uploaded_file.name}** into collection **{collection}**.")
    except Exception as exc:
        status_text.error(f"Ingestion failed: {exc}")
    finally:
        TraceCollector().collect(trace)
        # 清理临时文件
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


def render() -> None:
    """渲染 Ingestion Manager 页面。"""
    st.header("📥 Ingestion Manager")

    # ── 上传区域 ─────────────────────────────────────────────
    st.subheader("📤 Upload & Ingest")

    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded = st.file_uploader(
            "Select a file to ingest",
            type=["pdf", "txt", "md", "docx"],
            key="ingest_uploader",
        )
    with col2:
        collection = st.text_input("Collection", value="default", key="ingest_collection")

    if uploaded is not None:
        if st.button("🚀 Start Ingestion", key="btn_ingest"):
            progress_bar = st.progress(0, text="Preparing…")
            status_text = st.empty()
            _run_ingestion(uploaded, collection.strip() or "default", progress_bar, status_text)

    st.divider()

    # ── 文档管理区域 ────────────────────────────────
    st.subheader("🗑️ Manage Documents")

    try:
        svc = DataService()
        docs = svc.list_documents()
    except Exception as exc:
        st.error(f"Failed to load documents: {exc}")
        return

    if not docs:
        st.info(
            "**No documents ingested yet.** "
            "Upload a PDF, TXT, MD, or DOCX file above and click \"Start Ingestion\" to begin."
        )
        return

    for idx, doc in enumerate(docs):
        col_info, col_btn = st.columns([4, 1])
        with col_info:
            st.markdown(
                f"**{doc['source_path']}** — "
                f"collection: `{doc.get('collection', '—')}` | "
                f"chunks: {doc['chunk_count']} | "
                f"images: {doc['image_count']}"
            )
        with col_btn:
            if st.button("🗑️ Delete", key=f"del_{idx}"):
                try:
                    result = svc.delete_document(
                        source_path=doc["source_path"],
                        collection=doc.get("collection", "default"),
                        source_hash=doc.get("source_hash"),
                    )
                    if result.success:
                        st.success(
                            f"Deleted: {result.chunks_deleted} chunks, "
                            f"{result.images_deleted} images removed."
                        )
                        st.rerun()
                    else:
                        st.warning(f"Partial delete. Errors: {result.errors}")
                except Exception as exc:
                    st.error(f"Delete failed: {exc}")
