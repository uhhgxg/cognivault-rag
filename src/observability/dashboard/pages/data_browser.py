"""Data Browser 页面 —— 浏览已入库的文档、片段和图片。

布局：
1. 集合选择器（侧边栏）
2. 文档列表，显示片段数量
3. 可展开的文档详情 → 片段卡片，包含文本和元数据
4. 图片预览画廊
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.observability.dashboard.services.data_service import DataService


def render() -> None:
    """渲染 Data Browser 页面。"""
    st.header("🔍 Data Browser")

    try:
        svc = DataService()
    except Exception as exc:
        st.error(f"Failed to initialise DataService: {exc}")
        return

    # ── 集合选择器 ────────────────────────────────────────
    collections = svc.list_collections()
    if "default" not in collections:
        collections.insert(0, "default")
    collection = st.selectbox(
        "Collection",
        options=collections,
        index=0,
        key="db_collection_filter",
    )
    coll_arg = collection if collection else None

    # ── 危险区域：清空所有数据 ────────────────────────────────
    st.divider()
    with st.expander("⚠️ Danger Zone", expanded=False):
        st.warning(
            "This will **permanently delete** all data: "
            "ChromaDB collections, BM25 indexes, images, ingestion history, and trace logs."
        )
        col_btn, col_status = st.columns([1, 2])
        with col_btn:
            if st.button("🗑️ Clear All Data", type="primary", key="btn_clear_all"):
                st.session_state["confirm_clear"] = True

        if st.session_state.get("confirm_clear"):
            st.error("Are you sure? This action cannot be undone!")
            c1, c2, _ = st.columns([1, 1, 2])
            with c1:
                if st.button("✅ Yes, delete everything", key="btn_confirm_clear"):
                    result = svc.reset_all()
                    st.session_state["confirm_clear"] = False
                    if result["errors"]:
                        st.warning(
                            f"Cleared with {len(result['errors'])} error(s): "
                            + "; ".join(result["errors"])
                        )
                    else:
                        st.success(
                            f"All data cleared! "
                            f"{result['collections_deleted']} collection(s) deleted."
                        )
                    st.rerun()
            with c2:
                if st.button("❌ Cancel", key="btn_cancel_clear"):
                    st.session_state["confirm_clear"] = False
                    st.rerun()

    st.divider()

    # ── 文档列表 ──────────────────────────────────────────────
    try:
        docs = svc.list_documents(coll_arg)
    except Exception as exc:
        st.error(f"Failed to load documents: {exc}")
        return

    if not docs:
        st.info(
            "**No documents found in this collection.** "
            "Use the Ingestion Manager page to upload and ingest files, "
            "or select a different collection from the dropdown above."
        )
        return

    st.subheader(f"📄 Documents ({len(docs)})")

    for idx, doc in enumerate(docs):
        source_name = Path(doc["source_path"]).name
        label = f"📑 {source_name}  —  {doc['chunk_count']} chunks · {doc['image_count']} images"
        with st.expander(label, expanded=(len(docs) == 1)):
            # ── 文档元数据 ──────────────────────────────────
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Chunks", doc["chunk_count"])
            col_b.metric("Images", doc["image_count"])
            col_c.metric("Collection", doc.get("collection", "—"))
            st.caption(
                f"**Source:** {doc['source_path']}  ·  "
                f"**Hash:** `{doc['source_hash'][:16]}…`  ·  "
                f"**Processed:** {doc.get('processed_at', '—')}"
            )

            st.divider()

            # ── 片段卡片 ────────────────────────────────────────
            chunks = svc.get_chunks(doc["source_hash"], coll_arg)
            if chunks:
                st.markdown(f"### 📦 Chunks ({len(chunks)})")
                for cidx, chunk in enumerate(chunks):
                    text = chunk.get("text", "")
                    meta = chunk.get("metadata", {})
                    chunk_id = chunk["id"]

                    # 标题取自元数据，否则取首行文本
                    title = meta.get("title", "")
                    if not title:
                        title = text[:60].replace("\n", " ").strip()
                        if len(text) > 60:
                            title += "…"

                    with st.container(border=True):
                        st.markdown(
                            f"**Chunk {cidx + 1}** · `{chunk_id[-16:]}` · "
                            f"{len(text)} chars"
                        )
                        # 显示片段实际文本（可滚动）
                        _height = max(120, min(len(text) // 2, 600))
                        st.text_area(
                            "Content",
                            value=text,
                            height=_height,
                            disabled=True,
                            key=f"chunk_text_{idx}_{cidx}",
                            label_visibility="collapsed",
                        )
                        # 可展开的元数据
                        with st.expander("📋 Metadata", expanded=False):
                            st.json(meta)
            else:
                st.caption("No chunks found in vector store for this document.")

            # ── 图片预览 ──────────────────────────────────────
            images = svc.get_images(doc["source_hash"], coll_arg)
            if images:
                st.divider()
                st.markdown(f"### 🖼️ Images ({len(images)})")
                img_cols = st.columns(min(len(images), 4))
                for iidx, img in enumerate(images):
                    with img_cols[iidx % len(img_cols)]:
                        img_path = Path(img.get("file_path", ""))
                        if img_path.exists():
                            st.image(str(img_path), caption=img["image_id"], width=200)
                        else:
                            st.caption(f"{img['image_id']} (file missing)")
