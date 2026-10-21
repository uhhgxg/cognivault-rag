"""模块化 RAG 仪表盘 – 多页 Streamlit 应用。

入口：``streamlit run src/observability/dashboard/app.py``

页面通过 ``st.navigation()`` 注册，并由 ``pages/`` 下各自的
模块负责渲染。尚未实现的页面会显示占位提示信息。
"""

from __future__ import annotations

import streamlit as st


# ── 页面定义 ─────────────────────────────────────────────────

def _page_overview() -> None:
    from src.observability.dashboard.pages.overview import render
    render()


def _page_data_browser() -> None:
    from src.observability.dashboard.pages.data_browser import render
    render()


def _page_ingestion_manager() -> None:
    from src.observability.dashboard.pages.ingestion_manager import render
    render()


def _page_ingestion_traces() -> None:
    from src.observability.dashboard.pages.ingestion_traces import render
    render()


def _page_query_traces() -> None:
    from src.observability.dashboard.pages.query_traces import render
    render()


def _page_evaluation_panel() -> None:
    from src.observability.dashboard.pages.evaluation_panel import render
    render()


# ── 导航 ───────────────────────────────────────────────────────

pages = [
    st.Page(_page_overview, title="Overview", icon="📊", default=True),
    st.Page(_page_data_browser, title="Data Browser", icon="🔍"),
    st.Page(_page_ingestion_manager, title="Ingestion Manager", icon="📥"),
    st.Page(_page_ingestion_traces, title="Ingestion Traces", icon="🔬"),
    st.Page(_page_query_traces, title="Query Traces", icon="🔎"),
    st.Page(_page_evaluation_panel, title="Evaluation Panel", icon="📏"),
]


def main() -> None:
    st.set_page_config(
        page_title="Modular RAG Dashboard",
        page_icon="📊",
        layout="wide",
    )

    nav = st.navigation(pages)
    nav.run()


if __name__ == "__main__":
    main()
else:
    # 当通过 `streamlit run app.py` 直接运行时
    main()
