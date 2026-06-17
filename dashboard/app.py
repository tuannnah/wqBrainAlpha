"""Dashboard Streamlit theo dõi alpha, GA, submissions, correlation.

Chạy: streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Cho phép import config/src khi chạy từ thư mục gốc dự án.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text  # noqa: E402

from src.storage.db import make_engine  # noqa: E402

st.set_page_config(page_title="WQ Alpha Dashboard", layout="wide")
engine = make_engine()


def _read(query: str) -> pd.DataFrame:
    try:
        with engine.connect() as conn:
            return pd.read_sql(text(query), conn)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Không đọc được dữ liệu: {exc}")
        return pd.DataFrame()


st.title("WorldQuant Brain — Auto-Alpha Dashboard")

tab_overview, tab_explorer, tab_ga, tab_subs, tab_corr = st.tabs(
    ["Overview", "Explorer", "Tiến trình", "Submissions", "Correlation"]
)

with tab_overview:
    sims = _read("SELECT * FROM simulations")
    alphas = _read("SELECT * FROM alphas")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tổng alpha", len(alphas))
    col2.metric("Tổng simulation", len(sims))
    if not sims.empty:
        passed = (sims["status"] == "passed").sum()
        col3.metric("Pass rate", f"{passed / len(sims) * 100:.1f}%")
        col4.metric("Avg Sharpe", f"{sims['sharpe'].dropna().mean():.3f}")
        st.subheader("Phân phối Sharpe")
        st.bar_chart(sims["sharpe"].dropna())

with tab_explorer:
    sims = _read(
        "SELECT a.id, a.expression, a.hypothesis, a.description, a.source, a.parent_id, "
        "s.sharpe, s.fitness, s.turnover, s.drawdown, s.score, s.status "
        "FROM simulations s JOIN alphas a ON s.alpha_id = a.id ORDER BY s.score DESC"
    )
    if not sims.empty:
        col_f1, col_f2 = st.columns(2)
        min_sharpe = col_f1.slider("Sharpe tối thiểu", -2.0, 5.0, 0.0, 0.1)
        sources = sorted(sims["source"].dropna().unique().tolist())
        pick = col_f2.multiselect("Nguồn", sources, default=sources)
        view = sims[(sims["sharpe"].fillna(-99) >= min_sharpe) & (sims["source"].isin(pick))]
        st.dataframe(
            view[["expression", "source", "sharpe", "fitness", "turnover", "drawdown", "score", "status"]],
            use_container_width=True,
        )
        st.subheader("Chi tiết alpha (giả thuyết + mô tả)")
        if not view.empty:
            idx = st.selectbox(
                "Chọn alpha", view.index,
                format_func=lambda i: f"{view.loc[i, 'expression'][:60]} (score={view.loc[i, 'score']})",
            )
            row = view.loc[idx]
            st.code(row["expression"], language="text")
            hyp = row.get("hypothesis")
            if isinstance(hyp, str) and hyp.strip():
                try:
                    st.json(json.loads(hyp))
                except (ValueError, TypeError):
                    st.write(hyp)
            if isinstance(row.get("description"), str) and row["description"].strip():
                st.caption("Mô tả: " + row["description"])
    else:
        st.info("Chưa có simulation nào.")

with tab_ga:
    # T7.4: tiến trình cải thiện theo vòng — lineage tinh chỉnh qua parent_id.
    chain = _read(
        "SELECT a.id, a.parent_id, a.expression, s.score, s.sharpe "
        "FROM alphas a JOIN simulations s ON s.alpha_id = a.id WHERE a.source = 'llm'"
    )
    if not chain.empty and chain["parent_id"].notna().any():
        roots = chain[chain["parent_id"].isna()]
        st.caption("Chuỗi tinh chỉnh (mỗi seed → các đời cải tiến).")
        for _, root in roots.iterrows():
            lineage, cur = [], root
            while cur is not None:
                lineage.append(cur)
                children = chain[chain["parent_id"] == cur["id"]]
                cur = children.iloc[0] if not children.empty else None
            if len(lineage) > 1:
                df = pd.DataFrame(
                    {"step": range(len(lineage)), "score": [r["score"] for r in lineage]}
                ).set_index("step")
                st.line_chart(df)
                st.caption("Seed: " + str(root["expression"]))
    else:
        st.info("Chưa có chuỗi tinh chỉnh (chạy research để tạo lineage). GA: xem logs/.")

with tab_subs:
    subs = _read("SELECT * FROM submissions ORDER BY submitted_at DESC")
    if not subs.empty:
        st.dataframe(subs, use_container_width=True)
    else:
        st.info("Chưa có submission nào.")

with tab_corr:
    # T7.4: ma trận tương đồng cấu trúc (AST) của top alpha — KHÁC correlation thật WQ.
    top = _read(
        "SELECT a.expression, s.score FROM simulations s JOIN alphas a ON s.alpha_id = a.id "
        "WHERE s.status = 'passed' ORDER BY s.score DESC LIMIT 15"
    )
    st.caption("Tương đồng cấu trúc AST (bộ lọc local, KHÁC return-correlation thật của WQ).")
    if len(top) >= 2:
        from src.decorrelation.similarity import similarity_ratio

        exprs = top["expression"].tolist()
        n = len(exprs)
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                try:
                    matrix[i][j] = round(similarity_ratio(exprs[i], exprs[j]), 2)
                except ValueError:
                    matrix[i][j] = 0.0
        labels = [e[:24] for e in exprs]
        st.dataframe(
            pd.DataFrame(matrix, index=labels, columns=labels), use_container_width=True
        )
    else:
        st.info("Cần ít nhất 2 alpha đã pass để dựng ma trận.")
