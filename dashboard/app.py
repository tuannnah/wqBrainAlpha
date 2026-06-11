"""Dashboard Streamlit theo dõi alpha, GA, submissions, correlation.

Chạy: streamlit run dashboard/app.py
"""

from __future__ import annotations

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

tab_overview, tab_explorer, tab_ga, tab_subs = st.tabs(
    ["Overview", "Explorer", "GA Progress", "Submissions"]
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
        "SELECT a.expression, s.sharpe, s.fitness, s.turnover, s.drawdown, s.score, s.status "
        "FROM simulations s JOIN alphas a ON s.alpha_id = a.id ORDER BY s.score DESC"
    )
    if not sims.empty:
        min_sharpe = st.slider("Sharpe tối thiểu", -2.0, 5.0, 0.0, 0.1)
        st.dataframe(sims[sims["sharpe"].fillna(-99) >= min_sharpe], use_container_width=True)
    else:
        st.info("Chưa có simulation nào.")

with tab_ga:
    st.info("Theo dõi best/avg score theo generation trong logs/ (logger GA).")

with tab_subs:
    subs = _read("SELECT * FROM submissions ORDER BY submitted_at DESC")
    if not subs.empty:
        st.dataframe(subs, use_container_width=True)
    else:
        st.info("Chưa có submission nào.")
