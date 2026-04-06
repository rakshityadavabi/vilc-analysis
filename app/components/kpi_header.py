from __future__ import annotations

import streamlit as st
import pandas as pd

from app.services.aggregation_service import aggregate_for_chart, get_summary_metrics
from app.utils import fmt_million


def _largest_driver(df: pd.DataFrame, perf_col: str) -> tuple[str, float, str, float]:
    candidate_levels = [
        "Account_5_subpackage",
        "Account_5",
        "Account_4",
        "Account_3",
        "Entity_1",
    ]
    level = next((c for c in candidate_levels if c in df.columns), None)
    if not level or perf_col not in df.columns:
        return ("N/A", 0.0, "N/A", 0.0)

    agg = aggregate_for_chart(df, level, perf_col)
    if agg.empty:
        return ("N/A", 0.0, "N/A", 0.0)

    top_row = agg.iloc[0]
    bot_row = agg.sort_values(perf_col, ascending=True).iloc[0]
    return (str(top_row[level]), float(top_row[perf_col]), str(bot_row[level]), float(bot_row[perf_col]))


def render_kpi_header(df: pd.DataFrame, active_perf_col: str) -> None:
    st.markdown(
        """
        <style>
        div[data-testid=\"stVerticalBlockBorderWrapper\"] .kpi-sticky {
            position: sticky;
            top: 0.5rem;
            z-index: 80;
            background: #ffffff;
            border: 1px solid #e6edf5;
            border-radius: 12px;
            padding: 0.75rem 0.9rem 0.3rem 0.9rem;
            box-shadow: 0 8px 24px rgba(16, 24, 40, 0.06);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    mtd = get_summary_metrics(df, "MTH_Perf") if "MTH_Perf" in df.columns else {"variance": 0}
    ytd = get_summary_metrics(df, "YTD_Perf") if "YTD_Perf" in df.columns else {"variance": 0}

    region_col = "Zone" if "Zone" in df.columns else ("Country" if "Country" in df.columns else None)
    top_region_name = "N/A"
    top_region_val = 0.0
    low_region_name = "N/A"
    low_region_val = 0.0

    if region_col and active_perf_col in df.columns:
        reg = aggregate_for_chart(df, region_col, active_perf_col)
        if not reg.empty:
            top_region_name = str(reg.iloc[0][region_col])
            top_region_val = float(reg.iloc[0][active_perf_col])
            low = reg.sort_values(active_perf_col, ascending=True).iloc[0]
            low_region_name = str(low[region_col])
            low_region_val = float(low[active_perf_col])

    largest_driver, driver_val, largest_drag, drag_val = _largest_driver(df, active_perf_col)

    st.markdown('<div class="kpi-sticky">', unsafe_allow_html=True)
    cols = st.columns(6)
    cols[0].metric("MTD vs Budget", fmt_million(mtd.get("variance", 0.0)))
    cols[1].metric("YTD vs Budget", fmt_million(ytd.get("variance", 0.0)))
    cols[2].metric("Top Positive Region", f"{top_region_name} ({fmt_million(top_region_val)})")
    cols[3].metric("Top Negative Region", f"{low_region_name} ({fmt_million(low_region_val)})")
    cols[4].metric("Largest Driver", f"{largest_driver} ({fmt_million(driver_val)})")
    cols[5].metric("Largest Drag", f"{largest_drag} ({fmt_million(drag_val)})")
    st.markdown("</div>", unsafe_allow_html=True)
