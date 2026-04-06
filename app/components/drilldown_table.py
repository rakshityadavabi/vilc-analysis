from __future__ import annotations

import pandas as pd
import streamlit as st

from app.utils import arrow_safe, style_positive_blue


def _pick_column(df: pd.DataFrame, candidates: list[str], default: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    return default


def render_drilldown_table(df: pd.DataFrame, perf_col: str) -> None:
    pfx = "YTD_" if perf_col.startswith("YTD_") else "MTH_"
    bu_col = f"{pfx}BU"
    act_col = f"{pfx}ACT"

    supplier_col = _pick_column(df, ["Supplier", "Supplier_Name", "Supplier_1"], "Supplier")
    sku_col = _pick_column(df, ["SKU", "Material", "Account_5_subpackage", "Account_5"], "SKU")

    working = df.copy()
    if supplier_col not in working.columns:
        working[supplier_col] = "Unknown"
    if sku_col not in working.columns:
        working[sku_col] = "Unknown"

    if bu_col not in working.columns:
        working[bu_col] = 0.0
    if act_col not in working.columns:
        working[act_col] = 0.0

    if perf_col not in working.columns:
        working[perf_col] = working[act_col] - working[bu_col]

    detail = (
        working.groupby([supplier_col, sku_col], dropna=False)[[bu_col, act_col, perf_col]]
        .sum()
        .reset_index()
        .rename(columns={bu_col: "Budget", act_col: "Actual", perf_col: "Variance"})
    )

    denom = float(detail["Variance"].abs().sum())
    if denom == 0:
        detail["Contribution %"] = 0.0
    else:
        detail["Contribution %"] = (detail["Variance"] / denom) * 100

    detail = detail.sort_values("Variance", ascending=False)

    display_cols = [supplier_col, sku_col, "Budget", "Actual", "Variance", "Contribution %"]
    display = detail[display_cols]

    st.markdown("### Supplier / SKU Detail")
    st.dataframe(
        style_positive_blue(arrow_safe(display), ["Variance", "Contribution %"]),
        height=420,
        width="stretch",
    )
