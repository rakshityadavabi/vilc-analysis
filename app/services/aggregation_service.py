from __future__ import annotations

import pandas as pd


def aggregate_for_chart(df: pd.DataFrame, level: str, perf_col: str) -> pd.DataFrame:
    if level not in df.columns or perf_col not in df.columns:
        return pd.DataFrame(columns=[level, perf_col])

    out = (
        df.groupby(level, dropna=False)[perf_col]
        .sum()
        .reset_index()
        .sort_values(perf_col, ascending=False)
    )
    return out


def get_top_positive(df: pd.DataFrame, group_col: str, perf_col: str, n: int = 3) -> pd.DataFrame:
    agg = aggregate_for_chart(df, group_col, perf_col)
    if agg.empty:
        return agg
    return agg[agg[perf_col] > 0].head(n)


def get_top_negative(df: pd.DataFrame, group_col: str, perf_col: str, n: int = 3) -> pd.DataFrame:
    agg = aggregate_for_chart(df, group_col, perf_col)
    if agg.empty:
        return agg
    return agg[agg[perf_col] < 0].sort_values(perf_col, ascending=True).head(n)


def get_summary_metrics(df: pd.DataFrame, perf_col: str) -> dict:
    pfx = "YTD_" if perf_col.startswith("YTD_") else "MTH_"
    bu_col = f"{pfx}BU"
    act_col = f"{pfx}ACT"

    actual = float(df[act_col].sum()) if act_col in df.columns else 0.0
    budget = float(df[bu_col].sum()) if bu_col in df.columns else 0.0
    variance = float(df[perf_col].sum()) if perf_col in df.columns else 0.0

    return {
        "actual": actual,
        "budget": budget,
        "variance": variance,
        "bu_col": bu_col,
        "act_col": act_col,
    }
