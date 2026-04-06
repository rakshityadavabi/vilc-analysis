from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable, Optional

import pandas as pd
import streamlit as st

ALL_MTH_COLS = [
    "MTH_Perf",
    "MTH_Price",
    "MTH_BU",
    "MTH_ACT",
    "MTH_Volume",
    "MTH_Mix",
    "MTH_IntSu_Scope",
    "MTH_V_per_M",
    "MTH_FB",
    "MTH_Scope",
    "MTH_Scope_IN",
    "MTH_Scope_OUT",
]

ALL_YTD_COLS = [
    "YTD_Perf",
    "YTD_Price",
    "YTD_BU",
    "YTD_ACT",
    "YTD_Volume",
    "YTD_Mix",
    "YTD_IntSu_Scope",
    "YTD_V_per_M",
    "YTD_FB",
    "YTD_Scope",
    "YTD_Scope_IN",
    "YTD_Scope_OUT",
]

HIERARCHY = [
    "Zone",
    "Country",
    "Entity_1",
    "Account_3",
    "Account_4",
    "Account_5",
    "Account_5_subpackage",
]


_MONTH_ORDER: Dict[str, int] = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


def _normalize_year_token(token: str) -> Optional[int]:
    digits = "".join(ch for ch in str(token) if ch.isdigit())
    if not digits:
        return None
    try:
        return int(digits[-2:])
    except ValueError:
        return None


def fmt_num(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)

    abs_v = abs(v)
    if abs_v >= 1_000_000:
        return f"{v / 1_000_000:.2f} M"
    if abs_v >= 1_000:
        return f"{v / 1_000:.2f} K"
    return f"{v:.2f}"


def fmt_million(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{v:.1f}"


def arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include="object").columns:
        out[col] = out[col].astype(str)
    return out


def style_positive_blue(df: pd.DataFrame, metric_cols: list[str]):
    sty = df.style
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        non_metric_cols = [c for c in num_cols if c not in metric_cols]
        if non_metric_cols:
            sty = sty.format({c: fmt_num for c in non_metric_cols})
        metric_format_cols = [c for c in metric_cols if c in num_cols]
        if metric_format_cols:
            sty = sty.format({c: fmt_million for c in metric_format_cols})

    for col in metric_cols:
        if col not in df.columns:
            continue
        try:
            sty = sty.background_gradient(subset=[col], cmap="Blues")
        except Exception:
            pass

        def _neg(v):
            try:
                return "background-color: #ffd6d6" if float(v) < 0 else ""
            except Exception:
                return ""

        sty = sty.map(_neg, subset=[col])

    return sty


def month_sort_key(month_label: str) -> tuple[int, int]:
    parts = str(month_label).split()
    if len(parts) != 2:
        return (0, 0)

    yy = _normalize_year_token(parts[1])
    if yy is None:
        return (0, 0)

    return (yy, _MONTH_ORDER.get(parts[0], 0))


def get_ytd_last_month(years: list, available_months: list) -> Optional[str]:
    if not years:
        return None

    year_suffixes = set()
    for yr in years:
        yy = _normalize_year_token(str(yr).replace("FY", "").replace("fy", "").strip())
        if yy is not None:
            year_suffixes.add(f"{yy:02d}")

    if year_suffixes and available_months:
        matching = []
        for m in available_months:
            parts = str(m).split()
            if len(parts) != 2:
                continue
            yy = _normalize_year_token(parts[1])
            if yy is None:
                continue
            if f"{yy:02d}" in year_suffixes:
                matching.append(m)
        if matching:
            return max(matching, key=month_sort_key)

    if year_suffixes:
        suffix = sorted(year_suffixes, reverse=True)[0]
        return f"Dec {suffix}"

    if available_months:
        return max(available_months, key=month_sort_key)

    return None


def choose_mode_perf_col(mode: str) -> str:
    return "YTD_Perf" if str(mode).upper() == "YTD" else "MTH_Perf"


def mode_metric_columns(mode: str) -> list[str]:
    return ALL_YTD_COLS if str(mode).upper() == "YTD" else ALL_MTH_COLS


def safe_sum(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())


def available_hierarchy(df: pd.DataFrame) -> list[str]:
    return [c for c in HIERARCHY if c in df.columns]


def first_present(df: pd.DataFrame, candidates: Iterable[str], fallback: str = "") -> str:
    for c in candidates:
        if c in df.columns:
            return c
    return fallback


def ensure_log_state() -> None:
    if "logs" not in st.session_state:
        st.session_state["logs"] = []


def log_event(message: str) -> None:
    ensure_log_state()
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state["logs"].append(f"[{timestamp}] {message}")


def log_query(label: str, query: str) -> None:
    log_event(f"{label}\n{query}")


def log_dataframe(label: str, df: pd.DataFrame, max_rows: int = 5) -> None:
    ensure_log_state()
    preview = ""
    if df is not None and not df.empty:
        preview = df.head(max_rows).to_string(index=False)
    else:
        preview = "<empty dataframe>"
    log_event(f"{label} rows={0 if df is None else len(df):,} cols={0 if df is None else len(df.columns):,}\n{preview}")


def render_log_output(title: str = "Execution Log") -> None:
    ensure_log_state()
    with st.expander(title, expanded=False):
        if not st.session_state["logs"]:
            st.caption("No log entries yet.")
            return
        st.code("\n\n".join(st.session_state["logs"][-100:]), language="text")
