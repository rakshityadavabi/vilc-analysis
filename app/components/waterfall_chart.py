from __future__ import annotations

from typing import Optional

import plotly.graph_objects as go
import streamlit as st
import pandas as pd

from app.services.aggregation_service import aggregate_for_chart
from app.utils import fmt_million


def _selected_x(selection_result) -> Optional[str]:
    if not selection_result:
        return None

    selection = getattr(selection_result, "selection", None)
    if selection is None and isinstance(selection_result, dict):
        selection = selection_result.get("selection")

    if not selection:
        return None

    points = None
    if isinstance(selection, dict):
        points = selection.get("points")
    else:
        points = getattr(selection, "points", None)

    if not points:
        return None

    first = points[0]
    if isinstance(first, dict):
        x = first.get("x")
    else:
        x = getattr(first, "x", None)

    if x is None:
        return None

    label = str(x)
    if label == "Total":
        return None
    return label


def render_waterfall(
    df: pd.DataFrame,
    level: str,
    perf_col: str,
    title: str,
    key: str,
) -> tuple[Optional[str], pd.DataFrame]:
    chart_df = aggregate_for_chart(df, level, perf_col)
    if chart_df.empty:
        st.info(f"No data available for {level}.")
        return None, chart_df

    labels = chart_df[level].astype(str).tolist()
    values = chart_df[perf_col].tolist()

    fig = go.Figure(
        go.Waterfall(
            measure=["relative"] * len(values) + ["total"],
            x=labels + ["Total"],
            y=values + [sum(values)],
            connector={"line": {"color": "rgba(70, 70, 70, 0.25)"}},
            increasing={"marker": {"color": "#2e7d32"}},
            decreasing={"marker": {"color": "#c62828"}},
            totals={"marker": {"color": "#757575"}},
            text=[fmt_million(v) for v in values] + [fmt_million(sum(values))],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title,
        height=440,
        margin=dict(t=60, b=10, l=10, r=10),
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title="",
        yaxis_title=perf_col,
    )
    fig.update_yaxes(tickformat=".3s")

    selected = None
    try:
        selection_result = st.plotly_chart(
            fig,
            key=key,
            on_select="rerun",
            selection_mode=["points"],
            width="stretch",
        )
        selected = _selected_x(selection_result)
    except TypeError:
        st.plotly_chart(fig, key=key, width="stretch")
        st.caption("Chart click selection is not available in this Streamlit version.")

    return selected, chart_df
