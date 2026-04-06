from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.services.aggregation_service import aggregate_for_chart
from app.utils import fmt_million


POSITIVE = "#2e7d32"
NEGATIVE = "#c62828"
NEGATIVE_LAST = "#2e7d32"
TOTAL = "#757575"
NEUTRAL = "#546e7a"
ZONE_ORDER = ["EUR", "AFR", "SAZ", "APAC", "NAZ", "MAZ"]

PLOTLY_TITLE_SIZE = 64
PLOTLY_TICK_SIZE = 64
PLOTLY_VALUE_LABEL_SIZE = 64
MATRIX_TITLE_SIZE = 36
MATRIX_HEADER_LABEL_SIZE = 26
MATRIX_ROW_LABEL_SIZE = 22
MATRIX_VALUE_LABEL_SIZE = 23
MATRIX_ZONE_LABEL_SIZE = 24


def _ordered_labels(labels: list[str], preferred_order: list[str] | None = None, strict: bool = False) -> list[str]:
    ordered: list[str] = []
    preferred = preferred_order or []
    for value in preferred:
        if value in labels and value not in ordered:
            ordered.append(value)
    if not strict:
        for value in labels:
            if value not in ordered:
                ordered.append(value)
    return ordered


def _wrap_label(value: str, width: int = 12) -> str:
    text = str(value)
    if len(text) <= width:
        return text
    return "<br>".join(textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False))


def get_package_breakdown_order(df: pd.DataFrame, level: str = "Account_3", perf_col: str = "MTH_Price", max_items: int | None = None) -> list[str]:
    if level not in df.columns or perf_col not in df.columns:
        return []

    chart_df = aggregate_for_chart(df, level, perf_col)
    if max_items is not None:
        chart_df = _limit_other(chart_df, level, perf_col, max_items=max_items)
    chart_df = chart_df[chart_df[perf_col].abs() >= 0.1]
    chart_df = chart_df.sort_values(perf_col, ascending=False)
    return chart_df[level].astype(str).tolist()


def _wrap_label_lines(value: str, width: int = 12) -> str:
    text = str(value)
    if len(text) <= width:
        return text
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False))


def _draw_row_dividers(ax, row_count: int) -> None:
    for boundary in range(row_count + 1):
        ax.axhline(boundary - 0.5, color="#e6e6e6", linewidth=0.8, zorder=0)


def _limit_other(df: pd.DataFrame, label_col: str, value_col: str, max_items: int) -> pd.DataFrame:
    if df.empty or len(df) <= max_items:
        return df

    working = df.copy()
    working["_abs"] = working[value_col].abs()
    working = working.sort_values(["_abs", value_col], ascending=[False, False])

    head = working.head(max_items - 1).drop(columns=["_abs"])
    other_value = working.iloc[max_items - 1 :][value_col].sum()
    other = pd.DataFrame([{label_col: "Other", value_col: other_value}])
    return pd.concat([head, other], ignore_index=True)


def _coerce_ordered_chart_df(chart_df: pd.DataFrame, label_col: str, value_col: str, label_order: list[str] | None = None, strict: bool = False) -> pd.DataFrame:
    if chart_df.empty:
        return chart_df

    if label_order:
        ordered_labels = _ordered_labels(chart_df[label_col].astype(str).tolist(), label_order, strict=strict)
        if strict:
            chart_df = chart_df[chart_df[label_col].astype(str).isin(ordered_labels)].copy()
        chart_df = chart_df.assign(_order=chart_df[label_col].astype(str).map(lambda value: ordered_labels.index(value) if value in ordered_labels else len(ordered_labels)))
        chart_df = chart_df.sort_values(["_order", value_col], ascending=[True, False]).drop(columns=["_order"])
    return chart_df


def build_waterfall_figure(
    df: pd.DataFrame,
    level: str,
    perf_col: str,
    title: str,
    max_items: int = 12,
    total_label: str = "Total",
    label_order: list[str] | None = None,
) -> go.Figure:
    chart_df = aggregate_for_chart(df, level, perf_col)
    if chart_df.empty:
        fig = go.Figure()
        fig.add_annotation(text=f"No data available for {level}", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=460, paper_bgcolor="white", plot_bgcolor="white", margin=dict(t=60, b=25, l=20, r=20))
        return fig

    if level != "Account_3":
        chart_df = _limit_other(chart_df, level, perf_col, max_items=max_items)
    else:
        chart_df = chart_df[chart_df[perf_col].abs() >= 0.1]
    if label_order:
        chart_df = _coerce_ordered_chart_df(chart_df, level, perf_col, label_order=label_order, strict=(level == "Zone"))
    elif level == "Zone":
        chart_df = _coerce_ordered_chart_df(chart_df, level, perf_col, label_order=ZONE_ORDER, strict=True)

    if level == "Account_3":
        labels = [str(label) for label in chart_df[level].astype(str).tolist()]
    else:
        labels = [_wrap_label(label) for label in chart_df[level].astype(str).tolist()]
    values = [float(v) for v in chart_df[perf_col].tolist()]
    total = sum(values)

    fig = go.Figure(
        go.Waterfall(
            measure=["relative"] * len(values) + ["total"],
            x=labels + [total_label],
            y=values + [total],
            connector={"line": {"color": "rgba(0, 0, 0, 0)", "width": 0}},
            increasing={"marker": {"color": POSITIVE}},
            decreasing={"marker": {"color": NEGATIVE}},
            totals={"marker": {"color": TOTAL}},
            text=[fmt_million(v) for v in values] + [fmt_million(total)],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=PLOTLY_TITLE_SIZE, color="#111111"), x=0.5, xanchor="center"),
        height=1000,
        margin=dict(t=170, b=260, l=45, r=45),
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="DejaVu Sans"),
        xaxis=dict(showgrid=False, zeroline=False, showline=False, title="", tickfont=dict(color="#111111", size=PLOTLY_TICK_SIZE), tickangle=90 if level == "Account_3" else 0, automargin=True),
        yaxis=dict(showgrid=False, zeroline=False, showline=False, title="", showticklabels=False),
    )
    fig.update_traces(textfont=dict(color="#111111", size=PLOTLY_VALUE_LABEL_SIZE, family="DejaVu Sans"))
    return fig


def build_breakdown_bar_figure(
    df: pd.DataFrame,
    level: str,
    perf_col: str,
    title: str,
    max_items: int = 12,
    label_order: list[str] | None = None,
) -> go.Figure:
    chart_df = aggregate_for_chart(df, level, perf_col)
    if chart_df.empty:
        fig = go.Figure()
        fig.add_annotation(text=f"No data available for {level}", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=460, paper_bgcolor="white", plot_bgcolor="white", margin=dict(t=60, b=25, l=20, r=20))
        return fig

    if level != "Account_3":
        chart_df = _limit_other(chart_df, level, perf_col, max_items=max_items)
    else:
        chart_df = chart_df[chart_df[perf_col].abs() >= 0.1]
    if label_order:
        chart_df = _coerce_ordered_chart_df(chart_df, level, perf_col, label_order=label_order, strict=(level == "Zone"))
    else:
        chart_df = chart_df.sort_values(perf_col, ascending=True)

    colors = [POSITIVE if float(v) >= 0 else NEGATIVE for v in chart_df[perf_col].tolist()]
    labels = [_wrap_label(label, width=18) for label in chart_df[level].astype(str).tolist()]
    values = [float(v) for v in chart_df[perf_col].tolist()]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[fmt_million(v) for v in values],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=PLOTLY_TITLE_SIZE, color="#111111"), x=0.5, xanchor="center"),
        height=1000,
        margin=dict(t=170, b=260, l=45, r=45),
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="DejaVu Sans"),
        xaxis=dict(showgrid=False, zeroline=False, showline=False, title="", tickfont=dict(color="#111111", size=PLOTLY_TICK_SIZE), tickangle=0, automargin=True),
        yaxis=dict(showgrid=False, zeroline=False, showline=False, title="", showticklabels=False),
    )
    fig.update_traces(textfont=dict(color="#111111", size=PLOTLY_VALUE_LABEL_SIZE, family="DejaVu Sans"))
    return fig


def build_zone_package_matrix_figure(
    df: pd.DataFrame,
    row_level: str,
    col_level: str,
    perf_col: str,
    title: str,
    row_header_label: str | None = None,
    row_order: list[str] | None = None,
    col_order: list[str] | None = None,
    total_col_label: str = "ABI",
    max_rows: int = 12,
) -> plt.Figure:
    if row_level not in df.columns or col_level not in df.columns or perf_col not in df.columns:
        fig, ax = plt.subplots(figsize=(14, 4), facecolor="white")
        ax.axis("off")
        ax.text(0.5, 0.5, f"No breakdown data available for {row_level} vs {col_level}", ha="center", va="center", fontsize=12, color="#111111", fontweight="bold")
        fig.suptitle(title, x=0.01, ha="left", fontsize=MATRIX_TITLE_SIZE, fontweight="bold", color="#111111")
        return fig

    grouped = (
        df.groupby([row_level, col_level], dropna=False)[perf_col]
        .sum()
        .reset_index()
    )
    if grouped.empty:
        fig, ax = plt.subplots(figsize=(14, 4), facecolor="white")
        ax.axis("off")
        ax.text(0.5, 0.5, f"No breakdown data available for {row_level} vs {col_level}", ha="center", va="center", fontsize=12, color="#111111", fontweight="bold")
        fig.suptitle(title, x=0.01, ha="left", fontsize=MATRIX_TITLE_SIZE, fontweight="bold", color="#111111")
        return fig

    pivot = grouped.pivot_table(index=row_level, columns=col_level, values=perf_col, aggfunc="sum", fill_value=0)
    if pivot.empty:
        fig, ax = plt.subplots(figsize=(14, 4), facecolor="white")
        ax.axis("off")
        ax.text(0.5, 0.5, f"No breakdown data available for {row_level} vs {col_level}", ha="center", va="center", fontsize=12, color="#111111", fontweight="bold")
        fig.suptitle(title, x=0.01, ha="left", fontsize=MATRIX_TITLE_SIZE, fontweight="bold", color="#111111")
        return fig

    ordered_cols = [c for c in (col_order or ZONE_ORDER) if c in pivot.columns]
    pivot = pivot.reindex(columns=ordered_cols, fill_value=0)
    pivot[total_col_label] = pivot.sum(axis=1)

    if row_order:
        ordered_rows = [row for row in row_order if row in pivot.index]
        pivot = pivot.reindex(ordered_rows)
    else:
        pivot = pivot.sort_values(total_col_label, ascending=False)

    pivot = pivot.head(max_rows)
    display = pivot

    if display.empty:
        fig, ax = plt.subplots(figsize=(14, 4), facecolor="white")
        ax.axis("off")
        ax.text(0.5, 0.5, f"No breakdown data available for {row_level} vs {col_level}", ha="center", va="center", fontsize=12, color="#111111", fontweight="bold")
        fig.suptitle(title, x=0.01, ha="left", fontsize=MATRIX_TITLE_SIZE, fontweight="bold", color="#111111")
        return fig

    zone_cols = list(display.columns)
    n_zones = len(zone_cols)
    fig_width = max(24.0, 3.6 * n_zones)
    fig_height = max(11.0, 1.02 * len(display.index) + 4.2)
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")
    gs = fig.add_gridspec(2, n_zones + 1, height_ratios=[0.12, 1.0], width_ratios=[2.8] + [1] * n_zones, hspace=0.0, wspace=0.02)

    header_ax = fig.add_subplot(gs[0, :])
    header_ax.set_facecolor("black")
    header_ax.set_xticks([])
    header_ax.set_yticks([])
    for spine in header_ax.spines.values():
        spine.set_visible(False)
    header_ax.axhline(0.0, color="#ffffff", linewidth=1.4, clip_on=False)
    header_ax.text(0.015, 0.5, row_header_label or row_level, ha="left", va="center", fontsize=MATRIX_HEADER_LABEL_SIZE, fontweight="bold", color="white")

    # Keep zone labels inside the black header row so they remain visible.
    total_units = 2.8 + n_zones
    for zone_idx, zone in enumerate(zone_cols, start=1):
        zone_center_units = 2.8 + (zone_idx - 1) + 0.5
        zone_x = zone_center_units / total_units
        header_ax.text(zone_x, 0.5, str(zone), ha="center", va="center", fontsize=MATRIX_ZONE_LABEL_SIZE, fontweight="bold", color="white")

    label_ax = fig.add_subplot(gs[1, 0])
    label_ax.set_facecolor("white")
    label_ax.set_xticks([])
    label_ax.set_yticks([])
    for spine in label_ax.spines.values():
        spine.set_visible(False)

    y_positions = np.arange(len(display.index))
    y_labels = [str(idx) for idx in display.index.tolist()]
    label_ax.set_ylim(-0.5, len(y_positions) - 0.5)
    label_ax.invert_yaxis()
    for y_pos, label in zip(y_positions, y_labels):
        label_ax.text(0.02, y_pos, label, ha="left", va="center", fontsize=MATRIX_ROW_LABEL_SIZE, fontweight="bold", color="#111111")
    label_ax.text(0.02, -1.05, "Category", ha="left", va="center", fontsize=MATRIX_HEADER_LABEL_SIZE, fontweight="bold", color="#111111")
    _draw_row_dividers(label_ax, len(y_positions))

    max_x = float(np.nanmax(np.abs(display.to_numpy(dtype=float)))) if display.size else 1.0
    max_x = max(max_x, 1.0)

    for zone_idx, zone in enumerate(zone_cols, start=1):
        ax = fig.add_subplot(gs[1, zone_idx])
        values = display[zone].astype(float).to_numpy()
        colors = [POSITIVE if value > 0.1 else NEGATIVE if value < 0 else TOTAL for value in values]
        ax.barh(y_positions, values, color=colors, height=0.52)
        ax.axvline(0, color="#111111", linewidth=2.8, zorder=0)
        _draw_row_dividers(ax, len(y_positions))
        ax.set_xlim(-max_x * 1.15, max_x * 1.15)
        ax.set_ylim(-0.5, len(y_positions) - 0.5)
        ax.invert_yaxis()
        ax.set_facecolor("white")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for y_pos, value in zip(y_positions, values):
            if abs(value) < 0.1:
                continue
            x_pos = value + (0.03 * max_x if value > 0 else -0.03 * max_x)
            ha = "left" if value > 0 else "right"
            ax.text(x_pos, y_pos, f"{value:.1f}", va="center", ha=ha, fontsize=MATRIX_VALUE_LABEL_SIZE, fontweight="bold", color="#111111")

    fig.subplots_adjust(left=0.02, right=0.995, top=0.9, bottom=0.04)
    return fig