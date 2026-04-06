from __future__ import annotations

from io import BytesIO
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from app.utils import first_present, fmt_million


BRAND_BLACK = "#111111"
POSITIVE = "#16a34a"
NEGATIVE = "#b91c1c"
TOTAL = "#8b8b8b"
GRID = "#e6edf5"
ZONE_ORDER = ["AFR", "APAC", "SAZ", "NAZ", "MAZ", "EUR"]
PACKAGE_CANDIDATES = ["Account_3", "Account_4", "Account_5_subpackage", "Account_5"]


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return first_present(df, candidates, fallback="") or None


def _metric_cols(prefix: str) -> tuple[str, str, str]:
    return (f"{prefix}_Price", f"{prefix}_BU", f"{prefix}_ACT")


def _variance_frame(df: pd.DataFrame, group_col: str, prefix: str) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame(columns=[group_col, "variance"])

    price_col, bu_col, act_col = _metric_cols(prefix)
    working = df.copy()

    if price_col in working.columns and bu_col in working.columns:
        working["variance"] = pd.to_numeric(working[price_col], errors="coerce").fillna(0) - pd.to_numeric(
            working[bu_col], errors="coerce"
        ).fillna(0)
    elif f"{prefix}_Perf" in working.columns:
        working["variance"] = pd.to_numeric(working[f"{prefix}_Perf"], errors="coerce").fillna(0)
    elif act_col in working.columns and bu_col in working.columns:
        working["variance"] = pd.to_numeric(working[act_col], errors="coerce").fillna(0) - pd.to_numeric(
            working[bu_col], errors="coerce"
        ).fillna(0)
    else:
        return pd.DataFrame(columns=[group_col, "variance"])

    out = (
        working.groupby(group_col, dropna=False)["variance"]
        .sum()
        .reset_index()
        .sort_values("variance", ascending=False)
    )
    return out


def _pairwise_variance_frame(df: pd.DataFrame, row_col: str, col_col: str, prefix: str) -> pd.DataFrame:
    if row_col not in df.columns or col_col not in df.columns:
        return pd.DataFrame(columns=[row_col, col_col, "variance"])

    price_col, bu_col, act_col = _metric_cols(prefix)
    working = df.copy()

    if price_col in working.columns and bu_col in working.columns:
        working["variance"] = pd.to_numeric(working[price_col], errors="coerce").fillna(0) - pd.to_numeric(
            working[bu_col], errors="coerce"
        ).fillna(0)
    elif f"{prefix}_Perf" in working.columns:
        working["variance"] = pd.to_numeric(working[f"{prefix}_Perf"], errors="coerce").fillna(0)
    elif act_col in working.columns and bu_col in working.columns:
        working["variance"] = pd.to_numeric(working[act_col], errors="coerce").fillna(0) - pd.to_numeric(
            working[bu_col], errors="coerce"
        ).fillna(0)
    else:
        return pd.DataFrame(columns=[row_col, col_col, "variance"])

    out = (
        working.groupby([row_col, col_col], dropna=False)["variance"]
        .sum()
        .reset_index()
        .sort_values("variance", ascending=False)
    )
    return out


def _limit_categories(df: pd.DataFrame, group_col: str, limit: int = 8) -> pd.DataFrame:
    if df.empty or len(df) <= limit:
        return df

    working = df.copy()
    working["abs_variance"] = working["variance"].abs()
    top = working.sort_values("abs_variance", ascending=False).head(limit - 1).drop(columns=["abs_variance"])
    other_value = working.sort_values("abs_variance", ascending=False).iloc[limit - 1 :]["variance"].sum()
    other = pd.DataFrame([{group_col: "Other", "variance": other_value}])
    return pd.concat([top, other], ignore_index=True)


def _ordered_zones(zones: list[str]) -> list[str]:
    preferred = [z for z in ZONE_ORDER if z in zones]
    remainder = [z for z in zones if z not in preferred]
    return preferred + sorted(remainder)


def _waterfall_values(series: pd.Series) -> tuple[list[float], float]:
    values = [float(v) for v in series.tolist()]
    total = float(np.sum(values)) if values else 0.0
    return values, total


def _wrap_label(label: str, width: int = 11) -> str:
    text = str(label)
    if len(text) <= width:
        return text
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False))


def _draw_waterfall(ax: plt.Axes, frame: pd.DataFrame, label_col: str, value_col: str, title: str) -> None:
    ax.set_facecolor("white")
    labels = frame[label_col].astype(str).tolist()
    values, total = _waterfall_values(frame[value_col])

    positions = np.arange(len(values) + 1)
    running = 0.0
    bottoms: list[float] = []
    heights: list[float] = []
    colors: list[str] = []

    for value in values:
        if value >= 0:
            bottoms.append(running)
            heights.append(value)
            colors.append(POSITIVE)
        else:
            bottoms.append(running + value)
            heights.append(abs(value))
            colors.append(NEGATIVE)
        running += value

    ax.bar(positions[:-1], heights, bottom=bottoms, width=0.58, color=colors, edgecolor="none")
    ax.bar([positions[-1]], [abs(total)], bottom=[0], width=0.58, color=TOTAL, edgecolor="none")

    show_values = len(values) <= 8
    for idx, (value, bottom, height) in enumerate(zip(values, bottoms, heights)):
        if not show_values:
            continue
        label_y = bottom + height + (0.02 * max(1.0, abs(total))) if value >= 0 else bottom - (0.02 * max(1.0, abs(total)))
        va = "bottom" if value >= 0 else "top"
        ax.text(idx, label_y, fmt_million(value), ha="center", va=va, fontsize=7, fontweight="bold", color=BRAND_BLACK)

    total_y = abs(total) + (0.03 * max(1.0, abs(total)))
    ax.text(positions[-1], total_y, fmt_million(total), ha="center", va="bottom", fontsize=7, fontweight="bold", color=BRAND_BLACK)

    tick_labels = [_wrap_label(label) for label in labels] + ["Total"]
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=7)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#b9c4d0")
    ax.spines["bottom"].set_color("#b9c4d0")
    ax.set_ylabel("$Mio", fontsize=8, color="#555555")
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", pad=6)

    y_values = bottoms + [0.0, total]
    y_max = max(max(y_values), total, 0.0)
    y_min = min(min(y_values), total, 0.0)
    padding = max(0.5, (y_max - y_min) * 0.18)
    ax.set_ylim(y_min - padding, y_max + padding)


def _render_matrix(ax: plt.Axes, frame: pd.DataFrame, row_col: str, col_col: str, title: str) -> None:
    ax.set_facecolor("white")
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=10)
    ax.axis("off")

    if frame.empty or row_col not in frame.columns or col_col not in frame.columns:
        ax.text(0.5, 0.5, "No breakdown data available", ha="center", va="center", fontsize=11, color="#666666")
        return

    pivot = (
        frame.groupby([row_col, col_col], dropna=False)["variance"]
        .sum()
        .unstack(fill_value=0)
    )

    if pivot.empty:
        ax.text(0.5, 0.5, "No breakdown data available", ha="center", va="center", fontsize=11, color="#666666")
        return

    zone_cols = _ordered_zones(list(pivot.columns.astype(str)))
    pivot = pivot.reindex(columns=zone_cols, fill_value=0)
    pivot["__total__"] = pivot.sum(axis=1)
    pivot = pivot.reindex(pivot["__total__"].abs().sort_values(ascending=False).index)
    pivot = pivot.head(7)
    display = pivot.drop(columns=["__total__"])

    cell_text = [[fmt_million(v) for v in row] for row in display.to_numpy()]
    table = ax.table(
        cellText=cell_text,
        rowLabels=[_wrap_label(str(idx), width=16) for idx in display.index],
        colLabels=[str(col) for col in display.columns],
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.25)

    vmax = float(np.nanmax(np.abs(display.to_numpy()))) if display.size else 0.0
    vmax = vmax if vmax > 0 else 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdYlGn")

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if row_idx == 0:
            cell.set_facecolor(BRAND_BLACK)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            continue
        if col_idx == -1:
            cell.set_facecolor("#f3f4f6")
            cell.get_text().set_color(BRAND_BLACK)
            cell.get_text().set_fontweight("bold")
            continue

        value = float(display.iloc[row_idx - 1, col_idx])
        face = cmap(norm(value))
        cell.set_facecolor(face)
        luminance = 0.299 * face[0] + 0.587 * face[1] + 0.114 * face[2]
        cell.get_text().set_color("white" if luminance < 0.55 else BRAND_BLACK)


def build_vilc_report_figure(
    df: pd.DataFrame,
    year_label: str,
    month_label: str,
    subtitle: str = "Excl. ARG",
) -> plt.Figure:
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold"})

    region_col = _pick_col(df, ["Zone", "Country", "Entity_1"])
    package_col = _pick_col(df, PACKAGE_CANDIDATES)
    if package_col is None:
        package_col = region_col

    fig = plt.figure(figsize=(18, 22), facecolor="white")
    gs = fig.add_gridspec(
        nrows=4,
        ncols=2,
        height_ratios=[2.9, 2.9, 4.2, 4.2],
        hspace=0.68,
        wspace=0.22,
    )

    ax_mtd_region = fig.add_subplot(gs[0, 0])
    ax_ytd_region = fig.add_subplot(gs[0, 1])
    ax_mtd_pkg = fig.add_subplot(gs[1, 0])
    ax_ytd_pkg = fig.add_subplot(gs[1, 1])
    ax_matrix_mtd = fig.add_subplot(gs[2, :])
    ax_matrix_ytd = fig.add_subplot(gs[3, :])

    if region_col is not None:
        region_mtd = _variance_frame(df, region_col, "MTH")
        region_ytd = _variance_frame(df, region_col, "YTD")
        if not region_mtd.empty:
            region_mtd = region_mtd.sort_values("variance", ascending=False)
            if region_col == "Zone":
                region_mtd = region_mtd.assign(
                    _order=region_mtd[region_col].astype(str).map(lambda x: ZONE_ORDER.index(x) if x in ZONE_ORDER else 999)
                )
                region_mtd = region_mtd.sort_values(by=["_order", "variance"], ascending=[True, False]).drop(columns=["_order"], errors="ignore")
        if not region_ytd.empty:
            region_ytd = region_ytd.sort_values("variance", ascending=False)
        _draw_waterfall(ax_mtd_region, region_mtd, region_col, "variance", "MTD vs BGT ($Mio)")
        _draw_waterfall(ax_ytd_region, region_ytd, region_col, "variance", "YTD vs BGT ($Mio)")
    else:
        ax_mtd_region.axis("off")
        ax_ytd_region.axis("off")

    if package_col is not None:
        pkg_mtd = _variance_frame(df, package_col, "MTH")
        pkg_ytd = _variance_frame(df, package_col, "YTD")
        pkg_mtd = _limit_categories(pkg_mtd, package_col, limit=8)
        pkg_ytd = _limit_categories(pkg_ytd, package_col, limit=8)
        _draw_waterfall(ax_mtd_pkg, pkg_mtd, package_col, "variance", "MTD vs BGT by category ($Mio)")
        _draw_waterfall(ax_ytd_pkg, pkg_ytd, package_col, "variance", "YTD vs BGT by category ($Mio)")
    else:
        ax_mtd_pkg.axis("off")
        ax_ytd_pkg.axis("off")

    if package_col is not None and region_col is not None:
        matrix_frame_mtd = _pairwise_variance_frame(df, package_col, region_col, "MTH")
        matrix_frame_ytd = _pairwise_variance_frame(df, package_col, region_col, "YTD")
        _render_matrix(ax_matrix_mtd, matrix_frame_mtd, package_col, region_col, "MTD vs BGT category by Zone ($Mio)")
        _render_matrix(ax_matrix_ytd, matrix_frame_ytd, package_col, region_col, "YTD vs BGT category by Zone ($Mio)")
    else:
        for ax in [ax_matrix_mtd, ax_matrix_ytd]:
            ax.axis("off")
            ax.text(0.5, 0.5, "No package breakdown available", ha="center", va="center", fontsize=12, color="#666666")

    return fig


def figure_to_png_bytes(fig: plt.Figure, dpi: int = 180) -> bytes:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def save_report_png(fig: plt.Figure, export_path: Path, dpi: int = 180) -> Path:
    export_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(export_path, format="png", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    return export_path