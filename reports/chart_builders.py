from __future__ import annotations

from pathlib import Path

import matplotlib.figure
import pandas as pd
import plotly.graph_objects as go

from app.components.report_charts import build_waterfall_figure, build_zone_package_matrix_figure

from .data_queries import load_report_dataframe, normalize_report_period


REPORT_ZONE_ORDER = ["EUR", "AFR", "SAZ", "APAC", "NAZ", "MAZ"]


def _package_order(df: pd.DataFrame, value_columns: tuple[str, str] = ("MTH_Price", "YTD_Price")) -> list[str]:
    available_columns = [column for column in value_columns if column in df.columns]
    if "Account_3" not in df.columns or not available_columns:
        return []

    package_totals = df.groupby("Account_3", dropna=False)[available_columns].sum().reset_index()
    package_totals["has_value"] = package_totals[available_columns].abs().max(axis=1)
    package_totals = package_totals[package_totals["has_value"] >= 0.1]
    package_totals = package_totals.sort_values(["has_value", "Account_3"], ascending=[False, True])
    return [str(value) for value in package_totals["Account_3"].tolist()]


def get_all_package_order(df: pd.DataFrame, value_columns: tuple[str, str] = ("MTH_Price", "YTD_Price")) -> list[str]:
    return _package_order(df, value_columns=value_columns)


def _package_order_for_charts(df: pd.DataFrame, value_columns: tuple[str, str] = ("MTH_Price", "YTD_Price")) -> list[str]:
    return _package_order(df, value_columns=value_columns)


def _get_df(month, year, df: pd.DataFrame | None = None) -> pd.DataFrame:
    return df if df is not None else load_report_dataframe(month, year)


def _build_waterfall(
    df: pd.DataFrame,
    level: str,
    perf_col: str,
    title: str,
    value_columns: tuple[str, str] = ("MTH_Price", "YTD_Price"),
) -> go.Figure:
    label_order = REPORT_ZONE_ORDER if level == "Zone" else _package_order_for_charts(df, value_columns=value_columns)
    fig = build_waterfall_figure(df, level, perf_col, title, total_label="ABI", label_order=label_order)
    return fig


def _build_zone_table(
    df: pd.DataFrame,
    perf_col: str,
    title: str,
    value_columns: tuple[str, str] = ("MTH_Price", "YTD_Price"),
) -> matplotlib.figure.Figure:
    package_order = get_all_package_order(df, value_columns=value_columns)
    return build_zone_package_matrix_figure(
        df,
        "Account_3",
        "Zone",
        perf_col,
        title,
        row_header_label="Package",
        row_order=package_order,
        col_order=REPORT_ZONE_ORDER,
        total_col_label="ABI",
        max_rows=len(package_order) if package_order else 12,
    )


def build_mtd_vic_pp_mtd(month, year, df: pd.DataFrame | None = None) -> go.Figure:
    report_df = _get_df(month, year, df)
    return _build_waterfall(report_df, "Zone", "MTH_PnP", "MTD vs BGT ($Mio)")


def build_mtd_vic_pp_ytd(month, year, df: pd.DataFrame | None = None) -> go.Figure:
    report_df = _get_df(month, year, df)
    return _build_waterfall(report_df, "Zone", "YTD_PnP", "YTD vs BGT ($Mio)")


def build_mtd_vic_price_mtd(month, year, df: pd.DataFrame | None = None) -> go.Figure:
    report_df = _get_df(month, year, df)
    return _build_waterfall(report_df, "Zone", "MTH_Price", "MTD vs BGT ($Mio)")


def build_mtd_vic_price_ytd(month, year, df: pd.DataFrame | None = None) -> go.Figure:
    report_df = _get_df(month, year, df)
    return _build_waterfall(report_df, "Zone", "YTD_Price", "YTD vs BGT ($Mio)")


def build_mtd_category_mtd(month, year, df: pd.DataFrame | None = None) -> go.Figure:
    report_df = _get_df(month, year, df)
    return _build_waterfall(report_df, "Account_3", "MTH_Price", "MTD vs BGT by package ($Mio)")


def build_mtd_category_ytd(month, year, df: pd.DataFrame | None = None) -> go.Figure:
    report_df = _get_df(month, year, df)
    return _build_waterfall(report_df, "Account_3", "YTD_Price", "YTD vs BGT by package ($Mio)")


def build_mtd_zone_table(month, year, df: pd.DataFrame | None = None) -> matplotlib.figure.Figure:
    report_df = _get_df(month, year, df)
    return _build_zone_table(report_df, "MTH_Price", "MTD vs BGT category by Zone ($Mio)")


def build_mtd_perf_mtd(month, year, df: pd.DataFrame | None = None) -> go.Figure:
    report_df = _get_df(month, year, df)
    return _build_waterfall(
        report_df,
        "Zone",
        "MTH_Perf",
        "MTD vs BGT Performance ($Mio)",
        value_columns=("MTH_Perf", "YTD_Perf"),
    )


def build_mtd_perf_ytd(month, year, df: pd.DataFrame | None = None) -> go.Figure:
    report_df = _get_df(month, year, df)
    return _build_waterfall(
        report_df,
        "Zone",
        "YTD_Perf",
        "YTD vs BGT Performance ($Mio)",
        value_columns=("MTH_Perf", "YTD_Perf"),
    )


def build_mtd_perf_category_mtd(month, year, df: pd.DataFrame | None = None) -> go.Figure:
    report_df = _get_df(month, year, df)
    return _build_waterfall(
        report_df,
        "Account_3",
        "MTH_Perf",
        "MTD vs BGT performance by package ($Mio)",
        value_columns=("MTH_Perf", "YTD_Perf"),
    )


def build_mtd_perf_category_ytd(month, year, df: pd.DataFrame | None = None) -> go.Figure:
    report_df = _get_df(month, year, df)
    return _build_waterfall(
        report_df,
        "Account_3",
        "YTD_Perf",
        "YTD vs BGT performance by package ($Mio)",
        value_columns=("MTH_Perf", "YTD_Perf"),
    )


def build_mtd_perf_zone_table(month, year, df: pd.DataFrame | None = None) -> matplotlib.figure.Figure:
    report_df = _get_df(month, year, df)
    return _build_zone_table(
        report_df,
        "MTH_Perf",
        "MTD vs BGT performance category by Zone ($Mio)",
        value_columns=("MTH_Perf", "YTD_Perf"),
    )


def report_title_context(month, year) -> dict[str, str]:
    month_display, year_display, _ = normalize_report_period(month, year)
    return {
        "month_display": month_display,
        "year_display": year_display,
        "report_subtitle": "Excl. ARG",
        "report_period": f"{month_display}, {year_display}",
    }