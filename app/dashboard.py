from __future__ import annotations

import json

import streamlit as st

from app.components.report_charts import build_waterfall_figure, build_zone_package_matrix_figure
from reports.chart_builders import get_all_package_order
from app.data_loader import (
    DB_CATALOG,
    DB_HOST,
    DB_HTTP_PATH,
    DB_SCHEMA,
    DB_TABLE,
    DB_TOKEN,
    get_distinct_values,
    load_aggregated_databricks_data,
)
from app.services.aggregation_service import get_summary_metrics
from app.utils import fmt_million, log_event, month_sort_key


BREAKDOWN_OPTIONS = [
    "Zone",
    "Country",
    "Entity_1",
    "Account_3",
    "Account_4",
    "Account_5",
    "Account_5_subpackage",
]


def init_dashboard_state() -> None:
    log_event("init_dashboard_state()")
    if "active_df" not in st.session_state:
        st.session_state["active_df"] = None
    if "active_query_key" not in st.session_state:
        st.session_state["active_query_key"] = ""


def _qualified_table_name() -> str:
    return ".".join([part for part in [DB_CATALOG, DB_SCHEMA, DB_TABLE] if part])


def _year_sort_key(value) -> int:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if not digits:
        return -1
    try:
        return int(digits[-4:])
    except ValueError:
        return -1


def _distinct_options(column: str, filters: dict | None = None) -> list[str]:
    try:
        values = get_distinct_values(
            DB_HOST,
            DB_HTTP_PATH,
            DB_TOKEN,
            _qualified_table_name(),
            column,
            filters=filters,
        )
    except Exception:
        return []

    if column == "Month":
        return sorted(values, key=month_sort_key, reverse=True)
    if column == "Year":
        return sorted(values, key=_year_sort_key, reverse=True)
    return sorted(values)


def _collect_sidebar_filters() -> dict:
    with st.sidebar:
        mode = st.radio("Mode", options=["MTD", "YTD"], horizontal=True, key="mode_filter")

        year_options = _distinct_options("Year")
        if not year_options:
            st.error("No Year values were returned from Databricks.")
            return {}

        selected_year = st.selectbox("Year", options=year_options, index=0, key="year_filter")

        month_options = _distinct_options("Month", filters={"Year": [selected_year]})
        if not month_options:
            st.error("No Month values were returned for the selected Year.")
            return {}

        selected_month = st.selectbox("Month", options=month_options, index=0, key="month_filter")

        zone_options = _distinct_options("Zone", filters={"Year": [selected_year], "Month": [selected_month]})
        selected_zones = st.multiselect("Zone selection", options=zone_options, key="zone_filter")

        package_options = _distinct_options("Account_3", filters={"Year": [selected_year], "Month": [selected_month]})
        if not package_options:
            package_options = _distinct_options("Account_5", filters={"Year": [selected_year], "Month": [selected_month]})
        selected_packages = st.multiselect("Package selection", options=package_options, key="package_filter")

        if st.button("Refresh Databricks data", type="secondary"):
            log_event("Refresh Databricks data clicked")
            st.cache_data.clear()
            st.session_state.pop("active_df", None)
            st.session_state.pop("active_query_key", None)
            st.rerun()

    query_filters: dict[str, list[str]] = {
        "Year": [selected_year],
        "Month": [selected_month],
    }
    if selected_zones:
        query_filters["Zone"] = selected_zones
    if selected_packages:
        query_filters["Account_3"] = selected_packages

    return {"mode": mode, "filters": query_filters, "year": selected_year, "month": selected_month}


def _request_payload(ui_state: dict) -> dict:
    return {
        "mode": ui_state.get("mode", "MTD"),
        "filters": ui_state.get("filters", {}),
        "drill_path": {},
        "query_backend": "Databricks",
    }


def _validate_dataframe(df) -> list[str]:
    required = [
        "MTH_Price",
        "MTH_Perf",
        "MTH_BU",
        "MTH_ACT",
        "MTH_Mix",
        "MTH_Scope",
        "MTH_Volume",
        "YTD_Price",
        "YTD_Perf",
        "YTD_BU",
        "YTD_ACT",
        "YTD_Mix",
        "YTD_Scope",
        "YTD_Volume",
    ]
    errors = []
    for col in required:
        if col not in df.columns:
            errors.append(f"Missing required metric column: {col}")
    if not any(c in df.columns for c in ["Zone", "Country", "Entity_1"]):
        errors.append("No region hierarchy column found. Expected one of Zone, Country, Entity_1.")
    if not any(c in df.columns for c in ["Account_3", "Account_4", "Account_5", "Account_5_subpackage"]):
        errors.append("No package hierarchy column found. Expected one of Account_3, Account_4, Account_5, Account_5_subpackage.")
    return errors


def _summary_value(summary: dict, key: str) -> float:
    return float(summary.get(key, 0.0))


def _metric_total(df, col: str) -> float:
    return float(df[col].sum()) if col in df.columns else 0.0


def _render_metric_row(df, metric_prefix: str, display_prefix: str) -> None:
    metric_specs = [
        ("BU", f"{metric_prefix}_BU"),
        ("ACT", f"{metric_prefix}_ACT"),
        ("Delta ACT - BU", None),
        ("Price", f"{metric_prefix}_Price"),
        ("Perf", f"{metric_prefix}_Perf"),
        ("Mix", f"{metric_prefix}_Mix"),
        ("Scope", f"{metric_prefix}_Scope"),
        ("Volume", f"{metric_prefix}_Volume"),
    ]

    values = []
    bu_val = _metric_total(df, f"{metric_prefix}_BU")
    act_val = _metric_total(df, f"{metric_prefix}_ACT")
    delta_val = act_val - bu_val
    for label, col in metric_specs:
        if col is None:
            values.append((label, delta_val))
        else:
            values.append((label, _metric_total(df, col)))

    cols = st.columns(len(values))
    for idx, (label, value) in enumerate(values):
        cols[idx].metric(f"{display_prefix} {label}", fmt_million(value))


def _render_metric_rows(df) -> None:
    _render_metric_row(df, "MTH", "MTD")
    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    _render_metric_row(df, "YTD", "YTD")


def _render_chart_grid(df) -> None:
    st.markdown("<div style='text-align:center;font-size:1.7rem;font-weight:800;margin:0.35rem 0 0.45rem;'>MTD/YTD VIC P&amp;P vs BU</div>", unsafe_allow_html=True)
    row1 = st.columns([1, 0.03, 1])
    with row1[0]:
        st.plotly_chart(build_waterfall_figure(df, "Zone", "MTH_PnP", "MTD PnP by Zone ($Mio)"), use_container_width=True, key="mtd_pnp_zone")
    with row1[1]:
        st.markdown("<div style='border-left:2px solid #d0d0d0;height:100%;min-height:540px;margin:0 auto;'></div>", unsafe_allow_html=True)
    with row1[2]:
        st.plotly_chart(build_waterfall_figure(df, "Zone", "YTD_PnP", "YTD PnP by Zone ($Mio)"), use_container_width=True, key="ytd_pnp_zone")

    st.divider()
    st.markdown("<div style='text-align:center;font-size:1.7rem;font-weight:800;margin:0.35rem 0 0.45rem;'>MTD/YTD VIC Price vs BU</div>", unsafe_allow_html=True)
    row2 = st.columns([1, 0.03, 1])
    with row2[0]:
        st.plotly_chart(build_waterfall_figure(df, "Zone", "MTH_Price", "MTD Price by Zone ($Mio)"), use_container_width=True, key="mtd_price_zone")
    with row2[1]:
        st.markdown("<div style='border-left:2px solid #d0d0d0;height:100%;min-height:540px;margin:0 auto;'></div>", unsafe_allow_html=True)
    with row2[2]:
        st.plotly_chart(build_waterfall_figure(df, "Zone", "YTD_Price", "YTD Price by Zone ($Mio)"), use_container_width=True, key="ytd_price_zone")

    st.divider()
    st.markdown("<div style='text-align:center;font-size:1.7rem;font-weight:800;margin:0.35rem 0 0.45rem;'>MTD/YTD Package Level Breakdown</div>", unsafe_allow_html=True)
    package_order = get_all_package_order(df)
    row3 = st.columns([1, 0.03, 1])
    with row3[0]:
        st.plotly_chart(
            build_waterfall_figure(
                df,
                "Account_3",
                "MTH_Price",
                "MTD Package Price breakdown ($Mio)",
                label_order=package_order,
            ),
            use_container_width=True,
            key="mtd_account3_price",
        )
    with row3[1]:
        st.markdown("<div style='border-left:2px solid #d0d0d0;height:100%;min-height:540px;margin:0 auto;'></div>", unsafe_allow_html=True)
    with row3[2]:
        st.plotly_chart(
            build_waterfall_figure(
                df,
                "Account_3",
                "YTD_Price",
                "YTD Package Price breakdown ($Mio)",
                label_order=package_order,
            ),
            use_container_width=True,
            key="ytd_account3_price",
        )

    st.divider()
    zone_matrix_fig = build_zone_package_matrix_figure(
        df,
        "Account_3",
        "Zone",
        "MTH_Price",
        "MTD vs BGT package by Zone ($Mio)",
        row_order=package_order,
        col_order=["EUR", "AFR", "SAZ", "APAC", "NAZ", "MAZ"],
        total_col_label="ABI",
        max_rows=len(package_order) if package_order else 8,
    )
    st.pyplot(zone_matrix_fig, use_container_width=True)


def render_driver_dashboard() -> None:
    log_event("render_driver_dashboard()")

    ui_state = _collect_sidebar_filters()
    if not ui_state:
        return

    request_payload = _request_payload(ui_state)
    request_key = json.dumps(request_payload, sort_keys=True)
    log_event(f"request_payload={request_payload}")

    if st.session_state.get("active_query_key") != request_key:
        try:
            with st.spinner("Loading Databricks report data..."):
                st.session_state["active_df"] = load_aggregated_databricks_data(request_key=request_key)
                st.session_state["active_query_key"] = request_key
        except Exception as exc:
            st.error(f"Databricks load failed: {exc}")
            return

    df = st.session_state.get("active_df")
    if df is None:
        st.info("Load data from Databricks to begin.")
        return

    errors = _validate_dataframe(df)
    if errors:
        for err in errors:
            st.error(err)
        return

    log_event(f"Data ready for render: rows={len(df):,}, cols={len(df.columns):,}")

    st.caption(f"{ui_state['mode']} • Year {ui_state['year']} • Month {ui_state['month']}")
    _render_metric_rows(df)

    _render_chart_grid(df)
