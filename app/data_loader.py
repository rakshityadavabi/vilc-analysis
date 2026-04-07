from __future__ import annotations

import configparser
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st

from app.utils import ALL_MTH_COLS, ALL_YTD_COLS, log_dataframe, log_event, log_query, month_sort_key

try:
    from databricks import sql as dbsql

    DATABRICKS_AVAILABLE = True
except Exception:
    DATABRICKS_AVAILABLE = False

def _load_local_config() -> dict[str, str]:
    config_path = Path(__file__).resolve().parents[1] / ".env"
    parser = configparser.ConfigParser()
    if not config_path.exists():
        return {}

    parser.read(config_path)
    if not parser.has_section("databricks"):
        return {}

    def _clean(value: str) -> str:
        return str(value).strip().strip('"').strip("'")

    values = {key: _clean(value) for key, value in parser.items("databricks")}
    if parser.has_section("performance"):
        values.update({f"performance_{key}": _clean(value) for key, value in parser.items("performance")})
    return values


_LOCAL_CONFIG = _load_local_config()

DB_HOST = os.environ.get("DATABRICKS_HOST", _LOCAL_CONFIG.get("host", ""))
DB_HTTP_PATH = os.environ.get("DATABRICKS_HTTP_PATH", _LOCAL_CONFIG.get("http_path", ""))
DB_TOKEN = os.environ.get("DATABRICKS_TOKEN", _LOCAL_CONFIG.get("token", ""))
DB_CATALOG = os.environ.get("DATABRICKS_CATALOG", _LOCAL_CONFIG.get("catalog", "brewdat_uc_supchn_prod"))
DB_SCHEMA = os.environ.get("DATABRICKS_SCHEMA", _LOCAL_CONFIG.get("schema", "slv_ghq_supply_anaplan"))
DB_TABLE = os.environ.get("DATABRICKS_TABLE", _LOCAL_CONFIG.get("table", "scfd3_consolidator_subpackagelevel_alldata"))

MAX_RAW_ROWS = min(int(os.environ.get("PERF_APP_MAX_RAW_ROWS", _LOCAL_CONFIG.get("performance_max_raw_rows", 1000))), 1000)

_CANONICAL_COLUMNS = [
    "Year",
    "Month",
    "period_1",
    "Zone",
    "Country",
    "Entity_1",
    "Account_3",
    "Account_4",
    "Account_5",
    "Account_5_subpackage",
    "BeverageType",
    "P_&_L_code",
] + ALL_MTH_COLS + ALL_YTD_COLS
_CANONICAL_MAP = {c.lower(): c for c in _CANONICAL_COLUMNS}
_MILLION_SCALE = 1000.0
_METRIC_COLUMNS = list(dict.fromkeys(ALL_MTH_COLS + ALL_YTD_COLS))


def _add_pnp_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for prefix in ("MTH", "YTD"):
        price_col = f"{prefix}_Price"
        perf_col = f"{prefix}_Perf"
        pnp_col = f"{prefix}_PnP"
        if price_col in out.columns or perf_col in out.columns:
            price = pd.to_numeric(out[price_col], errors="coerce").fillna(0) if price_col in out.columns else 0
            perf = pd.to_numeric(out[perf_col], errors="coerce").fillna(0) if perf_col in out.columns else 0
            out[pnp_col] = price + perf
    return out


def split_csv(txt: str) -> list[str]:
    if not txt:
        return []
    return [s.strip() for s in str(txt).split(",") if s.strip()]


def process_raw_df(df: pd.DataFrame, scale_metrics: float = 1.0) -> pd.DataFrame:
    out = df.copy()
    out.columns = out.columns.astype(str).str.strip()

    rename_map = {}
    for col in out.columns:
        canonical = _CANONICAL_MAP.get(col.lower())
        if canonical and canonical != col:
            rename_map[col] = canonical
    if rename_map:
        out = out.rename(columns=rename_map)

    for col in _METRIC_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
            if scale_metrics and scale_metrics != 1.0:
                out[col] = out[col] / float(scale_metrics)

    out = _add_pnp_columns(out)

    return out


@st.cache_data
def load_uploaded_data(file, sheet_name: str = "vilc") -> pd.DataFrame:
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    elif file.name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file, sheet_name=sheet_name)
    else:
        raise ValueError("Unsupported file format. Use CSV or Excel.")

    return process_raw_df(df, scale_metrics=_MILLION_SCALE)


@st.cache_data(ttl=3600)
def query_databricks(host: str, http_path: str, access_token: str, query: str, scale_metrics: float = 1.0) -> pd.DataFrame:
    if not DATABRICKS_AVAILABLE:
        raise RuntimeError("databricks-sql-connector not available in this environment")

    with dbsql.connect(server_hostname=host, http_path=http_path, access_token=access_token) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            cols = [c[0] for c in cur.description]
            rows = cur.fetchall()

    return process_raw_df(pd.DataFrame(rows, columns=cols), scale_metrics=scale_metrics)


@st.cache_data(ttl=3600)
def get_table_columns(host: str, http_path: str, access_token: str, qualified_table: str) -> list[str]:
    if not DATABRICKS_AVAILABLE:
        raise RuntimeError("databricks-sql-connector not available in this environment")

    query = f"DESCRIBE {qualified_table}"
    with dbsql.connect(server_hostname=host, http_path=http_path, access_token=access_token) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()

    return [r[0] for r in rows if r and r[0]]


def _sql_quote_list(vals: list) -> str:
    esc = [str(v).replace("'", "''") for v in vals]
    return ",".join([f"'{v}'" for v in esc])


def build_where_clause(filters: dict) -> str:
    parts = []
    for col, vals in filters.items():
        if not vals:
            continue
        if len(vals) == 1:
            v = str(vals[0]).replace("'", "''")
            parts.append(f"{col} = '{v}'")
        else:
            parts.append(f"{col} IN ({_sql_quote_list(vals)})")
    return ("WHERE " + " AND ".join(parts)) if parts else ""


@st.cache_data(ttl=600)
def get_distinct_values(
    host: str,
    http_path: str,
    access_token: str,
    qualified_table: str,
    col: str,
    filters: dict | None = None,
    limit: int = 1000,
) -> list[str]:
    where = build_where_clause(filters or {})
    q = f"SELECT DISTINCT {col} FROM {qualified_table} {where} LIMIT {int(limit)}"
    df = query_databricks(host, http_path, access_token, q)
    if col not in df.columns:
        return []
    vals = df[col].dropna().unique().tolist()
    return sorted([str(v) for v in vals])


class PerformanceQueryBuilder:
    PARTITION_COLS = ["Year", "Month"]
    HIERARCHY_COLS = ["period_1", "Zone", "Country", "Entity_1"]
    PACKAGE_COLS = ["Account_3", "Account_4", "Account_5", "Account_5_subpackage"]
    ALL_DIM_COLS = PARTITION_COLS + HIERARCHY_COLS + PACKAGE_COLS
    EXTRA_FILTER_COLS = ["BeverageType", "P_&_L_code"]
    DEFAULT_LIMIT = 500_000

    @staticmethod
    def _escape(value) -> str:
        if value is None:
            return "NULL"
        return "'" + str(value).replace("'", "''") + "'"

    @staticmethod
    def _quote_col(col: str) -> str:
        import re

        return f"`{col}`" if re.search(r"[^a-zA-Z0-9_]", col) else col

    @staticmethod
    def _to_list(param) -> Optional[List]:
        if param is None:
            return None
        if isinstance(param, list):
            flat = [str(v).strip() for v in param if v is not None and str(v).strip()]
            return flat if flat else None
        s = str(param).strip()
        return [s] if s else None

    @staticmethod
    def _in_clause(column: str, values: list) -> str:
        qcol = PerformanceQueryBuilder._quote_col(column)
        if len(values) == 1:
            return f"{qcol} = {PerformanceQueryBuilder._escape(values[0])}"
        escaped = ", ".join(PerformanceQueryBuilder._escape(v) for v in values)
        return f"{qcol} IN ({escaped})"

    @staticmethod
    def _not_in_clause(column: str, values: list) -> str:
        qcol = PerformanceQueryBuilder._quote_col(column)
        if len(values) == 1:
            return f"{qcol} <> {PerformanceQueryBuilder._escape(values[0])}"
        escaped = ", ".join(PerformanceQueryBuilder._escape(v) for v in values)
        return f"{qcol} NOT IN ({escaped})"

    @staticmethod
    def _scaled_sum_expr(column: str) -> str:
        qcol = PerformanceQueryBuilder._quote_col(column)
        return f"SUM(CAST({qcol} AS DOUBLE)) / 1000 AS {qcol}"

    @staticmethod
    def _scaled_value_expr(column: str) -> str:
        qcol = PerformanceQueryBuilder._quote_col(column)
        return f"CAST({qcol} AS DOUBLE) / 1000 AS {qcol}"

    @staticmethod
    def _scaled_sum_value_sql(column: str) -> str:
        qcol = PerformanceQueryBuilder._quote_col(column)
        return f"SUM(CAST({qcol} AS DOUBLE)) / 1000"

    def build_query(
        self,
        table: str,
        present_cols: list,
        filters: dict,
        mode: str = "MTD",
        groupby_cols: Optional[list] = None,
        present_mth_cols: Optional[list] = None,
        ytd_last_month: Optional[str] = None,
        limit: int = DEFAULT_LIMIT,
        include_all_metrics: bool = False,
    ) -> str:
        present_set = set(present_cols)
        col_lookup = {str(c).lower(): str(c) for c in present_cols}

        def _resolve(col_name: str) -> Optional[str]:
            return col_lookup.get(str(col_name).lower())

        def _present_metric_cols(candidates: list[str]) -> list[str]:
            cols = []
            for c in candidates:
                mapped = _resolve(c)
                if mapped and mapped in present_set:
                    cols.append(mapped)
            return cols

        month_col = _resolve("Month") or "Month"
        year_col = _resolve("Year") or "Year"

        month_filter = self._to_list(filters.get(month_col) or filters.get("Month"))
        ytd_mode = str(mode).upper() == "YTD"

        if include_all_metrics:
            present_mth_cols = _present_metric_cols(ALL_MTH_COLS)
            present_ytd_cols = _present_metric_cols(ALL_YTD_COLS)
        elif ytd_mode:
            present_ytd_cols = _present_metric_cols(ALL_YTD_COLS)
            present_mth_cols = []
        else:
            if present_mth_cols is None:
                present_mth_cols = _present_metric_cols(ALL_MTH_COLS)
            else:
                present_mth_cols = [c for c in present_mth_cols if c in present_set]
            present_ytd_cols = []

        if groupby_cols is None:
            groupby_cols = [mapped for c in self.ALL_DIM_COLS for mapped in [_resolve(c)] if mapped and mapped in present_set]
        else:
            groupby_cols = [mapped for c in groupby_cols for mapped in [_resolve(c)] if mapped and mapped in present_set]

        where_parts: List[str] = []

        year_vals = self._to_list(filters.get(year_col) or filters.get("Year"))
        if year_vals:
            where_parts.append(self._in_clause(year_col, year_vals))

        exclude_filters = filters.get("_exclude") or filters.get("exclude") or {}
        if isinstance(exclude_filters, dict):
            for col, vals in exclude_filters.items():
                resolved_vals = self._to_list(vals)
                mapped_col = _resolve(col)
                if mapped_col and resolved_vals:
                    where_parts.append(self._not_in_clause(mapped_col, resolved_vals))

        if ytd_mode:
            if ytd_last_month:
                where_parts.append(self._in_clause(month_col, [ytd_last_month]))
        else:
            if month_filter:
                where_parts.append(self._in_clause(month_col, month_filter))

        for col in self.HIERARCHY_COLS + self.PACKAGE_COLS + self.EXTRA_FILTER_COLS:
            mapped_col = _resolve(col)
            if not mapped_col:
                continue
            vals = self._to_list(filters.get(mapped_col) or filters.get(col))
            if vals:
                where_parts.append(self._in_clause(mapped_col, vals))

        where_clause = ("\n    WHERE " + "\n      AND ".join(where_parts)) if where_parts else ""

        sum_parts = [self._scaled_sum_expr(c) for c in present_mth_cols]
        sum_parts += [self._scaled_sum_expr(c) for c in present_ytd_cols]

        if groupby_cols:
            select_clause = (
                "SELECT "
                + ", ".join(groupby_cols)
                + (",\n        " + ",\n        ".join(sum_parts) if sum_parts else "")
            )
            group_by_clause = "\n    GROUP BY " + ", ".join(groupby_cols)
        else:
            select_clause = "SELECT " + (",\n        ".join(sum_parts) if sum_parts else "*")
            group_by_clause = ""

        if ytd_mode and "YTD_Perf" in present_ytd_cols:
            order_clause = "\n    ORDER BY YTD_Perf DESC"
        elif "MTH_Perf" in present_mth_cols:
            order_clause = "\n    ORDER BY MTH_Perf DESC"
        else:
            order_clause = ""

        query = (
            f"{select_clause}\n"
            f"    FROM {table}"
            f"{where_clause}"
            f"{group_by_clause}"
            f"{order_clause}"
            f"\n    LIMIT {int(limit)}"
        )
        return query

    def build_raw_query(
        self,
        table: str,
        present_cols: list,
        filters: dict,
        ytd_last_month: Optional[str] = None,
        limit: int = DEFAULT_LIMIT,
        include_all_metrics: bool = False,
    ) -> str:
        if limit > MAX_RAW_ROWS:
            raise ValueError(
                f"Raw fetch limit {limit:,} exceeds safety cap {MAX_RAW_ROWS:,}. "
                "Enable server-side aggregation or reduce the row limit."
            )

        present_set = set(present_cols)
        month_filter = self._to_list(filters.get("Month"))
        ytd_mode = not month_filter

        needed: List[str] = []
        for c in self.ALL_DIM_COLS:
            if c in present_set:
                needed.append(c)

        if include_all_metrics:
            metric_src = ALL_MTH_COLS + ALL_YTD_COLS
        else:
            metric_src = ALL_YTD_COLS if ytd_mode else ALL_MTH_COLS
        for c in metric_src:
            if c in present_set and c not in needed:
                needed.append(c)

        where_parts: List[str] = []
        year_vals = self._to_list(filters.get("Year"))
        if year_vals:
            where_parts.append(self._in_clause("Year", year_vals))

        exclude_filters = filters.get("_exclude") or filters.get("exclude") or {}
        if isinstance(exclude_filters, dict):
            for col, vals in exclude_filters.items():
                resolved_vals = self._to_list(vals)
                if resolved_vals:
                    where_parts.append(self._not_in_clause(col, resolved_vals))

        if ytd_mode:
            if ytd_last_month:
                where_parts.append(self._in_clause("Month", [ytd_last_month]))
        else:
            if month_filter:
                where_parts.append(self._in_clause("Month", month_filter))

        for col in self.HIERARCHY_COLS + self.PACKAGE_COLS + self.EXTRA_FILTER_COLS:
            vals = self._to_list(filters.get(col))
            if vals:
                where_parts.append(self._in_clause(col, vals))

        where_clause = ("\n    WHERE " + "\n      AND ".join(where_parts)) if where_parts else ""
        metric_selects = [self._scaled_value_expr(c) for c in metric_src if c in present_set]
        select_items = [c for c in needed if c not in metric_src] + metric_selects
        select_str = ", ".join(select_items) if select_items else "*"
        return f"SELECT {select_str}\n    FROM {table}{where_clause}\n    LIMIT {int(limit)}"

    def build_ranked_query(
        self,
        table: str,
        present_cols: list,
        filters: dict,
        rank_cols: list,
        perf_col: str = "MTH_Perf",
        top_n: int = 5,
        ytd_last_month: Optional[str] = None,
        limit: int = DEFAULT_LIMIT,
    ) -> str:
        present_set = set(present_cols)
        rank_cols = [c for c in rank_cols if c in present_set]
        if not rank_cols:
            raise ValueError("No ranking columns found in table.")

        month_filter = self._to_list(filters.get("Month"))
        ytd_mode = not month_filter

        if ytd_mode:
            present_mth = []
            present_ytd = [c for c in ALL_YTD_COLS if c in present_set]
        else:
            present_mth = [c for c in ALL_MTH_COLS if c in present_set]
            present_ytd = []

        sum_parts = [self._scaled_sum_expr(c) for c in present_mth]
        sum_parts += [self._scaled_sum_expr(c) for c in present_ytd]

        where_parts = []
        year_vals = self._to_list(filters.get("Year"))
        if year_vals:
            where_parts.append(self._in_clause("Year", year_vals))

        exclude_filters = filters.get("_exclude") or filters.get("exclude") or {}
        if isinstance(exclude_filters, dict):
            for col, vals in exclude_filters.items():
                resolved_vals = self._to_list(vals)
                if resolved_vals:
                    where_parts.append(self._not_in_clause(col, resolved_vals))

        if ytd_mode:
            if ytd_last_month:
                where_parts.append(self._in_clause("Month", [ytd_last_month]))
        else:
            if month_filter:
                where_parts.append(self._in_clause("Month", month_filter))

        for col in self.HIERARCHY_COLS + self.PACKAGE_COLS + self.EXTRA_FILTER_COLS:
            vals = self._to_list(filters.get(col))
            if vals:
                where_parts.append(self._in_clause(col, vals))

        where_clause = ("\n    WHERE " + "\n      AND ".join(where_parts)) if where_parts else ""

        perf_alias = perf_col if perf_col in (present_mth + present_ytd) else ("YTD_Perf" if ytd_mode else "MTH_Perf")
        group_str = ", ".join(rank_cols)

        select_inner = (
            "SELECT "
            + group_str
            + (",\n        " + ",\n        ".join(sum_parts) if sum_parts else "")
            + f",\n        ROW_NUMBER() OVER (ORDER BY {self._scaled_sum_value_sql(perf_alias)} DESC) AS _rn_desc"
            + f",\n        ROW_NUMBER() OVER (ORDER BY {self._scaled_sum_value_sql(perf_alias)} ASC)  AS _rn_asc"
        )

        inner_query = f"{select_inner}\n    FROM {table}{where_clause}\n    GROUP BY {group_str}\n"
        outer = (
            f"SELECT * FROM (\n{inner_query}) AS _ranked\n"
            f"WHERE _rn_desc <= {int(top_n)} OR _rn_asc <= {int(top_n)}\n"
            f"ORDER BY _rn_desc\n"
            f"LIMIT {int(limit)}"
        )
        return outer


@st.cache_data(ttl=3600)
def load_aggregated_databricks_data(
    request_key: str = "",
    host: str = DB_HOST,
    http_path: str = DB_HTTP_PATH,
    access_token: str = DB_TOKEN,
    catalog: str = DB_CATALOG,
    schema: str = DB_SCHEMA,
    table: str = DB_TABLE,
    row_limit: int = 500000,
) -> pd.DataFrame:
    if not host or not http_path or not access_token:
        raise RuntimeError("Missing Databricks connection details. Set them in .env or environment variables.")

    import json

    payload = json.loads(request_key) if request_key else {}
    mode = str(payload.get("mode", "MTD")).upper()
    query_filters = payload.get("filters", {}) or {}
    qualified_table = ".".join([part for part in [catalog, schema, table] if part])

    table_cols = get_table_columns(host, http_path, access_token, qualified_table)
    groupby_cols = [c for c in PerformanceQueryBuilder.ALL_DIM_COLS if c in table_cols]
    if not groupby_cols:
        groupby_cols = [c for c in ["Zone", "Country", "Entity_1", "Account_3"] if c in table_cols]

    log_event(f"load_aggregated_databricks_data -> mode={mode}, filters={query_filters}, groupby_cols={groupby_cols}")
    query = PerformanceQueryBuilder().build_query(
        table=qualified_table,
        present_cols=table_cols,
        filters=query_filters,
        mode=mode,
        groupby_cols=groupby_cols,
        limit=int(row_limit),
        include_all_metrics=True,
    )
    st.session_state["last_query"] = query
    log_query("SQL query", query)

    df = query_databricks(host, http_path, access_token, query)
    log_dataframe("Databricks result", df)
    return df
