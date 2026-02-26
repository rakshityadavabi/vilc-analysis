import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import time
import logging
from typing import List, Optional

# Optional Databricks SQL connector (handled gracefully if absent)
try:
    from databricks import sql as dbsql
    DATABRICKS_AVAILABLE = True
except Exception:
    DATABRICKS_AVAILABLE = False

SKLEARN_OK = True   # pure-numpy KMeans — no external dependency

# ── Databricks connection defaults ───────────────────────────────────────────
# Fill in your values here. Env vars override if set.
DB_HOST       = os.environ.get("DATABRICKS_HOST",)
DB_HTTP_PATH  = os.environ.get("DATABRICKS_HTTP_PATH",)
DB_TOKEN      = os.environ.get("DATABRICKS_TOKEN", "")  
DB_CATALOG    = os.environ.get("DATABRICKS_CATALOG", "brewdat_uc_supchn_prod")
DB_SCHEMA     = os.environ.get("DATABRICKS_SCHEMA", "slv_ghq_supply_anaplan")
DB_TABLE      = os.environ.get("DATABRICKS_TABLE", "scfd3_consolidator_subpackagelevel_alldata")

# ── Month ordering helpers ────────────────────────────────────────────────────
# ABI month labels follow the pattern "MMM YY", e.g. "Jan 25", "Dec 25".
# These helpers identify the LAST month of a fiscal year so that YTD queries
# are restricted to that single month and SUM(YTD_*) is never double-counted.
_MONTH_ORDER: dict = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,  "May": 5,  "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


def _month_sort_key(m: str) -> tuple:
    """Return (2-digit-year, month_number) for strings like 'Dec 25'."""
    parts = str(m).split()
    if len(parts) == 2:
        try:
            return (int(parts[1]), _MONTH_ORDER.get(parts[0], 0))
        except ValueError:
            pass
    return (0, 0)


def get_ytd_last_month(years: list, available_months: list) -> "Optional[str]":
    """Return the latest month that belongs to the given fiscal year labels.

    Rules (in priority order):
    1. Filter `available_months` to those whose 2-digit suffix matches the FY
       suffix (e.g. 'FY25' → months ending in '25'), then return the max.
    2. If no match, fall back to a heuristic: assume December is the last month
       of the fiscal year  (e.g. 'FY25' → 'Dec 25').
    3. Last resort: return the global max of all available_months.
    4. Return None only when both lists are empty.
    """
    if not years:
        return None
    # Extract 2-digit numeric suffix from FY labels like 'FY25' → '25'
    year_suffixes: set = set()
    for yr in years:
        s = str(yr).replace("FY", "").replace("fy", "").strip()
        if s.isdigit():
            year_suffixes.add(s)
    # 1. Match from available months
    if year_suffixes and available_months:
        matching = [
            m for m in available_months
            if len(str(m).split()) == 2 and str(m).split()[1] in year_suffixes
        ]
        if matching:
            return max(matching, key=_month_sort_key)
    # 2. Heuristic fallback: FY25 → Dec 25
    if year_suffixes:
        suffix = sorted(year_suffixes, reverse=True)[0]   # latest year first
        return f"Dec {suffix}"
    # 3. Global max of all available months
    if available_months:
        return max(available_months, key=_month_sort_key)
    return None

st.set_page_config(
    page_title="Performance Analyzer",
    page_icon="📊",
    layout="wide",
)

# ── Logging setup ─────────────────────────────────────────────────────────────
class _ListHandler(logging.Handler):
    """Logging handler that appends formatted records to session_state['logs']."""
    def emit(self, record):
        if "logs" not in st.session_state:
            st.session_state["logs"] = []
        st.session_state["logs"].append(self.format(record))

_logger = logging.getLogger("perf_app")
if not _logger.handlers:
    _h = _ListHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    _logger.addHandler(_h)
    _logger.setLevel(logging.DEBUG)

def log(msg: str, level: str = "info"):
    getattr(_logger, level)(msg)


def fmt_num(v) -> str:
    """Format a number to M / K / units with 2 decimal places.
    1 234 567  → '1.23 M'
      12 345   → '12.35 K'
         123   → '123.00'
    """
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    abs_v = abs(v)
    if abs_v >= 1_000_000:
        return f"{v / 1_000_000:.2f} M"
    elif abs_v >= 1_000:
        return f"{v / 1_000:.2f} K"
    else:
        return f"{v:.2f}"

st.title("📊 Dynamic Performance Analyzer")
st.markdown("---")

HIERARCHY     = ["Zone", "Country", "Entity_1"]
METRIC_COLS   = ["MTH_Perf", "MTH_BU", "MTH_ACT"]
# Structural columns that must always be present regardless of mode.
# Metric columns are NOT listed here because YTD mode uses YTD_* and
# MTH mode uses MTH_* — both sets are valid; at least one must be present.
REQUIRED_COLS = ["Zone", "Country", "Entity_1", "Account_3"]
# At least one of these perf columns must exist (checked at validation time)
_PERF_COL_CANDIDATES = ["MTH_Perf", "YTD_Perf"]
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all object-dtype columns to str to prevent Arrow serialisation errors."""
    out = df.copy()
    for col in out.select_dtypes(include="object").columns:
        out[col] = out[col].astype(str)
    return out


def style_positive_blue(df: pd.DataFrame, metric_cols: list):
    """Return a Styler where positive metrics use Blues gradient and negatives are highlighted red.
    All numeric columns are formatted with fmt_num (M / K / units, 2 dp).
    """
    sty = df.style
    # Format all numeric columns with M/K units
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        sty = sty.format({c: fmt_num for c in num_cols})
    for c in metric_cols:
        if c in df.columns:
            try:
                sty = sty.background_gradient(subset=[c], cmap="Blues")
            except Exception:
                pass
            def neg(v):
                try:
                    return "background-color: #ffd6d6" if float(v) < 0 else ""
                except Exception:
                    return ""
            sty = sty.map(neg, subset=[c])
    return sty


@st.cache_data
def load_data(file, sheet_name: str) -> pd.DataFrame:
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    elif file.name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file, sheet_name=sheet_name)
    else:
        raise ValueError("Unsupported file format. Use CSV or Excel.")

    df.columns = df.columns.astype(str).str.strip()

    # Coerce ALL known metric columns (both MTH and YTD) to numeric when present.
    # This is safe to do unconditionally — columns absent from df are skipped.
    _all_metric_candidates = list(dict.fromkeys(METRIC_COLS + ALL_MTH_COLS + ALL_YTD_COLS))
    for col in _all_metric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def process_raw_df(df: pd.DataFrame) -> pd.DataFrame:
    """Shared post-processing for any raw dataframe source: cleanup cols, coerce metrics."""
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    _all_metric_candidates = list(dict.fromkeys(METRIC_COLS + ALL_MTH_COLS + ALL_YTD_COLS))
    for col in _all_metric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


@st.cache_data(ttl=3600)
def query_databricks(host: str, http_path: str, access_token: str, query: str) -> pd.DataFrame:
    """Run a SQL query against Databricks using databricks-sql-connector.

    Requires `databricks-sql-connector` to be installed in the runtime.
    The connection parameters are passed in; leave empty in the UI for secrets.
    Results are cached for 1 hour keyed by (host, http_path, query).
    """
    if not DATABRICKS_AVAILABLE:
        raise RuntimeError("databricks-sql-connector not available in this environment")

    with dbsql.connect(server_hostname=host, http_path=http_path, access_token=access_token) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            cols = [c[0] for c in cur.description]
            rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=cols)
    return process_raw_df(df)


# ── Databricks helpers for schema + server-side query building ───────────
MAX_RAW_ROWS = int(os.environ.get("PERF_APP_MAX_RAW_ROWS", 200_000))


@st.cache_data(ttl=3600)
def get_table_columns(host: str, http_path: str, access_token: str, qualified_table: str) -> list:
    """Return list of column names for a table in Databricks. Uses DESCRIBE TABLE.

    qualified_table should be `catalog.schema.table` or `schema.table` or `table`.
    """
    if not DATABRICKS_AVAILABLE:
        raise RuntimeError("databricks-sql-connector not available in this environment")
    q = f"DESCRIBE {qualified_table}"
    with dbsql.connect(server_hostname=host, http_path=http_path, access_token=access_token) as conn:
        with conn.cursor() as cur:
            cur.execute(q)
            rows = cur.fetchall()
            # DESCRIBE returns rows like (col_name, data_type, comment)
            cols = [r[0] for r in rows if r and r[0]]
    return cols


def split_csv(txt: str) -> list:
    """Split a comma-separated user input into trimmed non-empty strings."""
    if not txt:
        return []
    return [s.strip() for s in str(txt).split(",") if s.strip()]


@st.cache_data(ttl=600)
def get_distinct_values(host: str, http_path: str, access_token: str, qualified_table: str, col: str, filters: dict | None = None, limit: int = 1000) -> list:
    """Fetch distinct values for `col` from `qualified_table`, applying optional filters.

    Returns a sorted list of string values. Cached for 10 minutes.
    """
    if not DATABRICKS_AVAILABLE:
        raise RuntimeError("databricks-sql-connector not available in this environment")
    where = build_where_clause(filters or {})
    q = f"SELECT DISTINCT {col} FROM {qualified_table} {where} LIMIT {int(limit)}"
    df = query_databricks(host, http_path, access_token, q)
    if col not in df.columns:
        return []
    vals = df[col].dropna().unique().tolist()
    # convert to strings for stable UI behaviour
    return sorted([str(v) for v in vals])


def _sql_quote_list(vals: list) -> str:
    """Return SQL list like ('a','b') with proper escaping."""
    esc = [str(v).replace("'", "''") for v in vals]
    return ",".join([f"'{v}'" for v in esc])


def build_where_clause(filters: dict) -> str:
    """Build WHERE clause from filters dict {col: [vals]}.
    Empty lists are ignored. Values are quoted as strings.
    """
    parts = []
    for col, vals in filters.items():
        if not vals:
            continue
        # For numeric-like values we still quote to be safe; Databricks will cast if needed
        if len(vals) == 1:
            v = str(vals[0]).replace("'", "''")
            parts.append(f"{col} = '{v}'")
        else:
            parts.append(f"{col} IN ({_sql_quote_list(vals)})")
    return ("WHERE " + " AND ".join(parts)) if parts else ""


class PerformanceQueryBuilder:
    """
    Dynamic, scale-safe SQL query builder for the Performance Analyzer on Databricks.

    Mirrors the pattern in VilcSummary (get_vilc_summary.py). Builds a fully
    server-side aggregated query so Databricks performs all heavy lifting
    (filter → partition-prune → aggregate) before any data crosses the network.

    Scale strategy
    ──────────────
    1. Partition pruning first — Year and Month always go FIRST in WHERE so the
       Spark planner can skip entire file partitions before touching any row data.
    2. Server-side aggregation — GROUP BY + SUM runs on the cluster; Streamlit
       receives only one aggregated row per dimension combination (not 100M+ rows).
    3. Dynamic column presence — caller supplies `present_cols` (from DESCRIBE TABLE);
       only columns that exist are selected or summed — no runtime errors on missing cols.
    4. Minimal SELECT / GROUP BY — only the requested groupby dimensions + metric sums;
       no SELECT * that would pull unwanted columns across the wire.
    5. LIMIT after aggregation — caps the result row count (not raw row count), acting
       as a safety net against unexpectedly high dimension cardinality.
    6. Safe escaping — every user-supplied value goes through _escape(); no raw
       f-string substitution of filter values, preventing SQL injection.
    """

    # Partition columns listed first — their WHERE predicates must come before
    # any other filter so the Spark optimizer knows to prune partitions early.
    PARTITION_COLS    = ["Year", "Month"]
    HIERARCHY_COLS    = ["period_1", "Zone", "Country", "Entity_1"]
    PACKAGE_COLS      = ["Account_3", "Account_4", "Account_5", "Account_5_subpackage"]
    ALL_DIM_COLS      = PARTITION_COLS + HIERARCHY_COLS + PACKAGE_COLS
    # Extra filter-only cols — added to WHERE but NOT to GROUP BY / SELECT
    EXTRA_FILTER_COLS = ["BeverageType", "P_&_L_code"]

    DEFAULT_LIMIT  = 500_000

    # ── Escaping helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _escape(value) -> str:
        """Single-quote a scalar value safely for SQL injection prevention."""
        if value is None:
            return "NULL"
        return "'" + str(value).replace("'", "''") + "'"

    @staticmethod
    def _quote_col(col: str) -> str:
        """Backtick-quote a column name that contains characters outside [a-zA-Z0-9_]."""
        import re
        return f"`{col}`" if re.search(r'[^a-zA-Z0-9_]', col) else col

    @staticmethod
    def _to_list(param) -> Optional[List]:
        """Normalise a filter param into a list of strings, or None if empty."""
        if param is None:
            return None
        if isinstance(param, list):
            flat = [str(v).strip() for v in param if v is not None and str(v).strip()]
            return flat if flat else None
        s = str(param).strip()
        return [s] if s else None

    @staticmethod
    def _in_clause(column: str, values: list) -> str:
        """Return single-value `col = 'x'` or multi-value `col IN ('x','y')` fragment.
        Column names with special characters (e.g. P_&_L_code) are backtick-quoted.
        """
        qcol = PerformanceQueryBuilder._quote_col(column)
        if len(values) == 1:
            return f"{qcol} = {PerformanceQueryBuilder._escape(values[0])}"
        escaped = ", ".join(PerformanceQueryBuilder._escape(v) for v in values)
        return f"{qcol} IN ({escaped})"

    # ── Main query builder ────────────────────────────────────────────────────

    def build_query(
        self,
        table: str,
        present_cols: list,
        filters: dict,
        groupby_cols: Optional[list] = None,
        present_mth_cols: Optional[list] = None,
        ytd_last_month: "Optional[str]" = None,
        limit: int = DEFAULT_LIMIT,
    ) -> str:
        """
        Build a fully server-side aggregated SQL query.

        Parameters
        ----------
        table            : Fully-qualified Databricks table (catalog.schema.table).
        present_cols     : Columns that exist in the remote table (from DESCRIBE TABLE).
                           Only these are ever referenced in the SQL.
        filters          : {col_name: [values]} pushed into WHERE before aggregation.
                           Year and Month predicates are always emitted first for
                           Spark partition pruning.
        groupby_cols     : Dimensions to GROUP BY. If None, all known dimension cols
                           present in the table are used.
        present_mth_cols : Override the MTH_* cols to SUM (MTH mode only).
        ytd_last_month   : When in YTD mode, the single Month value to restrict to
                           so that SUM(YTD_*) is never double-counted across months.
                           Derive via get_ytd_last_month() before calling this method.
        limit            : Row cap on the *aggregated* result — not on raw scanned rows.

        Mode detection
        --------------
        MTH mode  — filters["Month"] is non-empty  → SUM(MTH_*) only.
        YTD mode  — filters["Month"] is empty      → SUM(YTD_*) only, Month forced
                    to ytd_last_month to prevent double-counting.

        Returns
        -------
        str — complete, ready-to-execute SQL string.
        """
        present_set = set(present_cols)

        # ── 1. Mode detection — mutually exclusive column sets ────────────────
        # YTD mode: year selected, no month → SUM(YTD_*) restricted to last month.
        # MTH mode: month(s) selected      → SUM(MTH_*) only.
        month_filter     = self._to_list(filters.get("Month"))
        ytd_mode         = not month_filter

        if ytd_mode:
            # Full-year YTD: select YTD columns ONLY — mixing MTH would be wrong
            present_ytd_cols = [c for c in ALL_YTD_COLS if c in present_set]
            present_mth_cols = []            # excluded in YTD mode
        else:
            # Monthly MTH: select MTH columns ONLY — never mix with YTD
            if present_mth_cols is None:
                present_mth_cols = [c for c in ALL_MTH_COLS if c in present_set]
            else:
                present_mth_cols = [c for c in present_mth_cols if c in present_set]
            present_ytd_cols = []            # excluded in MTH mode

        # ── 2. Resolve groupby dimensions (only use cols that exist) ─────────
        if groupby_cols is None:
            groupby_cols = [c for c in self.ALL_DIM_COLS if c in present_set]
        else:
            groupby_cols = [c for c in groupby_cols if c in present_set]

        # ── 3. Build WHERE clause (partition cols go first for pruning) ───────
        where_parts: List[str] = []

        # Year — always first so Spark can skip entire file partitions
        year_vals = self._to_list(filters.get("Year"))
        if year_vals:
            where_parts.append(self._in_clause("Year", year_vals))

        # Month — YTD mode is restricted to a single last month to prevent
        # double-counting.  MTH mode uses the user-supplied month filter.
        if ytd_mode:
            if ytd_last_month:
                where_parts.append(self._in_clause("Month", [ytd_last_month]))
            # else: no month restriction — ytd_last_month not determinable
        else:
            if month_filter:
                where_parts.append(self._in_clause("Month", month_filter))

        # Remaining dimension filters in declaration order
        for col in self.HIERARCHY_COLS + self.PACKAGE_COLS + self.EXTRA_FILTER_COLS:
            vals = self._to_list(filters.get(col))
            if vals:
                where_parts.append(self._in_clause(col, vals))

        where_clause = (
            "\n    WHERE " + "\n      AND ".join(where_parts)
        ) if where_parts else ""

        # ── 4. Build SELECT clause ────────────────────────────────────────────
        sum_parts = [f"SUM({c}) AS {c}" for c in present_mth_cols]
        sum_parts += [f"SUM({c}) AS {c}" for c in present_ytd_cols]

        if groupby_cols:
            select_clause = (
                "SELECT " + ", ".join(groupby_cols)
                + (",\n        " + ",\n        ".join(sum_parts) if sum_parts else "")
            )
            group_by_clause = "\n    GROUP BY " + ", ".join(groupby_cols)
        else:
            # no groupby — just aggregate everything into one row
            select_clause = "SELECT " + (",\n        ".join(sum_parts) if sum_parts else "*")
            group_by_clause = ""

        # ── 5. ORDER BY primary perf column DESC (best performers first) ──────
        if ytd_mode and "YTD_Perf" in present_ytd_cols:
            order_clause = "\n    ORDER BY YTD_Perf DESC"
        elif "MTH_Perf" in present_mth_cols:
            order_clause = "\n    ORDER BY MTH_Perf DESC"
        else:
            order_clause = ""

        # ── 6. Assemble final query ───────────────────────────────────────────
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
        ytd_last_month: "Optional[str]" = None,
        limit: int = DEFAULT_LIMIT,
    ) -> str:
        """
        Build a non-aggregated SELECT for cases where raw rows are needed.
        Selects only columns relevant to the detected mode (MTH or YTD).
        Hard-blocked if limit > MAX_RAW_ROWS to prevent memory crashes.
        In YTD mode the Month is restricted to ytd_last_month to avoid
        returning cumulative rows from multiple months.
        """
        if limit > MAX_RAW_ROWS:
            raise ValueError(
                f"Raw fetch limit {limit:,} exceeds safety cap {MAX_RAW_ROWS:,}. "
                "Enable server-side aggregation or reduce the row limit."
            )
        present_set  = set(present_cols)
        month_filter = self._to_list(filters.get("Month"))
        ytd_mode     = not month_filter

        # Select dimension cols + only the relevant metric cols
        needed: List[str] = []
        for c in self.ALL_DIM_COLS:
            if c in present_set:
                needed.append(c)
        metric_src = ALL_YTD_COLS if ytd_mode else ALL_MTH_COLS
        for c in metric_src:
            if c in present_set and c not in needed:
                needed.append(c)

        # WHERE — partition cols first for pruning;
        # Month forced to last fiscal month in YTD mode.
        where_parts: List[str] = []
        year_vals = self._to_list(filters.get("Year"))
        if year_vals:
            where_parts.append(self._in_clause("Year", year_vals))
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

        where_clause = (
            "\n    WHERE " + "\n      AND ".join(where_parts)
        ) if where_parts else ""

        select_str = ", ".join(needed) if needed else "*"
        return f"SELECT {select_str}\n    FROM {table}{where_clause}\n    LIMIT {int(limit)}"

    # ── Optional server-side ranking ──────────────────────────────────────────

    def build_ranked_query(
        self,
        table: str,
        present_cols: list,
        filters: dict,
        rank_cols: list,
        perf_col: str = "MTH_Perf",
        top_n: int = 5,
        ytd_last_month: "Optional[str]" = None,
        limit: int = DEFAULT_LIMIT,
    ) -> str:
        """
        Server-side Top-N / Bottom-N ranking via ROW_NUMBER().

        Returns a query that aggregates to the `rank_cols` grain,
        computes a descending rank on `perf_col`, and returns only the
        top and bottom `top_n` rows — all inside Databricks.

        In YTD mode (no Month filter) the query is restricted to ytd_last_month
        and only SUM(YTD_*) columns are included. In MTH mode only SUM(MTH_*)
        columns are included. The two sets are never mixed.
        """
        present_set = set(present_cols)
        rank_cols = [c for c in rank_cols if c in present_set]
        if not rank_cols:
            raise ValueError("No ranking columns found in table.")

        month_filter = self._to_list(filters.get("Month"))
        ytd_mode     = not month_filter

        # Mutually exclusive column sets — never mix MTH and YTD
        if ytd_mode:
            present_mth = []
            present_ytd = [c for c in ALL_YTD_COLS if c in present_set]
        else:
            present_mth = [c for c in ALL_MTH_COLS if c in present_set]
            present_ytd = []

        # Always SUM — data is restricted to one month per mode
        sum_parts  = [f"SUM({c}) AS {c}" for c in present_mth]
        sum_parts += [f"SUM({c}) AS {c}" for c in present_ytd]

        # WHERE — Year first for pruning; Month forced to last fiscal month in YTD mode
        where_parts: list = []
        year_vals = self._to_list(filters.get("Year"))
        if year_vals:
            where_parts.append(self._in_clause("Year", year_vals))
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
        where_clause = (
            "\n    WHERE " + "\n      AND ".join(where_parts)
        ) if where_parts else ""

        perf_alias = perf_col if perf_col in (present_mth + present_ytd) else "MTH_Perf"
        group_str  = ", ".join(rank_cols)
        select_inner = (
            "SELECT " + group_str
            + (",\n        " + ",\n        ".join(sum_parts) if sum_parts else "")
            + f",\n        ROW_NUMBER() OVER (ORDER BY SUM({perf_alias}) DESC) AS _rn_desc"
            + f",\n        ROW_NUMBER() OVER (ORDER BY SUM({perf_alias}) ASC)  AS _rn_asc"
        )
        inner_query = (
            f"{select_inner}\n"
            f"    FROM {table}{where_clause}\n"
            f"    GROUP BY {group_str}\n"
        )
        outer = (
            f"SELECT * FROM (\n{inner_query}) AS _ranked\n"
            f"WHERE _rn_desc <= {int(top_n)} OR _rn_asc <= {int(top_n)}\n"
            f"ORDER BY _rn_desc\n"
            f"LIMIT {int(limit)}"
        )
        return outer


def _resolve_agg_cols(df: pd.DataFrame):
    """Return (mth_cols, ytd_cols) that exist in df, de-duplicated."""
    all_candidates = METRIC_COLS + ALL_MTH_COLS + ALL_YTD_COLS
    seen = set()
    mth_cols, ytd_cols = [], []
    for c in all_candidates:
        if c in df.columns and c not in seen:
            seen.add(c)
            if c.startswith("YTD_"):
                ytd_cols.append(c)
            else:
                mth_cols.append(c)
    return mth_cols, ytd_cols


def agg_by(df: pd.DataFrame, group_col: str, perf_col: str = "MTH_Perf") -> pd.DataFrame:
    """SUM all metric cols grouped by group_col.

    The caller is responsible for ensuring df contains only the correct month(s)
    for the active mode:
      - MTH mode: df already filtered to the selected month(s) → SUM(MTH_*) is correct.
      - YTD mode: df already filtered to the single last fiscal month →
        SUM(YTD_*) is correct (no double-counting across months).
    """
    mth_cols, ytd_cols = _resolve_agg_cols(df)
    all_cols = mth_cols + ytd_cols
    sort_col = perf_col if perf_col in all_cols else (all_cols[0] if all_cols else None)
    result = df.groupby(group_col)[all_cols].sum().reset_index()
    if sort_col:
        result = result.sort_values(sort_col, ascending=False)
    return result


def agg_by_multi(
    df: pd.DataFrame,
    group_cols: list,
    perf_col: str = "MTH_Perf",
) -> pd.DataFrame:
    """Like agg_by but groups by multiple columns. Returns sorted by perf_col DESC.

    The caller must ensure df is restricted to the correct month(s) for the mode;
    see agg_by docstring for the contract.
    """
    group_cols = [c for c in group_cols if c in df.columns]
    if not group_cols:
        return pd.DataFrame()
    mth_cols, ytd_cols = _resolve_agg_cols(df)
    all_cols = mth_cols + ytd_cols
    sort_col = perf_col if perf_col in all_cols else (all_cols[0] if all_cols else None)
    result = df.groupby(group_cols)[all_cols].sum().reset_index()
    if sort_col:
        result = result.sort_values(sort_col, ascending=False)
    return result


@st.cache_data
def compute_plant_clusters(df_json: str, n_clusters: int) -> pd.DataFrame:
    """Cluster Entity_1 plants by MTH_Volume (+MTH_Perf) using pure-numpy KMeans.
    Returns DataFrame with columns: Entity_1, MTH_Volume, MTH_Perf, Cluster.
    """
    df = pd.read_json(df_json, orient="split")
    cluster_cols = [c for c in ["MTH_Volume", "MTH_Perf"] if c in df.columns]
    if not cluster_cols or "Entity_1" not in df.columns:
        return pd.DataFrame()

    plant_agg = df.groupby("Entity_1")[cluster_cols].sum().reset_index()
    X = plant_agg[cluster_cols].values.astype(float)

    # Standardise
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    sigma[sigma == 0] = 1
    Xs = (X - mu) / sigma

    k = min(n_clusters, len(plant_agg))
    rng = np.random.default_rng(42)
    centers = Xs[rng.choice(len(Xs), k, replace=False)]

    for _ in range(100):
        dists   = np.linalg.norm(Xs[:, None] - centers[None], axis=2)  # (N, k)
        labels  = dists.argmin(axis=1)
        new_c   = np.array([
            Xs[labels == i].mean(axis=0) if (labels == i).any() else centers[i]
            for i in range(k)
        ])
        if np.allclose(new_c, centers):
            break
        centers = new_c

    plant_agg["Cluster"] = "Cluster " + pd.Series(labels).astype(str).values
    return plant_agg


def compute_plant_clusters_df(
    plant_df: pd.DataFrame,
    n_clusters: int,
    vol_col: str = "MTH_Volume",
    perf_col: str = "MTH_Perf",
) -> pd.DataFrame:
    """Cluster pre-aggregated plant-level dataframe by volume (+ perf).
    Clusters are named by volume tier: 'Large Plants', 'Mid-Size Plants', etc.
    Returns DataFrame with Entity_1, vol_col?, perf_col?, Cluster columns.
    """
    if "Entity_1" not in plant_df.columns:
        return pd.DataFrame()
    cluster_cols = [c for c in [vol_col, perf_col] if c in plant_df.columns]
    if not cluster_cols:
        return pd.DataFrame()

    plant_agg = plant_df.groupby("Entity_1")[cluster_cols].sum().reset_index()
    X = plant_agg[cluster_cols].values.astype(float)

    mu, sigma = X.mean(axis=0), X.std(axis=0)
    sigma[sigma == 0] = 1
    Xs = (X - mu) / sigma

    k = min(n_clusters, len(plant_agg))
    rng = np.random.default_rng(42)
    centers = Xs[rng.choice(len(Xs), k, replace=False)]

    for _ in range(100):
        dists  = np.linalg.norm(Xs[:, None] - centers[None], axis=2)
        labels = dists.argmin(axis=1)
        new_c  = np.array([
            Xs[labels == i].mean(axis=0) if (labels == i).any() else centers[i]
            for i in range(k)
        ])
        if np.allclose(new_c, centers):
            break
        centers = new_c

    # Name clusters by average volume (0 = primary feature → largest volume = "Large Plants")
    vol_idx = 0  # vol_col is first in cluster_cols if present, else perf_col
    cluster_mean_vol = {
        i: float(X[labels == i, vol_idx].mean()) if (labels == i).any() else 0.0
        for i in range(k)
    }
    rank = sorted(cluster_mean_vol, key=lambda i: cluster_mean_vol[i], reverse=True)
    tier_names = ["Large Plants", "Mid-Size Plants", "Small Plants", "Very Small Plants"]
    cluster_name_map = {
        rank[i]: tier_names[i] if i < len(tier_names) else f"Tier {i+1} Plants"
        for i in range(k)
    }
    plant_agg["Cluster"] = [cluster_name_map[lbl] for lbl in labels]
    return plant_agg


def make_stacked_chart(df_raw: pd.DataFrame, x_col: str, title: str,
                       chart_type: str, top_n: int, perf_col: str = "MTH_Perf",
                       seg_col: str = "Account_3"):
    """Stacked bar segmented by seg_col (default Account_3)."""
    top_x = (
        df_raw.groupby(x_col)[perf_col].sum()
        .nlargest(top_n).index.tolist()
    )
    df_plot = df_raw[df_raw[x_col].isin(top_x)].copy()
    if seg_col not in df_plot.columns:
        seg_col = "Account_3" if "Account_3" in df_plot.columns else df_plot.columns[0]

    grp = (
        df_plot.groupby([x_col, seg_col])[perf_col]
        .sum()
        .reset_index()
    )
    order = (
        grp.groupby(x_col)[perf_col].sum()
        .sort_values(ascending=False).index.tolist()
    )

    if chart_type == "Horizontal Bar":
        fig = px.bar(
            grp, x=perf_col, y=x_col,
            color=seg_col,
            orientation="h",
            title=title,
            barmode="stack",
            category_orders={x_col: list(reversed(order))},
        )
        fig.update_layout(yaxis_title=x_col)
    else:
        fig = px.bar(
            grp, x=x_col, y=perf_col,
            color=seg_col,
            title=title,
            barmode="stack",
            category_orders={x_col: order},
        )
        fig.update_layout(xaxis_tickangle=-35)

    fig.update_layout(
        legend_title_text=seg_col,
        margin=dict(t=50, b=20, l=10, r=10),
    )
    # Format numeric axis ticks
    fig.update_xaxes(tickformat=".3s")
    fig.update_yaxes(tickformat=".3s")
    return fig


def make_chart(df: pd.DataFrame, x_col: str, title: str,
               chart_type: str, color_scale: str, top_n: int,
               color_col: str | None = None, perf_col: str = "MTH_Perf"):
    """Plain bar/treemap chart.
    color_col: if set, use categorical coloring by that column instead of perf_col heatmap.
    """
    df = df.head(top_n)
    use_cat = color_col is not None and color_col in df.columns
    color_kwarg  = dict(color=color_col)                            if use_cat else dict(color=perf_col, color_continuous_scale=color_scale)
    layout_extra = dict(coloraxis_showscale=False)                  if not use_cat else {}

    if chart_type == "Bar":
        fig = px.bar(
            df, x=x_col, y=perf_col,
            title=title,
            text_auto=".3s",
            **color_kwarg,
        )
        fig.update_layout(xaxis_tickangle=-35, **layout_extra)
    elif chart_type == "Horizontal Bar":
        fig = px.bar(
            df, x=perf_col, y=x_col,
            orientation="h",
            title=title,
            text_auto=".3s",
            **color_kwarg,
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            **layout_extra,
        )
    else:  # Treemap
        fig = px.treemap(
            df, path=[x_col], values=perf_col,
            title=title,
            **color_kwarg,
        )
    fig.update_layout(margin=dict(t=50, b=20, l=10, r=10))
    fig.update_xaxes(tickformat=".3s")
    fig.update_yaxes(tickformat=".3s")
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    data_source = st.radio("Data source", options=["Upload file", "Databricks"], index=0)

    df = None
    uploaded_file = None
    sheet_name = "vilc"

    if data_source == "Upload file":
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
        sheet_name    = st.text_input("Excel sheet name (ignored for CSV)", value="vilc")

    else:
        st.markdown("### Databricks connection")
        st.caption("Leave sensitive fields empty in the UI; use env vars or secrets in production.")
        db_host      = st.text_input("Databricks host (server hostname)", value=os.environ.get("DATABRICKS_HOST", DB_HOST))
        db_http_path = st.text_input("HTTP Path", value=os.environ.get("DATABRICKS_HTTP_PATH", DB_HTTP_PATH))
        db_token     = st.text_input("Access token", value=os.environ.get("DATABRICKS_TOKEN", DB_TOKEN), type="password")
        db_catalog   = st.text_input("Catalog (optional)", value=os.environ.get("DATABRICKS_CATALOG", DB_CATALOG))
        db_schema    = st.text_input("Schema/Database (optional)", value=os.environ.get("DATABRICKS_SCHEMA", DB_SCHEMA))
        db_table     = st.text_input("Table name (optional)", value=DB_TABLE)
        sql_query   = st.text_area("SQL query (optional)", value="")
        server_agg  = st.checkbox("Use server-side aggregation (recommended for huge data)", value=True)
        row_limit   = st.number_input("Max rows to fetch (when not aggregating)", min_value=1000, max_value=10000000, value=200000)

        st.markdown("---")
        st.markdown("**Filters** *(all optional — leave empty to fetch all data)*")
        st.caption(
            "Tip: filtering on **Year** and/or **Month** is strongly recommended for large tables "
            "because Databricks uses those columns for partition pruning (much faster + cheaper)."
        )

        auto_populate = st.checkbox("Auto-populate filter options from Databricks (runs small queries)", value=False)

        # placeholder options if auto-populate is enabled
        db_filter_years_opts = db_filter_months_opts = db_filter_periods_opts = []
        db_filter_zone_opts = db_filter_country_opts = db_filter_entity1_opts = []
        db_filter_acc3_opts = []
        db_filter_bevtype_opts = []
        db_filter_plcode_opts  = []
        if auto_populate and db_table.strip():
            qualified = (
                f"{db_catalog}." if db_catalog else ""
            ) + (f"{db_schema}." if db_schema else "") + db_table
            try:
                table_cols = get_table_columns(db_host, db_http_path, db_token, qualified)
                if "Year" in table_cols:
                    db_filter_years_opts = get_distinct_values(db_host, db_http_path, db_token, qualified, "Year")
                if "Month" in table_cols:
                    db_filter_months_opts = get_distinct_values(db_host, db_http_path, db_token, qualified, "Month")
                if "period_1" in table_cols:
                    db_filter_periods_opts = get_distinct_values(db_host, db_http_path, db_token, qualified, "period_1")
                if "Zone" in table_cols:
                    db_filter_zone_opts = get_distinct_values(db_host, db_http_path, db_token, qualified, "Zone")
                if "Country" in table_cols:
                    db_filter_country_opts = get_distinct_values(db_host, db_http_path, db_token, qualified, "Country")
                if "Entity_1" in table_cols:
                    db_filter_entity1_opts = get_distinct_values(db_host, db_http_path, db_token, qualified, "Entity_1")
                if "Account_3" in table_cols:
                    db_filter_acc3_opts = get_distinct_values(db_host, db_http_path, db_token, qualified, "Account_3")
                if "BeverageType" in table_cols:
                    db_filter_bevtype_opts = get_distinct_values(db_host, db_http_path, db_token, qualified, "BeverageType")
                if "P_&_L_code" in table_cols:
                    db_filter_plcode_opts = get_distinct_values(db_host, db_http_path, db_token, qualified, "P_&_L_code")
            except Exception as e:
                st.warning(f"Could not auto-populate filter options: {e}")

        # Provide either multiselect options (auto-populated) or free-text CSV inputs
        if auto_populate:
            fb_years    = st.multiselect("Year values", options=db_filter_years_opts)
            fb_months   = st.multiselect("Month values", options=db_filter_months_opts)
            fb_periods  = st.multiselect("Period (period_1) values", options=db_filter_periods_opts)
            fb_zone     = st.multiselect("Zone values", options=db_filter_zone_opts)
            fb_country  = st.multiselect("Country values", options=db_filter_country_opts)
            fb_entity1  = st.multiselect("Entity_1 values", options=db_filter_entity1_opts)
            fb_acc3     = st.multiselect("Account_3 values", options=db_filter_acc3_opts)
            fb_bevtype  = st.multiselect("Beverage category (BeverageType)", options=db_filter_bevtype_opts)
            fb_plcode   = st.multiselect("P&L Code (P_&_L_code)", options=db_filter_plcode_opts)
        else:
            db_filter_years    = st.text_input("Year values (comma-separated, e.g. FY25)", value="")
            db_filter_months   = st.text_input("Month values (comma-separated)", value="")
            db_filter_periods  = st.text_input("Period (period_1) values (comma-separated)", value="")
            db_filter_zone     = st.text_input("Zone values (comma-separated)", value="")
            db_filter_country  = st.text_input("Country values (comma-separated)", value="")
            db_filter_entity1  = st.text_input("Entity_1 values (comma-separated)", value="")
            db_filter_acc3     = st.text_input("Account_3 values (comma-separated)", value="")
            db_filter_bevtype  = st.text_input("Beverage category — e.g. Beer", value="")
            db_filter_plcode   = st.text_input("P&L Code — e.g. VIC", value="")

        fetch_click = st.button("Fetch from Databricks", type="secondary")

        if fetch_click:
            try:
                # derive final filter lists from either multiselects or CSV inputs
                if auto_populate:
                    fb_years   = fb_years   if 'fb_years'   in locals() else []
                    fb_months  = fb_months  if 'fb_months'  in locals() else []
                    fb_periods = fb_periods if 'fb_periods' in locals() else []
                    fb_zone    = fb_zone    if 'fb_zone'    in locals() else []
                    fb_country = fb_country if 'fb_country' in locals() else []
                    fb_entity1 = fb_entity1 if 'fb_entity1' in locals() else []
                    fb_acc3    = fb_acc3    if 'fb_acc3'    in locals() else []
                    fb_bevtype = fb_bevtype if 'fb_bevtype' in locals() else []
                    fb_plcode  = fb_plcode  if 'fb_plcode'  in locals() else []
                else:
                    fb_years   = split_csv(db_filter_years)
                    fb_months  = split_csv(db_filter_months)
                    fb_periods = split_csv(db_filter_periods)
                    fb_zone    = split_csv(db_filter_zone)
                    fb_country = split_csv(db_filter_country)
                    fb_entity1 = split_csv(db_filter_entity1)
                    fb_acc3    = split_csv(db_filter_acc3)
                    fb_bevtype = split_csv(db_filter_bevtype)
                    fb_plcode  = split_csv(db_filter_plcode)

                # Warn when no partition filters are set — this will do a full table scan
                if not fb_years and not fb_months and not sql_query.strip():
                    st.warning(
                        "⚠️ No Year or Month filter set — this will scan the **entire table**. "
                        "For large tables, add at least a Year filter to avoid slow queries and high costs."
                    )

                _qb = PerformanceQueryBuilder()

                if sql_query.strip():
                    # User-supplied SQL — trust it as-is (supports custom JOINs etc.)
                    q = sql_query
                elif db_table.strip():
                    qualified = (
                        f"{db_catalog}." if db_catalog else ""
                    ) + (f"{db_schema}." if db_schema else "") + db_table

                    # Inspect table schema — only reference columns that actually exist
                    try:
                        table_cols = get_table_columns(db_host, db_http_path, db_token, qualified)
                    except Exception as e:
                        st.error(f"Unable to read table schema: {e}")
                        q = ""
                        table_cols = []

                    filters = {
                        "Year":         fb_years,
                        "Month":        fb_months,
                        "period_1":     fb_periods,
                        "Zone":         fb_zone,
                        "Country":      fb_country,
                        "Entity_1":     fb_entity1,
                        "Account_3":    fb_acc3,
                        "BeverageType": fb_bevtype,
                        "P_&_L_code":   fb_plcode,
                    }

                    # Determine YTD last month for the SQL WHERE clause.
                    # When the user selects a Year but no Month the query is in YTD
                    # mode and must restrict to the single last fiscal month so that
                    # SUM(YTD_*) is mathematically correct (no double-counting).
                    _db_ytd_lm: "Optional[str]" = None
                    if fb_years and not fb_months:
                        # Use auto-populated month options when available;
                        # falls back to 'Dec YY' heuristic via get_ytd_last_month.
                        _db_ytd_lm = get_ytd_last_month(fb_years, db_filter_months_opts)
                        log(f"YTD mode (Databricks): restricting Month to '{_db_ytd_lm}'")

                    if server_agg:
                        # Server-side aggregated query — partition pruning + SUM on cluster.
                        q = _qb.build_query(
                            table=qualified,
                            present_cols=table_cols,
                            filters=filters,
                            ytd_last_month=_db_ytd_lm,
                            limit=int(row_limit),
                        )
                        st.info(f"📋 Generated query preview:\n```sql\n{q}\n```")
                    else:
                        # Non-aggregated fetch — minimal columns, hard limit enforced
                        try:
                            q = _qb.build_raw_query(
                                table=qualified,
                                present_cols=table_cols,
                                filters=filters,
                                ytd_last_month=_db_ytd_lm,
                                limit=int(row_limit),
                            )
                        except ValueError as ve:
                            st.error(str(ve))
                            q = ""
                else:
                    st.error("Provide either a SQL query or a table name to fetch from Databricks.")
                    q = ""

                if q:
                    log(f"Executing query:\n{q}")
                    df = query_databricks(db_host, db_http_path, db_token, q)
                    st.session_state["db_df"] = df
                    log(f"Fetched {len(df):,} rows — columns: {list(df.columns)}")
                    st.success(f"✅ Fetched {len(df):,} rows from Databricks")
            except Exception as e:
                log(f"Databricks fetch error: {e}", "error")
                st.error(f"Error querying Databricks: {e}")

    # Restore previously fetched Databricks data if not re-fetching this run
    if data_source == "Databricks" and df is None and not fetch_click:
        df = st.session_state.get("db_df")
        if df is not None:
            log(f"Restored cached Databricks data ({len(df):,} rows)")

    st.markdown("---")

    # Logs expander
    with st.expander("🪵 Logs", expanded=False):
        logs = st.session_state.get("logs", [])
        if logs:
            st.code("\n".join(logs[-200:]), language="")  # show last 200 entries
        else:
            st.caption("No log entries yet.")
        if st.button("Clear logs", key="clear_logs"):
            st.session_state["logs"] = []
            st.rerun()

    # If using uploaded file, load it now
    if uploaded_file:
        try:
            df = load_data(uploaded_file, sheet_name)
        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.stop()

    if df is not None:
        # Check structural columns
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            st.error(
                f"Missing required structural columns: {missing}\n\n"
                f"Required: {REQUIRED_COLS}\n"
                f"Found: {df.columns.tolist()}"
            )
            st.stop()
        # Check that at least one perf column exists (MTH_Perf for monthly mode,
        # YTD_Perf for full-year mode — the data only needs to have one of them)
        if not any(c in df.columns for c in _PERF_COL_CANDIDATES):
            st.error(
                f"Data must contain at least one performance column: "
                f"{_PERF_COL_CANDIDATES}\n"
                f"Found columns: {df.columns.tolist()}"
            )
            st.stop()

        st.success(f"✅ Loaded {len(df):,} rows")
        with st.expander("Detected columns"):
            st.code(", ".join(df.columns.tolist()))

        # ── Time filters ──────────────────────────────────────────────────────
        st.markdown("### 🕒 Time Filters")

        sel_years   = []
        sel_months  = []
        sel_periods = []

        if "Year" in df.columns:
            year_opts = sorted(df["Year"].dropna().unique().tolist(), reverse=True)
            sel_years = st.multiselect("Year (multi-select)", options=year_opts)

        if "Month" in df.columns:
            month_opts = sorted(df["Month"].dropna().unique().tolist())
            sel_months = st.multiselect("Month (multi-select)", options=month_opts)

        if "period_1" in df.columns:
            period_opts = sorted(df["period_1"].dropna().unique().tolist())
            sel_periods = st.multiselect("Period / period_1 (multi-select)", options=period_opts)

        # ── Hierarchy filter ──────────────────────────────────────────────────
        st.markdown("### 📍 Hierarchy Filter")

        level = st.selectbox("Analysis level", options=["Global"] + HIERARCHY)

        sel_values = []
        if level != "Global":
            unique_vals = sorted(df[level].dropna().unique().tolist())
            sel_values  = st.multiselect(f"Select {level} (multi-select)", options=unique_vals)

        # convenience single value for titles/labels
        value = ", ".join(str(v) for v in sel_values) if sel_values else None

        # ── Chart options ─────────────────────────────────────────────────────
        st.markdown("### 📊 Chart Options")
        top_n       = st.slider("Top N per chart", min_value=5, max_value=100, value=20, step=5)
        chart_type  = st.radio("Chart type", options=["Bar", "Horizontal Bar", "Treemap"])
        color_scale = st.selectbox(
            "Color scale",
            options=["Blues", "Viridis", "Plasma", "Teal", "Reds", "Sunset"],
        )

        # ── Cluster options ───────────────────────────────────────────────────
        st.markdown("### 🔵 Plant Clustering")
        has_volume = "MTH_Volume" in df.columns or "YTD_Volume" in df.columns

        if not has_volume:
            st.info("MTH_Volume column not found — clustering disabled.")
            cluster_mode = False
            n_clusters   = 4
        else:
            cluster_mode = st.toggle("Enable Volume-based Clustering", value=False)
            n_clusters   = st.slider(
                "Number of clusters", min_value=2, max_value=8, value=4,
                disabled=not cluster_mode,
            )

        run_clicked = st.button("▶ Run Analysis", type="primary", width="stretch")

        # Persist run state so that widget interactions (which trigger reruns)
        # don't reset the page back to the prompt.
        if run_clicked:
            log(f"Run Analysis clicked — data_source={data_source}, rows={len(df):,}, level={level}, sel_values={sel_values}")
            st.session_state["run"] = True

        run = st.session_state.get("run", False)

    else:
        st.info("Provide a data source (Upload or Databricks) to get started.")
        run          = False
        cluster_mode = False
        n_clusters   = 4
        sel_years    = []
        sel_months   = []
        sel_periods  = []
        sel_values   = []
        level        = "Global"
        value        = None


# ── Main panel ────────────────────────────────────────────────────────────────
if df is None:
    st.markdown(
        """
        ### Getting started
        1. **Upload file** — use the sidebar file uploader, or choose **Databricks** and click **Fetch from Databricks**.
        2. Apply **time / hierarchy filters**.
        3. Click **▶ Run Analysis**.

        **Required structural columns:** `Zone`, `Country`, `Entity_1`, `Account_3`

        **Monthly mode** (Month filter selected): uses `MTH_Perf`, `MTH_BU`, `MTH_ACT`, …

        **Full-year YTD mode** (Year selected, no Month): uses `YTD_Perf`, `YTD_BU`, `YTD_ACT`, … restricted to the last fiscal month to avoid double-counting.
        """
    )
    st.stop()

if not run:
    st.info("Configure your filters in the sidebar and click **▶ Run Analysis**.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PROCESSING ORDER:
#   1. Load  →  2. Time filter  →  3. Hierarchy filter  →  4. Aggregate
# ══════════════════════════════════════════════════════════════════════════════

# Step 1 — already loaded above via load_data()
log(f"Analysis starting — {len(df):,} rows, columns: {list(df.columns)}")

# ── Mode detection ────────────────────────────────────────────────────────────
# MTH mode  = month(s) selected        → SUM MTH_* cols.
# YTD mode  = year selected, no month  → SUM YTD_* cols for the LAST month only.
# The two column sets are NEVER mixed in the same aggregation.
use_ytd         = bool(sel_years) and not sel_months
PERF_COL        = "YTD_Perf" if use_ytd else "MTH_Perf"
ACTIVE_ALL_COLS = ALL_YTD_COLS if use_ytd else ALL_MTH_COLS
log(f"Mode: {'YTD' if use_ytd else 'MTD'} — primary perf column: {PERF_COL}")

# Step 2 — Time filter (BEFORE any aggregation)
df_filtered = df.copy()

if sel_years:   df_filtered = df_filtered[df_filtered["Year"].isin(sel_years)]
if sel_months:  df_filtered = df_filtered[df_filtered["Month"].isin(sel_months)]
if sel_periods: df_filtered = df_filtered[df_filtered["period_1"].isin(sel_periods)]

# ── YTD safety filter ────────────────────────────────────────────────────────
# YTD columns are already cumulative (Jan = Jan, Dec = full year).  If the
# dataframe spans multiple months (e.g. all 12 months of FY25), summing
# YTD_* across them would multiply the figure by 12.  Prevent this by
# keeping only the LAST fiscal month, which already holds the full-year total.
# For file-upload data: applied here.
# For Databricks: the SQL query is built with WHERE Month = ytd_last_month.
if use_ytd and "Month" in df_filtered.columns:
    _avail_months   = df_filtered["Month"].dropna().unique().tolist()
    _ytd_last_month = get_ytd_last_month(sel_years, _avail_months)
    if _ytd_last_month:
        df_filtered = df_filtered[df_filtered["Month"] == _ytd_last_month]
        log(f"YTD mode: restricted to last fiscal month '{_ytd_last_month}' — double-count prevented")
    else:
        log("YTD mode: could not determine last fiscal month; all months retained (risk of double-count)", "warning")

# Step 3 — Hierarchy filter
if level != "Global":
    if sel_values:
        df_filtered = df_filtered[df_filtered[level].isin(sel_values)]
    if df_filtered.empty:
        st.warning("⚠️ No data found for the current filter combination.")
        st.stop()


# ── Cluster computation (uses full-time-filtered data, hierarchy-agnostic) ───
vol_col = "YTD_Volume" if use_ytd else "MTH_Volume"
plant_cluster_map: dict = {}   # Entity_1 → natural cluster name
cluster_df       = pd.DataFrame()

avail_vol = vol_col in df_filtered.columns
if cluster_mode and avail_vol:
    df_for_cluster = df.copy()
    if sel_years:   df_for_cluster = df_for_cluster[df_for_cluster["Year"].isin(sel_years)]
    if sel_months:  df_for_cluster = df_for_cluster[df_for_cluster["Month"].isin(sel_months)]
    if sel_periods: df_for_cluster = df_for_cluster[df_for_cluster["period_1"].isin(sel_periods)]

    plant_agg_for_cluster = df_for_cluster.groupby("Entity_1")[
        [c for c in [vol_col, PERF_COL] if c in df_for_cluster.columns]
    ].sum().reset_index()

    cluster_df = compute_plant_clusters_df(plant_agg_for_cluster, n_clusters,
                                           vol_col=vol_col, perf_col=PERF_COL)
    if not cluster_df.empty:
        plant_cluster_map = dict(zip(cluster_df["Entity_1"], cluster_df["Cluster"]))

# Natural cluster colours: green→amber→orange→red by volume tier
CLUSTER_COLORS = {
    "Large Plants":      "#2ecc71",
    "Mid-Size Plants":   "#f39c12",
    "Small Plants":      "#e74c3c",
    "Very Small Plants": "#8e44ad",
}


# Sub-header shows exactly which filters are active
filter_label = level + (f" → {value}" if value else "")
time_parts   = (
    [f"Year: {', '.join(str(v) for v in sel_years)}"  ] if sel_years   else []
) + (
    [f"Month: {', '.join(str(v) for v in sel_months)}"] if sel_months  else []
) + (
    [f"Period: {', '.join(str(v) for v in sel_periods)}"] if sel_periods else []
)
time_label = " | ".join(time_parts) if time_parts else "All Time"

mode_badge = "🗓️ YTD mode" if use_ytd else "📅 MTD mode"
st.subheader(f"📌 {filter_label}  ·  🕒 {time_label}  ·  {mode_badge}")

# ── KPI metric cards — priority order then remainder ─────────────────────────
_pfx      = "YTD_" if use_ytd else "MTH_"
_bu_kpi   = _pfx + "BU"
_act_kpi  = _pfx + "ACT"
_pri_kpi  = _pfx + "Price"
_perf_kpi = _pfx + "Perf"
_priority = [_bu_kpi, _act_kpi, _pri_kpi, _perf_kpi]   # shown in row 1 (+ delta)

_present_prio = [c for c in _priority if c in df_filtered.columns]
_rest_cols    = [c for c in ACTIVE_ALL_COLS
                 if c in df_filtered.columns and c not in _priority]

# Row 1: BU | ACT | Δ(BU−ACT) | Price | Perf
_bu_val  = df_filtered[_bu_kpi].sum()  if _bu_kpi  in df_filtered.columns else None
_act_val = df_filtered[_act_kpi].sum() if _act_kpi in df_filtered.columns else None
_delta   = (_bu_val - _act_val) if (_bu_val is not None and _act_val is not None) else None

_row1_items = []
for c in _present_prio:
    _row1_items.append((c, df_filtered[c].sum()))
# Insert Δ after ACT
_delta_pos = next(
    (i + 1 for i, (c, _) in enumerate(_row1_items) if c == _act_kpi), len(_row1_items)
)
if _delta is not None:
    _row1_items.insert(_delta_pos, ("Δ BU − ACT", _delta))

if _row1_items:
    _r1_cols = st.columns(len(_row1_items))
    for _ci, (_lbl, _val) in enumerate(_row1_items):
        _r1_cols[_ci].metric(_lbl, fmt_num(_val))

# Row 2+: all remaining ACTIVE_ALL_COLS
if _rest_cols:
    _r2_cols = st.columns(min(6, len(_rest_cols)))
    for _ci, _cname in enumerate(_rest_cols):
        _r2_cols[_ci % 6].metric(_cname, fmt_num(df_filtered[_cname].sum()))

st.markdown("---")

# ── Breakdown levels (hierarchy logic) ───────────────────────────────────────
# Global     → Zone, Country, Entity_1, Account_3
# Zone       → Country, Entity_1, Account_3
# Country    → Entity_1, Account_3
# Entity_1   → Account_3 only
if level == "Global":
    breakdown_levels = HIERARCHY
elif level == "Entity_1":
    breakdown_levels = []
else:
    idx = HIERARCHY.index(level)
    breakdown_levels = HIERARCHY[idx + 1:]


# ── Best & Worst Performers ───────────────────────────────────────────────────
st.markdown("## 🏆 Best & Worst Performers")

# All dimensions available in the dataset (hierarchy + package)
_ALL_DIMS = ["Zone", "Country", "Entity_1",
             "Account_3", "Account_4", "Account_5", "Account_5_subpackage"]
_avail_dims = [d for d in _ALL_DIMS if d in df_filtered.columns]

_bu_t = "YTD_BU"  if use_ytd else "MTH_BU"
_ac_t = "YTD_ACT" if use_ytd else "MTH_ACT"

if not _avail_dims:
    st.info("No recognisable dimension columns found in the data.")
else:
    # ── Controls row ──────────────────────────────────────────────────────
    _ctrl1, _ctrl2 = st.columns([3, 1])
    with _ctrl1:
        _rank_dim = st.selectbox(
            "Rank by dimension",
            options=_avail_dims,
            index=min(2, len(_avail_dims) - 1),   # default: Entity_1
            key="rank_dim_select",
            help="Choose the level you want to rank — Zone, Country, Plant, or any package level. "
                 "Rankings are computed across the entire filtered dataset.",
        )
    with _ctrl2:
        _n_each = st.number_input(
            "Top / Bottom N", min_value=1, max_value=20, value=5, step=1,
            key="rank_n_each",
        )

    # Dimensions *below* the ranked one (used for drill-down tabs)
    _rank_idx    = _ALL_DIMS.index(_rank_dim) if _rank_dim in _ALL_DIMS else 0
    _drill_dims  = [d for d in _ALL_DIMS[_rank_idx + 1:] if d in df_filtered.columns]

    # For quadrant scatter backward-compat
    _top5_dim = _rank_dim

    # ── Aggregate at the selected dimension ──────────────────────────────
    _top5_agg = agg_by(df_filtered, _rank_dim, perf_col=PERF_COL)

    if _top5_agg.empty:
        st.info("No data available for the current selection.")
    else:
        _n_ent = len(_top5_agg)
        _n_top = min(int(_n_each), max(1, _n_ent // 2))
        _n_bot = min(int(_n_each), max(1, _n_ent - _n_top))

        _top_rows   = _top5_agg.head(_n_top)
        _bot_rows   = _top5_agg.tail(_n_bot).iloc[::-1]
        _top_names  = set(_top_rows[_rank_dim].tolist())
        _bot_names  = set(_bot_rows[_rank_dim].tolist())
        _featured   = _top_names | _bot_names

        # For quadrant backward-compat
        _grain_cols = [_rank_dim]

        st.caption(
            f"Top **{_n_top}** and bottom **{_n_bot}** **{_rank_dim}** values "
            f"ranked by **{PERF_COL}** across the entire filtered dataset."
        )

        # ── Summary tables side by side ───────────────────────────────────
        _metric_cols = [c for c in [PERF_COL, _bu_t, _ac_t] if c in _top5_agg.columns]
        _tbl_cols    = [_rank_dim] + _metric_cols

        _tbl_top = _top_rows[_tbl_cols].reset_index(drop=True).copy()
        _tbl_bot = _bot_rows[_tbl_cols].reset_index(drop=True).copy()
        _tbl_top.insert(0, "Rank", range(1, len(_tbl_top) + 1))
        _tbl_bot.insert(0, "Rank", range(1, len(_tbl_bot) + 1))

        _tc, _bc = st.columns(2, gap="large")
        with _tc:
            st.markdown("##### 🟢 Best performers")
            st.dataframe(
                style_positive_blue(arrow_safe(_tbl_top), _metric_cols),
                width="stretch",
                height=min(400, 42 + len(_tbl_top) * 38),
            )
        with _bc:
            st.markdown("##### 🔴 Needs attention")
            st.dataframe(
                style_positive_blue(arrow_safe(_tbl_bot), _metric_cols),
                width="stretch",
                height=min(400, 42 + len(_tbl_bot) * 38),
            )

        # ── Overview bar ──────────────────────────────────────────────────
        _ent_order = (
            _bot_rows[_rank_dim].tolist()
            + list(reversed(_top_rows[_rank_dim].tolist()))
        )
        _all_pkg_lvl = [c for c in ["Account_3", "Account_4", "Account_5", "Account_5_subpackage"]
                        if c in df_filtered.columns and c != _rank_dim]

        if _all_pkg_lvl:
            _ov_pkg_opts = ["(No breakdown)"] + _all_pkg_lvl
            _ov_pkg_sel  = st.radio(
                "Stack bars by",
                options=_ov_pkg_opts,
                horizontal=True,
                key="t5_ov_pkg",
            )
            _ov_pkg = None if _ov_pkg_sel == "(No breakdown)" else _ov_pkg_sel
        else:
            _ov_pkg = None

        _bar_h = max(320, (_n_top + _n_bot) * 54 + 90)

        if _ov_pkg and _ov_pkg in df_filtered.columns:
            _featured_src = df_filtered[df_filtered[_rank_dim].isin(_featured)]
            _ov_grp = (
                _featured_src.groupby([_rank_dim, _ov_pkg])[PERF_COL]
                .sum().reset_index()
            )
            _fig_compare = px.bar(
                _ov_grp, x=PERF_COL, y=_rank_dim,
                color=_ov_pkg, orientation="h", barmode="stack",
                category_orders={_rank_dim: _ent_order},
                labels={PERF_COL: "", _rank_dim: ""},
                height=_bar_h,
            )
            _fig_compare.update_traces(marker_line_width=0)
            _fig_compare.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="left", x=0, title_text=""),
            )
        else:
            _compare_df = pd.concat([_top_rows, _bot_rows]).drop_duplicates(subset=_rank_dim)
            _compare_df["Group"] = _compare_df[_rank_dim].apply(
                lambda v: "Top performer" if v in _top_names else "Needs attention"
            )
            _fig_compare = px.bar(
                _compare_df, x=PERF_COL, y=_rank_dim,
                color="Group",
                color_discrete_map={"Top performer": "#2ecc71", "Needs attention": "#e74c3c"},
                orientation="h", text=PERF_COL,
                category_orders={_rank_dim: _ent_order},
                labels={PERF_COL: "", _rank_dim: ""},
                height=_bar_h,
            )
            _fig_compare.update_traces(
                texttemplate="%{text:.3s}", textposition="inside", marker_line_width=0,
            )
            _fig_compare.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1, title_text=""),
            )

        _fig_compare.update_layout(
            title=None,
            xaxis=dict(tickformat=".3s", showgrid=True, gridcolor="#f0f0f0",
                       zeroline=False, title=""),
            yaxis=dict(title="", automargin=True),
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=55, b=20, l=10, r=20),
        )
        if _n_bot > 0 and _n_top > 0:
            _fig_compare.add_shape(
                type="line", x0=0, x1=1, xref="paper",
                y0=_n_bot - 0.5, y1=_n_bot - 0.5, yref="y",
                line=dict(color="#bdc3c7", width=1.5, dash="dot"),
            )
            _fig_compare.add_annotation(
                x=1, y=_n_bot - 0.5, xref="paper",
                text="top ▲  ·  ▼ bottom ",
                showarrow=False, font=dict(size=10, color="#95a5a6"),
                xanchor="right", yanchor="middle",
            )
        st.plotly_chart(_fig_compare, width="stretch", key="t5_compare_chart")

        # ── Deep-dive expanders ───────────────────────────────────────────
        def _deep_dive_expander(row_series, group: str, rank_i: int):
            """
            Render one expander for a ranked entity with layered drill-down tabs.

            Tabs cover every dimension *below* the ranked one, in order:
              Zone → Country → Entity_1 → Account_3 → Account_4 → Account_5 → Account_5_subpackage
            Each tab shows a bar chart + summary table for that sub-dimension.
            A final 'All metrics' tab shows the full KPI card set.
            """
            _ent_val  = row_series[_rank_dim]
            _pval     = row_series[PERF_COL]
            _icons    = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]
            _icon     = _icons[rank_i] if rank_i < len(_icons) else f"{rank_i+1}."
            _kb       = f"t5_{group}_{rank_i}"

            with st.expander(
                f"{_icon}  **{_ent_val}**   ·   {PERF_COL}: {fmt_num(_pval)}",
                expanded=(rank_i == 0),
            ):
                # Slice df_filtered to just this entity
                df_ent = df_filtered[df_filtered[_rank_dim] == _ent_val]

                # ── Top KPI row ───────────────────────────────────────────
                _bu_v  = df_ent[_bu_t].sum()  if _bu_t  in df_ent.columns else None
                _ac_v  = df_ent[_ac_t].sum()  if _ac_t  in df_ent.columns else None
                _pr_v  = df_ent[_pfx + "Price"].sum() if (_pfx + "Price") in df_ent.columns else None
                _delta_v = (_bu_v - _ac_v) if (_bu_v is not None and _ac_v is not None) else None

                _kpi_items = []
                if _bu_v  is not None: _kpi_items.append((_bu_t,              _bu_v))
                if _ac_v  is not None: _kpi_items.append((_ac_t,              _ac_v))
                if _delta_v is not None: _kpi_items.append(("Δ BU − ACT",     _delta_v))
                if _pr_v  is not None: _kpi_items.append((_pfx + "Price",     _pr_v))
                _kpi_items.append((PERF_COL, _pval))

                _kc = st.columns(len(_kpi_items))
                for _ci, (_kl, _kv) in enumerate(_kpi_items):
                    _kc[_ci].metric(_kl, fmt_num(_kv))

                # ── Drill-down tabs ───────────────────────────────────────
                # Build tab list: one per sub-dimension + one "All metrics" tab
                _sub_dims_present = [d for d in _drill_dims if d in df_ent.columns
                                     and df_ent[d].nunique() > 0]

                if not _sub_dims_present:
                    # Leaf node – just show remaining metrics
                    st.markdown("*No further breakdown dimensions available for this entry.*")
                    _rest_m = [c for c in ACTIVE_ALL_COLS
                               if c in df_ent.columns and c not in [_bu_t, _ac_t, PERF_COL,
                                                                     _pfx + "Price"]]
                    if _rest_m:
                        _rm_cols = st.columns(min(6, len(_rest_m)))
                        for _ci, _cn in enumerate(_rest_m):
                            _rm_cols[_ci % 6].metric(_cn, fmt_num(df_ent[_cn].sum()))
                    return

                _tab_labels = [f"📊 By {d.replace('_', ' ')}" for d in _sub_dims_present]
                _tab_labels += ["📋 All metrics"]
                _tab_objs = st.tabs(_tab_labels)

                # One tab per sub-dimension
                for _ti, _sdim in enumerate(_sub_dims_present):
                    with _tab_objs[_ti]:
                        _sd_agg = agg_by(df_ent, _sdim, perf_col=PERF_COL)
                        if _sd_agg.empty:
                            st.caption("No data.")
                            continue

                        _sd_metric_cols = [c for c in [PERF_COL, _bu_t, _ac_t]
                                           if c in _sd_agg.columns]

                        # Chart + table side by side
                        _sc, _st_ = st.columns([2, 1])
                        with _sc:
                            # Stacked by next sub-dim if available, else plain bar
                            _next_dims = [d for d in _sub_dims_present
                                          if _ALL_DIMS.index(d) > _ALL_DIMS.index(_sdim)
                                          and d in df_ent.columns]
                            if _next_dims:
                                st.plotly_chart(
                                    make_stacked_chart(
                                        df_ent, _sdim,
                                        f"{_ent_val} — {_sdim} coloured by {_next_dims[0]}",
                                        chart_type, top_n=top_n, perf_col=PERF_COL,
                                        seg_col=_next_dims[0],
                                    ),
                                    width="stretch",
                                    key=f"{_kb}_{_sdim}_stk",
                                )
                            else:
                                st.plotly_chart(
                                    make_chart(
                                        _sd_agg, _sdim,
                                        f"{_ent_val} — by {_sdim}",
                                        chart_type, color_scale, top_n=top_n,
                                        perf_col=PERF_COL,
                                    ),
                                    width="stretch",
                                    key=f"{_kb}_{_sdim}_bar",
                                )
                        with _st_:
                            _disp = _sd_agg[[_sdim] + _sd_metric_cols].head(top_n)
                            st.markdown(f"**Top {top_n}  ·  {_sdim}**")
                            st.dataframe(
                                style_positive_blue(arrow_safe(_disp), _sd_metric_cols),
                                width="stretch",
                                height=min(420, 42 + len(_disp) * 36),
                            )
                            if not _sd_agg.empty:
                                _best = _sd_agg.iloc[0]
                                st.success(
                                    f"🏆 Best: **{_best[_sdim]}**  "
                                    f"({PERF_COL} = {fmt_num(_best[PERF_COL])})"
                                )
                                _worst = _sd_agg.iloc[-1]
                                st.error(
                                    f"⚠️ Worst: **{_worst[_sdim]}**  "
                                    f"({PERF_COL} = {fmt_num(_worst[PERF_COL])})"
                                )

                # "All metrics" tab
                with _tab_objs[-1]:
                    _all_m = [c for c in ACTIVE_ALL_COLS if c in df_ent.columns]
                    if _all_m:
                        _am_cols = st.columns(min(6, len(_all_m)))
                        for _ci, _cn in enumerate(_all_m):
                            _am_cols[_ci % 6].metric(_cn, fmt_num(df_ent[_cn].sum()))
                    # Full aggregated table at the rank dimension
                    _full_tbl = df_ent[[c for c in _sub_dims_present + _all_m
                                        if c in df_ent.columns]].copy()
                    _full_agg = agg_by_multi(df_ent, _sub_dims_present, perf_col=PERF_COL)
                    if not _full_agg.empty:
                        _fa_m = [c for c in _all_m if c in _full_agg.columns]
                        st.dataframe(
                            style_positive_blue(
                                arrow_safe(_full_agg[_sub_dims_present + _fa_m].head(50)),
                                _fa_m,
                            ),
                            width="stretch",
                            height=min(500, 42 + len(_full_agg) * 34),
                        )

        st.markdown("---")
        col_top5, col_bot5 = st.columns(2, gap="large")
        with col_top5:
            st.markdown("#### 🟢 Best performers")
            for _ri, (_idx, _row) in enumerate(_top_rows.iterrows()):
                _deep_dive_expander(_row, "top", _ri)
        with col_bot5:
            st.markdown("#### 🔴 Needs attention")
            for _ri, (_idx, _row) in enumerate(_bot_rows.iterrows()):
                _deep_dive_expander(_row, "bot", _ri)

        # ── 4-Quadrant: strictly top N + bottom N plants ──────────────────
        st.markdown("---")
        st.markdown("## 📍 Plant Positioning")
        st.caption(
            f"Top {_n_top} and bottom {_n_bot} **plants** (Entity\_1) by {PERF_COL}. "
            "Dashed lines show medians — use the quadrants to prioritise action."
        )

        if "Entity_1" in df_filtered.columns and vol_col in df_filtered.columns:
            _plant_perf = df_filtered.groupby("Entity_1")[[vol_col, PERF_COL]].sum().reset_index()

            if not _plant_perf.empty:
                _n_quad_each = min(int(_n_each), max(1, len(_plant_perf) // 2))
                _top_plants  = set(_plant_perf.nlargest(_n_quad_each, PERF_COL)["Entity_1"])
                _bot_plants  = set(_plant_perf.nsmallest(_n_quad_each, PERF_COL)["Entity_1"])
                _quad_df     = _plant_perf[
                    _plant_perf["Entity_1"].isin(_top_plants | _bot_plants)
                ].copy()
                _quad_df["Group"] = _quad_df["Entity_1"].apply(
                    lambda v: "Top plant" if v in _top_plants else "Needs attention"
                )
                _q_color_map = {"Top plant": "#2ecc71", "Needs attention": "#e74c3c"}

                if _rank_dim != "Entity_1" and _rank_dim in df_filtered.columns:
                    _pp_map = (
                        df_filtered[["Entity_1", _rank_dim]].drop_duplicates()
                        .set_index("Entity_1")[_rank_dim].to_dict()
                    )
                    _quad_df["Parent"] = _quad_df["Entity_1"].map(_pp_map)
                    _hover_cols = ["Group", "Parent"]
                else:
                    _hover_cols = ["Group"]

                _med_vol  = _quad_df[vol_col].median()
                _med_perf = _quad_df[PERF_COL].median()
                _xpad = (_quad_df[vol_col].max()  - _quad_df[vol_col].min())  * 0.12 or 1
                _ypad = (_quad_df[PERF_COL].max() - _quad_df[PERF_COL].min()) * 0.12 or 1
                _xmin = _quad_df[vol_col].min()  - _xpad
                _xmax = _quad_df[vol_col].max()  + _xpad
                _ymin = _quad_df[PERF_COL].min() - _ypad
                _ymax = _quad_df[PERF_COL].max() + _ypad

                _fig_quad = px.scatter(
                    _quad_df, x=vol_col, y=PERF_COL,
                    color="Group", color_discrete_map=_q_color_map,
                    text="Entity_1", hover_data=_hover_cols,
                    labels={vol_col: "Volume", PERF_COL: "Performance"},
                    height=520,
                )
                _fig_quad.update_traces(
                    textposition="top center",
                    marker=dict(size=16, line=dict(width=1.5, color="white")),
                )
                _fig_quad.add_vline(
                    x=_med_vol, line_dash="dash", line_color="#bdc3c7", line_width=1.5,
                    annotation_text=f"  Med. Vol: {fmt_num(_med_vol)}",
                    annotation_font=dict(size=10, color="#7f8c8d"),
                    annotation_position="top left",
                )
                _fig_quad.add_hline(
                    y=_med_perf, line_dash="dash", line_color="#bdc3c7", line_width=1.5,
                    annotation_text=f"Med. Perf: {fmt_num(_med_perf)}  ",
                    annotation_font=dict(size=10, color="#7f8c8d"),
                    annotation_position="bottom right",
                )
                _quads = [
                    (_med_vol, _xmax, _med_perf, _ymax,
                     "rgba(46,204,113,0.06)", "High Volume · High Performance",
                     (_med_vol+_xmax)/2, _ymax - _ypad*0.3),
                    (_med_vol, _xmax, _ymin, _med_perf,
                     "rgba(231,76,60,0.06)",  "High Volume · Underperforming",
                     (_med_vol+_xmax)/2, _ymin + _ypad*0.3),
                    (_xmin, _med_vol, _med_perf, _ymax,
                     "rgba(241,196,15,0.06)",  "Low Volume · Outperforming",
                     (_xmin+_med_vol)/2, _ymax - _ypad*0.3),
                    (_xmin, _med_vol, _ymin, _med_perf,
                     "rgba(149,165,166,0.06)", "Low Volume · Low Performance",
                     (_xmin+_med_vol)/2, _ymin + _ypad*0.3),
                ]
                for _x0,_x1,_y0,_y1,_fc,_qlbl,_lx,_ly in _quads:
                    _fig_quad.add_shape(type="rect", x0=_x0, x1=_x1, y0=_y0, y1=_y1,
                                        fillcolor=_fc, line_width=0, layer="below")
                    _fig_quad.add_annotation(
                        x=_lx, y=_ly, text=f"<i>{_qlbl}</i>",
                        showarrow=False, font=dict(size=10, color="#95a5a6"),
                    )
                _fig_quad.update_layout(
                    title=None,
                    xaxis=dict(tickformat=".3s", title="Volume",
                               showgrid=True, gridcolor="#f5f5f5", range=[_xmin, _xmax]),
                    yaxis=dict(tickformat=".3s", title="Performance",
                               showgrid=True, gridcolor="#f5f5f5", range=[_ymin, _ymax]),
                    legend=dict(orientation="h", yanchor="bottom", y=1.01,
                                xanchor="right", x=1, title_text=""),
                    plot_bgcolor="white", paper_bgcolor="white",
                    margin=dict(t=30, b=40, l=10, r=10),
                )
                st.plotly_chart(_fig_quad, width="stretch", key="t5_quad_chart")
            else:
                st.info("No plant-level data available.")
        else:
            st.info(f"Columns `{vol_col}` or `Entity_1` not found in data.")

# ── Derive BU / ACT companion columns for the active mode
_bu_col  = "YTD_BU"  if use_ytd else "MTH_BU"
_act_col = "YTD_ACT" if use_ytd else "MTH_ACT"
agg_display_cols = [PERF_COL] + [c for c in [_bu_col, _act_col] if c in df_filtered.columns]

# Extra "By Plant" tab only when drilling into a Country
show_by_plant_tab = (level == "Country")

tab_labels = [f"📍 {h}" for h in breakdown_levels] + ["📦 Account_3"]
if show_by_plant_tab:
    tab_labels += ["🏭 Account_3 by Plant"]
tabs = st.tabs(tab_labels)


def render_tab(tab, group_col: str, title_override: str = ""):
    """Render one breakdown tab.

    - Chart side: stacked bar (Account_3 segments) for hierarchy cols,
      plain bar for Account_3 itself.
      When cluster_mode is ON and group_col == "Entity_1", bars are coloured
      by Volume-cluster instead of MTH_Perf heatmap.
    - Table side: aggregated totals + a drill-down selectbox that shows
      the Account_3 package breakdown for any selected member.
    """
    bd    = agg_by(df_filtered, group_col, perf_col=PERF_COL)
    title = title_override or f"Top {top_n} — {group_col}"
    is_leaf = (group_col in ["Account_3", "Account_4", "Account_5", "Account_5_subpackage"])

    # Attach cluster labels to bd when relevant
    use_cluster_color = (
        cluster_mode
        and group_col == "Entity_1"
        and bool(plant_cluster_map)
    )
    if use_cluster_color:
        bd["Cluster"] = bd["Entity_1"].map(plant_cluster_map).fillna("Unknown")

    with tab:
        # ── Package segment toggle (for non-leaf hierarchy tabs) ─────────────
        _pkg_cols = ["Account_3", "Account_4", "Account_5", "Account_5_subpackage"]
        available_pkg_cols = [c for c in _pkg_cols if c in df_filtered.columns]
        if not is_leaf and not use_cluster_color and available_pkg_cols:
            segment_dim = st.selectbox(
                "🔍 Segment bars by package level:",
                options=available_pkg_cols,
                key=f"seg_dim_{group_col}",
            )
        else:
            segment_dim = available_pkg_cols[0] if available_pkg_cols else "Account_3"

        if not is_leaf:
            if use_cluster_color:
                st.caption(
                    "🔵 **Cluster mode ON** — bars coloured by Volume-based size tier."
                )
            else:
                st.caption(
                    f"Bars segmented by **{segment_dim}** — "
                    f"showing the package mix driving each {group_col}'s {PERF_COL}."
                )

        col_chart, col_table = st.columns([2, 1])

        # ── Left: chart ──────────────────────────────────────────────────────
        with col_chart:
            if is_leaf:
                st.plotly_chart(
                    make_chart(bd, group_col, title, chart_type, color_scale, top_n,
                               perf_col=PERF_COL),
                    width="stretch",
                    key=f"tab_leaf_{group_col}",
                )
            elif use_cluster_color:
                # Plain bar coloured by cluster (not stacked by Account_3)
                st.plotly_chart(
                    make_chart(
                        bd, group_col,
                        title + " — coloured by Volume Cluster",
                        chart_type, color_scale, top_n,
                        color_col="Cluster", perf_col=PERF_COL,
                    ),
                    width="stretch",
                    key=f"tab_cluster_{group_col}",
                )
            else:
                st.plotly_chart(
                    make_stacked_chart(
                        df_filtered, group_col,
                        title + f" — by {segment_dim}",
                        chart_type, top_n, perf_col=PERF_COL,
                        seg_col=segment_dim,
                    ),
                    width="stretch",
                    key=f"tab_stacked_{group_col}_{segment_dim}",
                )

        # ── Right: totals table + drill-down selectbox ────────────────────────
        with col_table:
            all_bd_active = [c for c in ACTIVE_ALL_COLS if c in bd.columns]
            table_cols_full = [group_col] + (["Cluster"] if use_cluster_color else []) + all_bd_active
            table_cols_full = [c for c in table_cols_full if c in bd.columns]
            display_bd = bd[table_cols_full].head(top_n)
            prefix = "YTD_" if use_ytd else "MTH_"
            metric_cols = [c for c in display_bd.columns if c.startswith(prefix)]
            st.markdown(f"**{group_col} breakdown — Top {top_n}**")
            st.dataframe(
                style_positive_blue(arrow_safe(display_bd), metric_cols),
                width="stretch",
                height=260,
            )
            if not bd.empty:
                best_val  = bd.iloc[0][group_col]
                best_perf = bd.iloc[0][PERF_COL]
                st.success(f"🏆 Best: **{best_val}**  ({PERF_COL} = {fmt_num(best_perf)})")

            # ── Package drill-down (only for non-leaf tabs) ───────────────────
            if not is_leaf:
                st.markdown("---")
                st.markdown(f"**📦 {segment_dim} breakdown for a specific {group_col}**")
                members = bd[group_col].tolist()
                sel_member = st.selectbox(
                    f"Select {group_col}",
                    options=members,
                    key=f"drill_{group_col}",
                )
                df_member = df_filtered[df_filtered[group_col] == sel_member]
                acc3_member = agg_by(df_member, segment_dim, perf_col=PERF_COL)

                # Mini KPIs for the selected member
                m1, m2, m3 = st.columns(3)
                m1.metric(PERF_COL, fmt_num(df_member[PERF_COL].sum()) if PERF_COL in df_member.columns else "N/A")
                m2.metric(
                    _bu_col,
                    fmt_num(df_member[_bu_col].sum()) if _bu_col in df_member.columns else "N/A",
                )
                m3.metric(
                    _act_col,
                    fmt_num(df_member[_act_col].sum()) if _act_col in df_member.columns else "N/A",
                )

                member_display = acc3_member[[segment_dim] + agg_display_cols].head(top_n)
                member_metrics = [c for c in agg_display_cols if c in member_display.columns]
                st.dataframe(
                    style_positive_blue(arrow_safe(member_display), member_metrics),
                    width="stretch",
                    height=300,
                )
                if not acc3_member.empty:
                    bp = acc3_member.iloc[0]
                    st.info(
                        f"📦 Top **{segment_dim}** in **{sel_member}**: "
                        f"**{bp[segment_dim]}**  ({PERF_COL} = {fmt_num(bp[PERF_COL])})"
                    )


for i, h in enumerate(breakdown_levels):
    render_tab(tabs[i], h)

# Account_3 tab (always the last standard tab)
acc3_tab_idx = len(breakdown_levels)
render_tab(
    tabs[acc3_tab_idx],
    "Account_3",
    title_override=f"Account_3 Packages — {value or 'Global'}",
)

# ── By Plant tab (Country level only) ────────────────────────────────────────
if show_by_plant_tab:
    with tabs[-1]:
        st.markdown("### 🏭 Account_3 breakdown — drill into a specific Plant")

        plants_in_country = sorted(df_filtered["Entity_1"].dropna().unique().tolist())

        if not plants_in_country:
            st.warning("No plant data available for this selection.")
        else:
            # Plant-level KPI summary table above the dropdown
            plant_summary = agg_by(df_filtered, "Entity_1", perf_col=PERF_COL)
            st.markdown("#### Plant performance overview")
            summary_cols = st.columns([2, 1])
            with summary_cols[0]:
                st.plotly_chart(
                    make_chart(
                        plant_summary, "Entity_1",
                        f"{PERF_COL} by Plant — {value}",
                        chart_type, color_scale, top_n,
                        perf_col=PERF_COL,
                    ),
                    width="stretch",
                    key="byplant_summary_chart",
                )
            with summary_cols[1]:
                ps = plant_summary[["Entity_1"] + agg_display_cols].head(top_n)
                ps_metrics = [c for c in agg_display_cols if c in ps.columns]
                st.dataframe(
                    style_positive_blue(arrow_safe(ps), ps_metrics),
                    width="stretch",
                    height=380,
                )

            st.markdown("---")
            st.markdown("#### 📦 Account_3 packages for a specific Plant")

            selected_plant = st.selectbox(
                "Select Plant (Entity_1)",
                options=plants_in_country,
                key="by_plant_selectbox",
            )

            df_plant = df_filtered[df_filtered["Entity_1"] == selected_plant]
            acc3_plant = agg_by(df_plant, "Account_3", perf_col=PERF_COL)

            # Mini KPIs for that plant
            p1, p2, p3 = st.columns(3)
            p1.metric(PERF_COL,  fmt_num(df_plant[PERF_COL].sum())  if PERF_COL  in df_plant.columns else "N/A")
            p2.metric(_bu_col,   fmt_num(df_plant[_bu_col].sum())   if _bu_col   in df_plant.columns else "N/A")
            p3.metric(_act_col,  fmt_num(df_plant[_act_col].sum())  if _act_col  in df_plant.columns else "N/A")

            pc, pt = st.columns([2, 1])
            with pc:
                st.plotly_chart(
                    make_chart(
                        acc3_plant, "Account_3",
                        f"Account_3 Packages — {selected_plant}",
                        chart_type, color_scale, top_n,
                        perf_col=PERF_COL,
                    ),
                    width="stretch",
                    key="byplant_acc3_chart",
                )
            with pt:
                st.markdown(f"**Top {top_n} Packages in {selected_plant}**")
                apd = acc3_plant[["Account_3"] + agg_display_cols].head(top_n)
                prefix = "YTD_" if use_ytd else "MTH_"
                ap_metrics = [c for c in apd.columns if c.startswith(prefix)]
                st.dataframe(
                    style_positive_blue(arrow_safe(apd), ap_metrics),
                    width="stretch",
                    height=420,
                )
                if not acc3_plant.empty:
                    bp = acc3_plant.iloc[0]
                    st.success(
                        f"🏆 Top package: **{bp['Account_3']}**  "
                        f"({PERF_COL} = {fmt_num(bp[PERF_COL])})"
                    )
# ── Package Deep-Dive — Top 10 Entity_1 ──────────────────────────────────────
st.markdown("---")
st.subheader("🔬 Package Deep-Dive — Top 10 Plants (Entity_1)")
st.caption(
    "Select a package dimension below to see how each of the top 10 plants "
    "is broken down by that sub-package level."
)

# Which sub-package columns actually exist in the data?
PKG_DIMS   = ["Account_3", "Account_4", "Account_5", "Account_5_subpackage"]
avail_dims = [d for d in PKG_DIMS if d in df_filtered.columns]

if not avail_dims:
    st.info("No sub-package columns (Account_3 … Account_5_subpackage) found in this dataset.")
else:
    pkg_dim = st.radio(
        "Break down by",
        options=avail_dims,
        horizontal=True,
        key="pkg_deepdive_dim",
    )

    # Top 10 plants by PERF_COL
    top10_plants = (
        df_filtered.groupby("Entity_1")[PERF_COL]
        .sum().nlargest(10).index.tolist()
    )
    df_top10 = df_filtered[df_filtered["Entity_1"].isin(top10_plants)].copy()

    plant_order = (
        df_top10.groupby("Entity_1")[PERF_COL]
        .sum().sort_values(ascending=False).index.tolist()
    )

    # ── Stacked bar: Entity_1 × pkg_dim ───────────────────────────────────────
    grp_pkg = (
        df_top10.groupby(["Entity_1", pkg_dim])[PERF_COL]
        .sum().reset_index()
    )

    if chart_type == "Horizontal Bar":
        fig_pkg = px.bar(
            grp_pkg, x=PERF_COL, y="Entity_1",
            color=pkg_dim,
            orientation="h",
            barmode="stack",
            title=f"Top 10 Plants — {PERF_COL} broken down by {pkg_dim}",
            category_orders={"Entity_1": list(reversed(plant_order))},
        )
    else:
        fig_pkg = px.bar(
            grp_pkg, x="Entity_1", y=PERF_COL,
            color=pkg_dim,
            barmode="stack",
            title=f"Top 10 Plants — {PERF_COL} broken down by {pkg_dim}",
            category_orders={"Entity_1": plant_order},
        )
        fig_pkg.update_layout(xaxis_tickangle=-35)

    fig_pkg.update_layout(
        legend_title_text=pkg_dim,
        margin=dict(t=50, b=20, l=10, r=10),
    )
    st.plotly_chart(fig_pkg, width="stretch")

    # ── Pivot table: plants × pkg_dim ─────────────────────────────────────────
    st.markdown(f"**{PERF_COL} pivot: Plant × {pkg_dim}**")
    pivot = (
        grp_pkg.pivot_table(
            index="Entity_1", columns=pkg_dim,
            values=PERF_COL, aggfunc="sum", fill_value=0,
        )
        .loc[plant_order]          # keep plant sort order
    )
    pivot["TOTAL"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("TOTAL", ascending=False)

    pv = pivot.reset_index()
    st.dataframe(
        style_positive_blue(arrow_safe(pv), ["TOTAL"]),
        width="stretch",
        height=380,
    )

    # ── Per-plant selectbox drill-down ────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"**Zoom into one Plant — {pkg_dim} detail**")
    sel_plant_pkg = st.selectbox(
        "Select Plant",
        options=plant_order,
        key="pkg_deepdive_plant",
    )
    df_one_plant = df_top10[df_top10["Entity_1"] == sel_plant_pkg]
    one_plant_bd = (
        df_one_plant.groupby(pkg_dim)[PERF_COL]
        .sum().reset_index()
        .sort_values(PERF_COL, ascending=False)
    )

    pp1, pp2 = st.columns(2)
    with pp1:
        fig_one = px.bar(
            one_plant_bd.head(top_n),
            x=pkg_dim, y=PERF_COL,
            color=PERF_COL,
            color_continuous_scale=color_scale,
            title=f"{sel_plant_pkg} — by {pkg_dim}",
            text_auto=".3s",
        )
        fig_one.update_layout(
            coloraxis_showscale=False,
            xaxis_tickangle=-35,
            margin=dict(t=50, b=20, l=10, r=10),
        )
        fig_one.update_xaxes(tickformat=".3s")
        fig_one.update_yaxes(tickformat=".3s")
        st.plotly_chart(fig_one, width="stretch")
    with pp2:
        st.markdown(f"**{sel_plant_pkg} — {pkg_dim} ranking**")
        op = one_plant_bd.head(top_n)
        st.dataframe(
            style_positive_blue(arrow_safe(op), [PERF_COL]),
            width="stretch",
            height=400,
        )
        if not one_plant_bd.empty:
            top_pkg = one_plant_bd.iloc[0]
            st.success(
                f"🏆 Top: **{top_pkg[pkg_dim]}**  "
                f"({PERF_COL} = {fmt_num(top_pkg[PERF_COL])})"
            )


# ── Plant Clustering Analysis ─────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔵 Plant Size & Performance Clustering")
st.caption(
    "Plants are grouped by their production volume into size tiers. "
    "Within each tier, performance relative to peers is highlighted."
)

if not avail_vol:
    st.info(f"Volume column (`{vol_col}`) not found — clustering requires it.")
elif not cluster_mode:
    st.info("Enable **Volume-based Clustering** in the sidebar to see this section.")
elif cluster_df.empty:
    st.warning("Clustering produced no results for the current filter combination.")
else:
    plant_full = agg_by(df_filtered, "Entity_1", perf_col=PERF_COL)
    if vol_col in df_filtered.columns:
        vol_agg = df_filtered.groupby("Entity_1")[vol_col].sum().reset_index()
        # avoid duplicate vol_col if agg_by already picked it up
        if vol_col not in plant_full.columns:
            plant_full = plant_full.merge(vol_agg, on="Entity_1", how="left")
    plant_full = plant_full.merge(
        cluster_df[["Entity_1", "Cluster"]], on="Entity_1", how="left"
    )
    plant_full["Cluster"] = plant_full["Cluster"].fillna("Unknown")
    cluster_color_seq = [CLUSTER_COLORS.get(c, "#aaaaaa") for c in sorted(plant_full["Cluster"].unique())]

    # ── Scatter: Volume vs Perf coloured by size cluster ─────────────────────
    st.markdown("#### How do plant size and performance relate?")
    st.caption(
        "Each dot is a plant. Hover to see details. "
        "Plants are coloured by their **size tier** — revealing whether large or small plants "
        "tend to drive performance in your selection."
    )
    _x_col = vol_col if vol_col in plant_full.columns else PERF_COL
    _hover = ["Entity_1", "Cluster", PERF_COL] + \
             ([vol_col] if vol_col in plant_full.columns else []) + \
             [c for c in [_bu_col, _act_col] if c in plant_full.columns]

    fig_scatter = px.scatter(
        plant_full,
        x=_x_col, y=PERF_COL,
        color="Cluster",
        color_discrete_map=CLUSTER_COLORS,
        text="Entity_1",
        title=f"Plant Size vs Performance ({vol_col} × {PERF_COL})",
        hover_data={k: True for k in _hover},
    )
    fig_scatter.update_traces(textposition="top center", marker=dict(size=11, opacity=0.85,
                              line=dict(width=1, color="white")))
    fig_scatter.update_layout(margin=dict(t=50, b=20, l=10, r=10))
    fig_scatter.update_xaxes(tickformat=".3s", title=f"Volume ({vol_col})")
    fig_scatter.update_yaxes(tickformat=".3s", title=f"Performance ({PERF_COL})")
    st.plotly_chart(fig_scatter, width="stretch")

    # ── Cluster summary table ─────────────────────────────────────────────────
    st.markdown("#### Size tier summary")
    _agg_map = {c: "sum" for c in [PERF_COL, _bu_col, _act_col, vol_col] if c in plant_full.columns}
    _agg_map["Entity_1"] = "count"
    summary = (
        plant_full.groupby("Cluster")
        .agg(_agg_map)
        .rename(columns={"Entity_1": "Plant Count"})
        .reset_index()
        .sort_values(vol_col if vol_col in plant_full.columns else PERF_COL, ascending=False)
    )
    if PERF_COL in summary.columns and "Plant Count" in summary.columns:
        summary[f"Avg {PERF_COL} / Plant"] = summary[PERF_COL] / summary["Plant Count"]
    sty = summary.style
    for _, row in summary.iterrows():
        clr = CLUSTER_COLORS.get(row["Cluster"], "#eeeeee")
        sty = sty.set_properties(subset=pd.IndexSlice[[row.name], ["Cluster"]],
                                 **{"background-color": clr, "color": "white",
                                    "font-weight": "bold"})
    num_cols_sum = summary.select_dtypes(include="number").columns.tolist()
    sty = sty.format({c: fmt_num for c in num_cols_sum})
    st.dataframe(sty, width="stretch", height=220)

    # ── Within-cluster drill-down ─────────────────────────────────────────────
    st.markdown("#### Drill into a size tier")
    cluster_order = sorted(plant_full["Cluster"].unique().tolist(),
                           key=lambda c: list(CLUSTER_COLORS).index(c) if c in CLUSTER_COLORS else 99)
    sel_cluster = st.selectbox("Select a size tier to inspect", options=cluster_order,
                               key="cluster_inspect")
    plants_in_cluster = plant_full[plant_full["Cluster"] == sel_cluster]["Entity_1"].tolist()
    df_cluster_detail = df_filtered[df_filtered["Entity_1"].isin(plants_in_cluster)]

    disp_c = ["Entity_1"] + [c for c in [PERF_COL, vol_col, _bu_col, _act_col] if c in plant_full.columns]
    disp_c = list(dict.fromkeys(disp_c))
    cluster_plant_table = (
        plant_full[plant_full["Cluster"] == sel_cluster][disp_c]
        .sort_values(PERF_COL, ascending=False)
    )

    st.markdown(f"**{sel_cluster}** — {len(cluster_plant_table)} plants")
    cc1, cc2 = st.columns([2, 1])
    with cc1:
        tier_color = CLUSTER_COLORS.get(sel_cluster, "#3498db")
        fig_cl = px.bar(
            cluster_plant_table, x="Entity_1", y=PERF_COL,
            title=f"{sel_cluster} — Performance Comparison",
            text_auto=".3s",
        )
        fig_cl.update_traces(marker_color=tier_color)
        fig_cl.update_layout(coloraxis_showscale=False, xaxis_tickangle=-35,
                             margin=dict(t=50, b=20, l=10, r=10))
        fig_cl.update_xaxes(tickformat=".3s")
        fig_cl.update_yaxes(tickformat=".3s")
        st.plotly_chart(fig_cl, width="stretch", key=f"cl_bar_{sel_cluster}")
    with cc2:
        st.dataframe(style_positive_blue(arrow_safe(cluster_plant_table), [PERF_COL]),
                     width="stretch", height=380)

    st.markdown(f"**📦 Package mix within {sel_cluster}**")
    acc3_cluster = agg_by(df_cluster_detail, "Account_3", perf_col=PERF_COL)
    cg1, cg2 = st.columns([2, 1])
    with cg1:
        st.plotly_chart(
            make_stacked_chart(df_cluster_detail, "Entity_1",
                               f"{sel_cluster} — Plants by Package",
                               chart_type, top_n, perf_col=PERF_COL),
            width="stretch",
            key=f"cl_stacked_{sel_cluster}",
        )
    with cg2:
        acd = acc3_cluster[["Account_3"] + agg_display_cols].head(top_n)
        st.dataframe(style_positive_blue(arrow_safe(acd), [PERF_COL]),
                     width="stretch", height=380)



# ── Raw data viewer ──────────────────────────────────────────────────────────
with st.expander("🔍 View raw filtered data"):
    max_preview = 5000
    preview_df  = df_filtered.head(max_preview)
    if len(df_filtered) > max_preview:
        st.caption(f"Showing first {max_preview:,} of {len(df_filtered):,} rows.")
    st.dataframe(arrow_safe(preview_df), width="stretch")
    csv_bytes = df_filtered.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download filtered data as CSV",
        csv_bytes, "filtered_data.csv", "text/csv",
    )
