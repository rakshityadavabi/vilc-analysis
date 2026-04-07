from __future__ import annotations

import calendar
import json
from functools import lru_cache

import pandas as pd

from app.data_loader import load_aggregated_databricks_data


def _month_number(month) -> int:
    if month is None:
        raise ValueError("month is required")

    if isinstance(month, int):
        if 1 <= month <= 12:
            return month
        raise ValueError(f"Invalid month number: {month}")

    month_text = str(month).strip().lower()
    if month_text.isdigit():
        month_num = int(month_text)
        if 1 <= month_num <= 12:
            return month_num
        raise ValueError(f"Invalid month number: {month}")

    lookup = {calendar.month_name[i].lower(): i for i in range(1, 13)}
    lookup.update({calendar.month_abbr[i].lower(): i for i in range(1, 13)})
    if month_text in lookup:
        return lookup[month_text]

    short_key = month_text[:3]
    if short_key in lookup:
        return lookup[short_key]

    raise ValueError(f"Unrecognised month value: {month}")


def _year_value(year) -> str:
    if year is None:
        raise ValueError("year is required")
    year_text = str(year).strip()
    if not year_text:
        raise ValueError("year is required")
    return year_text


def normalize_report_period(month, year) -> tuple[str, str, str]:
    month_num = _month_number(month)
    year_text = _year_value(year)
    month_abbr = calendar.month_abbr[month_num]
    month_name = calendar.month_name[month_num]
    year_suffix = year_text[-2:]
    month_display = month_name
    period_label = f"{month_abbr} {year_suffix}"
    return month_display, year_text, period_label


def _request_payload(month, year, month_label: str, year_label: str) -> dict:
    return {
        "mode": "MTD",
        "filters": {
            "Year": [year_label],
            "Month": [month_label],
            "_exclude": {"Country": ["Argentina"]},
        },
        "drill_path": {},
        "query_backend": "Databricks",
    }


def _candidate_periods(month, year) -> list[tuple[str, str]]:
    month_num = _month_number(month)
    year_text = _year_value(year)
    month_abbr = calendar.month_abbr[month_num]
    month_name = calendar.month_name[month_num]
    year_suffix = year_text[-2:]
    year_candidates = [year_text]
    if not year_text.upper().startswith("FY"):
        year_candidates.append(f"FY{year_suffix}")

    month_candidates = [
        f"{month_abbr} {year_suffix}",
        f"{month_name} {year_text}",
        f"{month_abbr} {year_text}",
    ]
    combos: list[tuple[str, str]] = []
    for month_label in month_candidates:
        for year_label in year_candidates:
            combos.append((month_label, year_label))
    return combos


@lru_cache(maxsize=24)
def load_report_dataframe(month, year) -> pd.DataFrame:
    last_df = pd.DataFrame()
    for month_label, year_label in _candidate_periods(month, year):
        payload = _request_payload(month, year, month_label, year_label)
        request_key = json.dumps(payload, sort_keys=True)
        df = load_aggregated_databricks_data(request_key=request_key)
        last_df = df
        if not df.empty:
            return df
    return last_df