from __future__ import annotations

import json
from typing import Any

import pandas as pd

from app.prompts import NEGATIVE_PROMPT_TEMPLATE, POSITIVE_PROMPT_TEMPLATE
from app.services.aggregation_service import (
    aggregate_for_chart,
    get_summary_metrics,
    get_top_negative,
    get_top_positive,
)
from app.services.rootcause_service import enrich_root_cause
from app.utils import HIERARCHY


def _driver_column(df: pd.DataFrame) -> str:
    # Prefer package drivers first, then plant/country/zone.
    priority = [
        "Account_5_subpackage",
        "Account_5",
        "Account_4",
        "Account_3",
        "Entity_1",
        "Country",
        "Zone",
    ]
    for col in priority:
        if col in df.columns:
            return col
    return next(iter(df.columns), "")


def _dict_from_df(df: pd.DataFrame, key_col: str, val_col: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if df.empty:
        return out

    for _, row in df.iterrows():
        out[str(row[key_col])] = float(row[val_col])
    return out


def build_insight_payload(df: pd.DataFrame, drill_path: dict, perf_col: str) -> dict[str, Any]:
    summary = get_summary_metrics(df, perf_col)

    drv_col = _driver_column(df)
    top_pos_df = get_top_positive(df, drv_col, perf_col, n=3)
    top_neg_df = get_top_negative(df, drv_col, perf_col, n=3)

    reg_col = "Zone" if "Zone" in df.columns else ("Country" if "Country" in df.columns else drv_col)
    top_regions = aggregate_for_chart(df, reg_col, perf_col).head(3)

    selected_path = " > ".join([str(v) for _, v in drill_path.items()]) if drill_path else "Home"

    return {
        "selected_path": selected_path,
        "variance": summary["variance"],
        "actual": summary["actual"],
        "budget": summary["budget"],
        "top_positive_drivers": _dict_from_df(top_pos_df, drv_col, perf_col),
        "top_negative_drivers": _dict_from_df(top_neg_df, drv_col, perf_col),
        "top_regions": _dict_from_df(top_regions, reg_col, perf_col),
        "driver_level": drv_col,
        "region_level": reg_col,
        "perf_col": perf_col,
    }


def build_prompt_from_payload(payload: dict[str, Any]) -> str:
    payload_json = json.dumps(payload, indent=2)
    if float(payload.get("variance", 0.0)) >= 0:
        return POSITIVE_PROMPT_TEMPLATE.format(json_payload=payload_json)
    return NEGATIVE_PROMPT_TEMPLATE.format(json_payload=payload_json)


def _format_driver_list(drivers: dict[str, float], top_n: int = 2) -> str:
    if not drivers:
        return "none"

    items = sorted(drivers.items(), key=lambda x: abs(float(x[1])), reverse=True)[:top_n]
    return ", ".join([f"{name} ({value:+.2f})" for name, value in items])


def generate_insight(payload: dict[str, Any]) -> dict[str, Any]:
    variance = float(payload.get("variance", 0.0))
    selected_path = payload.get("selected_path", "Home")
    actual = float(payload.get("actual", 0.0))
    budget = float(payload.get("budget", 0.0))

    top_pos = payload.get("top_positive_drivers", {})
    top_neg = payload.get("top_negative_drivers", {})
    top_regions = payload.get("top_regions", {})

    pos_list = list(top_pos.keys())
    neg_list = list(top_neg.keys())
    region_list = list(top_regions.keys())

    dominant_pos = pos_list[0] if pos_list else "none"
    dominant_neg = neg_list[0] if neg_list else "none"
    dominant_region = region_list[0] if region_list else "no clear concentration"

    root_causes = enrich_root_cause(pos_list + neg_list)

    pos_summary = _format_driver_list(top_pos)
    neg_summary = _format_driver_list(top_neg)

    if variance >= 0:
        summary = (
            f"{selected_path} is above budget by {variance:+.2f} (Actual {actual:.2f} vs Budget {budget:.2f}). "
            f"Largest contributors are {pos_summary}. Offsets are {neg_summary}. "
            f"Impact is most visible in {dominant_region}."
        )
        recommended_action = (
            f"Scale the current winning levers ({dominant_pos}) to similar nodes and prevent erosion from {dominant_neg}. "
            "Track weekly variance to lock in gains."
        )
    else:
        summary = (
            f"{selected_path} is below budget by {variance:+.2f} (Actual {actual:.2f} vs Budget {budget:.2f}). "
            f"Main drags are {neg_summary}. Partial offsets are {pos_summary}. "
            f"Impact is concentrated in {dominant_region}."
        )
        recommended_action = (
            f"Prioritize corrective actions on {dominant_neg}, validate root-cause assumptions, and replicate offsetting levers like {dominant_pos}. "
            "Review corrective impact in the next cycle."
        )

    return {
        "summary": summary,
        "positive_drivers": top_pos,
        "negative_drivers": top_neg,
        "root_cause": root_causes,
        "recommended_action": recommended_action,
        "prompt": build_prompt_from_payload(payload),
        "generation_note": "Insight generated from current filtered dataset and deterministic rules.",
    }
