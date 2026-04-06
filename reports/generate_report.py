from __future__ import annotations

from pathlib import Path

from .export_graphs import export_all_report_graphs
from .data_queries import normalize_report_period
from .image_export import export_png_from_assets
from .render_report import render_monthly_report


def _validate_inputs(month, year) -> None:
    if month is None or str(month).strip() == "":
        raise ValueError("month is required")
    if year is None or str(year).strip() == "":
        raise ValueError("year is required")


def generate_monthly_report(month="March", year=2026):
    _validate_inputs(month, year)

    image_paths = export_all_report_graphs(month, year)
    html_path = render_monthly_report(month, year, image_paths)
    month_display, year_display, _ = normalize_report_period(month, year)
    png_path = export_png_from_assets(month_display, year_display, image_paths, html_path.with_suffix(".png"))
    return {"html": html_path, "png": png_path}