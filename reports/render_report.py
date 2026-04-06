from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .chart_builders import report_title_context


TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
GENERATED_DIR = Path(__file__).resolve().parents[1] / "generated"


def _template_environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )


def _image_ref(filename: str) -> str:
    return f"images/{filename}"


def _performance_image_ref(filename: str) -> str:
    return f"performance_images/{filename}"


def _base_report_context(month, year) -> dict[str, str]:
    context = report_title_context(month, year)
    context.update({"css_path": "../templates/report.css"})
    return context


def _render_report(template_name: str, month, year, output_path: str | Path | None, image_map: dict[str, str]) -> Path:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    report_path = Path(output_path) if output_path else GENERATED_DIR / template_name

    context = _base_report_context(month, year)
    context.update(image_map)

    template = _template_environment().get_template(template_name)
    html = template.render(**context)
    report_path.write_text(html, encoding="utf-8")
    return report_path


def render_monthly_report(month, year, image_paths: dict[str, Path], output_path: str | Path | None = None) -> Path:
    return _render_report(
        "monthly_report.html",
        month,
        year,
        output_path,
        {
            "mtd_vic_pp_mtd": _image_ref("mtd_vic_pp_mtd.png"),
            "mtd_vic_pp_ytd": _image_ref("mtd_vic_pp_ytd.png"),
            "mtd_vic_price_mtd": _image_ref("mtd_vic_price_mtd.png"),
            "mtd_vic_price_ytd": _image_ref("mtd_vic_price_ytd.png"),
            "mtd_category_mtd": _image_ref("mtd_category_mtd.png"),
            "mtd_category_ytd": _image_ref("mtd_category_ytd.png"),
            "mtd_zone_table": _image_ref("mtd_zone_table.png"),
            "mtd_perf_mtd": _performance_image_ref("mtd_perf_mtd.png"),
            "mtd_perf_ytd": _performance_image_ref("mtd_perf_ytd.png"),
            "mtd_perf_category_mtd": _performance_image_ref("mtd_perf_category_mtd.png"),
            "mtd_perf_category_ytd": _performance_image_ref("mtd_perf_category_ytd.png"),
            "mtd_perf_zone_table": _performance_image_ref("mtd_perf_zone_table.png"),
        },
    )


def render_performance_report(month, year, image_paths: dict[str, Path], output_path: str | Path | None = None) -> Path:
    return _render_report(
        "performance_report.html",
        month,
        year,
        output_path,
        {
            "mtd_perf_mtd": _performance_image_ref("mtd_perf_mtd.png"),
            "mtd_perf_ytd": _performance_image_ref("mtd_perf_ytd.png"),
            "mtd_perf_category_mtd": _performance_image_ref("mtd_perf_category_mtd.png"),
            "mtd_perf_category_ytd": _performance_image_ref("mtd_perf_category_ytd.png"),
            "mtd_perf_zone_table": _performance_image_ref("mtd_perf_zone_table.png"),
        },
    )