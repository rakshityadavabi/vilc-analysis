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


def render_monthly_report(month, year, image_paths: dict[str, Path], output_path: str | Path | None = None) -> Path:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    report_path = Path(output_path) if output_path else GENERATED_DIR / "report.html"

    context = report_title_context(month, year)
    context.update({
        "css_path": "../templates/report.css",
        "mtd_vic_pp_mtd": _image_ref("mtd_vic_pp_mtd.png"),
        "mtd_vic_pp_ytd": _image_ref("mtd_vic_pp_ytd.png"),
        "mtd_vic_price_mtd": _image_ref("mtd_vic_price_mtd.png"),
        "mtd_vic_price_ytd": _image_ref("mtd_vic_price_ytd.png"),
        "mtd_category_mtd": _image_ref("mtd_category_mtd.png"),
        "mtd_category_ytd": _image_ref("mtd_category_ytd.png"),
        "mtd_zone_table": _image_ref("mtd_zone_table.png"),
    })

    template = _template_environment().get_template("monthly_report.html")
    html = template.render(**context)
    report_path.write_text(html, encoding="utf-8")
    return report_path