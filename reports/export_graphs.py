from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from PIL import Image

from .chart_builders import (
    build_mtd_category_mtd,
    build_mtd_category_ytd,
    build_mtd_vic_pp_mtd,
    build_mtd_vic_pp_ytd,
    build_mtd_vic_price_mtd,
    build_mtd_vic_price_ytd,
    build_mtd_zone_table,
)
from .data_queries import load_report_dataframe


REPORT_IMAGE_SPECS: list[tuple[str, Callable[..., object], dict]] = [
    ("mtd_vic_pp_mtd.png", build_mtd_vic_pp_mtd, {"kind": "plotly", "width": 2800, "height": 1800, "scale": 2}),
    ("mtd_vic_pp_ytd.png", build_mtd_vic_pp_ytd, {"kind": "plotly", "width": 2800, "height": 1800, "scale": 2}),
    ("mtd_vic_price_mtd.png", build_mtd_vic_price_mtd, {"kind": "plotly", "width": 2800, "height": 1800, "scale": 2}),
    ("mtd_vic_price_ytd.png", build_mtd_vic_price_ytd, {"kind": "plotly", "width": 2800, "height": 1800, "scale": 2}),
    ("mtd_category_mtd.png", build_mtd_category_mtd, {"kind": "plotly", "width": 2800, "height": 1800, "scale": 2}),
    ("mtd_category_ytd.png", build_mtd_category_ytd, {"kind": "plotly", "width": 2800, "height": 1800, "scale": 2}),
    ("mtd_zone_table.png", build_mtd_zone_table, {"kind": "matplotlib", "dpi": 320}),
]

REQUIRED_IMAGE_FILENAMES = [filename for filename, _, _ in REPORT_IMAGE_SPECS]


def _save_plotly_figure(fig: go.Figure, path: Path, width: int, height: int, scale: int = 2) -> None:
    fig.write_image(str(path), width=width, height=height, scale=scale)


def _save_matplotlib_figure(fig: matplotlib.figure.Figure, path: Path, dpi: int = 220) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_figure(fig, path: Path, spec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    kind = spec["kind"]
    if kind == "plotly":
        _save_plotly_figure(fig, path, width=spec["width"], height=spec["height"], scale=spec.get("scale", 2))
    elif kind == "matplotlib":
        _save_matplotlib_figure(fig, path, dpi=spec.get("dpi", 220))
    else:
        raise ValueError(f"Unsupported figure kind: {kind}")


def validate_images(image_paths: dict[str, Path]) -> None:
    missing: list[str] = []
    corrupted: list[str] = []

    for filename in REQUIRED_IMAGE_FILENAMES:
        path = image_paths.get(filename)
        if path is None or not path.exists() or path.stat().st_size <= 0:
            missing.append(filename)
            continue
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception:
            corrupted.append(filename)

    if missing or corrupted:
        lines = ["Missing required graph exports:"]
        lines.extend(f"- {name}" for name in missing)
        if corrupted:
            lines.append("Corrupted or unreadable graph exports:")
            lines.extend(f"- {name}" for name in corrupted)
        raise RuntimeError("\n".join(lines))


def export_all_report_graphs(month, year, output_dir: str | Path = "generated/images", df: pd.DataFrame | None = None) -> dict[str, Path]:
    report_df = df if df is not None else load_report_dataframe(month, year)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_paths: dict[str, Path] = {}
    for filename, builder, spec in REPORT_IMAGE_SPECS:
        fig = builder(month, year, df=report_df)
        file_path = output_path / filename
        save_figure(fig, file_path, spec)
        image_paths[filename] = file_path

    validate_images(image_paths)
    return image_paths