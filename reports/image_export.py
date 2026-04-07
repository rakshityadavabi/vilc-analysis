from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


CANVAS_WIDTH = 3200
CANVAS_HEIGHT = 14500
MARGIN_X = 110
TOP_MARGIN = 38
HEADER_HEIGHT = 210
PILL_HEIGHT = 110
SECTION_GAP = 38
ROW_GAP = 56
CHART_ROW_HEIGHT = 1150
CHART_GAP = 56
FOOTER_HEIGHT = 56
HEADER_ART_PATH = Path(__file__).resolve().parents[1] / "header.png"
FOOTER_ART_PATH = Path(__file__).resolve().parents[1] / "Footer.png"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _load_rgba(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGBA")


def _paste_contained(canvas: Image.Image, image: Image.Image, box: tuple[int, int, int, int]) -> None:
    left, top, right, bottom = box
    max_width = right - left
    max_height = bottom - top
    contained = ImageOps.contain(image, (max_width, max_height), method=Image.Resampling.LANCZOS)
    offset_x = left + (max_width - contained.width) // 2
    offset_y = top + (max_height - contained.height) // 2
    canvas.alpha_composite(contained, (offset_x, offset_y))


def _paste_fit_width(canvas: Image.Image, image: Image.Image, left: int, right: int, top: int) -> None:
    target_width = right - left
    scale = target_width / float(image.width)
    target_height = max(1, int(round(image.height * scale)))
    resized = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    canvas.alpha_composite(resized, (left, top))


def _draw_section_title(draw: ImageDraw.ImageDraw, title: str, top: int) -> None:
    line_top = top + (PILL_HEIGHT // 2) - 3
    draw.rounded_rectangle((MARGIN_X, line_top, CANVAS_WIDTH - MARGIN_X, line_top + 6), radius=3, fill="#111111")
    font = _font(62, bold=True)
    bbox = draw.textbbox((0, 0), title, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    pill_width = max(text_width + 110, 420)
    pill_left = (CANVAS_WIDTH - pill_width) // 2
    pill_right = pill_left + pill_width
    draw.rounded_rectangle((pill_left, top, pill_right, top + PILL_HEIGHT), radius=20, fill="#ffc20e")
    text_x = (CANVAS_WIDTH - text_width) // 2
    text_y = top + (PILL_HEIGHT - text_height) // 2 - 4
    draw.text((text_x, text_y), title, fill="#111111", font=font)


def _draw_header(draw: ImageDraw.ImageDraw, month_display: str, year_display: str) -> None:
    band_left = MARGIN_X
    band_right = CANVAS_WIDTH - MARGIN_X
    band_bottom = TOP_MARGIN + HEADER_HEIGHT
    draw.rounded_rectangle((band_left, TOP_MARGIN, band_right, band_bottom), radius=6, fill="#ffc20e")

    title_font = _font(56, bold=True)
    meta_font = _font(36, bold=True)

    title = "Preliminary MTD\nVILC Results | P&P"
    date_text = f"{month_display}, {year_display}"
    date_box = (band_left + 46, TOP_MARGIN + 44, band_left + 420, TOP_MARGIN + 150)
    draw.rounded_rectangle(date_box, radius=18, fill="#fff2bf")

    date_bbox = draw.textbbox((0, 0), date_text, font=meta_font)
    date_width = date_bbox[2] - date_bbox[0]
    date_height = date_bbox[3] - date_bbox[1]
    date_x = int(date_box[0] + (date_box[2] - date_box[0] - date_width) / 2)
    date_y = int(date_box[1] + (date_box[3] - date_box[1] - date_height) / 2 - 4)

    title_box_left = date_box[2] + 48
    title_box_right = CANVAS_WIDTH - MARGIN_X - 280
    title_box_center = (title_box_left + title_box_right) / 2
    title_bbox = draw.multiline_textbbox((0, 0), title, font=title_font, spacing=-2)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = int(title_box_center - title_width / 2)
    title_y = TOP_MARGIN + 26

    draw.text((date_x, date_y), date_text, fill="#7a5a00", font=meta_font)
    draw.multiline_text((title_x, title_y), title, fill="#111111", font=title_font, spacing=-2)

    logo_x = CANVAS_WIDTH - MARGIN_X - 250
    logo_y = TOP_MARGIN + 22
    draw.rounded_rectangle((logo_x, logo_y, logo_x + 200, logo_y + 120), radius=2, fill="#3a3a3a")
    icon_font = _font(24, bold=True)
    logo_font = _font(34, bold=True)
    draw.text((logo_x + 20, logo_y + 12), "▮▮▮", fill="#ffc20e", font=icon_font)
    draw.multiline_text((logo_x + 52, logo_y + 44), "PSP\nTECH", fill="#ffffff", font=logo_font, spacing=-8, align="center")


def _draw_banner_art(canvas: Image.Image, month_display: str, year_display: str) -> int:
    draw = ImageDraw.Draw(canvas)
    meta_font = _font(36, bold=True)
    date_text = f"{month_display}, {year_display}"
    date_box = (MARGIN_X, TOP_MARGIN, MARGIN_X + 320, TOP_MARGIN + 78)
    draw.rounded_rectangle(date_box, radius=18, fill="#fff2bf")

    date_bbox = draw.textbbox((0, 0), date_text, font=meta_font)
    date_width = date_bbox[2] - date_bbox[0]
    date_height = date_bbox[3] - date_bbox[1]
    date_x = int(date_box[0] + (date_box[2] - date_box[0] - date_width) / 2)
    date_y = int(date_box[1] + (date_box[3] - date_box[1] - date_height) / 2 - 4)
    draw.text((date_x, date_y), date_text, fill="#7a5a00", font=meta_font)

    header_art = _load_rgba(HEADER_ART_PATH)
    banner_top = TOP_MARGIN + 92
    banner_left = MARGIN_X
    banner_right = CANVAS_WIDTH - MARGIN_X
    _paste_fit_width(canvas, header_art, banner_left, banner_right, banner_top)
    return banner_top + _fit_width_height(header_art)


def _draw_footer(draw: ImageDraw.ImageDraw) -> None:
    footer_font = _font(22, bold=False)
    footer_y = CANVAS_HEIGHT - 64
    draw.text((MARGIN_X, footer_y), "ABInBev", fill="#666666", font=footer_font)
    center_text = "Generated report"
    center_bbox = draw.textbbox((0, 0), center_text, font=footer_font)
    center_width = center_bbox[2] - center_bbox[0]
    draw.text(((CANVAS_WIDTH - center_width) // 2, footer_y), center_text, fill="#666666", font=footer_font)
    right_text = "Powered by PSP TECH"
    right_bbox = draw.textbbox((0, 0), right_text, font=footer_font)
    right_width = right_bbox[2] - right_bbox[0]
    draw.text((CANVAS_WIDTH - MARGIN_X - right_width, footer_y), right_text, fill="#c89d00", font=footer_font)


def _draw_footer_art(canvas: Image.Image) -> None:
    footer_art = _load_rgba(FOOTER_ART_PATH)
    footer_width = CANVAS_WIDTH - 2 * MARGIN_X
    scale = footer_width / float(footer_art.width)
    footer_height = max(1, int(round(footer_art.height * scale)))
    footer_top = CANVAS_HEIGHT - footer_height - 40
    _paste_fit_width(canvas, footer_art, MARGIN_X, CANVAS_WIDTH - MARGIN_X, footer_top)


def _two_up_row(canvas: Image.Image, left_image: Image.Image, right_image: Image.Image, top: int) -> None:
    half_width = (CANVAS_WIDTH - 2 * MARGIN_X - CHART_GAP) // 2
    left_box = (MARGIN_X, top, MARGIN_X + half_width, top + CHART_ROW_HEIGHT)
    right_box = (MARGIN_X + half_width + CHART_GAP, top, CANVAS_WIDTH - MARGIN_X, top + CHART_ROW_HEIGHT)
    _paste_contained(canvas, left_image, left_box)
    _paste_contained(canvas, right_image, right_box)
    divider_x = MARGIN_X + half_width + (CHART_GAP // 2)
    draw = ImageDraw.Draw(canvas)
    draw.line((divider_x, top + 12, divider_x, top + CHART_ROW_HEIGHT - 12), fill="#111111", width=4)


def _single_row(canvas: Image.Image, image: Image.Image, top: int) -> None:
    _paste_fit_width(canvas, image, MARGIN_X, CANVAS_WIDTH - MARGIN_X, top)


def _fit_width_height(image: Image.Image) -> int:
    target_width = CANVAS_WIDTH - 2 * MARGIN_X
    scale = target_width / float(image.width)
    return max(1, int(round(image.height * scale)))


def export_png_from_assets(month_display: str, year_display: str, image_paths: dict[str, Path], output_path: str | Path) -> Path:
    png_path = Path(output_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    canvas = Image.new("RGBA", (CANVAS_WIDTH, CANVAS_HEIGHT), "white")
    draw = ImageDraw.Draw(canvas)

    current_top = _draw_banner_art(canvas, month_display, year_display) + ROW_GAP

    _draw_section_title(draw, "MTD/YTD VIC P&P vs BU", current_top)
    current_top += PILL_HEIGHT + SECTION_GAP
    _two_up_row(canvas, _load_rgba(image_paths["mtd_vic_pp_mtd.png"]), _load_rgba(image_paths["mtd_vic_pp_ytd.png"]), current_top)
    current_top += CHART_ROW_HEIGHT + ROW_GAP

    _draw_section_title(draw, "MTD/YTD VIC Price vs BU", current_top)
    current_top += PILL_HEIGHT + SECTION_GAP
    _two_up_row(canvas, _load_rgba(image_paths["mtd_vic_price_mtd.png"]), _load_rgba(image_paths["mtd_vic_price_ytd.png"]), current_top)
    current_top += CHART_ROW_HEIGHT + ROW_GAP

    _draw_section_title(draw, "MTD vs BGT category by Zone ($Mio)", current_top)
    current_top += PILL_HEIGHT + SECTION_GAP
    _two_up_row(canvas, _load_rgba(image_paths["mtd_category_mtd.png"]), _load_rgba(image_paths["mtd_category_ytd.png"]), current_top)
    current_top += CHART_ROW_HEIGHT + ROW_GAP + 20

    zone_table_image = _load_rgba(image_paths["mtd_zone_table.png"])
    _single_row(canvas, zone_table_image, current_top)
    current_top += _fit_width_height(zone_table_image) + ROW_GAP + 20

    _draw_section_title(draw, "MTD/YTD VIC Performance vs BU", current_top)
    current_top += PILL_HEIGHT + SECTION_GAP
    _two_up_row(canvas, _load_rgba(image_paths["mtd_perf_mtd.png"]), _load_rgba(image_paths["mtd_perf_ytd.png"]), current_top)
    current_top += CHART_ROW_HEIGHT + ROW_GAP

    _draw_section_title(draw, "MTD/YTD VIC Performance by package", current_top)
    current_top += PILL_HEIGHT + SECTION_GAP
    _two_up_row(canvas, _load_rgba(image_paths["mtd_perf_category_mtd.png"]), _load_rgba(image_paths["mtd_perf_category_ytd.png"]), current_top)
    current_top += CHART_ROW_HEIGHT + ROW_GAP + 20

    _draw_section_title(draw, "MTD vs BGT performance category by Zone ($Mio)", current_top)
    current_top += PILL_HEIGHT + SECTION_GAP
    _single_row(canvas, _load_rgba(image_paths["mtd_perf_zone_table.png"]), current_top)

    _draw_footer_art(canvas)

    canvas.convert("RGB").save(png_path, format="PNG", optimize=True)
    return png_path


def export_performance_png_from_assets(month_display: str, year_display: str, image_paths: dict[str, Path], output_path: str | Path) -> Path:
    png_path = Path(output_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    canvas = Image.new("RGBA", (CANVAS_WIDTH, CANVAS_HEIGHT), "white")
    draw = ImageDraw.Draw(canvas)

    current_top = _draw_banner_art(canvas, month_display, year_display) + ROW_GAP

    _draw_section_title(draw, "MTD/YTD VIC Performance vs BU", current_top)
    current_top += PILL_HEIGHT + SECTION_GAP
    _two_up_row(canvas, _load_rgba(image_paths["mtd_perf_mtd.png"]), _load_rgba(image_paths["mtd_perf_ytd.png"]), current_top)
    current_top += CHART_ROW_HEIGHT + ROW_GAP

    _draw_section_title(draw, "MTD/YTD VIC Performance by package", current_top)
    current_top += PILL_HEIGHT + SECTION_GAP
    _two_up_row(canvas, _load_rgba(image_paths["mtd_perf_category_mtd.png"]), _load_rgba(image_paths["mtd_perf_category_ytd.png"]), current_top)
    current_top += CHART_ROW_HEIGHT + ROW_GAP + 20

    _draw_section_title(draw, "MTD vs BGT performance category by Zone ($Mio)", current_top)
    current_top += PILL_HEIGHT + SECTION_GAP
    _single_row(canvas, _load_rgba(image_paths["mtd_perf_zone_table.png"]), current_top)

    _draw_footer_art(canvas)

    canvas.convert("RGB").save(png_path, format="PNG", optimize=True)
    return png_path