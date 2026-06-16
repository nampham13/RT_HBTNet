from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "architecture.png"

W, H = 1400, 760

COLORS = {
    "page_border": "#1f2937",
    "ink": "#111827",
    "gray_fill": "#f1f0ea",
    "gray": "#77746c",
    "gray_arrow": "#6f7169",
    "blue_fill": "#eaf4ff",
    "blue_node": "#cfe5ff",
    "blue": "#256fc2",
    "amber_fill": "#fff6de",
    "amber_node": "#ffe2a7",
    "amber": "#b66b00",
    "coral_fill": "#fff0e7",
    "coral_node": "#ffd3c2",
    "coral": "#d04d2b",
    "purple_fill": "#f0eaff",
    "purple_node": "#d8ccff",
    "purple": "#6b4bd2",
    "teal_fill": "#e8fff8",
    "teal_node": "#b5eee0",
    "teal": "#007b68",
    "green_node": "#bbefb0",
    "green": "#2e9c45",
}


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


F_LABEL = load_font(16, True)
F_NODE = load_font(14, True)
F_TEXT = load_font(12)
F_TINY = load_font(11)


def center_multiline(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    lines: list[tuple[str, ImageFont.ImageFont, str]],
    gap: int = 2,
) -> None:
    x1, y1, x2, y2 = box
    heights = []
    widths = []
    for text, fnt, _ in lines:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=fnt)
        widths.append(right - left)
        heights.append(bottom - top)
    total_h = sum(heights) + gap * (len(lines) - 1)
    y = y1 + (y2 - y1 - total_h) / 2 - 1
    for (text, fnt, color), width, height in zip(lines, widths, heights):
        x = x1 + (x2 - x1 - width) / 2
        draw.text((x, y), text, font=fnt, fill=color)
        y += height + gap


def node(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: str,
    stroke: str,
    lines: list[str],
    *,
    radius: int = 7,
    title_count: int = 1,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=stroke, width=2)
    styled: list[tuple[str, ImageFont.ImageFont, str]] = []
    for i, line in enumerate(lines):
        styled.append((line, F_NODE if i < title_count else F_TEXT, COLORS["ink"]))
    center_multiline(draw, box, styled)


def label(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, color: str) -> None:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=F_LABEL)
    draw.text((x - (right - left) / 2, y), text, font=F_LABEL, fill=color)


def dashed_round_rect(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: str,
    stroke: str,
    *,
    radius: int = 14,
    dash: int = 8,
    gap: int = 7,
) -> None:
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=radius, fill=fill)
    dashed_line(draw, (x1 + radius, y1), (x2 - radius, y1), stroke, dash=dash, gap=gap)
    dashed_line(draw, (x1 + radius, y2), (x2 - radius, y2), stroke, dash=dash, gap=gap)
    dashed_line(draw, (x1, y1 + radius), (x1, y2 - radius), stroke, dash=dash, gap=gap)
    dashed_line(draw, (x2, y1 + radius), (x2, y2 - radius), stroke, dash=dash, gap=gap)
    draw.arc((x1, y1, x1 + 2 * radius, y1 + 2 * radius), 180, 270, fill=stroke, width=2)
    draw.arc((x2 - 2 * radius, y1, x2, y1 + 2 * radius), 270, 360, fill=stroke, width=2)
    draw.arc((x1, y2 - 2 * radius, x1 + 2 * radius, y2), 90, 180, fill=stroke, width=2)
    draw.arc((x2 - 2 * radius, y2 - 2 * radius, x2, y2), 0, 90, fill=stroke, width=2)


def arrow_head(draw: ImageDraw.ImageDraw, p0: tuple[float, float], p1: tuple[float, float], color: str, size: int = 10) -> None:
    angle = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    left = (p1[0] - size * math.cos(angle - 0.48), p1[1] - size * math.sin(angle - 0.48))
    right = (p1[0] - size * math.cos(angle + 0.48), p1[1] - size * math.sin(angle + 0.48))
    draw.polygon([p1, left, right], fill=color)


def dashed_line(
    draw: ImageDraw.ImageDraw,
    p0: tuple[float, float],
    p1: tuple[float, float],
    color: str,
    *,
    width: int = 2,
    dash: int = 8,
    gap: int = 7,
) -> None:
    x0, y0 = p0
    x1, y1 = p1
    length = math.hypot(x1 - x0, y1 - y0)
    if length == 0:
        return
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    t = 0.0
    while t < length:
        t2 = min(t + dash, length)
        draw.line((x0 + ux * t, y0 + uy * t, x0 + ux * t2, y0 + uy * t2), fill=color, width=width)
        t += dash + gap


def poly_arrow(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    color: str,
    *,
    width: int = 2,
    dashed: bool = False,
) -> None:
    if dashed:
        for a, b in zip(points, points[1:]):
            dashed_line(draw, a, b, color, width=width)
    else:
        draw.line(points, fill=color, width=width, joint="curve")
    arrow_head(draw, points[-2], points[-1], color)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, W - 1, H - 1), outline=COLORS["page_border"], width=2)

    label(draw, 700, 18, "RT-HBTNet: Hybrid Blur-Texture Speed Estimation", COLORS["page_border"])

    node(draw, (28, 325, 138, 383), COLORS["gray_fill"], COLORS["gray"], ["Video", "monocular"])
    node(draw, (165, 325, 295, 383), COLORS["gray_fill"], COLORS["gray"], ["Multi-ROI", "preprocess"])
    node(draw, (322, 325, 467, 383), COLORS["gray_fill"], COLORS["gray"], ["Descriptor", "I_t + |grad I_t|"])
    node(draw, (494, 325, 654, 383), COLORS["gray_fill"], COLORS["gray"], ["Shared encoder", "feature map F_t"])

    poly_arrow(draw, [(138, 354), (165, 354)], COLORS["gray_arrow"])
    poly_arrow(draw, [(295, 354), (322, 354)], COLORS["gray_arrow"])
    poly_arrow(draw, [(467, 354), (494, 354)], COLORS["gray_arrow"])
    draw.line((654, 354, 692, 354), fill=COLORS["gray_arrow"], width=2)
    draw.line((692, 172, 692, 568), fill=COLORS["gray_arrow"], width=2)

    dashed_round_rect(draw, (710, 78, 1070, 248), COLORS["blue_fill"], COLORS["blue"])
    label(draw, 890, 47, "Temporal Texture Branch", COLORS["blue"])
    dashed_round_rect(draw, (710, 294, 1070, 464), COLORS["amber_fill"], COLORS["amber"])
    label(draw, 890, 488, "Blur Physics Branch", COLORS["amber"])
    dashed_round_rect(draw, (710, 520, 1070, 638), COLORS["coral_fill"], COLORS["coral"])
    label(draw, 890, 662, "Context Encoder", COLORS["coral"])

    node(draw, (740, 124, 852, 182), COLORS["blue_node"], COLORS["blue"], ["TSM head", "texture cues"])
    node(draw, (882, 124, 1000, 182), COLORS["blue_node"], COLORS["blue"], ["(2+1)D conv", "temporal model"])
    node(draw, (1022, 110, 1094, 196), COLORS["teal_node"], COLORS["teal"], ["Speed", "+", "Conf."], title_count=2)

    node(draw, (740, 340, 852, 398), COLORS["amber_node"], COLORS["amber"], ["Blur head", "physics cues"])
    node(draw, (882, 340, 1000, 398), COLORS["amber_node"], COLORS["amber"], ["Blur latent", "z_blur"])
    node(draw, (1022, 326, 1094, 412), COLORS["teal_node"], COLORS["teal"], ["Speed", "+", "Conf."], title_count=2)

    node(draw, (742, 548, 870, 606), COLORS["coral_node"], COLORS["coral"], ["Context", "observation state"])
    node(draw, (918, 548, 1040, 606), COLORS["coral_node"], COLORS["coral"], ["Quality + bias", "q_t, b_t"])

    poly_arrow(draw, [(692, 172), (710, 153), (740, 153)], COLORS["gray_arrow"])
    poly_arrow(draw, [(852, 153), (882, 153)], COLORS["blue"])
    poly_arrow(draw, [(1000, 153), (1022, 153)], COLORS["blue"])

    poly_arrow(draw, [(692, 369), (710, 369), (740, 369)], COLORS["gray_arrow"])
    poly_arrow(draw, [(852, 369), (882, 369)], COLORS["amber"])
    poly_arrow(draw, [(1000, 369), (1022, 369)], COLORS["amber"])

    poly_arrow(draw, [(692, 568), (710, 568), (742, 577)], COLORS["gray_arrow"])
    poly_arrow(draw, [(870, 577), (918, 577)], COLORS["coral"])

    dashed_round_rect(draw, (1130, 154, 1268, 424), COLORS["purple_fill"], COLORS["purple"])
    label(draw, 1199, 121, "Fusion", COLORS["purple"])
    node(draw, (1150, 184, 1248, 242), COLORS["purple_node"], COLORS["purple"], ["Weights", "softmax"])
    node(draw, (1150, 284, 1248, 342), COLORS["purple_node"], COLORS["purple"], ["Weighted", "sum"])
    node(draw, (1150, 374, 1248, 408), COLORS["purple_node"], COLORS["purple"], ["v_hat"])

    poly_arrow(draw, [(1094, 153), (1130, 213), (1150, 213)], COLORS["purple"])
    poly_arrow(draw, [(1094, 369), (1130, 313), (1150, 313)], COLORS["amber"])
    poly_arrow(draw, [(1040, 577), (1120, 577), (1120, 213), (1150, 213)], COLORS["coral"], dashed=True)
    poly_arrow(draw, [(1199, 242), (1199, 284)], COLORS["purple"])
    poly_arrow(draw, [(1199, 342), (1199, 374)], COLORS["purple"])

    dashed_round_rect(draw, (1300, 218, 1372, 408), COLORS["teal_fill"], COLORS["teal"])
    label(draw, 1336, 186, "Stabilize", COLORS["teal"])
    node(draw, (1316, 244, 1356, 284), COLORS["teal_node"], COLORS["teal"], ["ROI"], title_count=1)
    node(draw, (1316, 302, 1356, 342), COLORS["teal_node"], COLORS["teal"], ["EMA"], title_count=1)
    node(draw, (1316, 360, 1356, 400), COLORS["teal_node"], COLORS["teal"], ["Gate"], title_count=1)
    node(draw, (1294, 444, 1386, 508), COLORS["green_node"], COLORS["green"], ["Final speed", "v_final m/s"])

    poly_arrow(draw, [(1248, 391), (1300, 391), (1316, 264)], COLORS["teal"])
    poly_arrow(draw, [(1336, 284), (1336, 302)], COLORS["teal"])
    poly_arrow(draw, [(1336, 342), (1336, 360)], COLORS["teal"])
    poly_arrow(draw, [(1336, 400), (1336, 444)], COLORS["green"])

    img.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
