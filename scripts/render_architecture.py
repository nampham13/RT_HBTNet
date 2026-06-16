from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "architecture.png"

W, H = 1600, 760

COLORS = {
    "ink": "#111827",
    "muted": "#374151",
    "gray_fill": "#f1f0ea",
    "gray": "#77746c",
    "blue_fill": "#e8f3ff",
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


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


F_LABEL = font(17, True)
F_NODE = font(15, True)
F_TEXT = font(13)
F_MATH = font(12, True)


def text_center(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], lines: list[tuple[str, ImageFont.ImageFont, str]]) -> None:
    x1, y1, x2, y2 = box
    heights = [draw.textbbox((0, 0), text, font=fnt)[3] for text, fnt, _ in lines]
    total = sum(heights) + 2 * (len(lines) - 1)
    y = y1 + (y2 - y1 - total) / 2 - 1
    for (text, fnt, color), h in zip(lines, heights):
        bbox = draw.textbbox((0, 0), text, font=fnt)
        x = x1 + (x2 - x1 - (bbox[2] - bbox[0])) / 2
        draw.text((x, y), text, font=fnt, fill=color)
        y += h + 2


def node(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: str,
    stroke: str,
    lines: list[str],
    *,
    width: int = 2,
) -> None:
    draw.rounded_rectangle(box, radius=8, fill=fill, outline=stroke, width=width)
    styled = []
    for i, line in enumerate(lines):
        styled.append((line, F_NODE if i == 0 else F_TEXT, COLORS["ink"]))
    text_center(draw, box, styled)


def dashed_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: str,
    stroke: str,
    label: str,
    label_pos: str = "top",
) -> None:
    draw.rounded_rectangle(box, radius=16, fill=fill)
    x1, y1, x2, y2 = box
    dash = 9
    gap = 7
    for a, b in [((x1 + 14, y1), (x2 - 14, y1)), ((x1 + 14, y2), (x2 - 14, y2)), ((x1, y1 + 14), (x1, y2 - 14)), ((x2, y1 + 14), (x2, y2 - 14))]:
        dashed_line(draw, a, b, stroke, 2, dash, gap)
    for arc_box, start, end in [
        ((x1, y1, x1 + 28, y1 + 28), 180, 270),
        ((x2 - 28, y1, x2, y1 + 28), 270, 360),
        ((x1, y2 - 28, x1 + 28, y2), 90, 180),
        ((x2 - 28, y2 - 28, x2, y2), 0, 90),
    ]:
        draw.arc(arc_box, start=start, end=end, fill=stroke, width=2)
    label_w = draw.textbbox((0, 0), label, font=F_LABEL)[2]
    label_x = x1 + (x2 - x1 - label_w) / 2
    label_y = y1 - 24 if label_pos == "top" else y2 + 8
    draw.text((label_x, label_y), label, font=F_LABEL, fill=stroke)


def dashed_line(draw: ImageDraw.ImageDraw, a: tuple[int, int], b: tuple[int, int], color: str, width: int = 2, dash: int = 9, gap: int = 7) -> None:
    ax, ay = a
    bx, by = b
    length = math.hypot(bx - ax, by - ay)
    if length <= 0:
        return
    ux, uy = (bx - ax) / length, (by - ay) / length
    t = 0.0
    while t < length:
        t2 = min(t + dash, length)
        draw.line((ax + ux * t, ay + uy * t, ax + ux * t2, ay + uy * t2), fill=color, width=width)
        t += dash + gap


def arrow_head(draw: ImageDraw.ImageDraw, p0: tuple[int, int], p1: tuple[int, int], color: str) -> None:
    angle = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    size = 12
    left = (p1[0] - size * math.cos(angle - 0.46), p1[1] - size * math.sin(angle - 0.46))
    right = (p1[0] - size * math.cos(angle + 0.46), p1[1] - size * math.sin(angle + 0.46))
    draw.polygon([p1, left, right], fill=color)


def arrow(draw: ImageDraw.ImageDraw, pts: list[tuple[int, int]], color: str = "#444444", width: int = 2, dash: bool = False) -> None:
    if dash:
        for a, b in zip(pts, pts[1:]):
            dashed_line(draw, a, b, color, width)
    else:
        draw.line(pts, fill=color, width=width, joint="curve")
    arrow_head(draw, pts[-2], pts[-1], color)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, W - 1, H - 1), outline="#1f2937", width=3)

    # Left input pipeline.
    node(draw, (28, 265, 148, 335), COLORS["gray_fill"], COLORS["gray"], ["Video", "stream", "(monocular)"])
    node(draw, (176, 265, 320, 335), COLORS["gray_fill"], COLORS["gray"], ["Multi-ROI", "extraction", "& preprocessing"])
    node(draw, (370, 265, 540, 335), COLORS["gray_fill"], COLORS["gray"], ["Frame", "descriptor", "I_t, |grad I_t|"])
    node(draw, (590, 265, 790, 335), COLORS["gray_fill"], COLORS["gray"], ["Shared encoder", "MobileNetV3-Small", "feature map F_t"])
    arrow(draw, [(148, 300), (176, 300)], COLORS["gray"])
    arrow(draw, [(320, 300), (370, 300)], COLORS["gray"])
    arrow(draw, [(540, 300), (590, 300)], COLORS["gray"])

    split_x = 835
    draw.line((790, 300, split_x, 300), fill=COLORS["gray"], width=2)
    draw.line((split_x, 130, split_x, 595), fill=COLORS["gray"], width=2)

    # Branch boxes.
    dashed_box(draw, (350, 58, 900, 246), COLORS["blue_fill"], COLORS["blue"], "Temporal Texture Branch")
    dashed_box(draw, (350, 350, 900, 538), COLORS["amber_fill"], COLORS["amber"], "Blur Physics Branch", label_pos="bottom")
    dashed_box(draw, (350, 585, 930, 700), COLORS["coral_fill"], COLORS["coral"], "Context Encoder", label_pos="bottom")
    dashed_box(draw, (940, 96, 1190, 420), COLORS["purple_fill"], COLORS["purple"], "Confidence-Aware Fusion")
    dashed_box(draw, (1215, 152, 1415, 420), COLORS["teal_fill"], COLORS["teal"], "Temporal Stabilization")

    # Temporal branch.
    node(draw, (375, 118, 535, 188), COLORS["blue_node"], COLORS["blue"], ["TSM texture", "head"])
    node(draw, (565, 118, 735, 188), COLORS["blue_node"], COLORS["blue"], ["(2+1)D conv", "+ multi-scale", "ROI pooling"])
    node(draw, (780, 86, 875, 160), COLORS["green_node"], COLORS["green"], ["Speed", "head", "v_tex"])
    node(draw, (780, 160, 875, 234), COLORS["teal_node"], COLORS["teal"], ["Conf.", "head", "c_tex"])
    arrow(draw, [(split_x, 130), (350, 130), (375, 153)], COLORS["gray"])
    arrow(draw, [(535, 153), (565, 153)], COLORS["blue"])
    arrow(draw, [(735, 153), (780, 123)], COLORS["blue"])
    arrow(draw, [(735, 153), (780, 197)], COLORS["blue"])

    # Blur branch.
    node(draw, (375, 410, 535, 480), COLORS["amber_node"], COLORS["amber"], ["Blur cue", "head"])
    node(draw, (565, 410, 735, 480), COLORS["amber_node"], COLORS["amber"], ["Key-frame blur", "representation", "z_blur"])
    node(draw, (780, 378, 875, 452), COLORS["green_node"], COLORS["green"], ["Speed", "head", "v_blur"])
    node(draw, (780, 452, 875, 526), COLORS["teal_node"], COLORS["teal"], ["Conf.", "head", "c_blur"])
    arrow(draw, [(split_x, 445), (350, 445), (375, 445)], COLORS["gray"])
    arrow(draw, [(535, 445), (565, 445)], COLORS["amber"])
    arrow(draw, [(735, 445), (780, 415)], COLORS["amber"])
    arrow(draw, [(735, 445), (780, 489)], COLORS["amber"])

    # Context branch.
    node(draw, (375, 615, 535, 675), COLORS["coral_node"], COLORS["coral"], ["Lightweight", "context encoder"])
    node(draw, (565, 615, 735, 675), COLORS["coral_node"], COLORS["coral"], ["Observation", "context", "z_ctx"])
    node(draw, (755, 615, 905, 675), COLORS["coral_node"], COLORS["coral"], ["Quality scores", "bias + stability"])
    arrow(draw, [(split_x, 595), (350, 595), (375, 645)], COLORS["gray"])
    arrow(draw, [(535, 645), (565, 645)], COLORS["coral"])
    arrow(draw, [(735, 645), (755, 645)], COLORS["coral"])

    # Fusion.
    node(draw, (970, 120, 1168, 190), COLORS["purple_node"], COLORS["purple"], ["Fusion weight MLP", "softmax(conf + bias)", "w_tex, w_blur"])
    node(draw, (970, 235, 1168, 305), COLORS["purple_node"], COLORS["purple"], ["Weighted sum", "v = w_tex*v_tex", "+ w_blur*v_blur"])
    node(draw, (990, 345, 1148, 405), COLORS["purple_node"], COLORS["purple"], ["Fused speed", "v_hat"])
    arrow(draw, [(875, 123), (940, 123), (970, 155)], COLORS["purple"])
    arrow(draw, [(875, 197), (940, 197), (970, 155)], COLORS["purple"])
    arrow(draw, [(875, 415), (940, 415), (970, 270)], COLORS["amber"])
    arrow(draw, [(875, 489), (940, 489), (970, 155)], COLORS["purple"])
    arrow(draw, [(905, 645), (960, 645), (960, 155), (970, 155)], COLORS["coral"], dash=True)
    arrow(draw, [(1069, 190), (1069, 235)], COLORS["purple"])
    arrow(draw, [(1069, 305), (1069, 345)], COLORS["purple"])

    # Stabilization.
    node(draw, (1240, 170, 1390, 235), COLORS["teal_node"], COLORS["teal"], ["Multi-ROI", "robust voting", "median"])
    node(draw, (1240, 265, 1390, 330), COLORS["teal_node"], COLORS["teal"], ["Temporal filter", "Kalman / EMA"])
    node(draw, (1240, 360, 1390, 425), COLORS["teal_node"], COLORS["teal"], ["Outlier rejection", "& update"])
    node(draw, (1445, 355, 1560, 425), COLORS["green_node"], COLORS["green"], ["Final speed", "v_final", "(m/s)"])
    arrow(draw, [(1148, 375), (1215, 375), (1240, 202)], COLORS["teal"])
    arrow(draw, [(1315, 235), (1315, 265)], COLORS["teal"])
    arrow(draw, [(1315, 330), (1315, 360)], COLORS["teal"])
    arrow(draw, [(1390, 392), (1445, 392)], COLORS["green"])

    img.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
