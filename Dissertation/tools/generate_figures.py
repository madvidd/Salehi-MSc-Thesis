"""Generate the dissertation diagrams as matching PNG and SVG files.

The renderer deliberately depends only on Pillow, which is available in the
bundled workspace runtime. Coordinates are expressed in pixels so that the
raster and vector outputs have the same composition.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from math import atan2, cos, pi, sin
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "images"
OUT.mkdir(parents=True, exist_ok=True)

WIDTH = 1800
HEIGHT = 980
SCALE = 2

COLORS = {
    "ink": "#1f2933",
    "muted": "#5c6670",
    "light": "#eef1f3",
    "line": "#718096",
    "teal": "#167c80",
    "green": "#3f7d44",
    "orange": "#c87533",
    "red": "#a84a44",
    "blue": "#3b6ea8",
    "purple": "#745b99",
    "gold": "#b58a2a",
    "white": "#ffffff",
}

FONT_REGULAR = Path(r"C:\Windows\Fonts\times.ttf")
FONT_BOLD = Path(r"C:\Windows\Fonts\timesbd.ttf")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    return ImageFont.truetype(str(path), size * SCALE)


@dataclass
class SvgCommand:
    markup: str


class Figure:
    def __init__(self, title: str, subtitle: str = "") -> None:
        self.image = Image.new("RGB", (WIDTH * SCALE, HEIGHT * SCALE), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.svg: list[SvgCommand] = []
        self.text(
            WIDTH / 2,
            65,
            title,
            size=32,
            bold=True,
            anchor="mm",
            color=COLORS["ink"],
        )
        if subtitle:
            self.text(
                WIDTH / 2,
                112,
                subtitle,
                size=20,
                anchor="mm",
                color=COLORS["muted"],
            )

    @staticmethod
    def _points(points: list[tuple[float, float]]) -> list[tuple[int, int]]:
        return [(round(x * SCALE), round(y * SCALE)) for x, y in points]

    def line(
        self,
        points: list[tuple[float, float]],
        color: str = COLORS["line"],
        width: int = 3,
        dash: str | None = None,
    ) -> None:
        scaled = self._points(points)
        if dash:
            # Pillow has no general dashed polyline primitive; all dashed uses
            # in this document are horizontal or vertical two-point segments.
            (x1, y1), (x2, y2) = points
            length = max(abs(x2 - x1), abs(y2 - y1))
            steps = max(1, int(length // 18))
            for index in range(steps):
                if index % 2:
                    continue
                t1 = index / steps
                t2 = min(1.0, (index + 1) / steps)
                ax = x1 + (x2 - x1) * t1
                ay = y1 + (y2 - y1) * t1
                bx = x1 + (x2 - x1) * t2
                by = y1 + (y2 - y1) * t2
                self.draw.line(
                    self._points([(ax, ay), (bx, by)]),
                    fill=color,
                    width=width * SCALE,
                )
        else:
            self.draw.line(scaled, fill=color, width=width * SCALE, joint="curve")
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.svg.append(
            SvgCommand(
                f'<polyline points="{path}" fill="none" stroke="{color}" '
                f'stroke-width="{width}" stroke-linecap="round" '
                f'stroke-linejoin="round"{dash_attr}/>'
            )
        )

    def arrow(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        color: str = COLORS["line"],
        width: int = 3,
    ) -> None:
        self.line([start, end], color=color, width=width)
        angle = atan2(end[1] - start[1], end[0] - start[0])
        length = 16
        wing = 0.52
        points = [
            end,
            (
                end[0] - length * cos(angle - wing),
                end[1] - length * sin(angle - wing),
            ),
            (
                end[0] - length * cos(angle + wing),
                end[1] - length * sin(angle + wing),
            ),
        ]
        self.draw.polygon(self._points(points), fill=color)
        svg_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        self.svg.append(SvgCommand(f'<polygon points="{svg_points}" fill="{color}"/>'))

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str = COLORS["white"],
        outline: str = COLORS["line"],
        width: int = 3,
        radius: int = 10,
    ) -> None:
        self.draw.rounded_rectangle(
            (x * SCALE, y * SCALE, (x + w) * SCALE, (y + h) * SCALE),
            radius=radius * SCALE,
            fill=fill,
            outline=outline,
            width=width * SCALE,
        )
        self.svg.append(
            SvgCommand(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" '
                f'height="{h:.1f}" rx="{radius}" fill="{fill}" '
                f'stroke="{outline}" stroke-width="{width}"/>'
            )
        )

    def circle(
        self,
        x: float,
        y: float,
        radius: float,
        fill: str,
        outline: str = COLORS["white"],
        width: int = 2,
    ) -> None:
        self.draw.ellipse(
            (
                (x - radius) * SCALE,
                (y - radius) * SCALE,
                (x + radius) * SCALE,
                (y + radius) * SCALE,
            ),
            fill=fill,
            outline=outline,
            width=width * SCALE,
        )
        self.svg.append(
            SvgCommand(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" '
                f'fill="{fill}" stroke="{outline}" stroke-width="{width}"/>'
            )
        )

    def text(
        self,
        x: float,
        y: float,
        value: str,
        *,
        size: int = 22,
        bold: bool = False,
        color: str = COLORS["ink"],
        anchor: str = "la",
        wrap: int | None = None,
        line_spacing: float = 1.22,
    ) -> None:
        lines: list[str] = []
        for paragraph in value.split("\n"):
            if wrap:
                lines.extend(
                    textwrap.wrap(
                        paragraph,
                        width=wrap,
                        break_long_words=False,
                        break_on_hyphens=False,
                    )
                    or [""]
                )
            else:
                lines.append(paragraph)

        pil_anchor = {
            "la": "la",
            "lm": "lm",
            "mm": "mm",
            "ma": "ma",
            "ra": "ra",
            "rm": "rm",
        }[anchor]
        fnt = font(size, bold)
        step = size * line_spacing
        total = step * (len(lines) - 1)
        start_y = y - total / 2 if anchor.endswith("m") or anchor == "mm" else y
        horizontal = anchor[0]
        pil_line_anchor = {"l": "la", "m": "ma", "r": "ra"}[horizontal]

        for index, line_value in enumerate(lines):
            line_y = start_y + index * step
            self.draw.text(
                (x * SCALE, line_y * SCALE),
                line_value,
                font=fnt,
                fill=color,
                anchor=pil_line_anchor,
            )

        svg_anchor = {"l": "start", "m": "middle", "r": "end"}[horizontal]
        weight = "700" if bold else "400"
        baseline_y = start_y + size * 0.36
        tspans = []
        for index, line_value in enumerate(lines):
            dy = 0 if index == 0 else step
            tspans.append(
                f'<tspan x="{x:.1f}" dy="{dy:.1f}">{escape(line_value)}</tspan>'
            )
        self.svg.append(
            SvgCommand(
                f'<text x="{x:.1f}" y="{baseline_y:.1f}" '
                f'font-family="Times New Roman, Times, serif" font-size="{size}" '
                f'font-weight="{weight}" fill="{color}" '
                f'text-anchor="{svg_anchor}">{"".join(tspans)}</text>'
            )
        )

    def box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        label: str,
        *,
        outline: str,
        fill: str = COLORS["white"],
        size: int = 22,
        bold: bool = False,
        wrap: int | None = None,
    ) -> None:
        self.rect(x, y, w, h, fill=fill, outline=outline)
        self.text(
            x + w / 2,
            y + h / 2,
            label,
            size=size,
            bold=bold,
            anchor="mm",
            wrap=wrap,
        )

    def save(self, stem: str) -> None:
        self.image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS).save(
            OUT / f"{stem}.png",
            dpi=(300, 300),
            optimize=True,
        )
        content = "\n".join(command.markup for command in self.svg)
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">\n'
            f'<rect width="100%" height="100%" fill="white"/>\n'
            f"{content}\n</svg>\n"
        )
        (OUT / f"{stem}.svg").write_text(svg, encoding="utf-8")


def experimental_programme() -> None:
    fig = Figure(
        "Experimental programme",
        "Evidence-led progression from reproduction to controlled architectural ablation",
    )
    boxes = [
        ("SHARP\nreproduction", COLORS["blue"]),
        ("Mamba placement\nstudies", COLORS["purple"]),
        ("Ten-test\nscreen", COLORS["orange"]),
        ("Attention\nablation", COLORS["teal"]),
        ("Final three-run\nconfirmation", COLORS["green"]),
    ]
    x_values = [85, 430, 775, 1120, 1465]
    for index, ((label, color), x) in enumerate(zip(boxes, x_values)):
        fig.box(x, 325, 250, 180, label, outline=color, fill=COLORS["light"], size=25, bold=True)
        fig.circle(x + 125, 275, 31, fill=color)
        fig.text(x + 125, 275, str(index + 1), size=25, bold=True, color="white", anchor="mm")
        if index < len(boxes) - 1:
            fig.arrow((x + 250, 415), (x_values[index + 1] - 20, 415), color=COLORS["line"])
    fig.text(
        900,
        650,
        "Constant evidence rules",
        size=27,
        bold=True,
        anchor="mm",
    )
    rules = [
        "same AV2 processed split",
        "same seed and optimiser schedule",
        "same global batch where controlled",
        "best checkpoint selected by minADE6",
        "all reported metrics are lower-is-better",
    ]
    for index, rule in enumerate(rules):
        x = 245 + (index % 3) * 550
        y = 735 + (index // 3) * 105
        fig.circle(x, y, 10, fill=COLORS["teal"], outline=COLORS["teal"])
        fig.text(x + 24, y, rule, size=22, anchor="lm")
    fig.save("experimental_programme")


def sharp_streaming_overview() -> None:
    fig = Figure(
        "SHARP incremental streaming architecture",
        "Short observation windows share instance-aware context while the prediction state is updated online",
    )
    windows = [("Window t-2", 95), ("Window t-1", 530), ("Window t", 965)]
    for label, x in windows:
        fig.box(x, 250, 320, 150, label + "\nagent and lane tokens", outline=COLORS["blue"], fill=COLORS["light"], size=24)
        fig.arrow((x + 160, 400), (x + 160, 500), color=COLORS["blue"])
        fig.box(x, 510, 320, 135, "shared SHARP encoder", outline=COLORS["teal"], size=23, bold=True)
    fig.box(1410, 250, 300, 150, "Target-specific\ntrajectory prediction", outline=COLORS["green"], fill="#edf7ee", size=24, bold=True)
    fig.box(1410, 510, 300, 135, "streamed context\nand trajectory relay", outline=COLORS["orange"], fill="#fff5ec", size=23, bold=True)
    for left, right in [(415, 530), (850, 965), (1285, 1410)]:
        fig.arrow((left, 325), (right - 18, 325), color=COLORS["line"])
        fig.arrow((left, 578), (right - 18, 578), color=COLORS["orange"])
    fig.arrow((1560, 510), (1560, 418), color=COLORS["green"])
    fig.text(900, 760, "Preserved SHARP contribution", size=28, bold=True, anchor="mm")
    fig.text(
        900,
        825,
        "incremental short-window processing  |  instance-aware context streaming  |  target-conditioned decoding",
        size=24,
        anchor="mm",
    )
    fig.save("sharp_streaming_overview")


def mamba_placements() -> None:
    fig = Figure(
        "Mamba placement hypotheses",
        "The placement changes the sequence semantics even when the state-space block is similar",
    )
    rows = [
        (
            250,
            "Scene-token residual Mamba",
            [
                ("temporal\nencoder", COLORS["blue"]),
                ("max pool", COLORS["line"]),
                ("scene-token\nMamba", COLORS["purple"]),
                ("scene\nattention", COLORS["teal"]),
                ("decoder", COLORS["green"]),
            ],
            "mixes the ordered scene token list after temporal pooling",
        ),
        (
            615,
            "Temporal-agent residual Mamba",
            [
                ("temporal\nblocks 1-2", COLORS["blue"]),
                ("agent-time\nMamba", COLORS["purple"]),
                ("temporal\nblocks 3-4", COLORS["blue"]),
                ("max pool", COLORS["line"]),
                ("decoder", COLORS["green"]),
            ],
            "models each valid agent history before temporal pooling",
        ),
    ]
    for y, heading, stages, explanation in rows:
        fig.text(80, y - 62, heading, size=26, bold=True, anchor="la")
        x = 85
        for index, (label, color) in enumerate(stages):
            fig.box(x, y, 260, 135, label, outline=color, fill=COLORS["light"], size=22, bold=index == 1 and "Mamba" in label)
            if index < len(stages) - 1:
                fig.arrow((x + 260, y + 67), (x + 310, y + 67), color=COLORS["line"])
            x += 330
        fig.text(900, y + 177, explanation, size=21, color=COLORS["muted"], anchor="mm")
    fig.save("mamba_placements")


def horizontal_bars(
    stem: str,
    title: str,
    subtitle: str,
    labels: list[str],
    values: list[float],
    colors: list[str],
    *,
    lower: float,
    upper: float,
    unit: str = "",
) -> None:
    fig = Figure(title, subtitle)
    x0, x1 = 560, 1640
    y0 = 235
    row = 130
    for tick_index in range(6):
        value = lower + (upper - lower) * tick_index / 5
        x = x0 + (x1 - x0) * tick_index / 5
        fig.line([(x, y0 - 35), (x, y0 + row * len(labels) - 30)], color="#d5d9dd", width=2)
        fig.text(x, y0 - 65, f"{value:.2f}", size=18, color=COLORS["muted"], anchor="mm")
    for index, (label, value, color) in enumerate(zip(labels, values, colors)):
        y = y0 + index * row
        fig.text(x0 - 30, y + 35, label, size=22, anchor="rm", wrap=28)
        length = max(3, (value - lower) / (upper - lower) * (x1 - x0))
        fig.rect(x0, y, length, 70, fill=color, outline=color, radius=5)
        value_text = f"{value:.1f}{unit}" if unit else f"{value:.3f}"
        fig.text(
            min(x1 - 12, x0 + length + 18),
            y + 35,
            value_text,
            size=21,
            bold=True,
            color=COLORS["ink"],
            anchor="lm" if x0 + length + 105 < x1 else "rm",
        )
    axis_label = "Lower is better" if not unit else "Elapsed time (hours)"
    fig.text((x0 + x1) / 2, 925, axis_label, size=20, color=COLORS["muted"], anchor="mm")
    fig.save(stem)


def mamba_placement_results() -> None:
    horizontal_bars(
        "mamba_placement_results",
        "Long-run Mamba placement comparison",
        "Best validation minADE6 on Argoverse 2; lower is better",
        [
            "SHARP without Mamba",
            "Residual scene-token Mamba",
            "Residual temporal-agent Mamba",
        ],
        [0.661463, 0.680362, 0.673736],
        [COLORS["blue"], COLORS["purple"], COLORS["orange"]],
        lower=0.64,
        upper=0.70,
    )


def vertical_bar_chart(
    stem: str,
    title: str,
    subtitle: str,
    labels: list[str],
    values: list[float],
    colors: list[str],
    *,
    lower: float,
    upper: float,
) -> None:
    fig = Figure(title, subtitle)
    x0, x1 = 125, 1710
    y_top, y_bottom = 205, 810
    for tick_index in range(6):
        value = lower + (upper - lower) * tick_index / 5
        y = y_bottom - (value - lower) / (upper - lower) * (y_bottom - y_top)
        fig.line([(x0, y), (x1, y)], color="#d5d9dd", width=2)
        precision = 3 if upper - lower < 0.05 else 2
        fig.text(
            x0 - 18,
            y,
            f"{value:.{precision}f}",
            size=18,
            color=COLORS["muted"],
            anchor="rm",
        )
    spacing = (x1 - x0) / len(labels)
    bar_width = spacing * 0.62
    for index, (label, value, color) in enumerate(zip(labels, values, colors)):
        x = x0 + spacing * index + (spacing - bar_width) / 2
        y = y_bottom - (value - lower) / (upper - lower) * (y_bottom - y_top)
        fig.rect(x, y, bar_width, y_bottom - y, fill=color, outline=color, radius=4)
        fig.text(x + bar_width / 2, y - 25, f"{value:.3f}", size=19, bold=True, anchor="mm")
        fig.text(x + bar_width / 2, y_bottom + 42, label, size=18, anchor="mm", wrap=15)
    fig.text(
        x0,
        y_top - 52,
        "minADE6 (m)",
        size=20,
        color=COLORS["muted"],
        anchor="la",
    )
    fig.save(stem)


def lab2_screen() -> None:
    vertical_bar_chart(
        "lab2_screen_minade6",
        "Lab 2 controlled 20-epoch screening study",
        "Validation minADE6 after identical short-run budgets; lower is better",
        [
            "Baseline",
            "Confidence\ngate",
            "Consistency",
            "Learned\npool",
            "Uncertainty",
            "Geometry",
            "Kinematic",
            "Endpoint",
            "Lane",
            "Full\nMamba",
        ],
        [0.752339, 0.755349, 0.776207, 0.757637, 0.749961, 0.749679, 0.785164, 0.755898, 0.754238, 0.780190],
        [
            COLORS["blue"],
            COLORS["teal"],
            COLORS["red"],
            COLORS["orange"],
            COLORS["green"],
            COLORS["gold"],
            COLORS["red"],
            COLORS["orange"],
            COLORS["teal"],
            COLORS["purple"],
        ],
        lower=0.72,
        upper=0.80,
    )


def lab3_attention() -> None:
    vertical_bar_chart(
        "lab3_attention_minade6",
        "Lab 3 attention ablation",
        "Best validation minADE6 under the 80-epoch attention study; lower is better",
        ["Baseline\nMHA", "QKNorm", "Talking\nHeads", "QKNorm +\nTalking Heads"],
        [0.673460, 0.669777, 0.674991, 0.674492],
        [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["purple"]],
        lower=0.665,
        upper=0.680,
    )


def training_time() -> None:
    horizontal_bars(
        "training_time_context",
        "Observed long-run wall-clock context",
        "Approximate end-to-end elapsed time; hardware and interruption history differ between runs",
        [
            "SHARP without Mamba",
            "Lab 3 baseline MHA",
            "Scene-token Mamba",
            "Temporal-agent Mamba",
        ],
        [109.0, 99.65, 129.28, 160.33],
        [COLORS["blue"], COLORS["teal"], COLORS["purple"], COLORS["orange"]],
        lower=80,
        upper=170,
        unit=" h",
    )


def main() -> None:
    experimental_programme()
    sharp_streaming_overview()
    mamba_placements()
    mamba_placement_results()
    lab2_screen()
    lab3_attention()
    training_time()
    print(f"Generated 7 PNG and 7 SVG figures in {OUT}")


if __name__ == "__main__":
    main()
