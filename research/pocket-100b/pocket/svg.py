"""Tiny SVG plotting, because a research note with no figures does not travel.

No dependencies, no fonts, no rasterisation: axes, gridlines, polylines, legends
and labels, sized in user units.  Enough for the two chart shapes this study
needs (log-log lines, grouped bars).
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

PALETTE = ["#0b6bcb", "#c2410c", "#15803d", "#7c3aed", "#b91c1c", "#0891b2", "#a16207"]
INK = "#1f2933"
MUTED = "#7b8794"
GRID = "#e4e7eb"


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def _nice_ticks(lo: float, hi: float, count: int = 5) -> List[float]:
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / max(1, count)
    mag = 10 ** math.floor(math.log10(raw))
    for mult in (1, 2, 2.5, 5, 10):
        step = mag * mult
        if step >= raw:
            break
    start = math.ceil(lo / step) * step
    ticks = []
    v = start
    while v <= hi + step * 1e-9:
        ticks.append(round(v, 10))
        v += step
    return ticks


def _fmt(v: float) -> str:
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1e9:
        return f"{v/1e9:g}B"
    if a >= 1e6:
        return f"{v/1e6:g}M"
    if a >= 1e3:
        return f"{v/1e3:g}k"
    if a >= 1:
        return f"{v:g}"
    return f"{v:.3g}"


class Chart:
    def __init__(self, width: int = 760, height: int = 460, title: str = "", subtitle: str = ""):
        self.w = width
        self.h = height
        self.title = title
        self.subtitle = subtitle
        self.body: List[str] = []
        self.pad = {"l": 78, "r": 168, "t": 64, "b": 62}

    # -- coordinate transforms -------------------------------------------- #
    def _scale(self, x0, x1, y0, y1, xlog, ylog):
        pl, pr, pt, pb = self.pad["l"], self.pad["r"], self.pad["t"], self.pad["b"]
        def fx(x):
            if xlog:
                t = (math.log10(x) - math.log10(x0)) / (math.log10(x1) - math.log10(x0))
            else:
                t = (x - x0) / (x1 - x0)
            return pl + t * (self.w - pl - pr)
        def fy(y):
            if ylog:
                t = (math.log10(y) - math.log10(y0)) / (math.log10(y1) - math.log10(y0))
            else:
                t = (y - y0) / (y1 - y0)
            return self.h - pb - t * (self.h - pt - pb)
        return fx, fy

    def line(
        self,
        series: Sequence[Tuple[str, Sequence[Tuple[float, float]]]],
        xlabel: str,
        ylabel: str,
        xlog: bool = True,
        ylog: bool = True,
        annotations: Sequence[Tuple[float, float, str]] = (),
    ) -> "Chart":
        xs = [p[0] for _, pts in series for p in pts if p[0] > 0]
        ys = [p[1] for _, pts in series for p in pts if p[1] > 0]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        if ylog:
            y0 = max(y0, 0.01)
        x0, x1 = x0 / 1.25, x1 * 1.25
        if ylog:
            y0, y1 = y0 / 1.6, y1 * 1.6
        else:
            y0, y1 = 0, y1 * 1.15
        fx, fy = self._scale(x0, x1, y0, y1, xlog, ylog)

        # grid + ticks
        if xlog:
            xticks = [10 ** e for e in range(math.floor(math.log10(x0)), math.ceil(math.log10(x1)) + 1)]
        else:
            xticks = _nice_ticks(x0, x1)
        if ylog:
            yticks = [10 ** e for e in range(math.floor(math.log10(y0)), math.ceil(math.log10(y1)) + 1)]
        else:
            yticks = _nice_ticks(y0, y1)
        for t in xticks:
            if not (x0 <= t <= x1):
                continue
            x = fx(t)
            self.body.append(f'<line x1="{x:.1f}" y1="{self.pad["t"]}" x2="{x:.1f}" y2="{self.h-self.pad["b"]}" stroke="{GRID}"/>')
            self.body.append(f'<text x="{x:.1f}" y="{self.h-self.pad["b"]+18}" fill="{MUTED}" font-size="11" text-anchor="middle">{_esc(_fmt(t))}</text>')
        for t in yticks:
            if not (y0 <= t <= y1):
                continue
            y = fy(t)
            self.body.append(f'<line x1="{self.pad["l"]}" y1="{y:.1f}" x2="{self.w-self.pad["r"]}" y2="{y:.1f}" stroke="{GRID}"/>')
            self.body.append(f'<text x="{self.pad["l"]-10}" y="{y+4:.1f}" fill="{MUTED}" font-size="11" text-anchor="end">{_esc(_fmt(t))}</text>')

        # axes
        self.body.append(
            f'<rect x="{self.pad["l"]}" y="{self.pad["t"]}" width="{self.w-self.pad["l"]-self.pad["r"]}" '
            f'height="{self.h-self.pad["t"]-self.pad["b"]}" fill="none" stroke="{MUTED}" stroke-width="1"/>'
        )
        self.body.append(
            f'<text x="{(self.pad["l"]+self.w-self.pad["r"])/2}" y="{self.h-14}" fill="{INK}" font-size="12" text-anchor="middle">{_esc(xlabel)}</text>'
        )
        self.body.append(
            f'<text x="20" y="{(self.pad["t"]+self.h-self.pad["b"])/2}" fill="{INK}" font-size="12" text-anchor="middle" transform="rotate(-90 20 {(self.pad["t"]+self.h-self.pad["b"])/2})">{_esc(ylabel)}</text>'
        )

        # series
        for idx, (label, pts) in enumerate(series):
            color = PALETTE[idx % len(PALETTE)]
            d = " ".join(f"{'M' if i == 0 else 'L'}{fx(x):.1f},{fy(y):.1f}" for i, (x, y) in enumerate(pts))
            self.body.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.2" stroke-linejoin="round"/>')
            lx = self.w - self.pad["r"] + 14
            ly = self.pad["t"] + 14 + idx * 19
            self.body.append(f'<line x1="{lx}" y1="{ly-4}" x2="{lx+16}" y2="{ly-4}" stroke="{color}" stroke-width="2.6"/>')
            self.body.append(f'<text x="{lx+22}" y="{ly}" fill="{INK}" font-size="11">{_esc(label)}</text>')

        for x, y, text in annotations:
            if x <= 0 or y <= 0:
                continue
            self.body.append(f'<circle cx="{fx(x):.1f}" cy="{fy(y):.1f}" r="3" fill="{INK}"/>')
            self.body.append(f'<text x="{fx(x)+7:.1f}" y="{fy(y)-6:.1f}" fill="{INK}" font-size="10">{_esc(text)}</text>')
        return self

    def bars(
        self,
        rows: Sequence[Tuple[str, float]],
        xlabel: str,
        ylabel: str,
        log: bool = False,
        value_suffix: str = "",
    ) -> "Chart":
        n = len(rows)
        pl, pr, pt, pb = self.pad["l"], self.pad["r"] - 120, self.pad["t"], self.pad["b"]
        plot_w = self.w - pl - pr
        vmax = max(v for _, v in rows) or 1.0
        if log:
            vmin = min(v for _, v in rows) or 0.01
        slot = plot_w / max(1, n)
        width = slot * 0.62
        base = self.h - pb
        for i, (label, v) in enumerate(rows):
            if log:
                frac = (math.log10(max(v, 1e-3)) - math.log10(max(min(x for _, x in rows), 1e-3))) / max(
                    1e-9, math.log10(vmax) - math.log10(max(min(x for _, x in rows), 1e-3))
                )
                frac = 0.12 + 0.88 * max(0.02, frac)
            else:
                frac = v / vmax
            hgt = frac * (self.h - pt - pb)
            x = pl + i * slot + (slot - width) / 2
            color = PALETTE[i % len(PALETTE)]
            self.body.append(
                f'<rect x="{x:.1f}" y="{base-hgt:.1f}" width="{width:.1f}" height="{hgt:.1f}" fill="{color}" rx="3"/>'
            )
            self.body.append(
                f'<text x="{x+width/2:.1f}" y="{base-hgt-8:.1f}" fill="{INK}" font-size="11" text-anchor="middle">{_esc(_fmt(v))}{_esc(value_suffix)}</text>'
            )
            self.body.append(
                f'<text x="{x+width/2:.1f}" y="{base+18:.1f}" fill="{MUTED}" font-size="10" text-anchor="middle">{_esc(label)}</text>'
            )
        self.body.append(
            f'<line x1="{pl}" y1="{base}" x2="{self.w-pr}" y2="{base}" stroke="{MUTED}"/>'
        )
        self.body.append(
            f'<text x="{pl+plot_w/2:.1f}" y="{self.h-12}" fill="{INK}" font-size="12" text-anchor="middle">{_esc(xlabel)}</text>'
        )
        self.body.append(
            f'<text x="20" y="{(pt+base)/2:.1f}" fill="{INK}" font-size="12" text-anchor="middle" transform="rotate(-90 20 {(pt+base)/2:.1f})">{_esc(ylabel)}</text>'
        )
        return self

    def render(self) -> str:
        head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
            f'viewBox="0 0 {self.w} {self.h}" font-family="ui-sans-serif, -apple-system, Segoe UI, Roboto, Helvetica, Arial">'
            f'<rect width="{self.w}" height="{self.h}" fill="#ffffff"/>'
        )
        title = f'<text x="{self.pad["l"]}" y="30" fill="{INK}" font-size="16" font-weight="600">{_esc(self.title)}</text>'
        sub = f'<text x="{self.pad["l"]}" y="49" fill="{MUTED}" font-size="11">{_esc(self.subtitle)}</text>'
        return head + title + sub + "".join(self.body) + "</svg>"

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.render() + "\n")
