"""Dependency-free SVG plots used by RBLA analysis scripts."""
from __future__ import annotations

import html
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]


def _scale(value: float, low: float, high: float, start: float, end: float) -> float:
    if high <= low:
        return (start + end) / 2
    return start + (value - low) * (end - start) / (high - low)


def write_line_plot(
    series: Dict[str, Sequence[Tuple[float, float]]],
    path: str | Path,
    *,
    xlabel: str,
    ylabel: str,
    title: str = "",
    width: int = 900,
    height: int = 560,
) -> str:
    path = str(path)
    points = [point for values in series.values() for point in values]
    xs = [float(point[0]) for point in points] or [0.0, 1.0]
    ys = [float(point[1]) for point in points] or [0.0, 1.0]
    x0, x1, y0, y1 = 85, width - 220, height - 70, 55
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    ypad = (ymax - ymin) * 0.08 or 0.05
    ymin, ymax = ymin - ypad, ymax + ypad
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    body += [f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#222"/>', f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#222"/>']
    for tick in range(6):
        fraction = tick / 5
        xv = xmin + fraction * (xmax - xmin)
        xp = _scale(xv, xmin, xmax, x0, x1)
        yv = ymin + fraction * (ymax - ymin)
        yp = _scale(yv, ymin, ymax, y0, y1)
        body.append(f'<line x1="{xp:.1f}" y1="{y0}" x2="{xp:.1f}" y2="{y0 + 5}" stroke="#555"/><text x="{xp:.1f}" y="{y0 + 22}" text-anchor="middle" font-size="11">{xv:.2f}</text>')
        body.append(f'<line x1="{x0 - 5}" y1="{yp:.1f}" x2="{x0}" y2="{yp:.1f}" stroke="#555"/><text x="{x0 - 9}" y="{yp + 4:.1f}" text-anchor="end" font-size="11">{yv:.3g}</text>')
        body.append(f'<line x1="{x0}" y1="{yp:.1f}" x2="{x1}" y2="{yp:.1f}" stroke="#ddd" stroke-width="0.7"/>')
    for index, (label, values) in enumerate(series.items()):
        color = COLORS[index % len(COLORS)]
        coords = [(_scale(float(x), xmin, xmax, x0, x1), _scale(float(y), ymin, ymax, y0, y1)) for x, y in values]
        polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
        body.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in coords:
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" fill="{color}"/>')
        ly = 75 + index * 20
        body.append(f'<line x1="{x1 + 18}" y1="{ly}" x2="{x1 + 42}" y2="{ly}" stroke="{color}" stroke-width="3"/><text x="{x1 + 48}" y="{ly + 4}" font-size="10">{html.escape(label)}</text>')
    body.append(f'<text x="{(x0 + x1) / 2}" y="{height - 20}" text-anchor="middle" font-size="14">{html.escape(xlabel)}</text>')
    body.append(f'<text x="20" y="{(y0 + y1) / 2}" transform="rotate(-90 20 {(y0 + y1) / 2})" text-anchor="middle" font-size="14">{html.escape(ylabel)}</text>')
    body.append(f'<text x="{(x0 + x1) / 2}" y="26" text-anchor="middle" font-size="17" font-weight="bold">{html.escape(title)}</text></svg>')
    Path(path).write_text("".join(body), encoding="utf-8")
    return path


def write_scatter_plot(
    points: Sequence[Tuple[float, float, float]],
    path: str | Path,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    width: int = 760,
    height: int = 540,
) -> str:
    path = str(path)
    xs = [p[0] for p in points] or [0.0, 1.0]
    ys = [p[1] for p in points] or [0.0, 1.0]
    x0, x1, y0, y1 = 80, width - 60, height - 65, 50
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    ypad = (ymax - ymin) * 0.08 or 0.05
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="white"/>', f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#222"/>', f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#222"/>']
    for x, y, color_value in points:
        xp = _scale(x, xmin, xmax, x0, x1)
        yp = _scale(y, ymin - ypad, ymax + ypad, y0, y1)
        blue = int(255 * max(0.0, min(1.0, color_value)))
        red = 255 - blue
        body.append(f'<circle cx="{xp:.1f}" cy="{yp:.1f}" r="4" fill="rgb({red},80,{blue})" fill-opacity="0.7"/>')
    body += [f'<text x="{(x0+x1)/2}" y="{height-18}" text-anchor="middle" font-size="14">{html.escape(xlabel)}</text>', f'<text x="18" y="{(y0+y1)/2}" transform="rotate(-90 18 {(y0+y1)/2})" text-anchor="middle" font-size="14">{html.escape(ylabel)}</text>', f'<text x="{width/2}" y="25" text-anchor="middle" font-size="17" font-weight="bold">{html.escape(title)}</text></svg>']
    Path(path).write_text("".join(body), encoding="utf-8")
    return path


def write_confusion_heatmap(matrix: Sequence[Sequence[int]], path: str | Path, *, title: str) -> str:
    path = str(path)
    n = len(matrix)
    cell = 42
    left, top = 70, 55
    width, height = left + n * cell + 40, top + n * cell + 70
    maximum = max((max(row) for row in matrix), default=1) or 1
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="white"/>', f'<text x="{width/2}" y="25" text-anchor="middle" font-size="16" font-weight="bold">{html.escape(title)}</text>']
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            intensity = int(245 - 190 * value / maximum)
            color = f"rgb({intensity},{intensity},{255})"
            x, y = left + j * cell, top + i * cell
            body.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}" stroke="white"/><text x="{x+cell/2}" y="{y+cell/2+4}" text-anchor="middle" font-size="10">{value}</text>')
    for i in range(n):
        body.append(f'<text x="{left+i*cell+cell/2}" y="{top+n*cell+18}" text-anchor="middle" font-size="11">{i}</text><text x="{left-12}" y="{top+i*cell+cell/2+4}" text-anchor="middle" font-size="11">{i}</text>')
    body.append(f'<text x="{left+n*cell/2}" y="{height-18}" text-anchor="middle" font-size="13">Predicted class</text><text x="18" y="{top+n*cell/2}" transform="rotate(-90 18 {top+n*cell/2})" text-anchor="middle" font-size="13">True class</text></svg>')
    Path(path).write_text("".join(body), encoding="utf-8")
    return path
