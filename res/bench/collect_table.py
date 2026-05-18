#!/usr/bin/env python3
"""Collect benchmark CSV files into one combined f128/f256 table."""

from __future__ import annotations

import argparse
import csv
import html
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BENCH_FILE_RE = re.compile(r"(?P<compiler>.+)_(?P<type>f128|f256)_typical_ratios\.csv$", re.IGNORECASE)

FP_TYPE_ORDER = {"f128": 0, "f256": 1}
FP_TYPE_LABELS = {"f128": "f128", "f256": "f256"}

PLATFORM_ORDER = {"windows": 0, "linux": 1, "macos": 2, "web": 3}
PLATFORM_LABELS = {"windows": "Windows", "linux": "Linux", "macos": "MacOS", "web": "Web"}

COMPILER_ORDER = {
    "msvc": 0,
    "mingw": 1,
    "gcc": 2,
    "clang": 3,
    "appleclang": 4,
    "emscripten": 5,
    "emcc": 5,
}
COMPILER_LABELS = {
    "msvc": "MSVC",
    "mingw": "MinGW",
    "gcc": "GCC",
    "clang": "Clang",
    "appleclang": "AppleClang",
    "emscripten": "Emscripten",
    "emcc": "Emscripten",
}

HEADER_BG = "#334155"
PLATFORM_HEADER_BG = "#475569"
SUBHEADER_BG = "#64748B"
GROUP_BG = "#E5E7EB"
GRID = "#D1D5DB"
TEXT = "#111827"
MUTED = "#4B5563"
WHITE = "#FFFFFF"
TYPE_SEPARATOR = "#F8FAFC"
SLOW_RATIO_COLOR = "#FF0000"
EVEN_RATIO_COLOR = "#D9F99D"
FAST_RATIO_COLOR = "#00B050"
FAST_RATIO_CEILING = 10.0
NATIVE_ARITHMETIC_GROUP = "Arithmetic"
NATIVE_ARITHMETIC_LABELS = {"add", "subtract", "multiply", "divide"}
VISIBLE_ARITHMETIC_LABELS = {"mixed recurrence"}
ARITHMETIC_GROUP_RE = re.compile(r"^f(?P<bits>128|256) <-> (?P<rhs>f128|f256|f64|f32|i64|i32)$")
HIDDEN_GROUPS = {"Mandelbrot"}
HIDDEN_ROWS: set[tuple[str, str]] = set()


@dataclass(frozen=True)
class ColumnKey:
    fp_type: str
    platform: str
    compiler: str


@dataclass
class BenchmarkCell:
    ns_per_iter: float | None = None
    ratio: float | None = None


@dataclass
class BenchmarkTable:
    columns: list[ColumnKey]
    groups: list[str]
    rows_by_group: dict[str, list[str]]
    cells: dict[tuple[str, str, ColumnKey], BenchmarkCell]


def normalize_key(value: str) -> str:
    return value.strip().replace("-", "").replace("_", "").lower()


def platform_key(path: Path) -> str:
    key = normalize_key(path.parent.name)
    if key in {"mac", "darwin", "osx"}:
        return "macos"
    if key in {"emscripten", "wasm"}:
        return "web"
    return key


def platform_label(key: str) -> str:
    return PLATFORM_LABELS.get(key, key[:1].upper() + key[1:])


def compiler_key(value: str) -> str:
    return normalize_key(value)


def fp_type_label(key: str) -> str:
    return FP_TYPE_LABELS.get(key, key)


def compiler_label(key: str) -> str:
    return COMPILER_LABELS.get(key, key)


def column_sort_key(column: ColumnKey) -> tuple[int, str, int, str, int, str]:
    return (
        FP_TYPE_ORDER.get(column.fp_type, 100),
        fp_type_label(column.fp_type).lower(),
        PLATFORM_ORDER.get(column.platform, 100),
        platform_label(column.platform).lower(),
        COMPILER_ORDER.get(column.compiler, 100),
        compiler_label(column.compiler).lower(),
    )


def find_column(fieldnames: Iterable[str], *candidates: str) -> str | None:
    field_map = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in field_map:
            return field_map[candidate.lower()]
    return None


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_csv(path: Path, fp_type: str) -> list[tuple[str, str, BenchmarkCell]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return []

        candidate_column = find_column(reader.fieldnames, f"{fp_type}_ns_per_iter")
        if candidate_column is None:
            for name in reader.fieldnames:
                lowered = name.lower()
                if lowered.endswith("_ns_per_iter") and not lowered.startswith(("mpfr_", "std_")):
                    candidate_column = name
                    break

        ratio_column = find_column(reader.fieldnames, f"mpfr_to_{fp_type}_ratio")
        if ratio_column is None:
            for name in reader.fieldnames:
                if name.lower().endswith("_ratio"):
                    ratio_column = name
                    break

        if candidate_column is None:
            return []

        out: list[tuple[str, str, BenchmarkCell]] = []
        for row in reader:
            group = (row.get("group") or "").strip() or "Benchmarks"
            label = (row.get("label") or "").strip()
            if not label:
                continue
            out.append((
                group,
                label,
                BenchmarkCell(parse_float(row.get(candidate_column)), parse_float(row.get(ratio_column)) if ratio_column else None),
            ))
        return out


def visible_table_entry(fp_type: str, group: str, label: str) -> tuple[str, str] | None:
    arithmetic_match = ARITHMETIC_GROUP_RE.match(group)
    if arithmetic_match:
        native_group = f"{fp_type} <-> {fp_type}"
        if group == native_group and label in NATIVE_ARITHMETIC_LABELS:
            return NATIVE_ARITHMETIC_GROUP, label
        return None

    if group == NATIVE_ARITHMETIC_GROUP and label not in VISIBLE_ARITHMETIC_LABELS:
        return None

    if group in HIDDEN_GROUPS or (group, label) in HIDDEN_ROWS:
        return None

    return group, label


def discover_table(bench_root: Path) -> BenchmarkTable:
    columns: set[ColumnKey] = set()
    groups: list[str] = []
    rows_by_group: dict[str, list[str]] = {}
    cells: dict[tuple[str, str, ColumnKey], BenchmarkCell] = {}

    for path in sorted(bench_root.rglob("*.csv")):
        match = BENCH_FILE_RE.match(path.name)
        if not match:
            continue

        fp_type = match.group("type").lower()
        column = ColumnKey(fp_type, platform_key(path), compiler_key(match.group("compiler")))
        entries = read_csv(path, fp_type)
        has_visible_entries = False
        for group, label, cell in entries:
            visible_entry = visible_table_entry(fp_type, group, label)
            if visible_entry is None:
                continue

            visible_group, visible_label = visible_entry
            has_visible_entries = True
            if visible_group not in rows_by_group:
                groups.append(visible_group)
                rows_by_group[visible_group] = []
            if visible_label not in rows_by_group[visible_group]:
                rows_by_group[visible_group].append(visible_label)
            cells[(visible_group, visible_label, column)] = cell

        if has_visible_entries:
            columns.add(column)

    return BenchmarkTable(
        columns=sorted(columns, key=column_sort_key),
        groups=groups,
        rows_by_group=rows_by_group,
        cells=cells,
    )


def mix_channel(a: int, b: int, t: float) -> int:
    return round(a + (b - a) * t)


def mix_color(a: str, b: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    ar, ag, ab = int(a[1:3], 16), int(a[3:5], 16), int(a[5:7], 16)
    br, bg, bb = int(b[1:3], 16), int(b[3:5], 16), int(b[5:7], 16)
    return f"#{mix_channel(ar, br, t):02X}{mix_channel(ag, bg, t):02X}{mix_channel(ab, bb, t):02X}"


def ratio_color(ratio: float | None) -> str:
    if ratio is None:
        return WHITE
    if ratio <= 0.0:
        return SLOW_RATIO_COLOR
    if ratio < 1.0:
        return mix_color(SLOW_RATIO_COLOR, EVEN_RATIO_COLOR, ratio)
    if ratio < FAST_RATIO_CEILING:
        t = math.log(ratio) / math.log(FAST_RATIO_CEILING)
        return mix_color(EVEN_RATIO_COLOR, FAST_RATIO_COLOR, t)
    return FAST_RATIO_COLOR


def text_color(background: str) -> str:
    red = int(background[1:3], 16) / 255.0
    green = int(background[3:5], 16) / 255.0
    blue = int(background[5:7], 16) / 255.0
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "#000000" if luminance >= 0.45 else "#FFFFFF"


def format_ns(value: float | None) -> str:
    if value is None:
        return ""
    if value >= 100.0:
        return f"{value:,.0f}ns"
    if value >= 10.0:
        return f"{value:.1f}ns"
    if value >= 1.0:
        return f"{value:.2f}ns"
    return f"{value:.3f}ns"


def format_ratio(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.2f}x"


def e(value: str) -> str:
    return html.escape(value, quote=True)


def is_type_boundary(columns: list[ColumnKey], index: int) -> bool:
    return index > 0 and columns[index].fp_type != columns[index - 1].fp_type


def type_boundary_class(columns: list[ColumnKey], index: int) -> str:
    return " type-start" if is_type_boundary(columns, index) else ""


def type_boundary_indices(columns: list[ColumnKey]) -> list[int]:
    return [index for index in range(1, len(columns)) if is_type_boundary(columns, index)]


def render_type_header(columns: list[ColumnKey]) -> str:
    parts = ["<tr>", '<th class="function" rowspan="3">Function</th>']
    index = 0
    while index < len(columns):
        fp_type = columns[index].fp_type
        count = 1
        while index + count < len(columns) and columns[index + count].fp_type == fp_type:
            count += 1
        classes = f"type{type_boundary_class(columns, index)}"
        parts.append(f'<th class="{classes}" colspan="{count}">{e(fp_type_label(fp_type))}</th>')
        index += count
    parts.append("</tr>")
    return "".join(parts)


def render_platform_header(columns: list[ColumnKey]) -> str:
    parts = ["<tr>"]
    index = 0
    while index < len(columns):
        fp_type = columns[index].fp_type
        platform = columns[index].platform
        count = 1
        while (
            index + count < len(columns)
            and columns[index + count].fp_type == fp_type
            and columns[index + count].platform == platform
        ):
            count += 1
        classes = f"platform{type_boundary_class(columns, index)}"
        parts.append(f'<th class="{classes}" colspan="{count}">{e(platform_label(platform))}</th>')
        index += count
    parts.append("</tr>")
    return "".join(parts)


def render_compiler_header(columns: list[ColumnKey]) -> str:
    return "<tr>" + "".join(
        f'<th class="compiler{type_boundary_class(columns, index)}">{e(compiler_label(column.compiler))}</th>'
        for index, column in enumerate(columns)
    ) + "</tr>"


def render_html_cell(cell: BenchmarkCell | None, boundary_class: str = "") -> str:
    if cell is None or cell.ns_per_iter is None:
        return f'<td class="missing{boundary_class}"></td>'
    if cell.ratio is None:
        return f'<td class="result no-ratio{boundary_class}"><div class="timer">{e(format_ns(cell.ns_per_iter))}</div></td>'

    background = ratio_color(cell.ratio)
    color = text_color(background)
    return (
        f'<td class="result{boundary_class}" style="background:{background};color:{color}">'
        f'<div class="timer">{e(format_ns(cell.ns_per_iter))}</div>'
        f'<div class="ratio">{e(format_ratio(cell.ratio))}</div>'
        "</td>"
    )


def render_html_table(table: BenchmarkTable) -> str:
    if not table.columns:
        return '<p class="empty">No benchmark CSV files were found for this type.</p>'

    column_count = len(table.columns) + 1
    parts = [
        '<table class="bench-table">',
        "<thead>",
        render_type_header(table.columns),
        render_platform_header(table.columns),
        render_compiler_header(table.columns),
        "</thead>",
        "<tbody>",
    ]
    for group in table.groups:
        parts.append(f'<tr class="group-row"><th colspan="{column_count}">{e(group)}</th></tr>')
        for label in table.rows_by_group.get(group, []):
            parts.append("<tr>")
            parts.append(f'<th class="function">{e(label)}</th>')
            for index, column in enumerate(table.columns):
                parts.append(render_html_cell(table.cells.get((group, label, column)), type_boundary_class(table.columns, index)))
            parts.append("</tr>")
    parts.extend(["</tbody>", "</table>"])
    return "\n".join(parts)


def render_html_document(table: BenchmarkTable) -> str:
    title = "fltx Benchmark Table"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(title)}</title>
<style>
:root {{ color-scheme: light; font-family: Inter, Segoe UI, Roboto, Helvetica, Arial, sans-serif; background: #f5f7fb; color: {TEXT}; }}
body {{ margin: 0; padding: 32px; }}
h1 {{ margin: 0 0 8px; font-size: 28px; }}
.note {{ max-width: 960px; margin: 0 0 24px; color: {MUTED}; line-height: 1.45; }}
.table-wrap {{ overflow-x: auto; }}
.bench-table {{ border-collapse: collapse; min-width: 760px; background: {WHITE}; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.12); font-size: 13px; }}
.bench-table th, .bench-table td {{ border: 1px solid {GRID}; padding: 6px 10px; text-align: center; vertical-align: middle; }}
.bench-table thead th {{ background: {HEADER_BG}; color: {WHITE}; font-weight: 700; }}
.bench-table thead th.platform {{ background: {PLATFORM_HEADER_BG}; }}
.bench-table thead th.compiler {{ background: {SUBHEADER_BG}; }}
.bench-table th.type-start, .bench-table td.type-start {{ border-left: 6px solid {TYPE_SEPARATOR} !important; }}
.bench-table th.function {{ min-width: 190px; text-align: left; white-space: nowrap; }}
.bench-table tbody th.function {{ font-size: 14px; }}
.group-row th {{ background: {GROUP_BG}; color: {TEXT}; text-align: left; font-size: 13px; letter-spacing: 0.02em; }}
.result {{ min-width: 92px; font-weight: 700; }}
.result.no-ratio, .missing {{ background: {WHITE}; color: #000000; }}
.timer {{ line-height: 1.05; }}
.ratio {{ margin-top: 2px; font-size: 11px; line-height: 1.05; opacity: 0.88; }}
.empty {{ color: {MUTED}; }}
</style>
</head>
<body>
<h1>{e(title)}</h1>
<p class="note">Each cell shows typical nanoseconds per iteration, then the reference-to-fltx speed ratio when present. Missing ratios are left white and show timing only.</p>
<div class="table-wrap">
{render_html_table(table)}
</div>
</body>
</html>
"""


def svg_text(x: int, y: int, text: str, size: int, color: str, weight: int = 400, anchor: str = "start") -> str:
    return (
        f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" font-weight="{weight}" '
        f'font-family="Segoe UI, Arial, sans-serif" text-anchor="{anchor}">{e(text)}</text>'
    )


def svg_rect(x: int, y: int, width: int, height: int, fill: str, stroke: str = GRID) -> str:
    return f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{fill}" stroke="{stroke}" stroke-width="1"/>'


def svg_line(x1: int, y1: int, x2: int, y2: int, color: str, width: int) -> str:
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" stroke-linecap="square"/>'


def table_row_count(table: BenchmarkTable) -> int:
    return sum(1 + len(table.rows_by_group.get(group, [])) for group in table.groups)


def render_svg_table(table: BenchmarkTable) -> str:
    title_h = 62
    header_h = 34
    header_rows = 3
    group_h = 28
    row_h = 40
    margin = 18

    longest_label = max([len("Function")] + [len(label) for group in table.groups for label in table.rows_by_group.get(group, [])])
    function_w = max(210, min(360, longest_label * 8 + 34))
    column_w = 116
    width = margin * 2 + function_w + column_w * max(1, len(table.columns))
    height = margin * 2 + title_h + header_h * header_rows + group_h * len(table.groups) + row_h * sum(len(table.rows_by_group.get(group, [])) for group in table.groups)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">fltx benchmark table</title>',
        '<desc id="desc">Benchmark table showing nanoseconds per iteration and reference-to-fltx ratio by platform and compiler.</desc>',
        f'<rect width="{width}" height="{height}" fill="#F5F7FB"/>',
        svg_text(margin, margin + 26, "fltx Benchmark Results", 24, TEXT, 700),
        svg_text(margin, margin + 50, "Typical ns/iter with reference-to-fltx ratio underneath", 13, MUTED),
    ]

    if not table.columns:
        parts.append(svg_text(margin, margin + title_h + 32, "No benchmark CSV files were found for this table.", 14, MUTED))
        parts.append("</svg>")
        return "\n".join(parts)

    table_x = margin
    y = margin + title_h
    table_top = y
    parts.append(svg_rect(table_x, y, function_w, header_h * header_rows, HEADER_BG))
    parts.append(svg_text(table_x + 14, y + 60, "Function", 13, WHITE, 700))

    x = table_x + function_w
    index = 0
    while index < len(table.columns):
        fp_type = table.columns[index].fp_type
        count = 1
        while index + count < len(table.columns) and table.columns[index + count].fp_type == fp_type:
            count += 1
        span_w = count * column_w
        parts.append(svg_rect(x, y, span_w, header_h, HEADER_BG))
        parts.append(svg_text(x + span_w // 2, y + 22, fp_type_label(fp_type), 13, WHITE, 700, "middle"))
        x += span_w
        index += count

    y += header_h
    x = table_x + function_w
    index = 0
    while index < len(table.columns):
        fp_type = table.columns[index].fp_type
        platform = table.columns[index].platform
        count = 1
        while (
            index + count < len(table.columns)
            and table.columns[index + count].fp_type == fp_type
            and table.columns[index + count].platform == platform
        ):
            count += 1
        span_w = count * column_w
        parts.append(svg_rect(x, y, span_w, header_h, PLATFORM_HEADER_BG))
        parts.append(svg_text(x + span_w // 2, y + 22, platform_label(platform), 13, WHITE, 700, "middle"))
        x += span_w
        index += count

    y += header_h
    x = table_x + function_w
    for column in table.columns:
        parts.append(svg_rect(x, y, column_w, header_h, SUBHEADER_BG))
        parts.append(svg_text(x + column_w // 2, y + 22, compiler_label(column.compiler), 13, WHITE, 700, "middle"))
        x += column_w

    y += header_h
    for group in table.groups:
        total_w = function_w + column_w * len(table.columns)
        parts.append(svg_rect(table_x, y, total_w, group_h, GROUP_BG))
        parts.append(svg_text(table_x + 14, y + 19, group, 12, TEXT, 700))
        y += group_h

        for label in table.rows_by_group.get(group, []):
            parts.append(svg_rect(table_x, y, function_w, row_h, WHITE))
            parts.append(svg_text(table_x + 14, y + 25, label, 13, TEXT, 600))
            x = table_x + function_w
            for column in table.columns:
                cell = table.cells.get((group, label, column))
                if cell is None or cell.ns_per_iter is None:
                    parts.append(svg_rect(x, y, column_w, row_h, WHITE))
                elif cell.ratio is None:
                    parts.append(svg_rect(x, y, column_w, row_h, WHITE))
                    parts.append(svg_text(x + column_w // 2, y + 25, format_ns(cell.ns_per_iter), 13, "#000000", 700, "middle"))
                else:
                    background = ratio_color(cell.ratio)
                    color = text_color(background)
                    parts.append(svg_rect(x, y, column_w, row_h, background))
                    parts.append(svg_text(x + column_w // 2, y + 17, format_ns(cell.ns_per_iter), 13, color, 700, "middle"))
                    parts.append(svg_text(x + column_w // 2, y + 33, format_ratio(cell.ratio), 11, color, 600, "middle"))
                x += column_w
            y += row_h

    for index in type_boundary_indices(table.columns):
        separator_x = table_x + function_w + index * column_w
        parts.append(svg_line(separator_x, table_top, separator_x, y, TYPE_SEPARATOR, 6))

    parts.append("</svg>")
    return "\n".join(parts)


def default_res_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_bench_root(res_root: Path) -> Path:
    if res_root.name.lower() == "bench":
        return res_root
    return res_root / "bench"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect fltx benchmark CSV files into one combined f128/f256 table file.")
    parser.add_argument("--res-root", type=Path, default=default_res_root(), help="fltx/res root folder. Defaults to this script's parent res folder.")
    parser.add_argument("--root", type=Path, default=None, help="Deprecated alias for --res-root.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults to <res-root>/bench.")
    parser.add_argument("--format", choices=("html", "svg", "both"), default="both", help="Output format.")
    return parser.parse_args()


def write_outputs(table: BenchmarkTable, output_dir: Path, output_format: str) -> list[Path]:
    written: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_format in {"html", "both"}:
        path = output_dir / "benchmark_table.html"
        path.write_text(render_html_document(table), encoding="utf-8")
        written.append(path)
    if output_format in {"svg", "both"}:
        path = output_dir / "benchmark_table.svg"
        path.write_text(render_svg_table(table), encoding="utf-8")
        written.append(path)
    return written


def main() -> None:
    args = parse_args()
    res_root = (args.root if args.root is not None else args.res_root).resolve()
    bench_root = resolve_bench_root(res_root)
    output_dir = args.output_dir.resolve() if args.output_dir else bench_root

    table = discover_table(bench_root)
    for path in write_outputs(table, output_dir, args.format):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
