#!/usr/bin/env python3
"""Collect metrics CSV files into one combined bl::f128/bl::f256 table."""

from __future__ import annotations

import argparse
import csv
import html
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


METRICS_FILE_RE = re.compile(r"(?P<compiler>.+)_(?P<type>f128|f256)\.csv$", re.IGNORECASE)

TABLE_TITLE = "fltx metrics"
TABLE_SUBTITLE = "Nanoseconds per iteration for bl::f128/bl::f256 metrics, colored by speed ratio vs the stronger available reference."

FP_TYPE_ORDER = {"f128": 0, "f256": 1}
FP_TYPE_LABELS = {"f128": "bl::f128", "f256": "bl::f256"}

PLATFORM_ORDER = {
    "windows": 0,
    "linux": 1,
    "macos": 2,
    "wasm32": 3,
}
PLATFORM_LABELS = {
    "windows": "Windows",
    "linux": "Linux",
    "macos": "MacOS",
    "wasm32": "WebAssembly",
}

COMPILER_ORDER = {
    "msvc": 0,
    "mingw": 1,
    "gcc": 2,
    "clang": 3,
    "appleclang": 4,
    "nodejs": 5,
    "node": 5,
    "chrome": 6,
    "browser": 6,
    "wasm32": 7,
}
COMPILER_LABELS = {
    "msvc": "MSVC",
    "mingw": "MinGW",
    "gcc": "GCC",
    "clang": "Clang",
    "appleclang": "AppleClang",
    "nodejs": "Node.js",
    "node": "Node.js",
    "chrome": "Chrome",
    "browser": "Chrome",
    "wasm32": "Wasm32",
}

HEADER_BG = "#3B4B63"
PLATFORM_HEADER_BG = "#475569"
SUBHEADER_BG = "#64748B"
GROUP_BG = "#C9CDD6"
GRID = "#D1D5DB"
TEXT = "#111827"
MUTED = "#4B5563"
WHITE = "#FFFFFF"
PAGE_BG = "#F5F7FB"
TYPE_GAP_WIDTH = 16
FUNCTION_COLUMN_MIN_WIDTH = 210
FUNCTION_COLUMN_MAX_WIDTH = 720
FUNCTION_COLUMN_CHAR_WIDTH = 8
FUNCTION_COLUMN_PADDING = 34
FUNCTION_COLUMN_WIDTH_SCALE = 0.75
RATIO_COLOR_STOPS: tuple[tuple[float, str], ...] = (
    (0.10, "#FF0000"),
    (0.50, "#FA5B11"),
    (0.95, "#D7DA81"),
    (1.00, "#B0EA86"),
    (2.00, "#74D673"),
    (3.00, "#58CD6B"),
    (4.00, "#40C564"),
    (8.00, "#0DB454"),
    (20.00, "#009D4A"),
)
NATIVE_ARITHMETIC_GROUP = "Arithmetic"
NATIVE_ARITHMETIC_LABELS = {"add", "subtract", "multiply", "divide"}
MIXED_WORKLOADS_GROUP = "Mixed Workloads"
LEGACY_MIXED_WORKLOADS_GROUPS = {"Mixed workloads"}
LEGACY_HIDDEN_GROUPS = {"Mandelbrot"}
ARITHMETIC_GROUP_RE = re.compile(r"^f(?P<bits>128|256) <-> (?P<rhs>f128|f256|f64|f32|i64|i32)$")
PRIMARY_METRICS_GROUP = "Primary"


@dataclass(frozen=True)
class ColumnKey:
    fp_type: str
    platform: str
    compiler: str


@dataclass
class BenchmarkCell:
    ns_per_iter: float | None = None
    reference_ns_per_iter: float | None = None
    ratio: float | None = None
    reference_name: str | None = None


@dataclass
class BenchmarkTable:
    columns: list[ColumnKey]
    groups: list[str]
    rows_by_group: dict[str, list[str]]
    cells: dict[tuple[str, str, ColumnKey], BenchmarkCell]


def normalize_key(value: str) -> str:
    return (
        value.strip()
        .replace("-", "")
        .replace("_", "")
        .replace(".", "")
        .replace("(", "")
        .replace(")", "")
        .replace(" ", "")
        .lower()
    )


def platform_key(path: Path, compiler: str = "") -> str:
    key = normalize_key(path.parent.name)
    if key in {"mac", "darwin", "osx"}:
        return "macos"
    if key in {
        "browser",
        "chrome",
        "emscripten",
        "node",
        "nodejs",
        "wasm",
        "wasm32",
        "wasm32browser",
        "wasm32chrome",
        "wasm32node",
        "wasm32nodejs",
        "wasmbrowser",
        "wasmchrome",
        "wasmnode",
        "wasmnodejs",
        "web",
    }:
        return "wasm32"
    return key


def platform_label(key: str) -> str:
    return PLATFORM_LABELS.get(key, key[:1].upper() + key[1:])


def compiler_key(value: str, platform: str = "") -> str:
    key = normalize_key(value)
    if platform == "wasm32":
        if key in {"clang", "emcc", "emscripten", "emscriptennode", "emscriptennodejs", "node", "nodejs"}:
            return "nodejs"
        if key in {"browser", "chrome", "emscriptenbrowser", "emscriptenchrome"}:
            return "chrome"
        if key in {"wasm", "wasm32"}:
            return "wasm32"
    return key


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


def parse_reference_name(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if not text or text == "-":
        return None
    return text


def read_csv(path: Path, fp_type: str) -> list[tuple[str, str, BenchmarkCell]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return []

        group_column = find_column(reader.fieldnames, "group")
        label_column = find_column(reader.fieldnames, "label")
        precision_column = find_column(reader.fieldnames, "precision")
        operation_column = find_column(reader.fieldnames, "operation")
        candidate_column = find_column(reader.fieldnames, "fltx_ns_iter", f"{fp_type}_ns_iter")
        ratio_column = find_column(reader.fieldnames, "preferred_speed_ratio")
        reference_column = find_column(reader.fieldnames, "preferred_reference")

        if operation_column is None or candidate_column is None:
            return []

        out: list[tuple[str, str, BenchmarkCell]] = []
        for row in reader:
            precision = (row.get(precision_column) or "").strip().lower() if precision_column else ""
            if precision and precision != fp_type:
                continue

            group = (row.get(group_column) or "").strip() if group_column else ""
            label = (row.get(label_column) or "").strip() if label_column else ""
            if not group:
                group = PRIMARY_METRICS_GROUP
            if not label:
                label = (row.get(operation_column) or "").strip()
            if not label:
                continue
            candidate_ns = parse_float(row.get(candidate_column))
            if candidate_ns is None:
                continue
            ratio = parse_float(row.get(ratio_column)) if ratio_column else None
            reference_name = parse_reference_name(row.get(reference_column)) if reference_column else None
            reference_ns = candidate_ns * ratio if candidate_ns is not None and ratio is not None else None
            out.append((
                group,
                label,
                BenchmarkCell(candidate_ns, reference_ns, ratio, reference_name),
            ))
        return out


def visible_table_entry(fp_type: str, group: str, label: str) -> tuple[str, str] | None:
    arithmetic_match = ARITHMETIC_GROUP_RE.match(group)
    if arithmetic_match:
        native_group = f"{fp_type} <-> {fp_type}"
        if group == native_group and label in NATIVE_ARITHMETIC_LABELS:
            return NATIVE_ARITHMETIC_GROUP, label
        return None

    if group == NATIVE_ARITHMETIC_GROUP:
        return None

    if group in LEGACY_HIDDEN_GROUPS:
        return None

    if group == MIXED_WORKLOADS_GROUP or group in LEGACY_MIXED_WORKLOADS_GROUPS:
        return MIXED_WORKLOADS_GROUP, label

    return group, label


def average_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def average_cells(cells: list[BenchmarkCell]) -> BenchmarkCell:
    ns_per_iter = average_or_none([cell.ns_per_iter for cell in cells if cell.ns_per_iter is not None])
    reference_ns_per_iter = average_or_none([
        cell.reference_ns_per_iter
        for cell in cells
        if cell.reference_ns_per_iter is not None
    ])
    ratio = average_or_none([cell.ratio for cell in cells if cell.ratio is not None])
    if ratio is None and ns_per_iter is not None and ns_per_iter > 0.0 and reference_ns_per_iter is not None:
        ratio = reference_ns_per_iter / ns_per_iter

    reference_names = [cell.reference_name for cell in cells if cell.reference_name is not None]
    reference_name = None
    if reference_names:
        reference_name = max(reference_names, key=reference_names.count)

    return BenchmarkCell(ns_per_iter, reference_ns_per_iter, ratio, reference_name)


def discover_table(metrics_root: Path) -> BenchmarkTable:
    columns: set[ColumnKey] = set()
    groups: list[str] = []
    rows_by_group: dict[str, list[str]] = {}
    cell_buckets: dict[tuple[str, str, ColumnKey], list[BenchmarkCell]] = {}

    for path in sorted(metrics_root.rglob("*.csv")):
        match = METRICS_FILE_RE.match(path.name)
        if not match:
            continue

        fp_type = match.group("type").lower()
        platform = platform_key(path, match.group("compiler"))
        column = ColumnKey(fp_type, platform, compiler_key(match.group("compiler"), platform))
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
            cell_buckets.setdefault((visible_group, visible_label, column), []).append(cell)

        if has_visible_entries:
            columns.add(column)

    cells = {
        key: average_cells(values)
        for key, values in cell_buckets.items()
    }

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


def smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def ratio_curve_t(low_ratio: float, high_ratio: float, ratio: float) -> float:
    if low_ratio > 0.0 and high_ratio > low_ratio:
        t = (math.log(ratio) - math.log(low_ratio)) / (math.log(high_ratio) - math.log(low_ratio))
    else:
        t = (ratio - low_ratio) / (high_ratio - low_ratio)
    return smoothstep(t)


def ratio_color(ratio: float | None) -> str:
    if ratio is None:
        return WHITE
    if math.isnan(ratio):
        return WHITE
    if ratio <= RATIO_COLOR_STOPS[0][0]:
        return RATIO_COLOR_STOPS[0][1]
    if ratio >= RATIO_COLOR_STOPS[-1][0] or math.isinf(ratio):
        return RATIO_COLOR_STOPS[-1][1]

    previous_ratio, previous_color = RATIO_COLOR_STOPS[0]
    for next_ratio, next_color in RATIO_COLOR_STOPS[1:]:
        if ratio <= next_ratio:
            return mix_color(
                previous_color,
                next_color,
                ratio_curve_t(previous_ratio, next_ratio, ratio),
            )
        previous_ratio, previous_color = next_ratio, next_color

    return RATIO_COLOR_STOPS[-1][1]


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


def reference_nickname(reference_name: str | None) -> str:
    if reference_name is None:
        return ""

    key = normalize_key(reference_name)
    if "cppdoubledouble" in key:
        return "cpp_dd"
    if "mpfrfloatbackend" in key or "mpfr" in key:
        return "mpfr"
    if "ddreal" in key:
        return "dd_real"
    if "qdreal" in key:
        return "qd_real"
    return reference_name


def reference_full_name(reference_name: str | None) -> str:
    if reference_name is None:
        return ""

    key = normalize_key(reference_name)
    if "cppdoubledouble" in key:
        return "boost::multiprecision::cpp_double_double"
    if "mpfrfloatbackend" in key or "mpfr" in key:
        return "boost::multiprecision::mpfr_float_backend<64>"
    if "ddreal" in key:
        return "qdpp (dd_real)"
    if "qdreal" in key:
        return "qdpp (qd_real)"
    return reference_name


def reference_sort_rank(reference_name: str) -> int:
    key = normalize_key(reference_name)
    if "cppdoubledouble" in key or "mpfrfloatbackend" in key or "mpfr" in key:
        return 0
    if "qdpp" in key or "ddreal" in key or "qdreal" in key:
        return 1
    return 10


def reference_legend_groups(table: BenchmarkTable) -> list[tuple[str, list[str]]]:
    groups: list[tuple[str, list[str]]] = []
    for start_index, _ in fp_type_spans(table.columns):
        fp_type = table.columns[start_index].fp_type
        references: list[str] = []
        for column in [column for column in table.columns if column.fp_type == fp_type]:
            for group in table.groups:
                for label in table.rows_by_group.get(group, []):
                    cell = table.cells.get((group, label, column))
                    if cell is None or cell.ratio is None or cell.reference_name is None:
                        continue
                    if cell.reference_name not in references:
                        references.append(cell.reference_name)

        references.sort(key=lambda reference: (reference_sort_rank(reference), reference_nickname(reference)))
        if references:
            groups.append((fp_type, references))
    return groups


def reference_legend_lines(table: BenchmarkTable) -> list[str]:
    lines: list[str] = []
    for group_index, (_, references) in enumerate(reference_legend_groups(table)):
        if group_index > 0 and lines:
            lines.append("")
        for reference in references:
            lines.append(f"{reference_nickname(reference)} = {reference_full_name(reference)}")
    return lines


def format_ratio_with_reference(cell: BenchmarkCell) -> str:
    ratio = format_ratio(cell.ratio)
    nickname = reference_nickname(cell.reference_name)
    return f"{nickname}: {ratio}" if nickname and ratio else ratio


def e(value: str) -> str:
    return html.escape(value, quote=True)


def is_type_boundary(columns: list[ColumnKey], index: int) -> bool:
    return index > 0 and columns[index].fp_type != columns[index - 1].fp_type


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
        if index > 0:
            parts.append('<th class="type-gap" rowspan="3" aria-hidden="true"></th>')
        parts.append(f'<th class="type" colspan="{count}">{e(fp_type_label(fp_type))}</th>')
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
        parts.append(f'<th class="platform" colspan="{count}">{e(platform_label(platform))}</th>')
        index += count
    parts.append("</tr>")
    return "".join(parts)


def render_compiler_header(columns: list[ColumnKey]) -> str:
    return "<tr>" + "".join(
        f'<th class="compiler">{e(compiler_label(column.compiler))}</th>'
        for column in columns
    ) + "</tr>"


def render_html_reference_legend(table: BenchmarkTable) -> str:
    lines = reference_legend_lines(table)
    if not lines:
        return ""

    parts = ['<div class="reference-legend" aria-label="Ratio reference legend">']
    for line in lines:
        if not line:
            parts.append('<div class="legend-gap" aria-hidden="true"></div>')
            continue

        nickname, _, full_name = line.partition(" = ")
        parts.append(
            '<div class="legend-line">'
            f'<span class="legend-nickname">{e(nickname)}</span>'
            '<span class="legend-equals">=</span>'
            f'<span>{e(full_name)}</span>'
            '</div>'
        )
    parts.append('</div>')
    return "".join(parts)


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
        f'<div class="ratio">{e(format_ratio_with_reference(cell))}</div>'
        "</td>"
    )


def fp_type_spans(columns: list[ColumnKey]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(columns):
        fp_type = columns[index].fp_type
        count = 1
        while index + count < len(columns) and columns[index + count].fp_type == fp_type:
            count += 1
        spans.append((index, count))
        index += count
    return spans


def render_group_row(group: str, columns: list[ColumnKey]) -> str:
    parts = ['<tr class="group-row">']
    spans = fp_type_spans(columns)
    for span_index, (_, count) in enumerate(spans):
        if span_index > 0:
            parts.append('<th class="type-gap" aria-hidden="true"></th>')

        if span_index == 0:
            parts.append(f'<th class="group-label" colspan="{count + 1}">{e(group)}</th>')
        else:
            parts.append(f'<th class="group-fill" colspan="{count}"></th>')
    parts.append("</tr>")
    return "".join(parts)


def render_html_table(table: BenchmarkTable) -> str:
    if not table.columns:
        return '<p class="empty">No benchmark metrics were found for this table.</p>'

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
        parts.append(render_group_row(group, table.columns))
        for label in table.rows_by_group.get(group, []):
            parts.append("<tr>")
            parts.append(f'<th class="function">{e(label)}</th>')
            for index, column in enumerate(table.columns):
                if is_type_boundary(table.columns, index):
                    parts.append('<td class="type-gap" aria-hidden="true"></td>')
                parts.append(render_html_cell(table.cells.get((group, label, column))))
            parts.append("</tr>")
    parts.extend(["</tbody>", "</table>"])
    return "\n".join(parts)


def function_column_width(table: BenchmarkTable) -> int:
    labels = [label for group in table.groups for label in table.rows_by_group.get(group, [])]
    longest_label = max([len("Function")] + [len(label) for label in labels])
    natural_width = max(
        FUNCTION_COLUMN_MIN_WIDTH,
        min(FUNCTION_COLUMN_MAX_WIDTH, longest_label * FUNCTION_COLUMN_CHAR_WIDTH + FUNCTION_COLUMN_PADDING),
    )
    return round(natural_width * FUNCTION_COLUMN_WIDTH_SCALE)


def render_html_document(table: BenchmarkTable) -> str:
    function_w = function_column_width(table)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(TABLE_TITLE)}</title>
<style>
:root {{ color-scheme: light; font-family: Inter, Segoe UI, Roboto, Helvetica, Arial, sans-serif; background: {PAGE_BG}; color: {TEXT}; }}
body {{ margin: 0; padding: 32px; }}
h1 {{ margin: 0 0 8px; font-size: 28px; }}
.note {{ max-width: 960px; margin: 0 0 24px; color: {MUTED}; line-height: 1.45; }}
.table-wrap {{ overflow-x: auto; }}
.bench-table {{ border-collapse: collapse; min-width: 760px; background: {WHITE}; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.12); font-size: 13px; }}
.bench-table th, .bench-table td {{ border: 1px solid {GRID}; padding: 6px 10px; text-align: center; vertical-align: middle; }}
.bench-table thead th {{ background: {HEADER_BG}; color: {WHITE}; font-weight: 700; }}
.bench-table thead th.platform {{ background: {PLATFORM_HEADER_BG}; }}
.bench-table thead th.compiler {{ background: {SUBHEADER_BG}; }}
.bench-table th.type-gap, .bench-table td.type-gap {{ width: {TYPE_GAP_WIDTH}px; min-width: {TYPE_GAP_WIDTH}px; padding: 0; border-top: 1px hidden transparent !important; border-bottom: 1px hidden transparent !important; border-left: 1px solid {GRID} !important; border-right: 1px solid {GRID} !important; background: {PAGE_BG} !important; }}
.bench-table th.function {{ width: {function_w}px; min-width: {function_w}px; max-width: {function_w}px; text-align: left; white-space: normal; overflow-wrap: anywhere; }}
.bench-table tbody th.function {{ font-size: 14px; }}
.group-row th {{ background: {GROUP_BG}; color: {TEXT}; text-align: left; font-size: 13px; letter-spacing: 0.02em; }}
.reference-legend {{ display: grid; gap: 3px; margin: -10px 0 18px; color: {MUTED}; font-size: 12px; line-height: 1.3; }}
.legend-line {{ display: grid; grid-template-columns: max-content max-content 1fr; gap: 6px; align-items: baseline; }}
.legend-gap {{ height: 6px; }}
.legend-nickname {{ color: {TEXT}; font-weight: 750; font-family: Consolas, "Cascadia Mono", monospace; }}
.legend-equals {{ color: {TEXT}; font-weight: 650; }}
.result {{ min-width: 104px; font-weight: 700; }}
.result.no-ratio, .missing {{ background: {WHITE}; color: #000000; }}
.timer {{ line-height: 1.05; }}
.ratio {{ margin-top: 2px; font-size: 11px; line-height: 1.05; opacity: 0.88; }}
.empty {{ color: {MUTED}; }}
</style>
</head>
<body>
<h1>{e(TABLE_TITLE)}</h1>
<p class="note">{e(TABLE_SUBTITLE)}</p>
{render_html_reference_legend(table)}
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


def svg_rect_no_stroke(x: int, y: int, width: int, height: int, fill: str) -> str:
    return f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{fill}"/>'


def svg_vertical_borders(x: int, y: int, width: int, height: int, color: str) -> str:
    return (
        f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + height}" stroke="{color}" stroke-width="1"/>'
        f'<line x1="{x + width}" y1="{y}" x2="{x + width}" y2="{y + height}" stroke="{color}" stroke-width="1"/>'
    )


def svg_reference_legend_height(table: BenchmarkTable) -> int:
    lines = reference_legend_lines(table)
    if not lines:
        return 0

    height = 8
    for line in lines:
        height += 8 if not line else 15
    return height


def render_svg_reference_legend(table: BenchmarkTable, x: int, baseline_y: int) -> list[str]:
    lines = reference_legend_lines(table)
    if not lines:
        return []

    parts: list[str] = []
    y = baseline_y
    for line in lines:
        if not line:
            y += 8
            continue

        nickname, _, full_name = line.partition(" = ")
        nickname_w = math.ceil(len(nickname) * 7.0)
        parts.append(svg_text(x, y, nickname, 11, TEXT, 700))
        parts.append(svg_text(x + nickname_w + 8, y, "=", 11, TEXT, 650))
        parts.append(svg_text(x + nickname_w + 22, y, full_name, 11, MUTED, 600))
        y += 15
    return parts


def table_row_count(table: BenchmarkTable) -> int:
    return sum(1 + len(table.rows_by_group.get(group, [])) for group in table.groups)


def render_svg_table(table: BenchmarkTable) -> str:
    legend_h = svg_reference_legend_height(table)
    title_h = 62 + legend_h
    header_h = 34
    header_rows = 3
    group_h = 28
    row_h = 40
    margin = 18

    function_w = function_column_width(table)
    column_w = 116
    boundaries = set(type_boundary_indices(table.columns))
    width = margin * 2 + function_w + column_w * max(1, len(table.columns)) + TYPE_GAP_WIDTH * len(boundaries)
    legend_width = max([0] + [math.ceil(len(line) * 6.4) for line in reference_legend_lines(table)])
    width = max(width, margin * 2 + math.ceil(len(TABLE_SUBTITLE) * 7.2), margin * 2 + legend_width)
    height = margin * 2 + title_h + header_h * header_rows + group_h * len(table.groups) + row_h * sum(len(table.rows_by_group.get(group, [])) for group in table.groups)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{e(TABLE_TITLE)}</title>',
        f'<desc id="desc">{e(TABLE_SUBTITLE)}</desc>',
        f'<rect width="{width}" height="{height}" fill="{PAGE_BG}"/>',
        svg_text(margin, margin + 26, TABLE_TITLE, 24, TEXT, 700),
        svg_text(margin, margin + 50, TABLE_SUBTITLE, 13, MUTED),
    ]
    parts.extend(render_svg_reference_legend(table, margin, margin + 76))

    if not table.columns:
        parts.append(svg_text(margin, margin + title_h + 32, "No benchmark metrics were found for this table.", 14, MUTED))
        parts.append("</svg>")
        return "\n".join(parts)

    table_x = margin
    y = margin + title_h
    parts.append(svg_rect(table_x, y, function_w, header_h * header_rows, HEADER_BG))
    parts.append(svg_text(table_x + 14, y + 60, "Function", 13, WHITE, 700))

    x = table_x + function_w
    index = 0
    while index < len(table.columns):
        fp_type = table.columns[index].fp_type
        count = 1
        while index + count < len(table.columns) and table.columns[index + count].fp_type == fp_type:
            count += 1
        if index in boundaries:
            x += TYPE_GAP_WIDTH
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
        if index in boundaries:
            x += TYPE_GAP_WIDTH
        span_w = count * column_w
        parts.append(svg_rect(x, y, span_w, header_h, PLATFORM_HEADER_BG))
        parts.append(svg_text(x + span_w // 2, y + 22, platform_label(platform), 13, WHITE, 700, "middle"))
        x += span_w
        index += count

    y += header_h
    x = table_x + function_w
    for index, column in enumerate(table.columns):
        if index in boundaries:
            x += TYPE_GAP_WIDTH
        parts.append(svg_rect(x, y, column_w, header_h, SUBHEADER_BG))
        parts.append(svg_text(x + column_w // 2, y + 22, compiler_label(column.compiler), 13, WHITE, 700, "middle"))
        x += column_w

    y += header_h
    for group in table.groups:
        group_x = table_x
        for span_index, (_, count) in enumerate(fp_type_spans(table.columns)):
            if span_index == 0:
                span_w = function_w + column_w * count
            else:
                parts.append(svg_rect_no_stroke(group_x, y, TYPE_GAP_WIDTH, group_h, PAGE_BG))
                parts.append(svg_vertical_borders(group_x, y, TYPE_GAP_WIDTH, group_h, GRID))
                group_x += TYPE_GAP_WIDTH
                span_w = column_w * count
            parts.append(svg_rect(group_x, y, span_w, group_h, GROUP_BG))
            group_x += span_w
        parts.append(svg_text(table_x + 14, y + 19, group, 12, TEXT, 700))
        y += group_h

        for label in table.rows_by_group.get(group, []):
            parts.append(svg_rect(table_x, y, function_w, row_h, WHITE))
            parts.append(svg_text(table_x + 14, y + 25, label, 13, TEXT, 600))
            x = table_x + function_w
            for index, column in enumerate(table.columns):
                if index in boundaries:
                    x += TYPE_GAP_WIDTH
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
                    parts.append(svg_text(x + column_w // 2, y + 33, format_ratio_with_reference(cell), 11, color, 600, "middle"))
                x += column_w
            y += row_h

    parts.append("</svg>")
    return "\n".join(parts)


def default_res_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_metrics_root(res_root: Path) -> Path:
    if res_root.name.lower() == "metrics":
        return res_root
    return res_root / "metrics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect fltx metrics CSV files into one combined bl::f128/bl::f256 table file.")
    parser.add_argument("--res-root", type=Path, default=default_res_root(), help="fltx/res root folder. Defaults to this script's parent res folder.")
    parser.add_argument("--root", type=Path, default=None, help="Deprecated alias for --res-root.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults to <res-root>/metrics.")
    parser.add_argument("--format", choices=("html", "svg", "both"), default="svg", help="Output format.")
    return parser.parse_args()


def write_outputs(table: BenchmarkTable, output_dir: Path, output_format: str) -> list[Path]:
    written: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_format in {"html", "both"}:
        path = output_dir / "metrics_table.html"
        path.write_text(render_html_document(table), encoding="utf-8")
        written.append(path)
    if output_format in {"svg", "both"}:
        path = output_dir / "metrics_table.svg"
        path.write_text(render_svg_table(table), encoding="utf-8")
        written.append(path)
    return written


def main() -> None:
    args = parse_args()
    res_root = (args.root if args.root is not None else args.res_root).resolve()
    metrics_root = resolve_metrics_root(res_root)
    output_dir = args.output_dir.resolve() if args.output_dir else metrics_root

    table = discover_table(metrics_root)
    for path in write_outputs(table, output_dir, args.format):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
