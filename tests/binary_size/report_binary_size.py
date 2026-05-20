import argparse
import csv
from pathlib import Path
from typing import Optional


def companion_wasm_path(path: Path) -> Optional[Path]:
    if path.suffix.lower() in {".js", ".html"}:
        return path.with_suffix(".wasm")
    return None


def file_size(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    return path.stat().st_size


def kib(value: Optional[int]) -> str:
    if value is None:
        return ""
    return f"{value / 1024.0:.2f}"


def load_rows(manifest: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with manifest.open(newline="") as handle:
        for row in csv.DictReader(handle):
            path = Path(row["path"])
            exe_bytes = file_size(path)
            wasm_path = companion_wasm_path(path)
            wasm_bytes = file_size(wasm_path) if wasm_path else None
            primary_bytes = wasm_bytes if wasm_bytes is not None else exe_bytes
            primary_path = wasm_path if wasm_bytes is not None and wasm_path is not None else path

            rows.append(
                {
                    "family": row["family"],
                    "group": row["group"],
                    "label": row["label"],
                    "target": row["target"],
                    "path": str(path),
                    "primary_path": str(primary_path),
                    "exe_bytes": exe_bytes,
                    "wasm_bytes": wasm_bytes,
                    "primary_bytes": primary_bytes,
                }
            )
    return rows


def write_report(rows: list[dict[str, object]], output: Path) -> list[dict[str, object]]:
    baselines: dict[str, int] = {}
    for row in rows:
        if row["label"] == "baseline" and isinstance(row["primary_bytes"], int):
            baselines[str(row["family"])] = int(row["primary_bytes"])

    report_rows: list[dict[str, object]] = []
    for row in rows:
        primary = row["primary_bytes"]
        baseline = baselines.get(str(row["family"]))
        delta = primary - baseline if isinstance(primary, int) and baseline is not None else None

        report_rows.append(
            {
                "family": row["family"],
                "group": row["group"],
                "label": row["label"],
                "target": row["target"],
                "primary_bytes": primary if primary is not None else "",
                "primary_kib": kib(primary if isinstance(primary, int) else None),
                "delta_bytes": delta if delta is not None else "",
                "delta_kib": kib(delta),
                "exe_bytes": row["exe_bytes"] if row["exe_bytes"] is not None else "",
                "wasm_bytes": row["wasm_bytes"] if row["wasm_bytes"] is not None else "",
                "primary_path": row["primary_path"],
                "target_path": row["path"],
            }
        )

    report_rows.sort(
        key=lambda row: (
            str(row["family"]),
            str(row["group"]),
            -(int(row["delta_bytes"]) if row["delta_bytes"] != "" else -1),
            str(row["label"]),
        )
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "family",
                "group",
                "label",
                "target",
                "primary_bytes",
                "primary_kib",
                "delta_bytes",
                "delta_kib",
                "exe_bytes",
                "wasm_bytes",
                "primary_path",
                "target_path",
            ],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    return report_rows


def print_summary(report_rows: list[dict[str, object]], output: Path, top: int) -> None:
    print(f"Wrote {output}")
    print("Primary size uses the companion .wasm file when present; otherwise it uses the target executable.")

    families = sorted({str(row["family"]) for row in report_rows})
    for family in families:
        baseline = next((row for row in report_rows if row["family"] == family and row["label"] == "baseline"), None)
        if baseline:
            print(f"\n{family} baseline: {baseline['primary_kib']} KiB")

        groups = [group for group in ("single", "callsite") if any(row["family"] == family and row["group"] == group for row in report_rows)]
        for group in groups:
            rows = [
                row for row in report_rows
                if row["family"] == family and row["group"] == group and row["delta_bytes"] != ""
            ]
            rows.sort(key=lambda row: int(row["delta_bytes"]), reverse=True)

            print(f"  {group}:")
            for row in rows[:top]:
                print(f"    {row['label']:<24} +{row['delta_kib']:>9} KiB  total {row['primary_kib']:>9} KiB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Report fltx isolated binary-size deltas.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top", default=20, type=int)
    args = parser.parse_args()

    rows = load_rows(args.manifest)
    report_rows = write_report(rows, args.output)
    print_summary(report_rows, args.output, args.top)


if __name__ == "__main__":
    main()
