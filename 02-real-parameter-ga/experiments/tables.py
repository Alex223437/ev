"""Writing result tables as CSV and Markdown."""

import csv
from pathlib import Path
from typing import Sequence


def write_csv(rows: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: Sequence[dict], columns: dict[str, str]) -> str:
    """Render `rows` as a Markdown table; `columns` maps row keys to column headers."""
    lines = [
        "| " + " | ".join(columns.values()) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row[key]) for key in columns) + " |")
    return "\n".join(lines) + "\n"


def format_value(value) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)
