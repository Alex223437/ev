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
        lines.append("| " + " | ".join(_format(row[key]) for key in columns) + " |")
    return "\n".join(lines) + "\n"


def _format(value) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        if value.is_integer():
            return f"{value:.0f}"
        # Keep small parameters such as pm = 0.005 exact, round the rest to two decimals.
        return f"{value:.3g}" if abs(value) < 1 else f"{value:.2f}"
    return str(value)
