"""Shared CLI helpers for composer_toolchain commands."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Set

from rich.console import Console
from rich.table import Table

console = Console()


def choose_score(
    src_dir: Path, filter_suffix: Set[str] = {".mxl", ".xml", ".krn"}
) -> Path:
    """Prompt the user to select a score inside `src_dir` via `sk`."""

    scores = [p for p in src_dir.glob("*.*") if p.suffix in filter_suffix]
    if not scores:
        raise ValueError(f"No files with suffix {filter_suffix} in {src_dir}")
    choices = [candidate.name for candidate in scores]
    try:
        result = subprocess.run(
            ["sk"],
            input="\n".join(choices),
            text=True,
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "sk executable not found; install sk to use interactive selection."
        ) from exc

    filename = result.stdout.strip()
    if not filename:
        raise ValueError("No score selected.")
    score_file = src_dir / filename
    if not score_file.exists():
        raise FileNotFoundError(f"Selected file does not exist: {score_file}")
    return score_file.resolve()


def render_score_metadata(spec, *, source_label: str) -> None:
    """Print high-level metadata for composer context."""

    table = Table(
        title=f"Score Overview · {source_label}",
        show_header=False,
        box=None,
    )
    table.add_row("Title", spec.title)
    if spec.composer:
        table.add_row("Composer", spec.composer)
    primary = spec.primary_movement()
    if primary and (primary.number is not None or primary.title):
        bits: list[str] = []
        if primary.number is not None:
            bits.append(f"No. {primary.number}")
        if primary.title:
            bits.append(primary.title)
        table.add_row("Movement", " · ".join(bits))
    if spec.movements:
        labels: list[str] = []
        for entry in spec.movements:
            label_bits: list[str] = []
            if entry.number is not None:
                label_bits.append(str(entry.number))
            if entry.title:
                label_bits.append(entry.title)
            if label_bits:
                labels.append(": ".join(label_bits) if len(label_bits) > 1 else label_bits[0])
        if labels:
            table.add_row("Movements", ", ".join(labels))
    table.add_row("Parts", str(len(spec.parts)))
    if spec.total_measures:
        table.add_row("Length", f"{spec.total_measures} measures")
    console.print(table)


def render_measure_landmarks(spec) -> None:
    """Display cue spans covering the full score length."""

    rows = spec.iter_cue_spans()
    if not rows:
        return
    table = Table(
        title="Structural Landmarks (changes in meter/key/tempo)",
        header_style="bold cyan",
        show_lines=False,
        expand=False,
    )
    table.add_column("Measures", style="yellow", no_wrap=True)
    table.add_column("Time Sig", style="magenta")
    table.add_column("Key", style="green")
    table.add_column("Tempo", style="cyan")
    for cue in rows:
        measure_label = (
            str(cue.start_measure)
            if cue.start_measure == cue.end_measure
            else f"{cue.start_measure}-{cue.end_measure}"
        )
        table.add_row(
            measure_label,
            cue.time_signature or "–",
            cue.key_signature or "–",
            f"{cue.tempo_bpm} bpm" if cue.tempo_bpm else "–",
        )
    console.print(table)


__all__ = [
    "choose_score",
    "console",
    "render_score_metadata",
    "render_measure_landmarks",
]
