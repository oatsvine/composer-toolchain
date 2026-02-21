"""Typer CLI interface to core toolchain."""

from __future__ import annotations
import shutil
from pathlib import Path
from typing import Literal, Optional

import typer
from loguru import logger
from music21.humdrum.spineParser import GlobalComment
from typing_extensions import Annotated


from composer_toolchain.cli_helpers import (
    choose_score,
    console,
)
from composer_toolchain.core import Context
from composer_toolchain.score import MeasureSpec, PartSpec

# Refactor to use environment variable later.
SCORES_CORPUS_DIR = Path("/data/workspace/in/mxl")
MIDI_EXPORT_DIR = Path("/data/workspace/out/midi")

app = typer.Typer()


@app.command()
def init_with_score(
    score_file: Annotated[
        Optional[Path],
        typer.Option(
            help="Path to the source score. If omitted, launch interactive chooser.",
            resolve_path=True,
        ),
    ] = None,
    cwd: Annotated[
        Path,
        typer.Option(
            help="Root directory where the workspace will be created.",
            exists=True,
            file_okay=False,
            resolve_path=True,
        ),
    ] = Path("/data/workspace/music"),
) -> Path:
    """Create a new workspace from an external score."""
    selected = (
        choose_score(SCORES_CORPUS_DIR)
        if score_file is None
        else (
            score_file if score_file.is_absolute() else SCORES_CORPUS_DIR / score_file
        )
    )
    selected = selected.resolve()
    if not selected.exists():
        raise typer.BadParameter(f"Score file not found: {selected}")
    created = Context.init_with_score(score_file=selected, cwd=cwd)
    console.print(f"[green]Workspace created[/green]: {created}")
    return created


@app.command()
def import_score(
    work_dir: Annotated[
        Path,
        typer.Argument(
            help="Workspace root directory.",
            exists=True,
            file_okay=False,
            resolve_path=True,
        ),
    ] = Path.cwd(),
    score_file: Annotated[
        Optional[Path],
        typer.Option(
            help="Score to import into the workspace library. If omitted, choose interactively.",
            resolve_path=True,
        ),
    ] = None,
) -> Path:
    """Normalize an external score into the workspace `scores/` library."""
    candidate = (
        choose_score(SCORES_CORPUS_DIR)
        if score_file is None
        else (
            score_file if score_file.is_absolute() else SCORES_CORPUS_DIR / score_file
        )
    )
    candidate = candidate.resolve()
    if not candidate.exists():
        raise typer.BadParameter(f"Score file not found: {candidate}")
    workspace = Context(work_dir=work_dir)
    target = workspace.import_score(score_file=candidate)
    console.print(
        f"[green]Imported[/green] {candidate.name} → {target.relative_to(work_dir)}"
    )
    return target


@app.command()
def change_master(
    work_dir: Annotated[
        Path,
        typer.Argument(
            help="Workspace root directory.",
            exists=True,
            file_okay=False,
            resolve_path=True,
        ),
    ] = Path.cwd(),
    filename: Annotated[
        Optional[str],
        typer.Option(
            help="Workspace score (under scores/) to promote as master. If omitted, choose interactively.",
        ),
    ] = None,
) -> None:
    """Promote a workspace score to master."""
    workspace = Context(work_dir=work_dir)
    scores_dir = workspace.subdir("scores")

    if not filename:
        filename = choose_score(scores_dir, filter_suffix={".krn"}).name

    workspace.change_master(filename=filename)
    console.print(f"[green]Master changed[/green] → scores/{filename}")


@app.command()
def render_midi(
    work_dir: Annotated[
        Path,
        typer.Argument(
            help="Workspace root directory.",
            exists=True,
            file_okay=False,
            resolve_path=True,
        ),
    ] = Path.cwd(),
    export: Annotated[
        bool,
        typer.Option(
            help=f"Export MIDI to {MIDI_EXPORT_DIR}.",
            is_flag=True,
            show_default=True,
        ),
    ] = False,
) -> Path:
    """Export the master score to MIDI."""
    workspace = Context(work_dir=work_dir)
    # Recurse the work_dir to render any excerpt, sketch, etc.
    source_file = choose_score(work_dir, filter_suffix={".krn"})
    midi = workspace.render_midi(source_file)
    console.print(f"[green]MIDI exported[/green]: {midi.relative_to(work_dir)}")
    if export:
        shutil.copy2(midi, MIDI_EXPORT_DIR / midi.name)
        console.print(f"[green]MIDI copied[/green]: {MIDI_EXPORT_DIR / midi.name}")
    return midi


@app.command()
def create_excerpt(
    work_dir: Annotated[
        Path,
        typer.Argument(
            help="Workspace root directory (the folder created via init-with-score).",
            exists=True,
            file_okay=False,
            resolve_path=True,
        ),
    ] = Path.cwd(),
    parts: Annotated[
        str,
        typer.Option(
            "--parts",
            help=(
                "Comma-separated canonical part ids to include "
                "(matches PartSpec tokens such as 'flute,vn1')."
            ),
        ),
    ],
    measures: Annotated[
        str,
        typer.Option(
            "--measures",
            help=(
                "Measure ranges to extract, e.g. '1-8,17-'. Uses MeasureSpec "
                "grammar with ascending, comma-separated spans."
            ),
        ),
    ],
    filename: Annotated[
        Optional[str],
        typer.Option(
            help=(
                "Workspace score filename under scores/ to carve from; "
                "defaults to the current master after confirmation."
            ),
        ),
    ] = None,
    suffix: Annotated[
        Optional[str],
        typer.Option(
            help=(
                "Optional suffix appended to the excerpt filename (snake_case)."
                " Helps tag motives like 'antecedent_phrase'."
            ),
        ),
    ] = None,
) -> Path:
    """Slice an excerpt from a workspace score with Rich-guided prompts.

    Flow
    ----
    1. Confirm which workspace score to carve (the manifest master is suggested first)
       and review its title/composer/movement metadata for context.
    2. Multi-select the parts via questionary checkboxes (press 'a' to toggle all).
    3. Review meter/key/tempo landmarks aggregated from ScoreSpec before entering the
       MeasureSpec string (open tails such as "32-" are still accepted).

    The resulting Humdrum file is written to `excerpts/` following the standard
    `<source>_<parts>_<measures>.krn` scheme so it can later be merged back with
    `merge-excerpt`. Use `--non-interactive` for scripts that supply every option explicitly.
    """

    workspace = Context(work_dir=work_dir)
    master_file = workspace._master_file()
    master_name = master_file.name
    source_name = filename.strip() if filename else master_name

    try:
        part_spec = PartSpec(tokens=parts.strip())
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="parts") from exc

    try:
        measure_spec = MeasureSpec(spec=measures.strip())
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="measures") from exc

    source_override = None if source_name == master_name else source_name

    excerpt_file = workspace.create_and_store_excerpt(
        part_spec=part_spec,
        measure_spec=measure_spec,
        filename=source_override,
        suffix=suffix.strip() if suffix else None,
    )
    console.print(
        f"[green]Excerpt created[/green]: {excerpt_file.relative_to(work_dir)}"
    )
    return excerpt_file


@app.command()
def merge_excerpt(
    work_dir: Annotated[
        Path,
        typer.Argument(
            help="Workspace root directory.",
            exists=True,
            file_okay=False,
            resolve_path=True,
        ),
    ] = Path.cwd(),
    filename: Annotated[
        Optional[Path],
        typer.Option(
            help="Humdrum excerpt under excerpts/ to merge. If omitted, choose interactively.",
            resolve_path=True,
        ),
    ] = None,
) -> None:
    """Merge a Humdrum excerpt (under excerpts/) back into the master."""
    workspace = Context(work_dir=work_dir)
    excerpts_dir = workspace.subdir("excerpts")
    candidate = (
        choose_score(excerpts_dir, filter_suffix={".krn"})
        if filename is None
        else (filename if filename.is_absolute() else excerpts_dir / filename)
    )
    candidate = candidate.resolve()
    if not candidate.exists():
        raise typer.BadParameter(f"Excerpt file not found: {candidate}")

    workspace.merge_excerpt(filename=candidate.name)
    console.print(f"[green]Excerpt merged[/green]: {candidate.relative_to(work_dir)}")


@app.command()
def remove(
    work_dir: Annotated[
        Path,
        typer.Argument(
            help="Workspace root directory.",
            exists=True,
            file_okay=False,
            resolve_path=True,
        ),
    ] = Path.cwd(),
    measure_spec: Annotated[
        str,
        typer.Option(
            "--measures",
            help="Measure specification to remove (e.g. '12-16').",
            prompt=True,
        ),
    ] = "",
    mode: Annotated[
        str,
        typer.Option(
            help="Removal mode ('blank' or 'drop_renumber').",
            case_sensitive=False,
        ),
    ] = "blank",
) -> None:
    """Remove measure ranges across all parts in the master."""
    if not measure_spec:
        raise typer.BadParameter("Measure specification cannot be empty")

    mode_normalized = mode.lower()
    if mode_normalized == "blank":
        removal_mode: Literal["blank", "drop_renumber"] = "blank"
    elif mode_normalized == "drop_renumber":
        removal_mode = "drop_renumber"
    else:
        raise typer.BadParameter("Mode must be 'blank' or 'drop_renumber'")

    workspace = Context(work_dir=work_dir)
    workspace.delete_measures(measure_spec=measure_spec, mode=removal_mode)
    console.print(f"[yellow]Removed measures[/yellow]: {measure_spec} ({removal_mode})")


@app.command()
def expand(
    work_dir: Annotated[
        Path,
        typer.Argument(
            help="Workspace root directory.",
            exists=True,
            file_okay=False,
            resolve_path=True,
        ),
    ] = Path.cwd(),
    at: Annotated[
        int,
        typer.Option(
            help="Measure index where blank measures should be inserted.", min=1
        ),
    ] = 1,
    count: Annotated[
        int,
        typer.Option(help="Number of blank measures to insert.", min=1),
    ] = 1,
) -> None:
    """Insert blank measures into the master score."""
    workspace = Context(work_dir=work_dir)
    workspace.expand_master(at=at, count=count)
    console.print(f"[yellow]Inserted blank measures[/yellow]: {count} @ {at}")


@app.command()
def print_comments(
    work_dir: Annotated[
        Path,
        typer.Argument(
            help="Workspace root directory.",
            exists=True,
            file_okay=False,
            resolve_path=True,
        ),
    ] = Path.cwd(),
) -> None:
    """Dump the global comments stream for the master score."""
    master = Context(work_dir=work_dir)._master_score()
    stream = master.getElementsByClass(GlobalComment).stream()
    stream.show(fmt="text", addEndTimes=True)


@app.command()
def show(
    work_dir: Annotated[
        Path,
        typer.Argument(
            help="Workspace root directory.",
            exists=True,
            file_okay=False,
            resolve_path=True,
        ),
    ] = Path.cwd(),
    fmt: Annotated[
        str,
        typer.Option(help="music21 show() format.", case_sensitive=False),
    ] = "text",
) -> None:
    """Relay music21.show for the master score."""
    master = Context(work_dir=work_dir)._master_score()
    master.show(fmt=fmt, addEndTimes=True)


@app.callback()
def main(ctx: typer.Context) -> None:
    """Log the invoked subcommand for observability."""
    logger.info("Executing CLI command {}", ctx.invoked_subcommand)


if __name__ == "__main__":  # pragma: no cover - manual entry point
    app()
