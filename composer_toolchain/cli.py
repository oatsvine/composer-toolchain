"""Typer CLI interface to core toolchain.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Literal, Optional, Set

import typer
from loguru import logger
from music21.humdrum.spineParser import GlobalComment
from music21.stream import Measure
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from typing_extensions import Annotated

from composer_toolchain.score import MeasureSpec, PartSpec, load_score, normalize
from composer_toolchain.core import Context

console = Console()
app = typer.Typer()


def choose_score(
    src_dir: Path, filter_suffix: Set[str] = {".mxl", ".xml", ".krn"}
) -> Path:
    """Prompt the user to select a score inside `src_dir`."""
    scores = [p for p in src_dir.glob("*.*") if p.suffix in filter_suffix]
    if not scores:
        raise ValueError(f"No files with suffix {filter_suffix} in {src_dir}")
    choices = [candidate.name for candidate in scores]
    result = subprocess.run(
        ["sk"],
        input="\n".join(choices),
        text=True,
        capture_output=True,
        check=True,
    )
    filename = result.stdout.strip()
    score_file = src_dir / filename
    if not score_file.exists():
        raise FileNotFoundError(f"Selected file does not exist: {score_file}")
    return score_file.resolve()


@app.command()
def init_from_score(
    score_file: Annotated[
        Optional[Path],
        typer.Option(
            help="Path to the source score. If omitted, launch interactive chooser.",
            resolve_path=True,
        ),
    ] = None,
    scores_dir: Annotated[
        Path,
        typer.Option(
            help="Directory containing candidate source scores.",
            exists=True,
            file_okay=False,
            resolve_path=True,
        ),
    ] = Path("/data/workspace/in/mxl"),
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
        choose_score(scores_dir)
        if score_file is None
        else (score_file if score_file.is_absolute() else scores_dir / score_file)
    )
    selected = selected.resolve()
    if not selected.exists():
        raise typer.BadParameter(f"Score file not found: {selected}")
    created = Context.init_from_score(score_file=selected, cwd=cwd)
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
    # NOTE: scores_dir is ever only relevant here in the CLI, and goes together with choose_score().
    scores_dir: Annotated[
        Path,
        typer.Option(
            help="Directory containing candidate source scores.",
            exists=True,
            file_okay=False,
            resolve_path=True,
        ),
    ] = Path("/data/workspace/in/mxl"),
) -> Path:
    """Normalize an external score into the workspace `scores/` library."""
    candidate = (
        choose_score(scores_dir)
        if score_file is None
        else (score_file if score_file.is_absolute() else scores_dir / score_file)
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
    scores_dir = work_dir / "scores"
    if not scores_dir.exists():
        raise typer.BadParameter(f"Workspace scores directory missing: {scores_dir}")

    if not filename:
        filename = choose_score(scores_dir, filter_suffix={".krn"}).name

    workspace = Context(work_dir=work_dir)
    workspace.change_master(filename=filename)
    console.print(f"[green]Master changed[/green] → scores/{filename}")


@app.command()
def export_midi(
    work_dir: Annotated[
        Path,
        typer.Argument(
            help="Workspace root directory.",
            exists=True,
            file_okay=False,
            resolve_path=True,
        ),
    ] = Path.cwd(),
) -> Path:
    """Export the master score to MIDI."""
    workspace = Context(work_dir=work_dir)
    midi = workspace.export_midi()
    console.print(f"[green]MIDI exported[/green]: {midi.relative_to(work_dir)}")
    return midi


@app.command()
def create_excerpt(
    work_dir: Annotated[
        Path,
        typer.Argument(
            help="Workspace root directory.",
            exists=True,
            file_okay=False,
            resolve_path=True,
        ),
    ] = Path.cwd(),
    parts: Annotated[
        str,
        typer.Option(
            help="Comma-separated canonical part ids (e.g. 'vn1,vn2').",
            prompt=True,
        ),
    ] = "",
    measures: Annotated[
        str,
        typer.Option(
            help="Measure specification such as '17-24'.",
            prompt=True,
        ),
    ] = "",
    other_score: Annotated[
        Optional[Path],
        typer.Option(
            help="Alternate workspace score (under scores/) to carve from.",
            resolve_path=True,
        ),
    ] = None,
) -> Path:
    """Extract a Humdrum excerpt from the master (or another workspace score)."""
    if not parts:
        raise typer.BadParameter("At least one part id is required")
    if not measures:
        raise typer.BadParameter("Measure specification cannot be empty")

    part_spec = PartSpec(tokens=parts)
    measure_spec = MeasureSpec(spec=measures)

    workspace = Context(work_dir=work_dir)

    source_filename: Optional[str]
    if other_score is not None:
        scores_dir = work_dir / "scores"
        candidate = (
            other_score if other_score.is_absolute() else scores_dir / other_score
        )
        candidate = candidate.resolve()
        if not candidate.exists():
            raise typer.BadParameter(f"Other score not found: {candidate}")
        try:
            candidate.relative_to(scores_dir)
        except ValueError:
            raise typer.BadParameter(f"Score must reside within {scores_dir}") from None
        source_filename = candidate.name
    else:
        source_filename = None

    excerpt_file = workspace.create_and_store_excerpt(
        part_spec=part_spec,
        measure_spec=measure_spec,
        other_score=source_filename,
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
    excerpts_dir = work_dir / "excerpts"
    if not excerpts_dir.exists():
        raise typer.BadParameter(f"No excerpts directory in workspace: {excerpts_dir}")

    candidate = (
        choose_score(excerpts_dir, filter_suffix={".krn"})
        if filename is None
        else (filename if filename.is_absolute() else excerpts_dir / filename)
    )
    candidate = candidate.resolve()
    if not candidate.exists():
        raise typer.BadParameter(f"Excerpt file not found: {candidate}")

    workspace = Context(work_dir=work_dir)
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
def info(
    work_dir: Annotated[
        Path,
        typer.Argument(
            help="Workspace root directory.",
            exists=True,
            file_okay=False,
            resolve_path=True,
        ),
    ] = Path.cwd(),
    other_score: Annotated[
        Optional[Path],
        typer.Option(
            help="Alternate workspace score (under scores/) to inspect.",
            resolve_path=True,
        ),
    ] = None,
) -> None:
    """Render workspace and score metadata via Rich tables."""
    client = Context(work_dir=work_dir)

    if other_score is None:
        score_path = client._master_file()
        score = client._master_score()
    else:
        scores_dir = work_dir / "scores"
        candidate = (
            other_score if other_score.is_absolute() else scores_dir / other_score
        )
        candidate = candidate.resolve()
        if not candidate.exists():
            raise typer.BadParameter(f"Score not found: {candidate}")
        if not candidate.is_relative_to(scores_dir):
            raise typer.BadParameter(f"Score must reside within {scores_dir}")
        score_path = candidate
        score = normalize(load_score(score_path))

    spec = client._build_spec(score)

    try:
        display_master = score_path.relative_to(work_dir)
    except ValueError:
        display_master = score_path

    workspace_panel = Table.grid(padding=(0, 2))
    workspace_panel.add_row("Master", str(display_master))
    workspace_panel.add_row("Duration (QL)", f"{score.highestTime:.2f}")
    console.print(Panel(workspace_panel, title="Workspace", expand=True))

    parts_table = Table(title="Parts", header_style="bold")
    parts_table.add_column("ID", style="cyan")
    parts_table.add_column("Name")
    parts_table.add_column("Measures", justify="right")
    parts_table.add_column("Instrument")
    for part, payload in zip(score.parts, spec.parts, strict=True):
        measure_numbers = [
            int(measure.number)
            for measure in part.getElementsByClass(Measure)
            if measure.number is not None
        ]
        measure_total = max(measure_numbers) if measure_numbers else 0
        parts_table.add_row(
            payload.part_id,
            payload.name,
            f"{measure_total}",
            payload.instrument or "-",
        )
    console.print(parts_table)

    comment_table = Table(title="Global Comments", header_style="bold cyan")
    comment_table.add_column("Measure", style="green")
    comment_table.add_column("Comment")
    if spec.comments:
        for (measure_idx, _), text in sorted(spec.comments.items()):
            measure_label = f"m{measure_idx}" if measure_idx else "-"
            comment_table.add_row(measure_label, text)
    else:
        comment_table.add_row("-", "No global comments")
    console.print(comment_table)


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
