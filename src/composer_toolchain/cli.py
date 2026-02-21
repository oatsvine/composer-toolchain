"""Typer CLI interface to core toolchain."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import typer
from loguru import logger
from music21.humdrum.spineParser import GlobalComment
from music21.stream import Measure
from rich.panel import Panel
from typing_extensions import Annotated

import questionary
from questionary import Choice

from composer_toolchain.cli_helpers import (
    choose_score,
    console,
    render_measure_landmarks,
    render_score_metadata,
)
from composer_toolchain.core import Context, ScoreSpec
from composer_toolchain.score import MeasureSpec, PartSpec, load_score, normalize
from composer_toolchain.segmentation import segmentation_app
from composer_toolchain.sketch import sketch_app

# Refactor to use environment variable later.
SCORES_CORPUS_DIR = Path("/data/workspace/in/mxl")

app = typer.Typer()
app.add_typer(segmentation_app)
app.add_typer(sketch_app)


def _prompt_required_text(message: str, *, default: Optional[str] = None) -> str:
    """Prompt the user for a non-empty text value using questionary."""

    default_text = default or ""
    while True:
        reply = questionary.text(message, default=default_text).unsafe_ask()
        value = (reply or "").strip()
        if value:
            return value
        console.print("[red]A value is required.[/red]")
        default_text = ""


def _interactive_source_filename(
    workspace: Context, master_file: Path, provided: Optional[str]
) -> str:
    """Resolve the source score filename via questionary menus and chooser."""

    scores_dir = workspace.subdir("scores")
    master_name = master_file.name

    if provided:
        candidate = provided
    else:
        console.print(
            Panel(
                f"Master score: [bold]scores/{master_name}[/bold]\n"
                "Choose how to source the excerpt.",
                title="Source selection",
                border_style="green",
            )
        )
        action = questionary.select(
            "Select excerpt source",
            choices=[
                Choice(title=f"Use master ({master_name})", value="master"),
                Choice(title="Select another workspace score", value="workspace"),
                Choice(title="Import from corpus before excerpting", value="import"),
            ],
            default="master",
        ).unsafe_ask()
        if action == "master":
            candidate = master_name
        elif action == "workspace":
            selected = choose_score(scores_dir, filter_suffix={".krn"})
            candidate = selected.name
        else:
            console.print(
                Panel(
                    "Choose a corpus score to import (sk opens next).",
                    border_style="cyan",
                )
            )
            external = choose_score(SCORES_CORPUS_DIR)
            imported = workspace.import_score(score_file=external)
            console.print(
                f"[green]Imported[/green] {external.name} → {imported.relative_to(workspace.work_dir)}"
            )
            candidate = imported.name

    source_path = scores_dir / candidate
    if not source_path.exists():
        raise typer.BadParameter(
            f"Score not found under scores/: {candidate}",
            param_name="filename",
        )
    return candidate


def _multiselect_parts(
    spec: ScoreSpec, *, default_ids: Optional[set[str]] = None
) -> list[str]:
    default_ids = default_ids or {part.part_id for part in spec.parts}
    while True:
        choices: list[Choice] = []
        for part in spec.parts:
            label = part.name
            if part.abbreviation and part.abbreviation != part.name:
                label = f"{label} ({part.abbreviation})"
            instrument = part.instrument or "–"
            title = f"{label} · {instrument} [{part.part_id}]"
            choices.append(
                Choice(
                    title=title, value=part.part_id, checked=part.part_id in default_ids
                )
            )
        selected = questionary.checkbox(
            "Select parts for the excerpt",
            choices=choices,
            instruction="↑/↓ move · Space toggle · 'a' toggles all · Enter accept",
        ).unsafe_ask()
        if selected:
            return selected
        console.print("[red]Select at least one part.[/red]")
        default_ids = {part.part_id for part in spec.parts}


def _interactive_part_selection(
    spec: ScoreSpec, preset_ids: Optional[list[str]]
) -> list[str]:
    if preset_ids:
        preset_label = ", ".join(preset_ids)
        reuse = questionary.confirm(
            f"Use provided part list? {preset_label}", default=True
        ).unsafe_ask()
        if reuse:
            return preset_ids
        console.print("[cyan]Adjust selection below.[/cyan]")
    return _multiselect_parts(spec, default_ids=None)


def _interactive_measure_spec(spec: ScoreSpec, preset: Optional[str]) -> str:
    render_measure_landmarks(spec)
    if preset:
        preset_clean = preset.strip()
        reuse = questionary.confirm(
            f"Use existing measure spec? {preset_clean}", default=True
        ).unsafe_ask()
        if reuse:
            return preset_clean
        console.print(
            "[cyan]Enter new measure spans informed by the landmarks above.[/cyan]"
        )
    return _prompt_required_text(
        "Measure ranges (e.g. '5-12,18'; use ascending spans per MeasureSpec)",
        default=preset.strip() if preset else None,
    )


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
            help="Workspace root directory (the folder created via init-with-score).",
            exists=True,
            file_okay=False,
            resolve_path=True,
        ),
    ] = Path.cwd(),
    parts: Annotated[
        Optional[str],
        typer.Option(
            "--parts",
            help=(
                "Comma-separated canonical part ids to include "
                "(matches PartSpec tokens such as 'flute,vn1')."
            ),
        ),
    ] = None,
    measures: Annotated[
        Optional[str],
        typer.Option(
            "--measures",
            help=(
                "Measure ranges to extract, e.g. '1-8,17-'. Uses MeasureSpec "
                "grammar with ascending, comma-separated spans."
            ),
        ),
    ] = None,
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
    non_interactive: Annotated[
        bool,
        typer.Option(
            help=(
                "Skip Rich prompts (automation mode). Requires --parts/--measures "
                "and falls back to the master score when --filename is omitted."
            ),
        ),
    ] = False,
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

    provided_parts = parts.strip() if parts else None
    provided_measures = measures.strip() if measures else None

    provided_part_ids: Optional[list[str]] = None
    if provided_parts:
        try:
            provided_part_ids = PartSpec(tokens=provided_parts).ids
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_name="parts") from exc

    if non_interactive:
        if not provided_parts:
            raise typer.BadParameter(
                "--parts is required when using --non-interactive.",
                param_name="parts",
            )
        if not provided_measures:
            raise typer.BadParameter(
                "--measures is required when using --non-interactive.",
                param_name="measures",
            )
        parts_value = provided_parts
        measures_value = provided_measures
        source_name = filename.strip() if filename else master_name
    else:
        source_name = _interactive_source_filename(
            workspace=workspace,
            master_file=master_file,
            provided=filename.strip() if filename else None,
        )

        try:
            spec = workspace.score_spec(
                None if source_name == master_name else source_name
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_name="filename") from exc

        label = f"scores/{source_name}"
        if source_name == master_name:
            label = f"master (scores/{source_name})"
        render_score_metadata(spec, source_label=label)
        selected_ids = _interactive_part_selection(spec, provided_part_ids)
        parts_value = ",".join(selected_ids)
        measures_value = _interactive_measure_spec(spec, provided_measures)

    try:
        part_spec = PartSpec(tokens=parts_value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_name="parts") from exc

    try:
        measure_spec = MeasureSpec(spec=measures_value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_name="measures") from exc

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

    spec = ScoreSpec.build(score)
    try:
        display_master = score_path.relative_to(work_dir)
    except ValueError:
        display_master = score_path

    workspace_panel = Table.grid(padding=(0, 2))
    workspace_panel.add_row("Master", str(display_master))
    workspace_panel.add_row("Duration (QL)", f"{score.highestTime:.2f}")
    console.print(Panel(workspace_panel, title="Workspace", expand=True))
    render_score_metadata(spec, source_label=str(display_master))

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
    render_measure_landmarks(spec)

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
