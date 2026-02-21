"""Segmentation analysis workflow powered by OpenAI structured outputs."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from textwrap import dedent
from typing import Iterable, Optional, Sequence

import typer
from music21.humdrum.spineParser import GlobalComment
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from rich.panel import Panel
from rich.table import Table
from typing_extensions import Annotated

from composer_toolchain.cli_helpers import (
    choose_score,
    console,
    render_measure_landmarks,
    render_score_metadata,
)
from composer_toolchain.core import Context, ScoreSpec
from composer_toolchain.prompting import HUMDRUM_KERN_PRIMER
from composer_toolchain.score import kern_to_score, load_score, normalize, snake_case


DEFAULT_SEGMENTATION_MODEL = "gpt-5.1"
_SEGMENT_COMMENT_PREFIXES = ("!! SEGMENT|", "!!SEGMENT|")


class SegmentLevel(str, Enum):
    """Hierarchical level consistent with Open Music Theory vernacular."""

    MOTIVE = "motive"
    IDEA = "idea"
    PHRASE = "phrase"
    SECTION = "section"


class SegmentLabel(BaseModel):
    """Structured payload describing a labeled musical span."""

    name: str = Field(description="Descriptive segment label (Antecedent idea, Motive a)")
    level: SegmentLevel = Field(description="Hierarchy level: motive, idea, phrase, or section")
    measure_start: int = Field(ge=1, description="Inclusive starting measure number in the excerpt")
    measure_end: int = Field(ge=1, description="Inclusive ending measure number in the excerpt")
    suffix: str = Field(description="snake_case identifier for future excerpt filenames")
    reasoning: str = Field(description="Music-theory rationale referencing motives, ideas, cadences")

    @field_validator("name", "reasoning", mode="after")
    @classmethod
    def _strip(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("value must not be empty")
        return cleaned

    @field_validator("suffix", mode="after")
    @classmethod
    def _canonical_suffix(cls, value: str) -> str:
        slug = snake_case(value)
        if not slug:
            raise ValueError("suffix must not be empty")
        return slug

    @model_validator(mode="after")
    def _validate_span(self) -> "SegmentLabel":
        if self.measure_end < self.measure_start:
            raise ValueError("measure_end must be >= measure_start")
        return self


class SegmentationPayload(BaseModel):
    """Shared payload describing segmentation spans and annotated **kern."""

    segments: list[SegmentLabel]
    annotated_kern: str = Field(
        description="Full **kern excerpt plus segmentation !! comments"
    )
    overview: str = Field(
        description="High-level description of phrase/idea/motive architecture"
    )

    @field_validator("annotated_kern")
    @classmethod
    def _require_content(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("annotated_kern must not be empty")
        return value


class SegmentationLLMResponse(SegmentationPayload):
    """Structured output schema handed back by the OpenAI Responses API."""


class SegmentationContext(BaseModel):
    """Context passed into the segmentation runner."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    spec: ScoreSpec
    kern_text: str
    excerpt_label: str


class SegmentationResult(SegmentationPayload):
    """Validated segmentation outcome ready for downstream display/storage."""

    model: str


def analyze_segmentation(
    context: SegmentationContext,
    *,
    model: str = DEFAULT_SEGMENTATION_MODEL,
) -> SegmentationResult:
    """Run OpenAI-powered segmentation with validation and integrity checks."""

    response = _call_openai(context, model=model)
    spec = context.spec
    segments = _validate_segment_ranges(response.segments, spec)
    _ensure_suffix_uniqueness(segments)
    annotated_text = response.annotated_kern
    score = kern_to_score(annotated_text)
    _assert_segment_comments(score, segments)
    return SegmentationResult(
        segments=segments,
        overview=response.overview.strip(),
        annotated_kern=annotated_text,
        model=model,
    )


def _call_openai(
    context: SegmentationContext,
    *,
    model: str,
) -> SegmentationLLMResponse:
    """Invoke OpenAI Responses API with structured output parsing."""

    client = OpenAI()
    system_prompt = dedent(
        """
        You are a classical-form analyst steeped in Open Music Theory's “Foundational Concepts for Phrase-Level Forms.”
        Work hierarchically: movements → sections → themes → phrases → ideas → motives.
        Identify cadential endings to delimit phrases, then describe the idea- and motive-level surface inside each phrase.
        Focus on motives and ideas first; annotate phrases only when cadential function must be named.
        Motives remain compact (a beat or two) and recur or transform. Ideas bundle motives (presentation/continuation/cadential) before phrases.
        Cite meter, key, and tempo pivots to justify segmentation choices.
        Use Humdrum **kern global comments (lines beginning with “!!”) per the Humdrum Reference on global comments.
        Insert segmentation markers as `!! SEGMENT|suffix|name|level|m<start>-m<end>|short synopsis` immediately before the `=<number>` barline that starts each segment.
        Never alter **kern tokens or metadata beyond inserting those exact comment lines.
        Keep suffix suggestions snake_case so users can reuse them as excerpt filenames.
        """
    ).strip()
    system_prompt = f"{system_prompt}\n\n{HUMDRUM_KERN_PRIMER}"
    user_prompt = _build_user_prompt(context)
    raw_response = client.responses.parse(  # type: ignore[attr-defined]
        model=model,
        temperature=0.2,
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": system_prompt,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": user_prompt,
                    }
                ],
            },
        ],
        text_format=SegmentationLLMResponse,
    )
    parsed = raw_response.output_parsed
    if parsed is None:
        raise ValueError("Segmentation model did not return parsed content")
    return parsed


def _build_user_prompt(context: SegmentationContext) -> str:
    spec = context.spec
    metadata_lines = spec.metadata_lines()
    cue_lines = spec.cue_lines()
    first_measure = spec.first_measure_number()
    last_measure = spec.total_measures
    requirements = dedent(
        f"""
        Requirements:
        - Cover the excerpt from measure {first_measure} through measure {last_measure} with contiguous, non-overlapping spans.
        - Give each span a meaningful musical name and explain how motives/ideas function (antecedent, consequent, continuation, cadential idea, codetta, etc.).
        - Focus on motive and idea levels first, then mention a phrase only when a cadence closes the passage.
        - Reiterate meter/key/tempo cues from the score summary when motivating cadences.
        - Provide suffix suggestions that musicians can reuse when carving more granular excerpts (letters, digits, underscores only).
        - Return the untouched **kern excerpt with nothing changed except the added segmentation comments described earlier.
        Base excerpt (read-only apart from the segmentation comments):
        """
    ).strip()
    sections = [
        f"Excerpt label: {context.excerpt_label}",
        "Score summary:",
        "\n".join(metadata_lines),
        "Structural cues (measure spans inclusive):",
        "\n".join(cue_lines) or "(No cues extracted)",
        requirements,
        context.kern_text,
    ]
    return "\n\n".join(section for section in sections if section)


def _validate_segment_ranges(segments: Sequence[SegmentLabel], spec: ScoreSpec) -> list[SegmentLabel]:
    if not segments:
        raise ValueError("Segmentation response must include at least one segment")
    ordered = sorted(segments, key=lambda seg: (seg.measure_start, seg.measure_end))
    first_measure = spec.first_measure_number()
    last_measure = spec.total_measures
    if ordered[0].measure_start != first_measure:
        raise ValueError(
            f"Segments must start at measure {first_measure} (excerpt open), got {ordered[0].measure_start}"
        )
    if ordered[-1].measure_end != last_measure:
        raise ValueError(
            f"Segments must end at measure {last_measure} (excerpt close), got {ordered[-1].measure_end}"
        )
    prev_end = first_measure - 1
    for seg in ordered:
        if seg.measure_start != prev_end + 1:
            raise ValueError(
                f"Segment at measures {seg.measure_start}-{seg.measure_end} is not contiguous after measure {prev_end}"
            )
        if seg.measure_end > last_measure:
            raise ValueError("Segment exceeds excerpt length")
        prev_end = seg.measure_end
    return list(ordered)


def _ensure_suffix_uniqueness(segments: Iterable[SegmentLabel]) -> None:
    seen: set[str] = set()
    for seg in segments:
        if seg.suffix in seen:
            raise ValueError(f"Duplicate segment suffix '{seg.suffix}'")
        seen.add(seg.suffix)


def _assert_segment_comments(score, segments: Sequence[SegmentLabel]) -> None:
    if not segments:
        raise ValueError("Segmentation response must contain segments")
    required_suffixes = {seg.suffix for seg in segments}
    available_suffixes: set[str] = set()
    for gc in score.recurse().getElementsByClass(GlobalComment):
        text = (gc.comment or "").strip()
        if not text.startswith("SEGMENT|"):
            continue
        parts = text.split("|", 2)
        if len(parts) >= 2:
            available_suffixes.add(parts[1])
    missing = required_suffixes - available_suffixes
    if missing:
        formatted = ", ".join(sorted(missing))
        raise ValueError(
            f"Annotated kern missing SEGMENT comments for suffixes: {formatted}"
        )


def _render_segmentation_overview(overview: str) -> None:
    summary = overview.strip()
    if not summary:
        return
    console.print(Panel(summary, title="Segmentation Overview", border_style="magenta"))


def _render_segmentation_segments(segments: list[SegmentLabel]) -> None:
    if not segments:
        return
    table = Table(title="Segments", header_style="bold magenta")
    table.add_column("Measures", style="yellow", no_wrap=True)
    table.add_column("Level", style="cyan", no_wrap=True)
    table.add_column("Name")
    table.add_column("Suffix", style="green", no_wrap=True)
    table.add_column("Rationale")
    for seg in segments:
        measures = (
            str(seg.measure_start)
            if seg.measure_start == seg.measure_end
            else f"{seg.measure_start}-{seg.measure_end}"
        )
        table.add_row(
            measures,
            seg.level.value,
            seg.name,
            seg.suffix,
            seg.reasoning,
        )
    console.print(table)


def _write_segmentation_file(source: Path, annotated_text: str) -> Path:
    candidate = source.with_name(f"{source.stem}_segmentation{source.suffix}")
    counter = 1
    while candidate.exists():
        candidate = source.with_name(f"{source.stem}_segmentation_{counter}{source.suffix}")
        counter += 1
    candidate.write_text(annotated_text, encoding="utf-8")
    return candidate


segmentation_app = typer.Typer(help="Segmentation workflows")


@segmentation_app.command(name="analyze-segmentation")
def analyze_segmentation_cli(
    work_dir: Annotated[
        Path,
        typer.Argument(
            help="Workspace root directory (initialized via init-with-score).",
            exists=True,
            file_okay=False,
            resolve_path=True,
        ),
    ] = Path.cwd(),
    filename: Annotated[
        Optional[str],
        typer.Option(
            help="Excerpt filename under excerpts/ to analyze; omit for interactive chooser.",
        ),
    ] = None,
    model: Annotated[
        str,
        typer.Option(
            help="OpenAI reasoning model (requires OPENAI_API_KEY).",
        ),
    ] = DEFAULT_SEGMENTATION_MODEL,
) -> Path:
    """Run OpenAI segmentation on an existing excerpt and emit annotated **kern."""

    workspace = Context(work_dir=work_dir)
    excerpts_dir = workspace.subdir("excerpts")
    if filename:
        candidate = excerpts_dir / Path(filename).name
    else:
        candidate = choose_score(excerpts_dir, filter_suffix={".krn"})
    if not candidate.exists():
        raise typer.BadParameter(f"Excerpt not found: {candidate}")

    kern_text = candidate.read_text(encoding="utf-8")
    score = normalize(load_score(candidate))
    spec = ScoreSpec.build(score)
    label = f"excerpts/{candidate.name}"
    render_score_metadata(spec, source_label=label)
    render_measure_landmarks(spec)

    context = SegmentationContext(
        spec=spec,
        kern_text=kern_text,
        excerpt_label=candidate.stem,
    )
    try:
        result = analyze_segmentation(context=context, model=model)
    except Exception as exc:  # pragma: no cover - defensive guard for OpenAI failures
        console.print(f"[red]Segmentation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    target = _write_segmentation_file(candidate, result.annotated_kern)
    _render_segmentation_overview(result.overview)
    _render_segmentation_segments(result.segments)

    rel = target.relative_to(work_dir)
    console.print(
        f"[green]Segmentation annotated[/green]: {rel} (model: {result.model})"
    )
    return target


__all__ = [
    "DEFAULT_SEGMENTATION_MODEL",
    "SegmentLabel",
    "SegmentationContext",
    "SegmentationResult",
    "analyze_segmentation",
    "segmentation_app",
]
