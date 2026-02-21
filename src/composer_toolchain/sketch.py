"""Sketch generation workflow powered by OpenAI structured outputs."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import Optional

import questionary
import typer
from music21.humdrum.spineParser import GlobalComment
from music21.stream.base import Measure
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, field_validator
from questionary import Choice
from rich.panel import Panel
from typing_extensions import Annotated

from composer_toolchain.cli_helpers import (
    choose_score,
    console,
    render_measure_landmarks,
    render_score_metadata,
)
from composer_toolchain.core import Context, ScoreSpec
from composer_toolchain.prompting import (
    HUMDRUM_KERN_PRIMER,
    TECHNIQUE_PROMPTS,
    technique_guidance,
)
from composer_toolchain.score import kern_to_score, load_score, normalize, snake_case


DEFAULT_SKETCH_MODEL = "gpt-5.1"
_SKETCH_COMMENT_PREFIX = "SKETCH|"


class SketchPayload(BaseModel):
    """Structured payload describing the generated sketch."""

    title: str = Field(description="Short title for the generated sketch")
    suffix: str = Field(
        description=("Snake_case identifier appended to filenames inside the workspace")
    )
    annotated_kern: str = Field(
        description="Complete **kern document containing !! SKETCH comments"
    )
    commentary: str = Field(
        description="Single concise sentence explaining how the sketch satisfies the brief"
    )
    metrics: "SketchMetrics"

    @field_validator("title", "commentary", mode="after")
    @classmethod
    def _strip(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("value must not be empty")
        return cleaned

    @field_validator("suffix", mode="after")
    @classmethod
    def _slugify(cls, value: str) -> str:
        slug = snake_case(value)
        if not slug:
            raise ValueError("suffix must not be empty")
        return slug

    @field_validator("annotated_kern", mode="after")
    @classmethod
    def _validate_kern(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("annotated_kern must not be empty")
        return value


class SketchLLMResponse(SketchPayload):
    """Structured output schema returned by the OpenAI Responses API."""


class SketchContext(BaseModel):
    """Inputs required to drive sketch generation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    spec: ScoreSpec
    kern_text: str
    excerpt_label: str
    composer_brief: str
    measures: int = Field(ge=1, description="Exact measure count for the sketch")
    target_parts: int = Field(ge=1, description="Exact number of **kern spines to output")
    techniques: list[str] = Field(default_factory=list)

    @field_validator("composer_brief", mode="after")
    @classmethod
    def _normalize_brief(cls, value: str) -> str:
        brief = value.strip()
        if not brief:
            raise ValueError("composer brief must not be empty")
        return brief


class SketchResult(SketchPayload):
    """Validated LLM result ready for persistence."""

    model: str


class SketchMetrics(BaseModel):
    measure_count: int = Field(ge=1)
    part_count: int = Field(ge=1)


def generate_sketch(
    context: SketchContext,
    *,
    model: str = DEFAULT_SKETCH_MODEL,
) -> SketchResult:
    """Generate a **kern sketch by delegating to the OpenAI Responses API."""

    response = _call_openai(context, model=model)
    annotated_text = response.annotated_kern
    score = kern_to_score(annotated_text)
    _assert_measure_requirements(score, expected=context.measures)
    _assert_part_requirements(score, expected=context.target_parts)
    _assert_sketch_comments(score)
    _validate_sketch_metrics(response.metrics, context, score)
    return SketchResult(
        title=response.title,
        suffix=response.suffix,
        annotated_kern=annotated_text,
        commentary=response.commentary,
        metrics=response.metrics,
        model=model,
    )


def _call_openai(
    context: SketchContext,
    *,
    model: str,
) -> SketchLLMResponse:
    """Invoke OpenAI Responses API with structured output parsing."""

    client = OpenAI()
    system_prompt = dedent(
        """
        You are a composition professor steeped in Open Music Theory's “Foundational Concepts for Phrase-Level Forms.”
        Work hierarchically (motives → ideas → phrases) and describe how each span functions (presentation, continuation, cadential, codetta, etc.).
        Follow the Humdrum **kern reference (https://www.humdrum.org/Humdrum/representations/kern.html) for syntax, exclusive interpretations, and !! global comments.
        Insert !! SKETCH|m<start>-m<end>|label|function commentary immediately before the barline (`=<number>`) that begins each motivic/idea span.
        Maintain the original instrumentation order and emulate its meter/key/tempo profile unless the composer brief explicitly overrides it.
        Return a fully formed **kern document; never truncate metadata, never omit *- terminators, and never echo the source excerpt verbatim.
        Populate the `metrics` object so `measure_count` equals the measures you produced and `part_count` equals the number of **kern spines; if you cannot guarantee this, return no output.
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
        text_format=SketchLLMResponse,
    )
    parsed = raw_response.output_parsed
    if parsed is None:
        raise ValueError("Sketch model did not return parsed content")
    return parsed


def _build_user_prompt(context: SketchContext) -> str:
    spec = context.spec
    cue_lines = spec.cue_lines()
    metadata_lines = spec.metadata_lines()
    technique_block = technique_guidance(context.techniques)
    requirements = dedent(
        f"""
        Requirements:
        - Deliver a brand-new sketch spanning exactly {context.measures} numbered measures (start at measure 1, end at measure {context.measures}).
        - Notate exactly {context.target_parts} parallel **kern spines; preserve existing part names when possible and label any new voices clearly.
        - Carry forward meter, key, and tempo cues from the score summary unless the composer brief overrides them.
        - Outline motive/idea/phrase labels in the !! SKETCH comments so orchestration or editing decisions can track each span.
        - Include dynamics or articulations only when they clarify phrase shaping; focus on harmonic rhythm, contrapuntal contour, and cadence types.
        - Provide prose commentary (the `commentary` field) summarizing how the sketch satisfies the brief, referencing cadence types or prolongations drawn from the hierarchy in Open Music Theory.
        Reference excerpt (read-only; analyze but do not copy verbatim):
        """
    ).strip()
    sections = [
        f"Excerpt label: {context.excerpt_label}",
        "Score summary:",
        "\n".join(metadata_lines),
        "Structural cues (measure spans inclusive):",
        "\n".join(cue_lines) or "(No cues extracted)",
        ("Technique brief:\n" + technique_block) if technique_block else "",
        f"Composer brief:\n{context.composer_brief}",
        requirements,
        context.kern_text,
    ]
    return "\n\n".join(section for section in sections if section)


def _assert_measure_requirements(score, expected: int) -> None:
    first_part = score.parts[0] if score.parts else None
    if first_part is None:
        raise ValueError("Sketch must contain at least one part")
    measures = [
        int(m.number)
        for m in first_part.getElementsByClass(Measure)
        if m.number is not None
    ]
    if not measures:
        raise ValueError("Sketch must contain numbered measures")
    ordered_unique = sorted(dict.fromkeys(measures))
    expected_sequence = list(range(1, expected + 1))
    if ordered_unique != expected_sequence:
        raise ValueError(
            f"Sketch measures must cover 1..{expected}; got {ordered_unique[:5]}..."
        )


def _assert_part_requirements(score, expected: int) -> None:
    actual_parts = len(score.parts)
    if actual_parts != expected:
        raise ValueError(
            f"Sketch must contain exactly {expected} parts/spines; got {actual_parts}"
        )


def _assert_sketch_comments(score) -> None:
    comments = [
        (gc.comment or "").strip()
        for gc in score.recurse().getElementsByClass(GlobalComment)
        if (gc.comment or "").strip().startswith(_SKETCH_COMMENT_PREFIX)
    ]
    if not comments:
        raise ValueError("Sketch must include !! SKETCH global comments")


def _validate_sketch_metrics(metrics: SketchMetrics, context: SketchContext, score) -> None:
    if metrics.measure_count != context.measures:
        raise ValueError(
            f"Sketch metrics mismatch: expected {context.measures} measures, got {metrics.measure_count}"
        )
    if metrics.part_count != context.target_parts:
        raise ValueError(
            f"Sketch metrics mismatch: expected {context.target_parts} parts, got {metrics.part_count}"
        )


def _write_sketch_file(sketch_dir: Path, stem: str, annotated_text: str) -> Path:
    sketch_dir.mkdir(parents=True, exist_ok=True)
    candidate = sketch_dir / f"{stem}.krn"
    counter = 1
    while candidate.exists():
        candidate = sketch_dir / f"{stem}_{counter}.krn"
        counter += 1
    candidate.write_text(annotated_text, encoding="utf-8")
    return candidate


sketch_app = typer.Typer(help="Sketch generation workflows")


@sketch_app.command(name="create-sketch")
def create_sketch_cli(
    filename: Annotated[
        Optional[str],
        typer.Option(
            help=(
                "Excerpt filename under excerpts/ to feed as context; omit to choose interactively."
            ),
        ),
    ] = None,
    instructions_file: Annotated[
        Path,
        typer.Option(
            "--instructions-file",
            help="Path to a text brief describing the desired sketch (required).",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = ...,  # type: ignore[assignment]
    measures: Annotated[
        int,
        typer.Option(
            "--measures",
            min=1,
            help="Exact length of the generated sketch in measures.",
        ),
    ] = 8,
    parts: Annotated[
        Optional[int],
        typer.Option(
            "--parts",
            min=1,
            help="Exact number of parts/voices in the generated sketch.",
        ),
    ] = None,
    techniques: Annotated[
        Optional[list[str]],
        typer.Option(
            "--technique",
            help="Technique lexicon entries (repeat flag to combine).",
        ),
    ] = None,
    model: Annotated[
        str,
        typer.Option(
            help="OpenAI reasoning model used for sketch generation.",
        ),
    ] = DEFAULT_SKETCH_MODEL,
    work_dir: Annotated[
        Path,
        typer.Option(
            help="Workspace root directory (initialized via init-with-score).",
            exists=True,
            file_okay=False,
            resolve_path=True,
        ),
    ] = Path.cwd(),
) -> Path:
    """Generate a new **kern sketch guided by a composer brief."""

    workspace = Context(work_dir=work_dir)
    excerpts_dir = workspace.subdir("excerpts")
    if filename:
        candidate = excerpts_dir / Path(filename).name
    else:
        candidate = choose_score(excerpts_dir, filter_suffix={".krn"})
    if not candidate.exists():
        raise typer.BadParameter(f"Excerpt not found: {candidate}")

    brief_text = instructions_file.read_text(encoding="utf-8").strip()
    if not brief_text:
        raise typer.BadParameter(
            "Instructions file is empty", param_hint="instructions_file"
        )

    kern_text = candidate.read_text(encoding="utf-8")
    score = normalize(load_score(candidate))
    spec = ScoreSpec.build(score)
    label = f"excerpts/{candidate.name}"
    render_score_metadata(spec, source_label=label)
    render_measure_landmarks(spec)
    selected_techniques = _resolve_techniques(techniques)
    target_parts = parts or len(spec.parts)

    context = SketchContext(
        spec=spec,
        kern_text=kern_text,
        excerpt_label=candidate.stem,
        composer_brief=brief_text,
        measures=measures,
        target_parts=target_parts,
        techniques=selected_techniques,
    )
    try:
        result = generate_sketch(context=context, model=model)
    except Exception as exc:  # pragma: no cover - OpenAI failures are external
        console.print(f"[red]Sketch generation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    sketches_dir = workspace.subdir("sketches")
    stem = f"{candidate.stem}_{result.suffix}"
    target = _write_sketch_file(sketches_dir, stem, result.annotated_kern)

    console.print(Panel(result.commentary, title=result.title, border_style="green"))
    console.print(
        f"[blue]Metrics[/blue]: measures={result.metrics.measure_count} · parts={result.metrics.part_count}"
    )
    rel = target.relative_to(work_dir)
    console.print(f"[green]Sketch stored[/green]: {rel} (model: {result.model})")
    return target


def _resolve_techniques(preset: Optional[list[str]]) -> list[str]:
    if preset:
        return [_normalize_technique(name) for name in preset]
    if not TECHNIQUE_PROMPTS:
        return []
    selected = questionary.checkbox(
        "Select sketch techniques",
        choices=[Choice(title=label.title(), value=label) for label in sorted(TECHNIQUE_PROMPTS)],
        instruction="↑/↓ move · Space toggle · 'a' toggles all · Enter accept",
    ).unsafe_ask()
    return selected or []


def _normalize_technique(name: str) -> str:
    canonical = name.strip().lower()
    if canonical not in TECHNIQUE_PROMPTS:
        raise typer.BadParameter(
            f"Unknown technique '{name}'. Choices: {', '.join(sorted(TECHNIQUE_PROMPTS))}"
        )
    return canonical


__all__ = [
    "DEFAULT_SKETCH_MODEL",
    "SketchContext",
    "SketchResult",
    "generate_sketch",
    "sketch_app",
]
