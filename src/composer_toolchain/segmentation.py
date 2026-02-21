"""Segmentation analysis workflow powered by OpenAI structured outputs."""

from __future__ import annotations

from enum import Enum
from textwrap import dedent
from typing import Callable, Iterable, Optional, Sequence

from music21.humdrum.spineParser import GlobalComment
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from composer_toolchain.core import ScoreSpec
from composer_toolchain.score import kern_to_score, snake_case


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


LLMRunner = Callable[[SegmentationContext], SegmentationPayload]


def analyze_segmentation(
    context: SegmentationContext,
    *,
    model: str = DEFAULT_SEGMENTATION_MODEL,
    client: Optional[OpenAI] = None,
    runner: Optional[LLMRunner] = None,
) -> SegmentationResult:
    """Run OpenAI-powered segmentation with validation and integrity checks."""

    response = runner(context) if runner else _call_openai(context, model=model, client=client)
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
    client: Optional[OpenAI] = None,
) -> SegmentationLLMResponse:
    """Invoke OpenAI Responses API with structured output parsing."""

    client = client or OpenAI()
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
    metadata_lines = _score_metadata_lines(spec)
    cue_lines = _cue_lines(spec)
    first_measure = _first_measure(spec)
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


def _score_metadata_lines(spec: ScoreSpec) -> list[str]:
    lines: list[str] = [f"• Title: {spec.title}"]
    if spec.composer:
        lines.append(f"• Composer: {spec.composer}")
    if spec.movements:
        entries = []
        for mv in spec.movements:
            parts: list[str] = []
            if mv.number is not None:
                parts.append(str(mv.number))
            if mv.title:
                parts.append(mv.title)
            if parts:
                entries.append(" ".join(parts))
        if entries:
            lines.append(f"• Movements: {', '.join(entries)}")
    part_listing = ", ".join(f"{p.part_id}:{p.name}" for p in spec.parts)
    lines.append(f"• Parts: {part_listing}")
    lines.append(f"• Total measures: {spec.total_measures}")
    return lines


def _cue_lines(spec: ScoreSpec) -> list[str]:
    rows: list[str] = []
    for span in spec.iter_cue_spans():
        measure_label = (
            str(span.start_measure)
            if span.start_measure == span.end_measure
            else f"{span.start_measure}-{span.end_measure}"
        )
        rows.append(
            f"  · m{measure_label}: meter={span.time_signature or '—'}, key={span.key_signature or '—'}, tempo={span.tempo_bpm or '—'}"
        )
    return rows


def _first_measure(spec: ScoreSpec) -> int:
    if spec.measure_cues:
        return min(spec.measure_cues)
    return 1


def _validate_segment_ranges(segments: Sequence[SegmentLabel], spec: ScoreSpec) -> list[SegmentLabel]:
    if not segments:
        raise ValueError("Segmentation response must include at least one segment")
    ordered = sorted(segments, key=lambda seg: (seg.measure_start, seg.measure_end))
    first_measure = _first_measure(spec)
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
