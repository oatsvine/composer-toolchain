
from pathlib import Path
from typing import Dict, Literal, Optional, Sequence, Tuple

from loguru import logger
from music21.humdrum.spineParser import GlobalComment
from music21.key import KeySignature
from music21.meter.base import TimeSignature
from music21.stream import Score
from music21.stream.base import Measure, Part
from music21.tempo import MetronomeMark
from pydantic import BaseModel, ConfigDict, Field

from composer_toolchain.score import (MeasureSpec, PartSpec, create_excerpt,
                                      delete_measures, insert_blank_measures,
                                      load_score, merge_excerpt, normalize,
                                      snake_case)

class PartInfo(BaseModel):
    part_id: str
    name: str
    abbreviation: str
    instrument: Optional[str] = None


class MovementInfo(BaseModel):
    number: Optional[int] = Field(
        default=None,
        description="Movement index within the work's formal hierarchy (1-based).",
    )
    title: Optional[str] = Field(
        default=None,
        description="Movement or section label (e.g., 'Allegro').",
    )


class MeasureCue(BaseModel):
    measure: int = Field(description="Measure number for this structural snapshot")
    time_signature: Optional[str] = Field(
        default=None, description="Active time signature ratio string"
    )
    key_signature: Optional[str] = Field(
        default=None, description="Key summary such as 'G major'"
    )
    tempo_bpm: Optional[int] = Field(
        default=None, description="Metronome marking in beats per minute"
    )


class CueSpan(BaseModel):
    start_measure: int
    end_measure: int
    time_signature: Optional[str]
    key_signature: Optional[str]
    tempo_bpm: Optional[int]


class ScoreSpec(BaseModel):
    """Composer-facing summary of a normalized score's structure.

    Inspired by the hierarchy outlined in *Foundational Concepts for Phrase-Level
    Forms*, `ScoreSpec` captures only the layers composers scan when shaping
    phrases: title/composer at the top, movements for sectional context,
    instrumentation, and a DRY map of measure-level cues (meter, key, tempo).
    """

    model_config = ConfigDict(extra="forbid")

    title: str = Field(
        description=(
            "Primary title suitable for labeling the movement/section (falls back"
            " to the file stem when metadata is missing)."
        )
    )
    composer: Optional[str] = Field(
        default=None,
        description=(
            "Composer credit—paired with the title, it hints at phrase and"
            " cadence conventions a musician might expect."
        ),
    )
    parts: list[PartInfo] = Field(
        description=(
            "Ordered list of parts/instrumentation. Canonical part_ids keep"
            " excerpts mergeable without guesswork."
        )
    )
    movements: list[MovementInfo] = Field(
        default_factory=list,
        description=(
            "Movement or section labels supplying the large-scale context above"
            " phrase-level edits."
        ),
    )
    measure_cues: Dict[int, MeasureCue] = Field(
        default_factory=dict,
        description=(
            "Aggregated structural cues per measure (time signature, key, tempo)"
            " derived from the normalized score."
        ),
    )
    comments: Dict[Tuple[int, float], str] = Field(
        default_factory=dict,
        description=(
            "(Measure, offset) → global comments for rehearsal or analysis" " markings."
        ),
    )
    total_measures: int = Field(
        ge=0,
        description="Highest numbered measure present in the normalized score",
    )

    def primary_movement(self) -> Optional[MovementInfo]:
        """Return the first movement entry, if any."""

        return self.movements[0] if self.movements else None

    def metadata_lines(self) -> list[str]:
        """Return bullet-style metadata lines for prompt contexts."""

        lines: list[str] = [f"• Title: {self.title}"]
        if self.composer:
            lines.append(f"• Composer: {self.composer}")
        if self.movements:
            entries: list[str] = []
            for mv in self.movements:
                bits: list[str] = []
                if mv.number is not None:
                    bits.append(str(mv.number))
                if mv.title:
                    bits.append(mv.title)
                if bits:
                    entries.append(" ".join(bits))
            if entries:
                lines.append(f"• Movements: {', '.join(entries)}")
        part_listing = ", ".join(f"{p.part_id}:{p.name}" for p in self.parts)
        if part_listing:
            lines.append(f"• Parts: {part_listing}")
        lines.append(f"• Total measures: {self.total_measures}")
        return lines

    def cue_lines(self) -> list[str]:
        """Return structural cue summaries for measure spans."""

        rows: list[str] = []
        for span in self.iter_cue_spans():
            measure_label = (
                str(span.start_measure)
                if span.start_measure == span.end_measure
                else f"{span.start_measure}-{span.end_measure}"
            )
            rows.append(
                f"  · m{measure_label}: meter={span.time_signature or '—'}, "
                f"key={span.key_signature or '—'}, tempo={span.tempo_bpm or '—'}"
            )
        return rows

    def first_measure_number(self) -> int:
        """Return the lowest numbered measure present in the score."""

        if self.measure_cues:
            return min(self.measure_cues)
        return 1

    def iter_cue_spans(self) -> list[CueSpan]:
        """Return contiguous measure ranges where structural cues stay constant."""

        if self.total_measures <= 0:
            return []

        spans: list[CueSpan] = []
        current_ts: Optional[str] = None
        current_key: Optional[str] = None
        current_tempo: Optional[int] = None
        span_start = 1

        for measure in range(1, self.total_measures + 1):
            cue = self.measure_cues.get(measure)
            ts = current_ts
            key = current_key
            tempo = current_tempo
            if cue:
                if cue.time_signature is not None:
                    ts = cue.time_signature
                if cue.key_signature is not None:
                    key = cue.key_signature
                if cue.tempo_bpm is not None:
                    tempo = cue.tempo_bpm

            if measure == span_start:
                current_ts, current_key, current_tempo = ts, key, tempo
                continue

            if ts != current_ts or key != current_key or tempo != current_tempo:
                spans.append(
                    CueSpan(
                        start_measure=span_start,
                        end_measure=measure - 1,
                        time_signature=current_ts,
                        key_signature=current_key,
                        tempo_bpm=current_tempo,
                    )
                )
                span_start = measure
                current_ts, current_key, current_tempo = ts, key, tempo

        spans.append(
            CueSpan(
                start_measure=span_start,
                end_measure=self.total_measures,
                time_signature=current_ts,
                key_signature=current_key,
                tempo_bpm=current_tempo,
            )
        )
        return spans

    @classmethod
    def build(cls, score: Score) -> "ScoreSpec":
        meta = score.metadata
        title_raw = meta.title if meta and meta.title else None
        best_title = meta.bestTitle if meta else None
        movement_name = meta.movementName if meta and meta.movementName else None
        movement_number = (
            str(meta.movementNumber)
            if meta and meta.movementNumber is not None
            else None
        )
        composer = meta.composer if meta else None
        title_display = title_raw or best_title or "Untitled"

        parts_payload: list[PartInfo] = []
        for part in score.parts:
            part_id = str(part.id) if part.id is not None else _canonical_part_id(part)
            instrument_name: Optional[str] = None
            instrument = part.getInstrument(returnDefault=False)
            if instrument is not None:
                instrument_name = instrument.instrumentName or instrument.partName
                if instrument_name is None:
                    try:
                        instrument_name = instrument.bestName()
                    except AttributeError:
                        instrument_name = None
            parts_payload.append(
                PartInfo(
                    part_id=part_id,
                    name=part.partName or part_id,
                    abbreviation=part.partAbbreviation or part.partName or part_id,
                    instrument=instrument_name,
                )
            )

        structural_part = score.parts[0] if score.parts else None
        movement_entries: list[MovementInfo] = []
        if movement_name or movement_number is not None:
            try:
                idx = int(movement_number) if movement_number is not None else None
            except ValueError:
                idx = None
            movement_entries.append(MovementInfo(number=idx, title=movement_name))

        measure_cues: Dict[int, MeasureCue] = {}
        last_measure = 0
        if structural_part is not None:
            for measure in structural_part.getElementsByClass(Measure):
                if measure.number is None:
                    continue
                m_number = int(measure.number)
                last_measure = max(last_measure, m_number)
                cue = measure_cues.get(m_number)
                if cue is None:
                    cue = MeasureCue(measure=m_number)
                    measure_cues[m_number] = cue

                ts = measure.getElementsByClass(TimeSignature).first()
                if ts is None:
                    ts = measure.getContextByClass(TimeSignature)
                if ts is not None:
                    cue.time_signature = ts.ratioString

                ks = measure.getElementsByClass(KeySignature).first()
                if ks is None:
                    ks = measure.getContextByClass(KeySignature)
                if ks is not None:
                    try:
                        key_obj = ks.asKey()
                        cue.key_signature = f"{key_obj.tonic.name} {key_obj.mode}"
                    except Exception:  # pragma: no cover - exotic key signature
                        cue.key_signature = str(ks.sharps)

                for mark in measure.recurse().getElementsByClass(MetronomeMark):
                    if mark.number is None:
                        continue
                    cue.tempo_bpm = int(round(mark.number))

        comments: Dict[Tuple[int, float], str] = {}
        for gc in score.getElementsByClass(GlobalComment):
            measure_ctx = gc.getContextByClass(Measure)
            m_number = (
                int(measure_ctx.number)
                if measure_ctx is not None and measure_ctx.number is not None
                else 0
            )
            try:
                offset = float(gc.getOffsetBySite(score))
            except Exception:
                offset = float(gc.offset or 0.0)
            key = (m_number, offset)
            if key in comments:
                comments[key] = f"{comments[key]}\n{gc.comment}"
            else:
                comments[key] = gc.comment

        # NOTE: Use first part with advertizing it anywhere because normalize guarantees all parts have same structure.
        part = score.getElementsByClass(Part).first()
        assert part, f"Score must contain at least one part: {score}"
        return cls(
            title=title_display,
            composer=composer,
            parts=parts_payload,
            movements=movement_entries,
            measure_cues=measure_cues,
            comments=comments,
            total_measures=last_measure,
        )
