"""Core toolchain logic for score workspace management and operations.

This module provides the `Context` class, which encapsulates operations
for managing a musical score workspace. It includes functionality for
importing scores, changing the master score, creating and merging excerpts,
and exporting MIDI files.

All toolchain operations are encapsulated here to maintain a clean separation
from CLI interfaces and higher-level application logic.
"""

import re
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

from composer_toolchain.score import (
    MeasureSpec,
    PartSpec,
    create_excerpt,
    delete_measures,
    insert_blank_measures,
    load_score,
    merge_excerpt,
    normalize,
    snake_case,
)


class Manifest(BaseModel):
    master: str


class WorkspaceMeta(Manifest):
    work_dir: Path


class PartInfo(BaseModel):
    part_id: str
    name: str
    abbreviation: str
    instrument: Optional[str] = None


class ScoreSpec(BaseModel):
    """Structured snapshot of a normalized score and its metadata."""

    model_config = ConfigDict(extra="forbid")

    # NOTE: This is more than enough medatata. We don't want music21 parity.
    title: str = Field(description="Primary title (metadata.title or fallback stem)")
    movement_number: Optional[str] = Field(
        default=None,
        description="Movement identifier as exposed by metadata.movementNumber",
    )
    movement_name: Optional[str] = Field(
        default=None, description="Movement title from metadata.movementName"
    )
    composer: Optional[str] = Field(
        default=None, description="Preferred composer summary string"
    )

    parts: list[PartInfo] = Field(
        description="List of parts in the score; part_id must be unique and non-empty"
    )

    # NOTE: use only the **first part** of a score as proxy to determine these maps.
    movements: Dict[int, str] = Field(
        description="Map movement number to movement title"
    )
    time_signatures: Dict[int, str] = Field(
        description="Map measure number to time signature"
    )
    key_signatures: Dict[int, str] = Field(
        description="Map measure number to key signature"
    )
    metronome_marks: Dict[Tuple[int, int], int] = Field(
        description="Map (measure number, beat) to metronome mark (number)"
    )
    comments: Dict[Tuple[int, float], str] = Field(
        description="Map (measure number, offset) to comment text"
    )

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
                instrument_name = (
                    instrument.instrumentName
                    or instrument.partName
                    or getattr(instrument, "bestName", lambda: None)()
                )
            parts_payload.append(
                PartInfo(
                    part_id=part_id,
                    name=part.partName or part_id,
                    abbreviation=part.partAbbreviation or part.partName or part_id,
                    instrument=instrument_name,
                )
            )

        structural_part = score.parts[0] if score.parts else None
        movements: Dict[int, str] = {}
        if movement_name:
            try:
                idx = int(movement_number) if movement_number is not None else 1
            except ValueError:
                idx = 1
            movements[idx] = movement_name

        time_signatures: Dict[int, str] = {}
        key_signatures: Dict[int, str] = {}
        metronome_marks: Dict[Tuple[int, int], int] = {}
        if structural_part is not None:
            for measure in structural_part.getElementsByClass(Measure):
                if measure.number is None:
                    continue
                m_number = int(measure.number)
                ts = measure.getElementsByClass(TimeSignature).first()
                if ts is None:
                    ts = measure.getContextByClass(TimeSignature)
                if ts is not None:
                    time_signatures[m_number] = ts.ratioString

                ks = measure.getElementsByClass(KeySignature).first()
                if ks is None:
                    ks = measure.getContextByClass(KeySignature)
                if ks is not None:
                    try:
                        key_obj = ks.asKey()
                        key_signatures[m_number] = (
                            f"{key_obj.tonic.name} {key_obj.mode}"
                        )
                    except Exception:  # pragma: no cover - exotic key signature
                        key_signatures[m_number] = str(ks.sharps)

                for mark in measure.recurse().getElementsByClass(MetronomeMark):
                    if mark.number is None:
                        continue
                    beat_length = 1.0
                    if ts is not None:
                        beat_length = float(ts.beatDuration.quarterLength or 1.0)
                        if beat_length <= 1e-6:
                            beat_length = 1.0
                    beat = int(mark.offset / beat_length) + 1 if beat_length > 0 else 1
                    metronome_marks[(m_number, beat)] = int(round(mark.number))

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
            movement_number=movement_number,
            movement_name=movement_name,
            composer=composer,
            parts=parts_payload,
            movements=movements,
            time_signatures=time_signatures,
            key_signatures=key_signatures,
            metronome_marks=metronome_marks,
            comments=comments,
        )


WORKSPACE_DIRS = Literal["scores", "excerpts", "midi", "versions"]


def init_workspace(work_dir: Path) -> None:
    """Initialize the workspace directory structure."""
    for dirname in ["scores", "excerpts", "midi", "versions"]:
        subdir = work_dir / dirname
        subdir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Initialized workspace subdirectory: {subdir}")


def workspace_subdir(work_dir: Path, dirname: WORKSPACE_DIRS) -> Path:
    """Return the excerpts directory path."""
    subdir = work_dir / dirname
    if not subdir.exists():
        raise FileNotFoundError(f"Excerpts directory does not exist: {subdir}")
    return subdir


class Context:
    """CLI utilities for score conversions and edit operations."""

    def __init__(
        self,
        work_dir: Path,
    ) -> None:
        # NOTE: This directory is critical, it is our working directory for all operations, no external files allowed after workspace created.
        resolved = work_dir.resolve()
        logger.info(f"Working directory '{resolved}'")
        self.work_dir = resolved
        assert self.work_dir.exists(), f"Work directory does not exist: {self.work_dir}"

    def subdir(self, dirname: WORKSPACE_DIRS) -> Path:
        return workspace_subdir(self.work_dir, dirname)

    # NOTE: This is how we create the workspace from a score, starts from a score file and creates a new subdirectory in `cwd`.
    @staticmethod
    def init_with_score(score_file: Path, cwd: Path) -> Path:
        """
        Initialize a working directory with a normalized score in **kern format.

        This is the first step in the workflow of this toolchain.
        This inits from an external score, future variant is to init from a blueprint.
        """
        source = score_file.resolve()
        if not source.exists():
            raise FileNotFoundError(f"Score file not found: {source}")
        root = cwd.resolve()
        score = normalize(load_score(source))
        # NOTE: We only save working scores in krn format. MusicXML is for external scores only, not to save work.
        stem = snake_case(source.stem)
        work_dir = root / stem
        init_workspace(work_dir)
        workspace_scores = workspace_subdir(work_dir, "scores")
        target = workspace_scores / f"{stem}.krn"
        score.write("humdrum", fp=target)
        manifest_file = work_dir / "MANIFEST.json"
        manifest = Manifest(master=target.name)
        manifest_file.write_text(manifest.model_dump_json(indent=2))
        logger.success("Imported master: {}", target.relative_to(work_dir))
        return work_dir

    def import_score(self, score_file: Path) -> Path:
        """Import external scores in the working directory."""
        if not score_file.exists():
            raise FileNotFoundError(f"Score file not found: {score_file}")
        stem = snake_case(score_file.stem)
        score = normalize(load_score(score_file))
        scores_dir = self.subdir("scores")
        assert scores_dir.exists(), f"Scores directory not initialized: {scores_dir}"
        target = scores_dir / f"{stem}.krn"
        score.write("humdrum", fp=target)
        logger.success("Imported to: {}", target.relative_to(self.work_dir))
        return target

    def _master_file(self) -> Path:
        """Return the path to the master score in the working directory."""
        manifest_file = self.work_dir / "MANIFEST.json"
        assert manifest_file.exists(), f"Manifest file does not exist: {manifest_file}"
        manifest = Manifest.model_validate_json(manifest_file.read_text())
        master_file = self.subdir("scores") / manifest.master
        assert master_file.exists(), f"Master score does not exist: {master_file}"
        return master_file

    def _master_score(self) -> Score:
        master_file = self._master_file()
        logger.info(f"Using master score: {master_file.relative_to(self.work_dir)}")
        return normalize(load_score(master_file))

    # NOTE: Using `filename: str` is the correct any only choice.
    def change_master(self, filename: str) -> None:
        """
        Change to **other** workspace score as master.

        Workflow to change master in the workspace:
        - Import external score via `import_score()`. (prerequisite - file must exist in `scores/`)
        - Use `change_master()` to load the imported score.
        """
        scores_dir = self.subdir("scores")
        score_file = scores_dir / filename
        if not score_file.exists():
            raise FileNotFoundError(f"Score file does not exist: {score_file}")
        # NOTE: change_master changes, not return the master.
        manifest_file = self.work_dir / "MANIFEST.json"
        manifest = Manifest(master=score_file.name)
        manifest_file.write_text(manifest.model_dump_json(indent=2))
        logger.success("Changed master to: {}", score_file.relative_to(self.work_dir))

    def _bump(self, relative: Path) -> int:
        """Compute the next archival version number for `relative`."""
        versions_root = self.subdir("versions") / relative.parent
        if not versions_root.exists():
            return 1
        suffix = relative.suffix
        pattern = re.compile(rf"^{re.escape(relative.stem)}_v(\d+){re.escape(suffix)}$")
        history = sorted(versions_root.glob(f"{relative.stem}_v*{suffix}"))
        for candidate in reversed(history):
            match = pattern.match(candidate.name)
            if match:
                return int(match.group(1)) + 1
        return 1

    def _supersede(self, src_file: Path) -> Optional[Path]:
        if not src_file.exists():
            return None
        try:
            relative = src_file.relative_to(self.work_dir)
        except ValueError as exc:  # pragma: no cover - protective guard
            raise ValueError(f"File outside workspace: {src_file}") from exc
        # NOTE: Work directly with ``src_file`` so versions mirror workspace layout (d7b4d6d fix).
        archive_dir = self.subdir("versions") / relative.parent
        archive_dir.mkdir(parents=True, exist_ok=True)
        version = self._bump(relative.parent / relative.name)
        bak_file = archive_dir / f"{src_file.stem}_v{version:04d}{src_file.suffix}"
        try:
            src_file.rename(bak_file)
        except FileNotFoundError:
            return None
        return bak_file

    # NOTE: For all operations in the workspace, the target is always implicitely the master score by convention (no optionality or ad-hoc configuraiton).
    def _save(self, score: Score, suffix: str = ".krn") -> Optional[Path]:
        """Replace master with saved score file; backs up previous version."""
        master_file = self._master_file()
        bak_file = self._supersede(master_file)
        fmt = "musicxml" if suffix == ".mxl" else "humdrum"
        logger.debug("Saving {}; superseded {}", master_file, bak_file)
        target_score = normalize(score) if fmt == "humdrum" else score
        target_score.write(fmt, fp=master_file)
        return bak_file

    def export_midi(self) -> Path:
        """
        Convert the master score or another specified score to MIDI.
        """
        master_file = self._master_file()
        score = load_score(master_file)
        midi_dir = self.subdir("midi")
        mid_file = midi_dir / f"{master_file.stem}.mid"
        bak = self._supersede(mid_file)
        score.write("midi", fp=mid_file)
        logger.success(
            "Exported MIDI to {}; superseded {}",
            mid_file.name,
            bak.name if bak else None,
        )
        return mid_file

    def create_and_store_excerpt(
        self,
        part_spec: PartSpec,
        measure_spec: MeasureSpec,
        other_score: Optional[str],
    ) -> Path:
        """
        Create a Humdrum excerpt (kern subscore) and persist it under `excerpts/`.
        """
        if other_score:
            scores_dir = self.subdir("scores")
            source = scores_dir / other_score
            if not source.exists():
                raise FileNotFoundError(f"Score file must exist in workspace: {source}")
            logger.info(
                "Extracting excerpt from specified score file: {}",
                source.relative_to(self.work_dir),
            )
        else:
            source = self._master_file()
        score = load_score(source)
        excerpt = create_excerpt(score, part_spec, measure_spec)
        canonical_parts = part_spec.ids
        stem = (
            f"{source.stem}_"
            f"{'_'.join(canonical_parts)}_"
            f"{snake_case(measure_spec.spec)}"
        )
        excerpts_dir = self.subdir("excerpts")
        out_file = excerpts_dir / f"{stem}.krn"
        excerpt.write("humdrum", fp=out_file)
        logger.success("Extracted excerpt to {}", out_file)
        return out_file

    def merge_excerpt(
        self,
        filename: str,
    ) -> None:
        """
        Merge a Humdrum excerpt into the master score.
        """
        excerpts_dir = self.subdir("excerpts")
        excerpt_path = excerpts_dir / filename
        if not excerpt_path.exists():
            raise FileNotFoundError(f"Excerpt file does not exist: {excerpt_path}")

        score = self._master_score()
        excerpt_score = load_score(excerpt_path)
        master_tokens = {_canonical_part_id(p) for p in score.parts}
        master_tokens.update(str(p.id) for p in score.parts if p.id is not None)
        excerpt_ids = [_canonical_part_id(p) for p in excerpt_score.parts]
        missing = [pid for pid in excerpt_ids if pid not in master_tokens]
        if missing:
            raise ValueError(f"Excerpt references unknown parts: {', '.join(missing)}")
        measure_numbers: list[int] = []
        for part in excerpt_score.parts:
            for measure in part.getElementsByClass(Measure):
                if measure.number is None:
                    continue
                measure_numbers.append(int(measure.number))
        measure_spec = MeasureSpec(spec=_numbers_to_spec(measure_numbers))
        part_spec = PartSpec(tokens=",".join(excerpt_ids))
        merged = merge_excerpt(score, excerpt_score, part_spec, measure_spec)
        bak = self._save(merged)
        logger.success(
            "Merged excerpt {} into master; backup {}",
            excerpt_path.relative_to(self.work_dir),
            bak,
        )

    def delete_measures(
        self,
        measure_spec: str,
        mode: Literal["blank", "drop_renumber"] = "blank",
    ) -> None:
        """Remove measure ranges (across all parts) from the master score."""
        master = self._master_score()
        spec_model = MeasureSpec(spec=measure_spec)
        updated = delete_measures(master, spec_model, mode=mode)
        bak = self._save(updated)
        logger.success("Removed measures {}; backup {}", spec_model.spec, bak)

    def expand_master(
        self,
        at: int,
        count: int,
    ) -> None:
        """
        Insert blank measures (across all parts) into the master score.
        """
        master = self._master_score()
        updated = insert_blank_measures(master, at=at, count=count)
        bak = self._save(updated)
        logger.success("Inserted blank measures at {}; backup {}", at, bak)

    def score_spec(self, other_score: Optional[str] = None) -> ScoreSpec:
        """
        Build structured metadata snapshot of the master score or another specified score.
        """
        # NOTE: Accept string filenames only; Path arguments enabled cross-workspace lookups before d7b4d6d.
        if other_score:
            scores_dir = self.subdir("scores")
            source = scores_dir / other_score
            if not source.exists():
                raise ValueError(f"Score file must exist in workspace: {source}")
        else:
            source = self._master_file()

        score = normalize(load_score(source))

        return ScoreSpec.build(score)


def _numbers_to_spec(numbers: Sequence[int]) -> str:
    ordered = sorted(dict.fromkeys(numbers))
    if not ordered:
        raise ValueError("Excerpt does not contain numbered measures")
    ranges: list[tuple[int, int]] = []
    start = prev = ordered[0]
    for value in ordered[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append((start, prev))
        start = prev = value
    ranges.append((start, prev))
    pieces = [str(a) if a == b else f"{a}-{b}" for a, b in ranges]
    return ",".join(pieces)


def _canonical_part_id(part: Part) -> str:
    label = part.partName or part.partAbbreviation or str(part.id)
    return snake_case(str(label))
