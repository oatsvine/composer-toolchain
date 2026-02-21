"""Core toolchain logic for score workspace management and operations.

This module provides the `Context` class, which encapsulates operations
for managing a musical score workspace. It includes functionality for
importing scores, changing the master score, creating and merging excerpts,
and exporting MIDI files.

All toolchain operations are encapsulated here to maintain a clean separation
from CLI interfaces and higher-level application logic.
"""

from pathlib import Path
from typing import Literal, Optional, Sequence

from loguru import logger
from music21.stream import Score
from music21.stream.base import Measure, Part
from pydantic import BaseModel

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


WORKSPACE_DIRS = Literal["scores", "excerpts", "midi", "versions", "sketches"]


def init_workspace(work_dir: Path) -> None:
    """Initialize the workspace directory structure."""
    for dirname in ["scores", "excerpts", "midi", "versions", "sketches"]:
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
    """Operations in the workspace."""

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

    # NOTE: For all operations in the workspace, the target is always implicitely the master score by convention (no optionality or ad-hoc configuraiton).
    def _save_master(self, score: Score, suffix: str = ".krn") -> Optional[Path]:
        """Replace master with saved score file; backs up previous version."""
        master_file = self._master_file()
        master_file.rename(master_file.with_name(f".{master_file.name}.bak"))
        fmt = "musicxml" if suffix == ".mxl" else "humdrum"
        logger.debug("Saved {}", master_file)
        target_score = normalize(score) if fmt == "humdrum" else score
        target_score.write(fmt, fp=master_file)

    def render_midi(self, source_path: Path) -> Path:
        """
        Convert the master score or another specified score to MIDI.
        """
        assert source_path.is_relative_to(
            self.work_dir
        ), "Source path must be within workspace"
        score = load_score(source_path)
        midi_dir = self.subdir("midi")
        mid_file = midi_dir / f"{source_path.stem}.mid"
        score.write("midi", fp=mid_file)
        logger.success(
            "Exported MIDI to {}",
            mid_file.name,
        )
        return mid_file

    def create_and_store_excerpt(
        self,
        part_spec: PartSpec,
        measure_spec: MeasureSpec,
        filename: Optional[str] = None,
        suffix: Optional[str] = None,
    ) -> Path:
        """
        Create a Humdrum excerpt (kern subscore) and persist it under `excerpts/`.

        Parameters
        ----------
        part_spec : PartSpec
            Canonical part identifiers to extract.
        measure_spec : MeasureSpec
            Measure ranges to carve out of the source score.
        filename : str | None
            Optional workspace score (under scores/) to slice instead of the
            current master.
        suffix : str | None
            Optional human label appended (snake_cased) to the excerpt filename
            so later workflows (e.g., segmentation) can reference the excerpt
            semantically ("motive_a", "antecedent_phrase").
        """
        if filename:
            scores_dir = self.subdir("scores")
            source = scores_dir / filename
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
        if suffix:
            stem = f"{stem}_{snake_case(suffix)}"
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
        bak = self._save_master(merged)
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
        bak = self._save_master(updated)
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
        bak = self._save_master(updated)
        logger.success("Inserted blank measures at {}; backup {}", at, bak)


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
