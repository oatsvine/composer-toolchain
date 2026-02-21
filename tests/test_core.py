import pytest

from pathlib import Path
from typing import Tuple

from music21.note import Note
from music21.stream import Score
from music21.stream.base import Measure

from composer_toolchain.score import (
    MeasureSpec,
    PartSpec,
    load_score,
    normalize,
    snake_case,
)
from composer_toolchain.core import Manifest, ScoreSpec, Context


def _first_notated_span(score: Score) -> Tuple[str, int, str]:
    for part in score.parts:
        for measure in part.getElementsByClass(Measure):
            candidate = measure.recurse().getElementsByClass(Note).first()
            if candidate is None or measure.number is None:
                continue
            label = part.partName or part.partAbbreviation or str(part.id)
            part_token = snake_case(label)
            return part_token, int(measure.number), candidate.nameWithOctave
    raise AssertionError("score must contain at least one notated note")


@pytest.fixture()
def sample_workspace(tmp_path: Path, sample_score_path: Path) -> Context:
    work_dir = Context.init_with_score(
        score_file=sample_score_path,
        cwd=tmp_path,
    )
    return Context(work_dir=work_dir)


@pytest.fixture()
def mozart_fragment_score(multi_movement_score_path: Path) -> Score:
    return normalize(load_score(multi_movement_score_path))


def test_export_midi(sample_workspace: Context) -> None:
    workspace = sample_workspace
    midi_path = workspace.export_midi()
    assert midi_path.exists()

    workspace.export_midi()
    midi_versions = workspace.work_dir / "versions" / "midi"
    assert any(midi_versions.glob(f"{midi_path.stem}_v*.mid"))


def test_create_excerpt(sample_workspace: Context) -> None:
    workspace = sample_workspace
    score = workspace._master_score()
    part_token, measure, _ = _first_notated_span(score)
    excerpt_path = workspace.create_and_store_excerpt(
        part_spec=PartSpec(tokens=part_token),
        measure_spec=MeasureSpec(spec=str(measure)),
        filename=None,
    )
    assert excerpt_path.exists()


def test_merge_excerpt(sample_workspace: Context) -> None:
    workspace = sample_workspace
    score = workspace._master_score()
    part_token, measure, original_pitch = _first_notated_span(score)

    excerpt_path = workspace.create_and_store_excerpt(
        part_spec=PartSpec(tokens=part_token),
        measure_spec=MeasureSpec(spec=str(measure)),
        filename=None,
    )
    excerpt_score = load_score(excerpt_path)
    excerpt_measure = excerpt_score.parts[0].measure(measure)
    assert excerpt_measure is not None
    excerpt_note = excerpt_measure.recurse().getElementsByClass(Note).first()
    assert excerpt_note is not None, "excerpt must expose a concrete note"
    excerpt_note.nameWithOctave = "C#4" if original_pitch != "C#4" else "D4"
    excerpt_score.write("humdrum", fp=excerpt_path)

    workspace.merge_excerpt(excerpt_path.name)

    updated_score = workspace._master_score()
    updated_measure = updated_score.parts[0].measure(measure)
    assert updated_measure is not None
    updated_note = updated_measure.recurse().getElementsByClass(Note).first()
    assert updated_note is not None
    assert updated_note.nameWithOctave != original_pitch

    versions_dir = workspace.work_dir / "versions" / "scores"
    master_stem = workspace._master_file().stem
    assert any(versions_dir.glob(f"{master_stem}_v*.krn"))


def test_remove(sample_workspace: Context) -> None:
    workspace = sample_workspace
    workspace.delete_measures("2", mode="blank")
    score = workspace._master_score()
    measure = score.parts[0].measure(2)
    assert measure is not None
    rests = list(measure.recurse().notesAndRests)
    assert rests and all(r.isRest for r in rests)


def test_expand(sample_workspace: Context) -> None:
    workspace = sample_workspace
    workspace.expand_master(at=1, count=1)
    score = workspace._master_score()
    inserted = score.parts[0].measure(1)
    assert inserted is not None
    rests = list(inserted.recurse().notesAndRests)
    assert rests and all(r.isRest for r in rests)


def test_spec(sample_workspace: Context) -> None:
    workspace = sample_workspace
    spec = workspace.score_spec()
    assert isinstance(spec, ScoreSpec)
    assert spec.parts, "spec must expose part metadata"
    assert all(part.part_id and not part.part_id.isdigit() for part in spec.parts)
    assert spec.time_signatures, "expected structural metadata"


def test_spec_multimovement(mozart_fragment_score: Score) -> None:
    spec = ScoreSpec.build(mozart_fragment_score)
    assert len(spec.parts) >= 10
    assert spec.movements, "fragment must expose movement metadata"
    assert len(spec.time_signatures) > 1
    assert len(spec.key_signatures) > 1


def test_normalized_ids(sample_workspace: Context) -> None:
    score = sample_workspace._master_score()
    ids = [p.id for p in score.parts]
    assert len(set(ids)) == len(ids)
    assert all(isinstance(pid, str) and not pid.isdigit() for pid in ids)


def test_import_score(sample_workspace: Context, alt_score_path: Path) -> None:
    workspace = sample_workspace
    target = workspace.import_score(score_file=alt_score_path)
    expected = workspace.work_dir / "scores" / f"{snake_case(alt_score_path.stem)}.krn"
    assert target == expected
    assert expected.exists()
    spec = workspace.score_spec(other_score=expected.name)
    assert spec.parts


def test_change_master(sample_workspace: Context, alt_score_path: Path) -> None:
    workspace = sample_workspace
    new_score = workspace.import_score(score_file=alt_score_path)
    manifest_path = workspace.work_dir / "MANIFEST.json"
    original = Manifest.model_validate_json(manifest_path.read_text())
    new_master_name = f"{snake_case(alt_score_path.stem)}.krn"
    assert original.master != new_master_name

    workspace.change_master(new_score.name)

    updated = Manifest.model_validate_json(manifest_path.read_text())
    assert updated.master == new_master_name
    assert workspace._master_file().name == new_master_name
