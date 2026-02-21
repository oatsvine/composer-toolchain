"""
Extract/Merge tests
-------------------
Strategy:
- Extract a span across two parts.
- Deterministically modify ~35% of notes in the subscore.
- Merge back and verify:
  * Inside the span: fingerprints differ as expected (some measures changed).
  * Outside the span: fingerprints are identical.
"""

from random import Random
from typing import Optional

from music21 import note as m21_note

from composer_toolchain.score import (
    MeasureSpec,
    PartSpec,
    create_excerpt,
    merge_excerpt,
)
from composer_toolchain.test_utils import (
    score_fingerprint,
    build_simple_score,
    add_structure,
)


def test_extract_and_merge_randomized_changes():
    """Edits must be confined to the targeted span and nowhere else."""
    s = build_simple_score(n_parts=2, n_measures=6)
    ids = [str(s.parts[0].id), str(s.parts[1].id)]
    span = "2-4"
    part_spec = PartSpec(tokens=",".join(ids))
    measure_spec = MeasureSpec(spec=span)
    fp_before = score_fingerprint(s)

    excerpt = create_excerpt(s, part_spec, measure_spec)

    rnd = Random(42)
    for p in excerpt.parts[:2]:
        for n in p.recurse().notes:
            if not isinstance(n, m21_note.Note):
                continue
            if rnd.random() < 0.35:
                # transpose by a step (up or down)
                n.transpose(2 if rnd.random() < 0.5 else -2, inPlace=True)

    merged = merge_excerpt(s, excerpt, part_spec, measure_spec)
    fp_after = score_fingerprint(merged)

    def in_span(measure_no: int) -> bool:
        return 2 <= measure_no <= 4

    changed_inside = False
    for key, v in fp_before.items():
        pi, mnum = key
        if in_span(mnum) and pi in (0, 1):
            if fp_after.get(key) != v:
                changed_inside = True
        else:
            assert fp_after.get(key) == v

    assert changed_inside, "Expected changes inside the edited span"


def test_merge_preserves_structure():
    """Structural markers (clef/key/time) should persist through extract/merge."""
    s = build_simple_score(n_parts=1, n_measures=5)
    add_structure(s.parts[0], 3, keysig_fifths=2, clef=True)
    span = "2-4"
    part_spec = PartSpec(tokens=str(s.parts[0].id))
    measure_spec = MeasureSpec(spec=span)
    excerpt = create_excerpt(s, part_spec, measure_spec)
    # Simple edit: transpose all notes in measure 3 by +2
    target_measure = excerpt.parts[0].measure(3)
    assert target_measure is not None
    for n in target_measure.recurse().notes:
        if isinstance(n, m21_note.Note):
            n.transpose(2, inPlace=True)
    merged = merge_excerpt(s, excerpt, part_spec, measure_spec)
    m3 = merged.parts[0].measure(3)
    assert m3 is not None
    from music21.clef import TrebleClef
    from music21.key import KeySignature
    from music21.meter.base import TimeSignature

    assert any(isinstance(el, TrebleClef) for el in m3)
    assert any(isinstance(el, KeySignature) for el in m3)
    assert any(isinstance(el, TimeSignature) for el in m3)


def test_extract_merge_on_corpus_sample(corpus_scores):
    """Extract/merge honors corpus fingerprints, touching only the requested span."""

    score = corpus_scores["keyboard_polyphony"]
    part_ids = [str(p.id) for p in score.parts[:2]]
    span = "10-12"
    part_spec = PartSpec(tokens=",".join(part_ids))
    measure_spec = MeasureSpec(spec=span)
    baseline = score_fingerprint(score)

    excerpt = create_excerpt(score, part_spec, measure_spec)
    mutated_measures = set()
    for part in excerpt.parts:
        for measure in part.getElementsByClass("Measure"):
            note_obj: Optional[m21_note.Note] = next(
                (n for n in measure.recurse().notes if isinstance(n, m21_note.Note)),
                None,
            )
            if note_obj is None:
                continue
            note_obj.transpose(1, inPlace=True)
            if measure.number is not None:
                mutated_measures.add(int(measure.number))
            break

    assert mutated_measures, "expected to mutate at least one measure in the excerpt"

    merged = merge_excerpt(score, excerpt, part_spec, measure_spec)
    fp_after = score_fingerprint(merged)

    changes = []
    for (pi, mnum), before in baseline.items():
        after = fp_after[(pi, mnum)]
        if pi in (0, 1) and mnum in mutated_measures:
            if after != before:
                changes.append((pi, mnum))
        else:
            assert after == before
    assert changes, "expected at least one fingerprint change within targeted span"
