"""Normalization and voice-flattening invariants."""

from __future__ import annotations

from copy import deepcopy
from textwrap import dedent
from typing import Optional

import pytest
from music21 import metadata as m21_metadata
from music21 import note, stream
from music21.meter.base import TimeSignature
from music21.humdrum.spineParser import GlobalComment
from music21.stream import Voice

from composer_toolchain.score import (
    flatten_voices_to_parts,
    kern_to_score,
    normalize,
    normalize_score,
    score_to_kern,
)
from .helpers import score_fingerprint


@pytest.mark.parametrize("sample_key", ["keyboard_polyphony"])
def test_flatten_voices_eliminates_voice_containers(raw_corpus_scores, sample_key):
    """Raw corpus examples include polyphony; flattening must hoist Voice layers."""

    raw = deepcopy(raw_corpus_scores[sample_key])
    has_voice = any(raw.recurse().getElementsByClass(Voice))
    assert has_voice, "fixture expectation: raw score should include Voice containers"

    flattened = flatten_voices_to_parts(raw)
    assert not any(
        flattened.recurse().getElementsByClass(Voice)
    ), "voices must be hoisted"


def test_normalize_score_aligns_and_numbers_parts(raw_corpus_scores):
    """normalize_score equalizes measure counts and numbering across parts."""

    raw = deepcopy(raw_corpus_scores["choral_mass"])
    first_part = raw.parts[0]
    measures = list(first_part.getElementsByClass("Measure"))
    assert measures, "corpus score should expose measures"
    first_part.remove(measures[-1])  # desync the first part intentionally

    normalized = normalize_score(raw)
    measure_sequences = [
        list(p.getElementsByClass("Measure")) for p in normalized.parts
    ]
    counts = {len(seq) for seq in measure_sequences}
    assert len(counts) == 1, "measure counts must be uniform"

    expected_numbers = list(range(1, len(measure_sequences[0]) + 1))
    for seq in measure_sequences:
        numbers = [m.number for m in seq]
        assert numbers == expected_numbers, "measure numbering must be continuous"

    # Idempotence: applying normalize_score again should not change structure
    rebound = normalize_score(deepcopy(normalized))
    rebound_sequences = [
        [m.number for m in part.getElementsByClass("Measure")] for part in rebound.parts
    ]
    for numbers in rebound_sequences:
        assert numbers == expected_numbers


def test_normalized_corpus_roundtrip_preserves_metadata(corpus_scores):
    """Our curated corpus samples remain stable after Humdrum round-trip."""

    for name, score in corpus_scores.items():
        if name == "fugue_krn":
            pytest.xfail("converter21 cannot equalize fugue_krn for Humdrum export")
        baseline_fp = score_fingerprint(score)
        roundtrip = kern_to_score(score_to_kern(score))
        assert len(roundtrip.parts) == len(score.parts)
        assert pytest.approx(roundtrip.highestTime, rel=1e-6) == score.highestTime
        for part in roundtrip.parts:
            assert part.partName, "normalized parts must retain names"
            assert part.partAbbreviation, "normalized parts must retain abbreviations"
        assert score_fingerprint(roundtrip) == baseline_fp


def test_normalize_assigns_part_ids_unique():
    score = stream.Score()

    def make_part(label: Optional[str]) -> stream.Part:
        p = stream.Part()
        if label is not None:
            p.partName = label
        m = stream.Measure(number=1)
        m.insert(0, TimeSignature("4/4"))
        m.append(note.Note("C4", quarterLength=4.0))
        p.append(m)
        return p

    for lbl in ["S", "S", "", None]:
        score.append(make_part(lbl))

    normalized = normalize(score)
    ids = [p.id for p in normalized.parts]
    assert ids[0] == "s"
    assert ids[1] == "s_2"
    assert ids[2] == "p03"
    assert ids[3] == "p04"


def _score_with_global_comments() -> stream.Score:
    text = dedent(
        """
        !! Header line
        **kern
        *M4/4
        =1
        1c
        !! Between measures
        =2
        1d
        *-
        """
    ).strip()
    score = kern_to_score(text)
    if score.metadata is None:
        score.metadata = m21_metadata.Metadata()
    score.metadata.filename = "comment_fixture"
    return score


def _comment_texts(score: stream.Score) -> list[str]:
    return [gc.comment for gc in score.recurse().getElementsByClass(GlobalComment)]


def test_normalize_preserves_global_comments():
    score = _score_with_global_comments()
    normalized = normalize(score)
    assert _comment_texts(normalized) == ["Header line", "Between measures"]


def test_normalize_score_preserves_global_comments():
    score = _score_with_global_comments()
    normalized = normalize_score(score)
    assert _comment_texts(normalized) == ["Header line", "Between measures"]
