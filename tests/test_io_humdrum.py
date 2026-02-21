
"""
I/O tests
---------
Invariants:
- Round-trip Score -> **kern -> Score preserves:
  * part count
  * highestTime
  * semantic fingerprints (per-part, per-measure content)
- Unsupported suffixes raise UnsupportedFormatError.
"""

from pathlib import Path
import pytest
from composer_toolchain.score import (
    kern_to_score,
    load_score,
    normalize,
    save_score,
    score_to_kern,
    UnsupportedFormatError,
)
from tests.composer_toolchain.test_utils import score_fingerprint, build_simple_score, build_ts_score

def test_roundtrip_kern_fingerprint():
    """Round-trip through **kern preserves structure and per-measure content for simple meters."""
    s = build_simple_score(n_parts=2, n_measures=4)
    fp0 = score_fingerprint(s)
    text = score_to_kern(s)
    s2 = kern_to_score(text)
    fp1 = score_fingerprint(s2)
    assert len(s.parts) == len(s2.parts)
    assert s.highestTime == s2.highestTime
    assert fp0 == fp1


def test_roundtrip_kern_with_ts_changes():
    """Round-trip still works when time signatures change mid-score (aligned across parts)."""
    s = build_ts_score(["3/4", "3/4", "4/4", "5/4"], n_parts=2)
    text = score_to_kern(s)
    s2 = kern_to_score(text)
    assert len(s.parts) == len(s2.parts)
    assert s.highestTime == s2.highestTime
    # Measure counts and offsets equal
    for pi in range(len(s.parts)):
        m1 = [int(m.number) for m in s.parts[pi].getElementsByClass("Measure") if m.number]
        m2 = [int(m.number) for m in s2.parts[pi].getElementsByClass("Measure") if m.number]
        assert m1 == m2

def test_bad_type(tmp_path: Path):
    """Unsupported suffixes should raise a clear error."""
    from pathlib import Path
    p = tmp_path / "bad.txt"
    p.write_text("not a score")
    try:
        load_score(p)
    except UnsupportedFormatError:
        pass
    else:
        raise AssertionError("expected UnsupportedFormatError for .txt")

def test_save_reparse_tmp(tmp_path: Path):
    """Saving to humdrum should produce a file that parses cleanly."""
    s = build_simple_score(n_parts=2, n_measures=3)
    out = tmp_path / "rt.krn"
    save_score(s, out, "humdrum")
    s2 = load_score(out)
    assert s2.highestTime == s.highestTime

    # Unsupported format should raise
    with pytest.raises(UnsupportedFormatError):
        save_score(s, tmp_path / "x.out", "abc")  # type: ignore[arg-type]


def test_mid_measure_ts_change_roundtrips():
    """Mid-measure time signature changes should emit inline *M tokens."""
    from music21.meter.base import TimeSignature

    s = build_simple_score(n_parts=1, n_measures=2)
    m1 = s.parts[0].measure(1)
    assert m1 is not None
    # Insert a TS change mid-bar
    m1.insert(2.0, TimeSignature("3/8"))
    text = score_to_kern(s)
    assert "*M3/8" in text.splitlines(), "expected inline time signature change"


def test_roundtrip_kern_corpus_samples(corpus_scores):
    """Every curated corpus score maintains structure when round-tripped through **kern."""

    for name, score in corpus_scores.items():
        if name == "fugue_krn":
            pytest.xfail("converter21 cannot equalize fugue_krn for Humdrum export")
        baseline = score_fingerprint(score)
        rebuilt = kern_to_score(score_to_kern(score))
        assert len(rebuilt.parts) == len(score.parts), name
        assert pytest.approx(rebuilt.highestTime, rel=1e-6) == score.highestTime
        assert score_fingerprint(rebuilt) == baseline


def test_normalized_score_survives_kern_roundtrip(tmp_path: Path) -> None:
    score = build_simple_score(n_parts=3, n_measures=4)
    normalized = normalize(score)
    text = score_to_kern(normalized)
    reloaded = kern_to_score(text)
    reloaded_norm = normalize(reloaded)
    expected_ids = [p.id for p in normalized.parts]
    actual_ids = [p.id for p in reloaded_norm.parts]
    assert actual_ids == expected_ids
