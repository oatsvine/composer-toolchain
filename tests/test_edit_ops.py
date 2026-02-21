"""
Delete/Insert operations
------------------------
This suite validates the new public APIs:

- delete_measures(..., mode="blank"): structure preserved; targeted measures become full rests; ties at boundary broken.
- delete_measures(..., mode="drop_renumber"): measures removed; renumbered continuously; outside content preserved (by fingerprint).
- insert_blank_measures(at, count): inserts rest-filled measures; renumbers; shifts content; ties at boundary broken.

Invariants tested:
- Measure numbers and fingerprints outside the edited region remain identical.
- Insertions shift fingerprints by `count` measures.
- Deletions in 'drop_renumber' remove the measures and renumber, preserving fingerprints for all others.
"""

from textwrap import dedent
from math import isclose

from music21.humdrum.spineParser import GlobalComment

from composer_toolchain.score import (
    MeasureSpec,
    delete_measures,
    insert_blank_measures,
    kern_to_score,
    score_to_kern,
)
from .helpers import (
    add_structure,
    add_tie_across,
    build_simple_score,
    build_ts_score,
    measure_fingerprint,
    score_fingerprint,
)


def test_delete_blank_mode():
    """Blanking should replace content with rests only and preserve other measures verbatim."""
    s = build_simple_score(n_parts=2, n_measures=6)
    fp0 = score_fingerprint(s)
    # Add a tie across the left boundary to ensure boundary tie clearing is exercised
    add_tie_across(s.parts[0], 1, 2)
    out = delete_measures(s, MeasureSpec(spec="2-3"), mode="blank")
    fp1 = score_fingerprint(out)

    for (pi, mnum), fp in fp1.items():
        if mnum in (2, 3):
            assert all(
                tok[1] == "R" for tok in fp
            ), f"expected rests in part {pi} measure {mnum}"
        else:
            assert (
                fp == fp0[(pi, mnum)]
            ), f"unexpected change outside blanked range @ {(pi,mnum)}"


def test_delete_drop_renumber():
    """Dropping should renumber measures continuously and preserve remaining fingerprints."""
    s = build_simple_score(n_parts=2, n_measures=6)
    fp0 = score_fingerprint(s)

    out = delete_measures(s, MeasureSpec(spec="2-3"), mode="drop_renumber")
    fp1 = score_fingerprint(out)

    for (pi, mnum), fp in fp0.items():
        if mnum >= 4:
            assert fp1[(pi, mnum - 2)] == fp
        elif mnum == 1:
            assert fp1[(pi, 1)] == fp


def test_insert_blank_measures_shift():
    """Insertion shifts measures right and inserts rest-only measures at the position."""
    s = build_simple_score(n_parts=2, n_measures=6)
    fp0 = score_fingerprint(s)

    out = insert_blank_measures(s, at=2, count=2)
    fp1 = score_fingerprint(out)

    for (pi, mnum), fp in fp0.items():
        if mnum >= 2:
            assert fp1[(pi, mnum + 2)] == fp
        else:
            assert fp1[(pi, 1)] == fp  # measure 1 unchanged

    for pi in range(len(out.parts)):
        for m in (2, 3):
            assert all(tok[1] == "R" for tok in fp1[(pi, m)])


def test_insert_respects_last_meter():
    """Inserted measures must use the last-known meter to size rests."""
    s = build_ts_score(["3/4", "3/4", "4/4"], n_parts=1)
    out = insert_blank_measures(s, at=3, count=1)
    # New measure 3 should be a 3/4 rest-only bar
    m3_fp = score_fingerprint(out)[(0, 3)]
    total_rest = sum(ql for _, kind, ql in m3_fp if kind == "R")
    assert abs(total_rest - 3.0) < 1e-6


def test_blank_uses_measure_meter():
    """Blanking uses each measure's bar duration for rests when specific meters exist."""
    s = build_ts_score(["3/4", "5/4", "4/4"], n_parts=1)
    out = delete_measures(s, MeasureSpec(spec="2"), mode="blank")
    m2_fp = score_fingerprint(out)[(0, 2)]
    total_rest = sum(ql for _, kind, ql in m2_fp if kind == "R")
    assert abs(total_rest - 5.0) < 1e-6


def test_blank_preserves_structure():
    """Blanking should preserve clef/key/time inside the measure while removing notes."""
    s = build_ts_score(["4/4", "4/4", "4/4"], n_parts=1)
    # Add structure to measure 2

    add_structure(s.parts[0], 2, keysig_fifths=3, clef=True)
    out = delete_measures(s, MeasureSpec(spec="2"), mode="blank")
    m2 = out.parts[0].measure(2)
    assert m2 is not None
    # Structural preserved
    from music21.clef import TrebleClef
    from music21.key import KeySignature
    from music21.meter.base import TimeSignature

    assert any(isinstance(el, TrebleClef) for el in m2)
    assert any(isinstance(el, KeySignature) for el in m2)
    assert any(isinstance(el, TimeSignature) for el in m2)
    # Only rests as musical atoms

    fp = measure_fingerprint(m2)
    assert all(kind == "R" for _, kind, _ in fp)


def test_delete_blank_on_normalized_corpus(corpus_scores):
    """Blank deletions behave on real normalized scores without disturbing neighbors."""

    score = corpus_scores["fugue_krn"]
    fp_before = score_fingerprint(score)
    result = delete_measures(score, MeasureSpec(spec="2"), mode="blank")
    fp_after = score_fingerprint(result)

    for (pi, mnum), fp in fp_before.items():
        current = fp_after[(pi, mnum)]
        if mnum == 2:
            assert all(kind == "R" for _, kind, _ in current)
        else:
            assert current == fp


def test_kern_to_score_global_comments():
    """Global comments (!!) in **kern should surface as GlobalComment objects on the Score."""
    text = dedent(
        """
        !! Title line
        !! Subtitle line
        **kern
        *M4/4
        =1
        1c
        *-
        """
    ).strip()

    score = kern_to_score(text)
    comments = list(score.getElementsByClass(GlobalComment))
    assert [gc.comment for gc in comments] == ["Title line", "Subtitle line"]


def test_score_to_kern_global_comments(tmp_path):
    """Editing and adding GlobalComment objects on the Score should emit matching !! lines."""
    base = dedent(
        """
        !! Title line
        !! Subtitle line
        **kern
        *M4/4
        =1
        1c
        *-
        """
    ).strip()

    score = kern_to_score(base)
    comments = list(score.getElementsByClass(GlobalComment))
    assert comments, "expected parsed score to expose global comments"

    comments[0].comment = "Title line (edited)"
    max_priority = max((gc.priority or 0) for gc in comments)
    extra = GlobalComment("!! Additional note")
    extra.priority = max_priority + 1
    score.insert(0, extra)

    kern = score_to_kern(score)
    comment_lines: list[str] = []
    for line in kern.splitlines():
        if line.startswith("!!"):
            comment_lines.append(line)
        else:
            break

    assert comment_lines == [
        "!! Title line (edited)",
        "!! Subtitle line",
        "!! Additional note",
    ]


def test_global_comment_positions_roundtrip():
    """Global comments across multiple measures retain offsets and order through roundtrip."""
    text = dedent(
        """
        !! Header A
        !! Header B
        **kern
        *M4/4
        =1
        1c
        !! Between m1 m2
        =2
        1d
        !! Between m2 m3
        =3
        1e
        *-
    """
    ).strip()

    score = kern_to_score(text)
    comments = list(score.getElementsByClass(GlobalComment))
    assert [gc.comment for gc in comments] == [
        "Header A",
        "Header B",
        "Between m1 m2",
        "Between m2 m3",
    ]

    offsets = {gc.comment: float(gc.getOffsetBySite(score) or 0.0) for gc in comments}
    assert isclose(offsets["Header A"], 0.0, abs_tol=1e-6)
    assert isclose(offsets["Header B"], 0.0, abs_tol=1e-6)

    part0 = score.parts[0]
    measure_offsets = {
        int(m.number): float(m.offset or 0.0)
        for m in part0.getElementsByClass("Measure")
        if m.number is not None
    }
    assert isclose(offsets["Between m1 m2"], measure_offsets[2], abs_tol=1e-6)
    assert isclose(offsets["Between m2 m3"], measure_offsets[3], abs_tol=1e-6)

    kern_roundtrip = score_to_kern(score)
    lines = kern_roundtrip.splitlines()

    def idx(token: str) -> int:
        try:
            return lines.index(token)
        except ValueError as exc:
            raise AssertionError(f"expected {token!r} in lines") from exc

    assert idx("!! Header A") < idx("**kern")
    assert idx("!! Header B") < idx("**kern")
    assert idx("!! Between m1 m2") > idx("1c")
    assert idx("!! Between m1 m2") < idx("=2")
    assert idx("!! Between m2 m3") > idx("1d")
    assert idx("!! Between m2 m3") < idx("=3")
