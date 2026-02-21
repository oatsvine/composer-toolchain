"""Test helper utilities for building toy scores and computing fingerprints."""

from __future__ import annotations

from typing import Dict, List

from music21 import chord, note
from music21.clef import TrebleClef
from music21.key import KeySignature
from music21.meter.base import TimeSignature
from music21.stream import Measure, Part, Score, Voice
from music21.tie import Tie


def measure_fingerprint(m: Measure) -> List[tuple[float, str, float]]:
    """Return a stable (offset, pitch_repr, duration) fingerprint for a measure."""
    out: List[tuple[float, str, float]] = []
    for el in m.recurse().notesAndRests:
        off = float(el.offset)
        ql = float(el.duration.quarterLength)
        if el.isRest:
            out.append((off, "R", ql))
        elif isinstance(el, chord.Chord):
            mids = sorted(int(p.midi) for p in el.pitches)
            out.append((off, "C[" + ",".join(str(x) for x in mids) + "]", ql))
        elif isinstance(el, note.Note):
            out.append((off, str(int(el.pitch.midi)), ql))
    return out


def score_fingerprint(score: Score) -> Dict[tuple[int, int], List[tuple[float, str, float]]]:
    """Map (part index, measure number) to fingerprints for quick equality checks."""
    fingerprints: Dict[tuple[int, int], List[tuple[float, str, float]]] = {}
    for pi, part in enumerate(score.parts):
        for measure in part.getElementsByClass("Measure"):
            if measure.number is None:
                continue
            fingerprints[(pi, int(measure.number))] = measure_fingerprint(measure)
    return fingerprints


def build_simple_score(n_parts: int = 2, n_measures: int = 4) -> Score:
    """Create a simple monophonic multi-part score with ascending quarter notes."""
    score = Score()
    for pi in range(n_parts):
        part = Part(id=str(pi + 1))
        base = 60 + 12 * pi
        for mnum in range(1, n_measures + 1):
            measure = Measure(number=mnum)
            if mnum == 1:
                measure.insert(0, TimeSignature("4/4"))
            for i in range(4):
                n = note.Note(base + i)
                n.quarterLength = 1.0
                measure.append(n)
            part.insert((mnum - 1) * 4.0, measure)
        score.insert(0, part)
    return score


def build_keyboard_polyphony(n_measures: int = 2) -> Score:
    """Create a keyboard-like score with two Voices per measure."""
    score = Score()
    part = Part(id="P1")
    for mnum in range(1, n_measures + 1):
        measure = Measure(number=mnum)
        if mnum == 1:
            measure.insert(0, TimeSignature("4/4"))
        upper = Voice(id="1")
        lower = Voice(id="2")
        for i in range(4):
            high = note.Note(72 + i)
            high.quarterLength = 1.0
            upper.append(high)
            low = note.Note(48 + i)
            low.quarterLength = 1.0
            lower.append(low)
        measure.append(upper)
        measure.append(lower)
        part.insert((mnum - 1) * 4.0, measure)
    score.insert(0, part)
    return score


def add_tie_across(part: Part, left_measure: int, right_measure: int) -> None:
    """Tie the last note of left_measure to the first note of right_measure."""
    left = part.measure(left_measure)
    right = part.measure(right_measure)
    if left is None or right is None:
        return
    last = next((n for n in left.recurse().notes[::-1]), None)
    first = next((n for n in right.recurse().notes), None)
    if last is not None and first is not None:
        last.tie = Tie("start")
        first.tie = Tie("stop")


def add_structure(part: Part, measure_number: int, keysig_fifths: int = 0, clef: bool = True) -> None:
    """Insert clef/key/time structure at the start of a measure for assertions."""
    measure = part.measure(measure_number)
    if measure is None:
        return
    if clef:
        measure.insert(0, TrebleClef())
    ks = KeySignature(keysig_fifths)
    measure.insert(0, ks)
    measure.insert(0, TimeSignature("4/4"))


def build_ts_score(ts_sequence: list[str], n_parts: int = 2) -> Score:
    """Create a score where each measure follows the provided time signatures."""
    score = Score()
    bar_lengths = []
    for ts in ts_sequence:
        ts_obj = TimeSignature(ts)
        bar_lengths.append(float(ts_obj.barDuration.quarterLength))
    offsets = [0.0]
    for idx in range(1, len(bar_lengths)):
        offsets.append(offsets[-1] + bar_lengths[idx - 1])

    for pi in range(n_parts):
        part = Part(id=str(pi + 1))
        for idx, ts in enumerate(ts_sequence, start=1):
            measure = Measure(number=idx)
            if idx == 1 or ts_sequence[idx - 1] != ts_sequence[idx - 2]:  # type: ignore[index]
                measure.insert(0, TimeSignature(ts))
            remaining = bar_lengths[idx - 1]
            pitch_base = 60 + 12 * pi
            while remaining >= 1.0:
                n = note.Note(pitch_base)
                n.quarterLength = 1.0
                measure.append(n)
                remaining -= 1.0
            if remaining > 0:
                rest = note.Rest()
                rest.quarterLength = remaining
                measure.append(rest)
            part.insert(offsets[idx - 1], measure)
        score.insert(0, part)
    return score

__all__ = [
    "measure_fingerprint",
    "score_fingerprint",
    "build_simple_score",
    "build_keyboard_polyphony",
    "add_tie_across",
    "add_structure",
    "build_ts_score",
]
