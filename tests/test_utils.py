"""
Utilities for robust, semantic integrity checks across operations.
We intentionally avoid fragile IDs and rely on per-measure content fingerprints.
"""

from __future__ import annotations

from typing import Dict, List

from music21 import chord, note
from music21.clef import TrebleClef
from music21.key import KeySignature
from music21.meter.base import TimeSignature
from music21.stream import Measure, Part, Score, Voice
from music21.tie import Tie


def measure_fingerprint(m: Measure) -> List[tuple[float, str, float]]:
    """
    Semantic fingerprint for a single measure.
    Returns a stable list of (offsetQL, pitchRepr, durQL) for each Note/Rest/Chord:
      - Note: pitchRepr = MIDI integer as str
      - Rest: pitchRepr = "R"
      - Chord: pitchRepr = "C[" + sorted MIDI list + "]"
    """
    out: List[tuple[float, str, float]] = []
    for el in m.recurse().notesAndRests:
        off = float(el.offset)
        ql = float(el.duration.quarterLength)
        if el.isRest:
            out.append((off, "R", ql))
        elif isinstance(el, chord.Chord):
            mids = sorted([p.midi for p in el.pitches])
            out.append((off, "C[" + ",".join(str(x) for x in mids) + "]", ql))
        elif isinstance(el, note.Note):
            out.append((off, str(int(el.pitch.midi)), ql))
        else:
            continue
    return out


def score_fingerprint(
    score: Score,
) -> Dict[tuple[int, int], list[tuple[float, str, float]]]:
    """
    Map (part_index, measure_number) -> measure_fingerprint. Part index is positional
    to avoid issues with ID normalization. Only measures with explicit numbers are considered.
    """
    d: Dict[tuple[int, int], list[tuple[float, str, float]]] = {}
    for pi, p in enumerate(score.parts):
        for m in p.getElementsByClass("Measure"):
            if m.number is None:
                continue
            d[(pi, int(m.number))] = measure_fingerprint(m)
    return d


def build_simple_score(n_parts: int = 2, n_measures: int = 4) -> Score:
    """
    Build a small Score with `n_parts` monophonic parts and `n_measures` numbered measures.
    Each measure contains four quarter notes ascending from MIDI 60.
    """
    s = Score()
    for pi in range(n_parts):
        p = Part(id=str(pi + 1))
        base = 60 + 12 * pi
        for mnum in range(1, n_measures + 1):
            m = Measure(number=mnum)
            if mnum == 1:
                m.insert(0, TimeSignature("4/4"))
            for i in range(4):
                n = note.Note(base + i)
                n.quarterLength = 1.0
                m.append(n)
            # Place measure at exact bar-aligned offsets to keep parts synchronized
            p.insert((mnum - 1) * 4.0, m)
        s.insert(0, p)
    return s


def build_keyboard_polyphony(n_measures: int = 2) -> Score:
    """
    Build a single-part keyboard-like score with two Voices per measure.
    Voice 1: upper line; Voice 2: lower line. Ensures Voice containers exist.
    """
    s = Score()
    p = Part(id="P1")
    for mnum in range(1, n_measures + 1):
        m = Measure(number=mnum)
        if mnum == 1:
            m.insert(0, TimeSignature("4/4"))
        v1 = Voice(id="1")
        v2 = Voice(id="2")
        for i in range(4):
            n1 = note.Note(72 + i)
            n1.quarterLength = 1.0
            v1.append(n1)
            n2 = note.Note(48 + i)
            n2.quarterLength = 1.0
            v2.append(n2)
        m.append(v1)
        m.append(v2)
        p.insert((mnum - 1) * 4.0, m)
    s.insert(0, p)
    return s


def add_tie_across(p: Part, left_measure: int, right_measure: int) -> None:
    """
    Add a tie from the last note of `left_measure` to the first note of `right_measure`.
    Used to validate boundary tie clearing in structural edits.
    """
    lm = p.measure(left_measure)
    rm = p.measure(right_measure)
    if lm is None or rm is None:
        return
    last = next((n for n in lm.recurse().notes[::-1]), None)
    first = next((n for n in rm.recurse().notes), None)
    if last is not None and first is not None:
        last.tie = Tie("start")
        first.tie = Tie("stop")


def add_structure(
    p: Part, mnum: int, keysig_fifths: int = 0, clef: bool = True
) -> None:
    """Insert a key signature and/or treble clef at the start of a measure."""
    m = p.measure(mnum)
    if m is None:
        return
    if clef:
        m.insert(0, TrebleClef())
    ks = KeySignature(keysig_fifths)
    m.insert(0, ks)
    # Insert a default 4/4 TS marker for structural presence
    m.insert(0, TimeSignature("4/4"))


def build_ts_score(ts_sequence: list[str], n_parts: int = 2) -> Score:
    """
    Build a score with a per-measure time signature sequence applied identically across parts.
    Measures are inserted at exact bar-aligned offsets; each measure is filled with notes that
    sum to the bar duration (quarterLength). Only signatures with quarter-note denominators are recommended.
    """
    s = Score()
    # Pre-compute offsets
    qls = []
    for ts in ts_sequence:
        ts_obj = TimeSignature(ts)
        qls.append(float(ts_obj.barDuration.quarterLength))
    offsets = [0.0]
    for i in range(1, len(qls)):
        offsets.append(offsets[-1] + qls[i - 1])

    for pi in range(n_parts):
        p = Part(id=str(pi + 1))
        for idx, ts in enumerate(ts_sequence, start=1):
            m = Measure(number=idx)
            # Insert TS only when it changes or at first measure
            if idx == 1 or ts_sequence[idx - 1] != ts_sequence[idx - 2]:  # type: ignore[index]
                m.insert(0, TimeSignature(ts))
            # Fill measure with quarter notes then a final rest if needed
            remaining = qls[idx - 1]
            pitch_base = 60 + pi * 12
            while remaining >= 1.0:
                n = note.Note(pitch_base)
                n.quarterLength = 1.0
                m.append(n)
                remaining -= 1.0
            if remaining > 0:
                r = note.Rest()
                r.quarterLength = remaining
                m.append(r)
            p.insert(offsets[idx - 1], m)
        s.insert(0, p)
    return s
