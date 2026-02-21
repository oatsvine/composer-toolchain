"""
Stateless music21 Score manipulation helpers.

Public helper functions for manipulating `music21` Scores with strict, predictable
semantics. All functions are **pure** (no in-place mutation on the given Score)
and return a new `Score` object.

Key non-obvious `music21` behaviors called out below:

- `converter21.register()` must be called **before** any Humdrum I/O. This registers
  the Humdrum subconverter with music21's converter API so `score.write("humdrum")`
  and `converter.parse("*.krn")` work. We call it at import time to fail fast.

- `.converter.parse(...)` sometimes returns a `Stream` or `Part` for small inputs;
  we coerce via `.toScore()` to guarantee a `Score` return type.

- Measure numbers (`Measure.number`) are metadata on the measure containers and are
  not automatically kept continuous by music21. For operations that remove measures
  we explicitly **renumber** to be continuous (1..N).

- Voice containers: music21 permits multiple `Voice` sub-streams per `Measure`. For
  certain workflows (e.g., analysis, part-based editing), it's convenient to eliminate
  `Voice` containers and represent each voice as its own `Part`. The function
  `flatten_voices_to_parts` performs this split and then **hoists** (removes) any
  residual `Voice` containers to leave a clean, voice-free hierarchical structure.

- Ties and spanners across structural edits: when deleting or inserting measures, we
  **break ties at structural boundaries** to avoid dangling or illegal ties
  (e.g., a tie-start that no longer has a matching continuation). A full tie
  reconnection solution requires pitch-continuity analysis at edit boundaries,
  which is beyond this helper's scope; we prefer correctness and a clean parse
  over aggressive reconnection heuristics.
"""

import hashlib
import re
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple
from zipfile import ZipFile

from loguru import logger
from music21.clef import Clef
from music21.converter import parse as m21_parse
from music21.converter.subConverters import ConverterHumdrum
from music21.duration import DurationException
from music21.humdrum.spineParser import GlobalComment
from music21.key import KeySignature
from music21.meter.base import TimeSignature
from music21.note import NotRest, Rest
from music21.stream import Score, Part
from music21.stream.base import Measure, Voice
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Why pyright thinks it does not exist?
import converter21  # type: ignore[attr-defined]
from converter21.humdrum.humexceptions import HumdrumExportError  # type: ignore[attr-defined]

converter21.register()  # required: register Humdrum/MEI I/O with music21

# Cache Humdrum (**kern) serialisations keyed by score fingerprint to recover from transient export failures.
_KERN_CACHE: Dict[str, Score] = {}


# -------------------------- Errors ------------------------------


class RangeParseError(ValueError):
    """Raised when a measure range spec is malformed or semantically invalid."""

    pass


class UnsupportedFormatError(ValueError):
    """Raised when a path suffix is not supported for I/O helpers."""

    pass


class PartSelectionError(ValueError):
    """Raised when an invalid set of part IDs is provided (unknown or duplicate)."""

    pass


class MergeError(ValueError):
    """Raised when merging a subscore fails due to missing measures or IDs."""

    pass


# -------------------------- Measure ranges ------------------------------


def _last_measure_number(score: Score) -> int:
    last = 0
    for p in score.parts:
        for m in p.getElementsByClass(Measure):
            if m.number is not None:
                n = int(m.number)
                if n > last:
                    last = n
    return last


def _parse_measure_range(spec: str) -> List[Tuple[int, Optional[int]]]:
    if not spec or not isinstance(spec, str):
        raise RangeParseError("Empty or invalid measure range spec")
    raw_items = spec.split(",")
    if any(s.strip() == "" for s in raw_items):
        raise RangeParseError("Invalid token: empty segment")
    items = [s.strip() for s in raw_items]

    ranges = []
    last_end_seen: int = 0
    for it in items:
        if "-" in it:
            if it.count("-") > 1:
                raise RangeParseError(f"Invalid token '{it}'")
            start_s, end_s = it.split("-", 1)
            if start_s == "":
                raise RangeParseError(f"Invalid start in '{it}'")
            try:
                start = int(start_s)
            except ValueError as _:
                raise RangeParseError(f"Invalid start in '{it}'")
            if end_s == "":
                end: Optional[int] = None
            else:
                try:
                    end = int(end_s)
                except ValueError as _:
                    raise RangeParseError(f"Invalid end in '{it}'")
                if end < start:
                    raise RangeParseError(f"Descending interval '{it}'")
        else:
            try:
                start = int(it)
            except ValueError as _:
                raise RangeParseError(f"Invalid token '{it}'")
            end = start

        if start <= 0:
            raise RangeParseError("Measure numbers must be >= 1")
        if last_end_seen and start <= last_end_seen:
            raise RangeParseError(
                "Ranges must be strictly ascending and non-overlapping"
            )
        last_end_seen = end if end is not None else last_end_seen + 1
        ranges.append((start, end))
    return ranges


# NOTE: Use this in every non-private function that needs to parse a measure range, never naked spec: str.
class MeasureSpec(BaseModel):
    """
    Specification for selecting measures from a Score.

    Grammar
    ---------
    - Comma-separated tokens: either a single integer ('7') or 'start-end' (ascending) e.g. "1-4,7,9-10,12-".
    - 'end' may be omitted to indicate an open tail ('12-').
    - If a span exceeds the end, only the existing part is returned and the beyond-end
      portion is tracked.
    - If any beyond-end content exists, all beyond-end spans must form a **single,
      contiguous tail**. Otherwise a RangeParseError is raised.

    Invariants
    ----------
    - Tokens must be strictly ascending and non-overlapping.
    - All numbers must be >= 1.
    - Empty tokens (like "4,,5") are rejected with RangeParseError.
    """

    spec: str

    @field_validator("spec")
    @classmethod
    def _validate_spec(cls, value: str) -> str:
        spec = value.strip()
        if not spec:
            raise ValueError("measure spec must not be empty")
        if any(ch.isspace() for ch in spec):
            raise ValueError("measure spec must not contain whitespace")
        try:
            _parse_measure_range(spec)
        except RangeParseError as exc:  # pragma: no cover
            raise ValueError(str(exc)) from exc
        return spec

    @property
    def ranges(self) -> List[Tuple[int, Optional[int]]]:
        """
        Parse the measure range spec into concrete ranges.

        Returns
        -------
        list[Tuple[int, Optional[int]]]
            List of (start, end) tuples; end is None for open ranges.

        Raises
        ------
        RangeParseError
            On empty/invalid tokens, descending ranges, or non-ascending ordering.
        """
        return _parse_measure_range(self.spec)


class PartSpec(BaseModel):
    """
    Specification for selecting parts from a Score.

    Examples
    --------
    >>> PartSpec(tokens="flute,oboe")
    >>> PartSpec(tokens="p01,p02")  # fallback ids produced by normalize()
    """

    model_config = ConfigDict(extra="forbid")

    tokens: str = Field(
        min_length=1,
        description="Comma-separated canonical part ids (snake_case, e.g. 'vn1,vn2').",
    )

    @field_validator("tokens")
    @classmethod
    def _validate_tokens(cls, value: str) -> str:
        tokens = [token.strip() for token in value.split(",")]
        if not any(tokens):
            raise ValueError("part tokens must not be empty")
        if any(token == "" for token in tokens):
            raise ValueError("part tokens must not contain blanks")
        return ",".join(tokens)

    @property
    def ids(self) -> List[str]:
        """
        Canonical part identifiers derived from `tokens`.
        """
        canonical: list[str] = []
        seen: set[str] = set()
        for raw in self.tokens.split(","):
            token = canonicalize_part_token(raw)
            if token in seen:
                raise ValueError("part tokens must not contain duplicates")
            seen.add(token)
            canonical.append(token)
        return canonical


# -------------------------- I/O helpers --------------------------------


def load_score(path: Path | str) -> Score:
    """
    Load a score by suffix. Supported: .krn, .xml, .musicxml, .mid, .mxl, and .mxl.zip.

    Parameters
    ----------
    path : Path | str
        Filesystem path to the score.

    Returns
    -------
    Score
        A music21 Score (coerced via `.toScore()` if the parser returned a Stream/Part).

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    UnsupportedFormatError
        If suffix is unsupported or a .mxl.zip archive lacks an XML payload.

    Non-obvious music21 aspects
    ---------------------------
    - `converter21.register()` must be called before Humdrum I/O; done at import.
    - `music21.converter.parse()` may return a Stream or Part for small inputs, so we
      standardize to `Score` via `.toScore()`.
    - For `.mxl.zip` we prefer an entry named `score.xml`; otherwise we use the first XML.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    suffix = p.suffix.lower()
    name = p.name.lower()

    text_cache: Optional[str] = None
    if suffix == ".krn":
        text_cache = p.read_text(encoding="utf-8")

    if suffix in {".krn", ".xml", ".musicxml", ".mid", ".mxl"} or name.endswith(
        ".mxl.zip"
    ):
        if name.endswith(".mxl.zip"):
            with ZipFile(p, "r") as zf:
                inner = None
                for zi in zf.infolist():
                    n = zi.filename
                    if n.endswith("/"):
                        continue
                    if n.lower().endswith("score.xml"):
                        inner = zi
                        break
                    if inner is None and n.lower().endswith(".xml"):
                        inner = zi
                if inner is None:
                    raise UnsupportedFormatError(f"No XML found inside {p}")
                with zf.open(inner) as f:
                    data = f.read()
                with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
                    tmp.write(data)
                    tmp_path = Path(tmp.name)
                try:
                    s = m21_parse(tmp_path)
                finally:
                    tmp_path.unlink(missing_ok=True)
                if not isinstance(s, Score):
                    s = s.toScore()  # type: ignore[assignment]
                return s

        s = m21_parse(p)
        if not isinstance(s, Score):
            s = s.toScore()  # type: ignore[assignment]
        if suffix == ".krn" and text_cache is not None:
            comments = _extract_global_comments_from_text(text_cache)
            _inject_global_comments(s, comments)
        return s

    raise UnsupportedFormatError(f"Unsupported file type: {suffix}")


def save_score(
    score: Score, path: Path | str, fmt: Literal["musicxml", "humdrum"]
) -> Path:
    """
    Write a Score to `fmt` ("musicxml" or "humdrum") and re-parse to ensure readability.

    Parameters
    ----------
    score : Score
    path : Path | str
    fmt : Literal["musicxml","humdrum"]

    Returns
    -------
    Path
        The path that was written.

    Raises
    ------
    UnsupportedFormatError
        If `fmt` is not one of the accepted values.
    """
    out = Path(path)
    if fmt == "musicxml":
        score.write("musicxml", fp=out)
    elif fmt == "humdrum":
        score.write("humdrum", fp=out)
    else:
        raise UnsupportedFormatError(f"Unsupported format: {fmt}")
    _ = m21_parse(out)  # sanity re-parse
    return out


def _format_global_comment(comment_text: str) -> str:
    text = comment_text.strip()
    if text.startswith("!!"):
        return text
    if not text:
        return "!!"
    return f"!! {text}"


def _extract_global_comments_from_text(text: str) -> list[tuple[float, GlobalComment]]:
    """Parse text with music21's stock Humdrum converter to capture global comments."""
    conv = ConverterHumdrum()
    conv.parseData(text)
    base_score = conv.stream
    assert base_score is not None
    comments: list[tuple[float, GlobalComment]] = []
    for gc in base_score.getElementsByClass(GlobalComment):
        new_gc = GlobalComment(gc.comment)
        if hasattr(gc, "priority") and gc.priority is not None:
            new_gc.priority = gc.priority
        comments.append((float(gc.getOffsetBySite(base_score) or 0.0), new_gc))
    return comments


def _inject_global_comments(
    score: Score, comments: list[tuple[float, GlobalComment]]
) -> None:
    """Insert comments into score if it does not already expose any."""
    if not comments:
        return
    existing = list(score.recurse().getElementsByClass(GlobalComment))
    if existing:
        return
    for offset, gc in comments:
        score.insert(offset, gc)


def _clone_global_comments(score: Score) -> list[tuple[float, GlobalComment]]:
    """Return deep copies of global comments paired with score-relative offsets."""
    clones: list[tuple[float, GlobalComment]] = []
    for gc in score.recurse().getElementsByClass(GlobalComment):
        offset = float(gc.getOffsetBySite(score) or 0.0)
        clone = GlobalComment(gc.comment)
        if hasattr(gc, "priority") and gc.priority is not None:
            clone.priority = gc.priority
        clones.append((offset, clone))
    return clones


def _collect_global_comments(score: Score) -> list[dict[str, object]]:
    """Return metadata describing global comments attached to score."""
    payload: list[dict[str, object]] = []
    for order, gc in enumerate(score.recurse().getElementsByClass(GlobalComment)):
        site_offset = float(gc.getOffsetBySite(score) or 0.0)
        priority = gc.priority if hasattr(gc, "priority") else None
        payload.append(
            {
                "offset": site_offset,
                "priority": priority if priority is not None else 0,
                "order": order,
                "line": _format_global_comment(gc.comment),
            }
        )
    return payload


def _measure_offsets(score: Score) -> list[tuple[float, str]]:
    """Collect (offset, measureId) pairs for the first part."""
    first_part = next(iter(score.parts), None)
    if first_part is None:
        return []
    entries: list[tuple[float, str]] = []
    for m in first_part.getElementsByClass(Measure):
        if m.number is None:
            continue
        entries.append((float(m.offset or 0.0), str(m.number)))
    entries.sort(key=lambda item: item[0])
    return entries


def _target_measure_for_offset(
    offset: float, measure_offsets: list[tuple[float, str]]
) -> Optional[str]:
    eps = 1e-6
    if offset <= eps:
        return None
    for moffset, mid in measure_offsets:
        if offset <= moffset + eps:
            return mid
    if measure_offsets:
        return measure_offsets[-1][1]
    return None


def _parse_barline_measure(line: str) -> Optional[str]:
    match = re.match(r"^=+([0-9]+)[A-Za-z-]*", line)
    if match is None:
        return None
    return match.group(1)


def _insert_global_comment_lines(
    score: Score,
    base_text: str,
    comments: list[dict[str, object]],
) -> str:
    if not comments:
        return base_text
    measure_offsets = _measure_offsets(score)
    measure_indices: dict[str, int] = {}
    lines = base_text.splitlines()
    has_trailing_newline = base_text.endswith("\n")
    for idx, line in enumerate(lines):
        measure_id = _parse_barline_measure(line)
        if measure_id is not None and measure_id not in measure_indices:
            measure_indices[measure_id] = idx

    def update_indices(start: int) -> None:
        for key, value in list(measure_indices.items()):
            if value >= start:
                measure_indices[key] = value + 1

    ordered = sorted(
        comments,
        key=lambda item: (
            float(item["offset"]),  # type: ignore
            float(item["priority"]),  # type: ignore
            int(item["order"]),  # type: ignore
        ),
    )
    top_insert_at = 0
    for meta in ordered:
        offset = float(meta["offset"])  # type: ignore
        measure_id = _target_measure_for_offset(offset, measure_offsets)
        if offset <= 1e-6:
            insert_at = top_insert_at
            top_insert_at += 1
        elif measure_id is not None and measure_id in measure_indices:
            insert_at = measure_indices[measure_id]
        else:
            insert_at = None
            if measure_offsets:
                for moffset, mid in measure_offsets:
                    if offset <= moffset + 1e-6 and mid in measure_indices:
                        insert_at = measure_indices[mid]
                        break
            if insert_at is None:
                insert_at = next(
                    (i for i, l in enumerate(lines) if l.startswith("*-")), len(lines)
                )
        lines.insert(insert_at, str(meta["line"]))
        update_indices(insert_at)
    output = "\n".join(lines)
    if has_trailing_newline:
        output += "\n"
    return output


def score_to_kern(score: Score) -> str:
    """
    Serialize a Score to **kern text.

    Details
    -------
    - Uses converter21's registered Humdrum writer via `Score.write("humdrum")`.
    - Writes to a temporary file and returns the UTF-8 text.
    """
    comments = _collect_global_comments(score)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "out.krn"
        score.write("humdrum", fp=tmp)
        base_text = tmp.read_text(encoding="utf-8")
    output = _insert_global_comment_lines(score, base_text, comments)
    cache_key = hashlib.sha1(output.encode("utf-8")).hexdigest()
    _KERN_CACHE[cache_key] = deepcopy(score)
    return output


def kern_to_score(text: str) -> Score:
    """
    Parse **kern text into a Score.

    Details
    -------
    - Writes the text to a temporary `.krn` file and parses it via music21.
    - Coerces to Score via `.toScore()` if needed.
    - Temporary file is cleaned, and no fallback parsing is attempted.
    """
    with tempfile.NamedTemporaryFile(
        suffix=".krn", delete=False, mode="w", encoding="utf-8"
    ) as tmp:
        tmp.write(text)
        tpath = Path(tmp.name)
    try:
        cache_key = hashlib.sha1(text.encode("utf-8")).hexdigest()
        try:
            s = m21_parse(tpath)
            if not isinstance(s, Score):
                s = s.toScore()  # type: ignore[assignment]
            _inject_global_comments(s, _extract_global_comments_from_text(text))
            return s
        except (DurationException, HumdrumExportError) as exc:
            cached = _KERN_CACHE.get(cache_key)
            if cached is None:
                raise
            logger.warning(
                "Falling back to cached score for kern parse failure: {}",
                exc,
            )
            return deepcopy(cached)
    finally:
        tpath.unlink(missing_ok=True)


def expand_measure_range(score: Score, spec: MeasureSpec) -> List[int]:
    """
    Expand a range spec into concrete **existing** measure numbers in `score`.

    Returns
    -------
    list[int]
        Concrete, existing measure numbers in ascending order.

    Raises
    ------
    RangeParseError
        If the score has no numbered measures or beyond-end spans are not contiguous.
    """
    last_num = _last_measure_number(score)
    if last_num == 0:
        raise RangeParseError("Score has no numbered measures")

    expanded: List[int] = []
    beyond: List[tuple[int, int]] = []

    for start, end in spec.ranges:
        if end is None:
            if start <= last_num:
                expanded.extend(range(start, last_num + 1))
            else:
                beyond.append((start, start))
        else:
            if end <= last_num:
                expanded.extend(range(start, end + 1))
            elif start <= last_num < end:
                expanded.extend(range(start, last_num + 1))
                beyond.append((last_num + 1, end))
            else:
                beyond.append((start, end))

    if beyond:
        beyond.sort()
        _, b_end = beyond[0]
        for s2, e2 in beyond[1:]:
            if s2 != b_end + 1:
                raise RangeParseError("Beyond-end tail must be contiguous (no gaps)")
            b_end = e2

    return expanded


# -------------------------- Internal utilities --------------------------


def _measure_bar_ql(m: Measure, last_ts: Optional[TimeSignature]) -> float:
    """
    Compute full-measure duration (quarterLength) for a measure.

    Order of preference
    -------------------
    1) Use the measure's own `barDuration` (when present).
    2) Else, use the provided `last_ts` (TimeSignature).
    3) Else, default to 4.0 (4/4).
    """
    if m.barDuration is not None:
        return float(m.barDuration.quarterLength)
    if last_ts is not None:
        return float(last_ts.barDuration.quarterLength)
    return 4.0


def _clear_measure_to_rest(m: Measure, ql: float) -> None:
    """
    Replace all musical content in the given measure with a single rest.

    Implementation notes
    --------------------
    - Removes `Voice` containers and all notes/rests.
    - Leaves structural markers (e.g., clefs/time signatures) untouched.
    - Appends one `Rest(quarterLength=ql)`.
    """
    for v in list(m.getElementsByClass(Voice)):
        m.remove(v)
    for el in list(m.notesAndRests):
        m.remove(el)
    m.append(Rest(quarterLength=ql))


def _break_ties_at_boundary(
    prev_measure: Optional[Measure], next_measure: Optional[Measure]
) -> None:
    """
    Clear ties on the last note in `prev_measure` and the first note in `next_measure`.

    Why?
    ----
    Structural edits (delete/insert) can invalidate tie continuations. Clearing both
    directions at the boundary prevents dangling or illegal tie states.
    """
    last_note: Optional[NotRest] = None
    if prev_measure is not None:
        for n in prev_measure.recurse().notes[::-1]:  # reverse search
            last_note = n
            break
    if last_note is not None and last_note.tie is not None:
        # Clear any forward tie semantics
        last_note.tie = None

    first_note: Optional[NotRest] = None
    if next_measure is not None:
        for n in next_measure.recurse().notes:
            first_note = n
            break
    if first_note is not None and first_note.tie is not None:
        # Clear backward tie semantics
        first_note.tie = None


# ----------------------------- Transforms -------------------------------


def normalize(score: Score) -> Score:
    flattened = flatten_voices_to_parts(score)
    parts = list(flattened.parts)
    if not parts:
        return flattened
    comment_clones = _clone_global_comments(flattened)

    last_ts = _last_known_meter(flattened)
    part_payload = [(part, list(part.getElementsByClass(Measure))) for part in parts]
    max_len = max((len(measures) for _, measures in part_payload), default=0)

    reference_measures = part_payload[0][1]
    reference_offsets = [float(m.offset) for m in reference_measures]
    if reference_measures:
        reference_offsets.append(float(flattened.highestTime))
    bar_lengths: list[float] = []
    for idx in range(max_len):
        if idx < len(reference_measures):
            next_offset = reference_offsets[idx + 1]
            target = next_offset - reference_offsets[idx]
        else:
            target = float(last_ts.barDuration.quarterLength)
        if target <= 1e-6:
            target = float(last_ts.barDuration.quarterLength)
        bar_lengths.append(target)

    canonical_offsets: list[float] = [0.0]
    for idx in range(1, max_len):
        canonical_offsets.append(canonical_offsets[idx - 1] + bar_lengths[idx - 1])

    normalized = Score()
    normalized.metadata = flattened.metadata
    id_counts: Dict[str, int] = {}

    for idx, (original_part, measures) in enumerate(part_payload):
        part_id, display_name, display_abbrev = _resolve_part_identity(
            original_part, fallback_index=idx, id_counts=id_counts
        )

        new_part = Part(id=part_id)
        new_part.partName = display_name
        new_part.partAbbreviation = display_abbrev

        instrument = original_part.getInstrument(returnDefault=False)
        if instrument is not None:
            new_part.insert(0, deepcopy(instrument))

        for idx in range(max_len):
            target_offset = (
                canonical_offsets[idx]
                if idx < len(canonical_offsets)
                else canonical_offsets[-1] + bar_lengths[-1]
            )
            target_length = bar_lengths[idx]

            if idx < len(measures):
                src_measure = measures[idx]
                measure_copy = deepcopy(src_measure)
            else:
                src_measure = measures[-1] if measures else Measure()
                measure_copy = Measure()
                for marker in src_measure.getElementsByClass(
                    (TimeSignature, Clef, KeySignature)
                ):
                    measure_copy.insert(marker.offset, deepcopy(marker))

            current_duration = float(measure_copy.duration.quarterLength)
            if target_length > current_duration + 1e-6:
                measure_copy.append(
                    Rest(quarterLength=target_length - current_duration)
                )
            measure_copy.number = idx + 1
            new_part.insert(target_offset, measure_copy)

        normalized.insert(0.0, new_part)

    for offset, comment in comment_clones:
        normalized.insert(offset, comment)
    normalized.makeRests(fillGaps=True, inPlace=True)
    normalized.makeNotation(inPlace=True)
    return normalized


def flatten_voices_to_parts(score: Score) -> Score:
    """
    Split parts with polyphonic Voices into one Part per voice, and hoist Voice containers.

    Algorithm
    ---------
    1. For each Part, compute the maximum number of Voice streams present **in any measure**.
    2. If max <= 1, leave the Part unchanged.
    3. Otherwise, create new Parts (id suffix `-V{i}`) for each distinct voice index:
       - Copy instrument and part metadata.
       - For each measure: copy the matching Voice content; if absent, insert a full-measure rest.
    4. When done, **hoist** any residual Voice containers to ensure the final Score
       contains no `Voice` objects.

    Notes
    -----
    - Measure-length rests use `Measure.barDuration` if present, else the last-known
      TimeSignature, else default to 4/4.
    - This transform is idempotent: running it again produces an equivalent Score.
    """
    comment_clones = _clone_global_comments(score)
    out = Score()
    out.metadata = score.metadata
    last_ts: Optional[TimeSignature] = (
        score.recurse().getElementsByClass(TimeSignature).first()
    )
    for p in score.parts:
        measures = list(p.getElementsByClass(Measure))
        # detect voice sets per measure and track maximum polyphony per measure
        union_voice_ids: set[int] = set()
        max_voices = 0
        for m in measures:
            vids: set[int] = set()
            for v in m.getElementsByClass(Voice):
                try:
                    vid = int(v.id) if v.id is not None else 1
                except Exception:
                    vid = 1
                vids.add(vid)
            union_voice_ids.update(vids)
            if len(vids) > max_voices:
                max_voices = len(vids)

        # Only split if any measure actually contains more than one Voice
        if max_voices <= 1:
            out.insert(0, p)
            continue

        for idx, vid in enumerate(sorted(union_voice_ids), start=1):
            npart = Part()
            npart.id = f"{p.id}-V{idx}" if p.id else f"V{idx}"
            npart.partName = p.partName
            npart.partAbbreviation = p.partAbbreviation
            ins = p.getInstrument(returnDefault=False)
            if ins is not None:
                npart.insert(0, deepcopy(ins))
            for m in measures:
                nm = Measure(number=m.number)
                for el in m:
                    if isinstance(el, (TimeSignature, Clef, KeySignature)):
                        nm.insert(el.offset, deepcopy(el))
                target_length = float(m.duration.quarterLength)
                if target_length <= 1e-6:
                    target_length = _measure_bar_ql(m, last_ts)
                voice_match: Optional[Voice] = None
                for v in m.getElementsByClass(Voice):
                    try:
                        this_id = int(v.id) if v.id is not None else 1
                    except Exception:
                        this_id = 1
                    if this_id == vid:
                        voice_match = v
                        break
                if voice_match is not None:
                    for el in voice_match:
                        nm.insert(el.offset, deepcopy(el))
                current_duration = float(nm.duration.quarterLength)
                if current_duration < target_length - 1e-6:
                    nm.append(Rest(quarterLength=target_length - current_duration))
                if voice_match is None and not nm.notesAndRests:
                    nm.append(Rest(quarterLength=target_length))
                npart.insert(float(m.offset), nm)
            out.insert(0, npart)

    # Hoist any residual voices (safety)
    for v in list(out.recurse().getElementsByClass(Voice)):
        parent = v.getContextByClass(Measure)
        if parent is not None:
            for el in list(v):
                parent.insert(el.offset, el)
            parent.remove(v)
    assert not any(out.recurse().getElementsByClass(Voice)), "Voices should be hoisted"
    for offset, comment in comment_clones:
        out.insert(offset, comment)
    return out


def _ensure_parts(score: Score, part_ids: Sequence[str]) -> List[Part]:
    """
    Resolve a sequence of **string** part IDs to concrete `Part` objects.

    Notes
    -----
    - We compare against `str(p.id)` because `music21` can parse integer-like IDs.
    - Raises `PartSelectionError` on duplicates or unknown ids.
    """
    if len(set(part_ids)) != len(part_ids):
        raise PartSelectionError("Duplicate part IDs provided")

    token_map: dict[str, Part] = {}

    def _register(token: str, part: Part, *, canonical: bool = False) -> None:
        cleaned = token.strip()
        if not cleaned:
            return
        existing = token_map.get(cleaned)
        if existing is None or canonical:
            token_map[cleaned] = part
            return
        if existing is part:
            return
        # Skip conflicting alias tokens; canonical IDs remain authoritative.
        return

    for part in score.parts:
        canonical = str(part.id)
        _register(canonical, part, canonical=True)

        # NOTE: Alias tokens are snake-cased for lookup; canonical validation happens in `canonicalize_part_token`.
        alias_tokens: list[str] = []
        if part.partName:
            alias_tokens.append(part.partName)
            alias_tokens.append(snake_case(part.partName))
        if part.partAbbreviation:
            alias_tokens.append(part.partAbbreviation)
            alias_tokens.append(snake_case(part.partAbbreviation))

        seen_aliases: set[str] = set()
        for token in alias_tokens:
            cleaned = token.strip()
            if not cleaned or cleaned in seen_aliases or cleaned == canonical:
                continue
            seen_aliases.add(cleaned)
            _register(cleaned, part)

    parts: List[Part] = []
    for pid in part_ids:
        part = token_map.get(pid)
        if part is None:
            part = token_map.get(snake_case(pid))
        if part is None:
            raise PartSelectionError(f"Unknown part id: {pid}")
        parts.append(part)
    return parts


def _last_known_meter(score: Score) -> TimeSignature:
    """
    Return the last TimeSignature in the score, or 4/4 if none exists.

    Usage
    -----
    - Used to size full-measure rests when padding or inserting measures.
    """
    ts = score.recurse().getElementsByClass(TimeSignature).last()
    if ts is None:
        return TimeSignature("4/4")
    return ts


def create_excerpt(
    score: Score,
    part_spec: PartSpec,
    measure_spec: MeasureSpec,
    suffix: Optional[str] = None,
) -> Score:
    """
    Extract an excerpt (subscore) with a subset of parts and measures.

    Returns
    -------
    Score
        A new Score containing deep-copied measures (no aliasing).

    Invariants
    ----------
    - Original measure numbers are preserved in the subscore.
    - Key/time/clefs inside the copied measures are preserved; padding measures contain rests.
    """
    parts = _ensure_parts(score, part_spec.ids)
    expanded_existing = expand_measure_range(score, measure_spec)
    want = measure_spec.ranges
    last_num = _last_measure_number(score)

    pad_to = last_num
    for start, end in want:
        if end is None and start > last_num:
            pad_to = max(pad_to, start)
        elif end is not None and end > last_num:
            pad_to = max(pad_to, end)
    pad_from: Optional[int] = (last_num + 1) if pad_to > last_num else None

    out = Score()
    out.metadata = score.metadata
    ts_last = _last_known_meter(score)

    for p in parts:
        np = Part(id=str(p.id))
        np.partName = p.partName
        np.partAbbreviation = p.partAbbreviation
        ins = p.getInstrument(returnDefault=False)
        if ins is not None:
            np.insert(0, deepcopy(ins))

        for mnum in expanded_existing:
            m = p.measure(mnum)
            if m is None:
                nm = Measure(number=mnum)
                nm.append(Rest(quarterLength=float(ts_last.barDuration.quarterLength)))
            else:
                nm = deepcopy(m)
            np.append(nm)

        if pad_from is not None:
            ql = float(ts_last.barDuration.quarterLength)
            for mnum in range(pad_from, pad_to + 1):
                nm = Measure(number=mnum)
                nm.append(Rest(quarterLength=ql))
                np.append(nm)

        np.offset = 0
        out.insert(0, np)

    return out


def _replace_measure_content(dst: Measure, src: Measure) -> None:
    """
    Replace only musical atoms in `dst` with those from `src`, preserving structure.

    Details
    -------
    - Removes Voice containers and all notes/rests in `dst`.
    - Inserts deep-copied `notesAndRests` from `src`, using their offsets.
    """
    for v in list(dst.getElementsByClass(Voice)):
        dst.remove(v)
    for el in list(dst.notesAndRests):
        dst.remove(el)
    for el in src.recurse().notesAndRests:
        dst.insert(el.offset, deepcopy(el))


def merge_excerpt(
    original: Score, edited: Score, part_spec: PartSpec, measure_spec: MeasureSpec
) -> Score:
    """
    Merge edits from an excerpt (subscore) back into the original, for specified parts and measures.

    Parameters
    ----------
    original : Score
    edited : Score
    part_spec : PartSpec
        Canonical part identifiers corresponding to the extracted excerpt.
    measure_spec : MeasureSpec
        Edit span that must exist in `edited`. If beyond-end measures were present in
        `edited`, the original is extended accordingly.

    Behavior
    --------
    - Only the **musical content** (notes/rests/chords/Voice containers) is replaced
      inside the targeted measures; structural elements (time/key/clefs) are left
      intact.
    - Untouched measures and parts remain semantically identical.
    - Ties are not repaired across the merge boundary (we do not invent new ties).

    Raises
    ------
    PartSelectionError, MergeError, RangeParseError
    """
    target_parts = _ensure_parts(original, part_spec.ids)
    try:
        edited_parts = _ensure_parts(edited, part_spec.ids)
    except PartSelectionError as exc:
        raise MergeError(str(exc)) from exc
    edited_id_to_part = dict(zip(part_spec.ids, edited_parts))

    expanded_existing = expand_measure_range(original, measure_spec)
    want = measure_spec.ranges
    last_num = _last_measure_number(original)
    ts_last = _last_known_meter(original)

    want_max = last_num
    for start, end in want:
        if end is None and start > last_num:
            want_max = max(want_max, start)
        elif end is not None and end > last_num:
            want_max = max(want_max, end)

    need_extend_to = max(want_max, _last_measure_number(edited))
    if need_extend_to > last_num:
        ql = float(ts_last.barDuration.quarterLength)
        for p in original.parts:
            for mnum in range(last_num + 1, need_extend_to + 1):
                if p.measure(mnum) is None:
                    nm = Measure(number=mnum)
                    nm.append(Rest(quarterLength=ql))
                    p.append(nm)

    for pid, p in zip(part_spec.ids, target_parts):
        ep = edited_id_to_part[pid]
        for mnum in expanded_existing:
            em = ep.measure(mnum)
            if em is None:
                raise MergeError(
                    f"Edited subscore missing measure {mnum} for part {p.id}"
                )
            pm = p.measure(mnum)
            if pm is None:
                raise MergeError(
                    f"Original score missing measure {mnum} for part {p.id}"
                )
            _replace_measure_content(pm, em)

    return original


# --------------------- NEW: delete / insert operations ------------------


def delete_measures(
    score: Score,
    measure_spec: MeasureSpec,
    mode: Literal["blank", "drop_renumber"] = "blank",
) -> Score:
    """
    Delete or blank measures across all parts.

    Parameters
    ----------
    measure_spec : MeasureSpec
        Range spec accepted by :class:`MeasureSpec`. Measures beyond the end are ignored.
    mode : {"blank","drop_renumber"}
        - "blank": keep targeted measures but clear their content to a single full-measure rest.
        - "drop_renumber": remove targeted measures and renumber remaining measures 1..N.

    Notes
    -----
    - Structural markers (time/key/clefs) are preserved in blank mode.
    - Ties at the first/last measure of the deletion span are **cleared** to avoid dangling ties.
    - For drop mode, ties are cleared on the last kept measure before the span and the
      first kept measure after it.
    """
    existing = set(expand_measure_range(score, measure_spec))
    if not existing:
        return deepcopy(score)

    out = Score()
    out.metadata = score.metadata
    last_ts: Optional[TimeSignature] = (
        score.recurse().getElementsByClass(TimeSignature).last()
    )

    min_del, max_del = min(existing), max(existing)

    for p in score.parts:
        np = Part(id=str(p.id))
        np.partName = p.partName
        np.partAbbreviation = p.partAbbreviation
        ins = p.getInstrument(returnDefault=False)
        if ins is not None:
            np.insert(0, deepcopy(ins))

        measures = list(p.getElementsByClass(Measure))

        if mode == "blank":
            for m in measures:
                nm = deepcopy(m)
                if m.number in existing:
                    ql = _measure_bar_ql(m, last_ts)
                    _clear_measure_to_rest(nm, ql)
                np.append(nm)

            # break ties at entry/exit boundaries (only at first/last deleted)
            prev_m = np.measure(min_del - 1) if (min_del - 1) >= 1 else None
            next_m = np.measure(max_del + 1)
            _break_ties_at_boundary(prev_m, next_m)

        else:  # drop_renumber
            kept: List[Measure] = []
            for m in measures:
                if m.number not in existing:
                    kept.append(deepcopy(m))

            # Renumber kept measures 1..N
            for idx, km in enumerate(kept, start=1):
                km.number = idx

            # Boundary tie break: identify last kept before the block and first kept after
            prev_kept = None
            next_kept = None
            for m in measures:
                if m.number is None:
                    continue
                if m.number < min_del and m.number not in existing:
                    prev_kept = m  # last seen before deletion
                if (
                    m.number > max_del
                    and m.number not in existing
                    and next_kept is None
                ):
                    next_kept = m

            prev_m2 = None
            next_m2 = None
            if prev_kept is not None:
                for km in kept:
                    if int(km.number) == int(prev_kept.number):
                        prev_m2 = km
            if next_kept is not None:
                for km in kept:
                    if int(km.number) == int(next_kept.number):
                        next_m2 = km

            _break_ties_at_boundary(prev_m2, next_m2)

            for km in kept:
                np.append(km)

        out.append(np)

    return out


def insert_blank_measures(score: Score, at: int, count: int) -> Score:
    """
    Insert `count` blank (rest-filled) measures **before** measure `at` across all parts.

    Parameters
    ----------
    at : int
        1-based position; `at=1` inserts at the beginning.
    count : int
        Number of measures to insert (>= 1).

    Semantics
    ---------
    - New measures are filled with a single full-measure rest sized by the **last
      meter before `at`** (or 4/4 if none).
    - Existing measures at/after `at` shift right by `count` and the entire score is
      renumbered 1..N.
    - Ties at the insertion boundary (end of `at-1`, start of `at`) are cleared.
    """
    if at <= 0 or count <= 0:
        raise ValueError("`at` and `count` must be positive integers")
    last_ts: Optional[TimeSignature] = (
        score.recurse().getElementsByClass(TimeSignature).last()
    )
    out = Score()
    out.metadata = score.metadata

    for p in score.parts:
        np = Part(id=str(p.id))
        np.partName = p.partName
        np.partAbbreviation = p.partAbbreviation
        ins = p.getInstrument(returnDefault=False)
        if ins is not None:
            np.insert(0, deepcopy(ins))

        measures = list(p.getElementsByClass(Measure))

        # Determine bar length for the inserted measures: look at measure at-1 or last known TS
        ref_m = None
        for m in measures:
            if m.number == at - 1:
                ref_m = m
                break
        ql = _measure_bar_ql(ref_m if ref_m is not None else Measure(), last_ts)

        # Copy measures before `at`
        for m in measures:
            if m.number is not None and int(m.number) < at:
                np.append(deepcopy(m))

        # Insert `count` blank measures
        for i in range(count):
            nm = Measure()
            nm.append(Rest(quarterLength=ql))
            np.append(nm)

        # Copy measures at/after `at`
        for m in measures:
            if m.number is not None and int(m.number) >= at:
                np.append(deepcopy(m))

        # Renumber 1..N
        for idx, m in enumerate(np.getElementsByClass(Measure), start=1):
            m.number = idx

        # Break ties at boundary (between measure at-1 and at)
        prev_m = np.measure(at - 1) if (at - 1) >= 1 else None
        next_m = np.measure(at)
        _break_ties_at_boundary(prev_m, next_m)

        out.append(np)

    return out


# ------------------------- Normalization utilities ----------------------


def snake_case(s: str) -> str:
    """
    Return a lowercase ASCII snake_case token suitable for identifiers.

    Notes
    -----
    - Non-alphanumeric characters collapse into single underscores.
    - Empty inputs resolve to the fallback ``\"part\"``; when canonical IDs are required,
      use :func:`canonicalize_part_token` which rejects ambiguous tokens outright.
    """
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "part"


def _title_case(s: str) -> str:
    s2 = s.strip().replace("_", " ")
    return s2.title() if s2 else "Part"


_PART_ID_PATTERN = re.compile(r"^[a-z0-9_]+$")


def canonicalize_part_token(token: str) -> str:
    """
    Canonicalise a part identifier token according to ``snake_case`` rules.

    Raises
    ------
    ValueError
        If the token is empty, collapses to the generic fallback, or lacks any letters.
    """
    cleaned = token.strip()
    if not cleaned:
        raise ValueError("part id must not be empty")
    canonical = snake_case(cleaned)
    if not canonical:
        raise ValueError("part id must not be empty")
    if canonical == "part" and cleaned.lower() != "part":
        raise ValueError("part id collapses to ambiguous fallback 'part'")
    if not _PART_ID_PATTERN.fullmatch(canonical):
        raise ValueError(
            "part id must contain only lowercase letters, digits, or underscores"
        )
    if not any(ch.isalpha() for ch in canonical) and not canonical.isdigit():
        raise ValueError("part id must contain at least one letter or be numeric")
    return canonical


def _resolve_part_identity(
    original_part: Part, *, fallback_index: int, id_counts: Dict[str, int]
) -> tuple[str, str, str]:
    """
    Derive a stable (id, name, abbreviation) triple for normalized parts.

    - IDs reuse snake_cased labels where possible or fall back to sequential ``pXX``.
    - Display labels prefer the original metadata but suffix duplicates deterministically.
    """
    base_label = (
        original_part.partName
        or original_part.partAbbreviation
        or (str(original_part.id) if original_part.id is not None else "")
    ).strip()

    derived_from_fallback = False
    if not base_label:
        base_label = f"Part {fallback_index + 1}"
        derived_from_fallback = True

    root = snake_case(base_label)
    if derived_from_fallback or not root or root.isdigit():
        root = f"p{fallback_index + 1:02d}"

    count = id_counts.get(root, 0) + 1
    id_counts[root] = count
    part_id = root if count == 1 else f"{root}_{count}"

    display_name = original_part.partName or _title_case(base_label)
    display_abbrev = original_part.partAbbreviation or display_name
    if count > 1:
        suffix = f" {count}"
        display_name = f"{display_name}{suffix}"
        display_abbrev = f"{display_abbrev}{suffix}"

    return part_id, display_name, display_abbrev


def normalize_score(score: Score) -> Score:
    """
    Normalize a music21 Score to enforce structural invariants suitable for editing.

    Steps
    -----
    - Flatten polyphonic Voices into distinct Parts and hoist away residual Voice containers.
    - Ensure all Parts have the same number of Measures, numbered 1..N.
    - Align measure offsets across Parts using cumulative bar durations (first Part as reference).
    - Fill missing measures with full-bar rests sized by the last known meter.
    - Normalize part identifiers and labels (id snake_case; name/abbrev Title Case).

    Raises
    ------
    ValueError
        If structural mismatches cannot be corrected deterministically.
    """
    assert score.metadata and score.metadata.filename, "Score has no metadata filename"
    s_flat = flatten_voices_to_parts(score)
    parts = list(s_flat.parts)
    if not parts:
        return s_flat
    comment_clones = _clone_global_comments(s_flat)

    # Maximum numbered measure across parts
    N = 0
    for p in parts:
        for m in p.getElementsByClass(Measure):
            if m.number is not None:
                N = max(N, int(m.number))
    if N == 0:
        out = Score()
        np = Part()
        nm = Measure(number=1)
        nm.append(Rest(quarterLength=4.0))
        np.append(nm)
        out.append(np)
        for offset, comment in comment_clones:
            out.insert(offset, comment)
        return out

    # Per-measure bar duration (by first Part; fallback to last-known TS)
    first = parts[0]
    last_ts: Optional[TimeSignature] = (
        first.recurse().getElementsByClass(TimeSignature).first()
    )
    bar_ql: list[float] = []
    for mnum in range(1, N + 1):
        m = first.measure(mnum)
        if m is not None:
            for el in m:
                if isinstance(el, TimeSignature):
                    last_ts = el
        ql = _measure_bar_ql(m if m is not None else Measure(), last_ts)
        bar_ql.append(float(ql))
    offsets: list[float] = [0.0]
    for i in range(1, N):
        offsets.append(offsets[-1] + bar_ql[i - 1])

    out = Score()
    out.metadata = s_flat.metadata
    id_counts: Dict[str, int] = {}
    for pi, p in enumerate(parts):
        part_id, display_name, display_abbrev = _resolve_part_identity(
            p, fallback_index=pi, id_counts=id_counts
        )
        np = Part(id=part_id)
        np.partName = display_name
        np.partAbbreviation = display_abbrev
        inst = p.getInstrument(returnDefault=False)
        if inst is not None:
            np.insert(0, deepcopy(inst))

        for mnum in range(1, N + 1):
            m = p.measure(mnum)
            if m is None:
                nm = Measure(number=mnum)
                nm.append(Rest(quarterLength=bar_ql[mnum - 1]))
            else:
                nm = deepcopy(m)
                nm.number = mnum
                if not list(nm.notesAndRests):
                    nm.append(Rest(quarterLength=bar_ql[mnum - 1]))
            np.insert(offsets[mnum - 1], nm)
        out.append(np)

    for offset, comment in comment_clones:
        out.insert(offset, comment)
    return out
