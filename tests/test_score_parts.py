from music21.note import Rest
from music21.stream import Measure, Part, Score

from composer_toolchain.score import (
    MeasureSpec,
    PartSelectionError,
    PartSpec,
    _ensure_parts,
    create_excerpt,
)


def _build_part(part_id: str, name: str, abbreviation: str) -> Part:
    part = Part()
    part.id = part_id
    part.partName = name
    part.partAbbreviation = abbreviation
    return part


def test_ensure_parts_prefers_canonical_ids_over_duplicate_aliases() -> None:
    """Canonical ids should remain selectable even when aliases collide."""

    score = Score()
    score.insert(0, _build_part("s", "S", "S"))
    score.insert(0, _build_part("s_2", "S 2", "S"))

    parts = _ensure_parts(score, ["s", "s_2"])
    assert [str(part.id) for part in parts] == ["s", "s_2"]


def test_ensure_parts_skips_conflicting_alias_tokens() -> None:
    """Duplicate abbreviations should not make unrelated aliases fail."""

    score = Score()
    first = _build_part("s", "S", "S")
    second = _build_part("s_2", "S 2", "S")
    score.insert(0, first)
    score.insert(0, second)

    # Alias that does not collide remains usable.
    result = _ensure_parts(score, ["S 2"])
    assert result == [second]

    # Unknown ids should still raise.
    try:
        _ensure_parts(score, ["alto"])
    except PartSelectionError:
        pass
    else:  # pragma: no cover - guard for unexpected behaviour
        raise AssertionError("Unknown part id should raise PartSelectionError")


def test_extract_excerpt_normalises_measure_offsets() -> None:
    score = Score()

    def _part_with_offsets(part_id: str) -> Part:
        part = Part()
        part.id = part_id
        for idx in range(1, 3):
            measure = Measure(number=idx)
            measure.offset = float(idx * 8)
            measure.append(Rest(quarterLength=4.0))
            part.append(measure)
        return part

    score.insert(0, _part_with_offsets("s"))
    score.insert(0, _part_with_offsets("s_2"))

    excerpt = create_excerpt(
        score,
        PartSpec(tokens="s,s_2"),
        MeasureSpec(spec="1-2"),
    )
    for part in excerpt.parts:
        offsets = [measure.offset for measure in part.getElementsByClass(Measure)]
        assert offsets == [0.0, 4.0]
    first_offsets = [
        [
            measure.getOffsetInHierarchy(excerpt)
            for measure in part.getElementsByClass(Measure)
        ]
        for part in excerpt.parts
    ]
    assert first_offsets == [[0.0, 4.0], [0.0, 4.0]]
