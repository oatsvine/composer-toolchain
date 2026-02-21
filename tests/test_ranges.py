
"""
Range parsing/expansion tests
-----------------------------
Invariants:
- Grammar: accepts open-ended and fixed spans; rejects empty tokens and descending intervals.
- Expansion: returns only existing measures; beyond-end tail must be a single contiguous block.
"""

import pytest
from pydantic import ValidationError

from composer_toolchain.score import MeasureSpec, RangeParseError, expand_measure_range
from tests.composer_toolchain.test_utils import build_simple_score

def test_parse_measure_range_valid():
    """Happy path examples should parse to the expected tuples (including open-ended)."""
    spec = MeasureSpec(spec="1-4,7,9-10,12-")
    assert spec.ranges == [(1,4),(7,7),(9,10),(12,None)]

def test_parse_measure_range_invalid():
    """Invalid fragments must raise RangeParseError, not be silently accepted."""
    bad = ["--3", "abc", "4,,5", "10-5", "0", "-3", "2-2-2"]
    for b in bad:
        with pytest.raises(ValidationError):
            MeasureSpec(spec=b)

def test_expand_measure_range_contiguity_and_beyond():
    """Non-contiguous beyond-end tails must be rejected."""
    s = build_simple_score(n_parts=1, n_measures=8)
    # Find last existing measure, and ask for a non-contiguous beyond tail
    last = 8
    spec = f"{last+1},{last+3}"
    try:
        expand_measure_range(s, MeasureSpec(spec=spec))
    except RangeParseError:
        return
    raise AssertionError("expected RangeParseError for non-contiguous beyond-end tail")
