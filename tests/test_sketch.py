"""Tests for OpenAI-driven sketch workflow."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from music21.humdrum.spineParser import GlobalComment

from composer_toolchain.core import ScoreSpec
from composer_toolchain.score import kern_to_score, load_score, normalize
from composer_toolchain.sketch import (
    DEFAULT_SKETCH_MODEL,
    SketchContext,
    generate_sketch,
)


FIXTURE_PATH = Path(__file__).parent / "data" / "corpus" / "chor-bwv266.krn"


@pytest.mark.skipif("OPENAI_API_KEY" not in os.environ, reason="OPENAI_API_KEY required")
def test_sketch_live():
    score = normalize(load_score(FIXTURE_PATH))
    kern_text = FIXTURE_PATH.read_text(encoding="utf-8")
    spec = ScoreSpec.build(score)
    context = SketchContext(
        spec=spec,
        kern_text=kern_text,
        excerpt_label="bwv266_full",
        composer_brief=(
            "Sketch a contredanse-inspired answer phrase that sequences the"
            " antecedent motive through circle-of-fifths motion while keeping"
            " the chorale's B-flat major cadence profile."
        ),
        measures=4,
        target_parts=len(spec.parts),
        techniques=["variations"],
    )
    model = os.environ.get("SKETCH_TEST_MODEL", DEFAULT_SKETCH_MODEL)
    result = generate_sketch(context=context, model=model)

    assert result.title
    assert result.suffix
    assert result.commentary
    assert result.metrics.measure_count == 4
    assert result.metrics.part_count == len(spec.parts)

    sketch_score = kern_to_score(result.annotated_kern)
    comments = [
        (gc.comment or "").strip()
        for gc in sketch_score.recurse().getElementsByClass(GlobalComment)
        if (gc.comment or "").startswith("SKETCH|")
    ]
    assert comments, "Sketch must embed !! SKETCH global comments"
