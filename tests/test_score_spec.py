from copy import deepcopy

import pytest

from tests.conftest import (
    _STRESS_CORPUS_LIBRARY,
    _corpus_catalog,
    _normalized_corpus_score_cached,
)

from composer_toolchain.core import ScoreSpec


def _assert_cues_cover(scores: dict[str, object]) -> None:
    for name, score in scores.items():
        spec = ScoreSpec.build(score) if not isinstance(score, ScoreSpec) else score
        spans = spec.iter_cue_spans()
        assert spans, f"expected cue spans for {name}"
        assert spans[0].start_measure == 1
        assert spans[-1].end_measure == spec.total_measures
        prev_end = 0
        for span in spans:
            assert span.start_measure == prev_end + 1
            assert span.end_measure >= span.start_measure
            prev_end = span.end_measure
        assert prev_end == spec.total_measures


def test_measure_cues_cover_corpus(corpus_scores):
    """ScoreSpec cue spans must cover every corpus score end-to-end."""

    _assert_cues_cover(corpus_scores)


@pytest.fixture()
def corpus_scores_with_stress(corpus_root):
    catalog = _corpus_catalog(corpus_root, include_stress=True)
    stress_keys = {"multi_movement"}
    return {
        key: deepcopy(_normalized_corpus_score_cached(path))
        for key, path in catalog.items()
        if key in stress_keys
    }


def test_measure_cues_cover_stress_corpus(corpus_scores_with_stress):
    """Cue spans also cover the stress corpus (multi-movement and long works)."""

    _assert_cues_cover(corpus_scores_with_stress)
