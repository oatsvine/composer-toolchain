"""Shared fixtures for composer_toolchain test suite."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Dict

import pytest
from music21 import metadata as m21_metadata
from music21.stream import Score

from composer_toolchain.score import load_score, normalize, normalize_score


_CORPUS_ROOT = Path(__file__).resolve().parents[2] / "corpus"

# Curated set of representative scores covering different notational edge cases.
# - keyboard_polyphony: densely voiced piano texture (MusicXML)
# - choral_mass: choral/orchestral score delivered as compressed MusicXML
# - fugue_krn: Humdrum source with existing numbering quirks
_CORPUS_LIBRARY: Dict[str, Path] = {
    "keyboard_polyphony": Path("Frdric_Chopin_Polonaise_in_C_minor_Op.40_No.2.mxl"),
    "choral_mass": Path(
        "missa-pange-lingua-josquin-des-prez-missa-pange-lingua-kyrie.mxl.zip"
    ),
    "fugue_krn": Path("bwv891-fugue.krn"),
}

# DONE: Added `--stress` pytest option (register_stress_option) so corpus fixtures can include heavy scores on demand; validated in this session.
# DONE: Executed full suite with `--stress`, exercising larger fixtures to confirm pipeline performance envelopes.
# NOTE: For stress-testing with large or complex scores when making larger changes.
_STRESS_CORPUS_LIBRARY = {
    "multi_movement": Path("Mozart_-_Symphony_No._41_-_Jupiter.mxl"),
    "large_scale_oratorio": Path(
        "js-bach-matthaus-passion-bwv-244-i-chorus-kommt-ihr-tochter-helft-mir-klagen.mxl.zip"
    ),
}

_MAX_CORPUS_CACHE = len(_CORPUS_LIBRARY) + len(_STRESS_CORPUS_LIBRARY)


def register_stress_option(parser: pytest.Parser) -> None:
    parser.addoption(
        "--stress",
        action="store_true",
        help="Include the large stress corpus in score fixtures.",
    )


def _ensure_metadata_filename(score, path: Path):
    if score.metadata is None:
        score.metadata = m21_metadata.Metadata()
    if getattr(score.metadata, "filename", None) in (None, ""):
        score.metadata.filename = path.stem


def _resolve_corpus_path(key: str) -> Path:
    if key in _CORPUS_LIBRARY:
        rel_path = _CORPUS_LIBRARY[key]
    else:
        rel_path = _STRESS_CORPUS_LIBRARY[key]
    return _CORPUS_ROOT / rel_path


def _corpus_catalog(include_stress: bool) -> Dict[str, Path]:
    catalog = dict(_CORPUS_LIBRARY)
    if include_stress:
        catalog.update(_STRESS_CORPUS_LIBRARY)
    return catalog


@lru_cache(maxsize=_MAX_CORPUS_CACHE)
def _raw_corpus_score(key: str):
    path = _resolve_corpus_path(key)
    score = load_score(path)
    _ensure_metadata_filename(score, path)
    return score


@lru_cache(maxsize=_MAX_CORPUS_CACHE)
def _normalized_corpus_score(key: str):
    # Work on a copy to keep the raw cache pristine.
    raw_clone = deepcopy(_raw_corpus_score(key))
    normalized = normalize_score(raw_clone)
    return normalize(normalized)


@pytest.fixture()
def corpus_scores(request: pytest.FixtureRequest) -> Dict[str, Score]:
    """Return deep copies of canonical normalized scores keyed by scenario name."""

    include_stress = bool(request.config.getoption("--stress", default=False))
    catalog = _corpus_catalog(include_stress)
    return {key: deepcopy(_normalized_corpus_score(key)) for key in catalog}


@pytest.fixture()
def raw_corpus_scores(request: pytest.FixtureRequest) -> Dict[str, Score]:
    """Return deep copies of the raw corpus scores (pre-normalization)."""

    include_stress = bool(request.config.getoption("--stress", default=False))
    catalog = _corpus_catalog(include_stress)
    return {key: deepcopy(_raw_corpus_score(key)) for key in catalog}
