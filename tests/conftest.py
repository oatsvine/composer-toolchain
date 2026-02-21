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


_DATA_ROOT = Path(__file__).parent / "data"
_CORPUS_SUBDIR = Path("corpus")

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


def _corpus_catalog(root: Path, include_stress: bool) -> Dict[str, Path]:
    catalog: Dict[str, Path] = {key: root / rel for key, rel in _CORPUS_LIBRARY.items()}
    if include_stress:
        catalog.update({key: root / rel for key, rel in _STRESS_CORPUS_LIBRARY.items()})
    return catalog


@lru_cache(maxsize=_MAX_CORPUS_CACHE)
def _raw_corpus_score_cached(path: Path) -> Score:
    score = load_score(path)
    _ensure_metadata_filename(score, path)
    return score


@lru_cache(maxsize=_MAX_CORPUS_CACHE)
def _normalized_corpus_score_cached(path: Path) -> Score:
    raw_clone = deepcopy(_raw_corpus_score_cached(path))
    normalized = normalize_score(raw_clone)
    return normalize(normalized)


def _require_file(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"Corpus file missing: {path}")
    return path


@pytest.fixture()
def corpus_root(shared_datadir: Path) -> Path:
    base = shared_datadir if shared_datadir.exists() else _DATA_ROOT
    target = base / _CORPUS_SUBDIR
    if not target.exists():
        pytest.skip(f"Corpus directory missing: {target}")
    return target


@pytest.fixture()
def sample_score_path(corpus_root: Path) -> Path:
    return _require_file(corpus_root / "bwv891-prelude.krn")


@pytest.fixture()
def alt_score_path(corpus_root: Path) -> Path:
    return _require_file(corpus_root / "wtc1p16.krn")


@pytest.fixture()
def multi_movement_score_path(corpus_root: Path) -> Path:
    return _require_file(corpus_root / "Mozart_-_Symphony_No._41_-_Jupiter.mxl")


@pytest.fixture()
def corpus_scores(
    request: pytest.FixtureRequest, corpus_root: Path
) -> Dict[str, Score]:
    """Return deep copies of canonical normalized scores keyed by scenario name."""

    include_stress = bool(request.config.getoption("--stress", default=False))
    catalog = _corpus_catalog(corpus_root, include_stress)
    return {
        key: deepcopy(_normalized_corpus_score_cached(path))
        for key, path in catalog.items()
    }


@pytest.fixture()
def raw_corpus_scores(
    request: pytest.FixtureRequest, corpus_root: Path
) -> Dict[str, Score]:
    """Return deep copies of the raw corpus scores (pre-normalization)."""

    include_stress = bool(request.config.getoption("--stress", default=False))
    catalog = _corpus_catalog(corpus_root, include_stress)
    return {
        key: deepcopy(_raw_corpus_score_cached(path)) for key, path in catalog.items()
    }
