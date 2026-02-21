"""Tests for OpenAI-driven segmentation workflow."""

from __future__ import annotations

import os

import pytest
from music21.humdrum.spineParser import GlobalComment

from composer_toolchain.core import ScoreSpec
from composer_toolchain.segmentation import (
    DEFAULT_SEGMENTATION_MODEL,
    SegmentationContext,
    analyze_segmentation,
)
from composer_toolchain.score import kern_to_score


_SEGMENTATION_FIXTURE_KERN_RAW = """!!!COM: Bach, Johann Sebastian
!!!OTL: Chorale 14. Als vierzig Tag' nach Ostern war'n
!!!OPR: Joh. Seb. Bachs vierstimmige Choralgesänge
!!!YEC: Copyright (c) 1994, 2000 Center for Computer Assisted Research in the Humanities
**kern\t**text
*part1\t*part1
*staff1\t*staff1
*I"[SOPRANO]\t*
*I'S\t*
*clefG2\t*
*k[f#]\t*
*e:\t*
!!LO:TX:omd:t=
*M3/4\t*
=1\t=1
4e\tAls
4e\tvier-
4e\t-zig
=2\t=2
2b\tTag'
4b\tnach
=3\t=3
4a\tO-
4b\t.
8gL\t.
8aJ\t.
=4\t=4
4b\t.
4a\t.
4a\t-stern
=5\t=5
2b;<\twar'n
4g\tund
=6\t=6
4f#\tChri-
4g\t.
4a\t-stus
=7\t=7
4b\twollt'
4a\t.
4g\tgen
=8\t=8
2g\tHim-
4f#\t-mel
=9\t=9
2g;<\tfahr'n,
4a\tb'schied
=10\t=10
2b\ter
4b\tsein'
=11\t=11
2a\tJün-
4a\t-ger
=12\t=12
2g\tauf
4g\tein
=13\t=13
2.f#;<\tBerg,
=14\t=14
4e\tauf
4f#\t.
4d#X\tein
=15\t=15
2e;<\tBerg,
4e\tvoll-
=16\t=16
2a\t-en-
4a\tdet
=17\t=17
2b\tda
4b\tsein
=18\t=18
2g\tAmt
4g\tund
=19\t=19
2.f#;<\tWerk.
=20\t=20
8eL\tHal-
8f#\t.
8g\t.
8aJ\t.
4b\t.
=21\t=21
2b\t-le-
4a\t-lu-
=22\t=22
2.b;<\t-ja!
==\t==
*-\t*-
!!!software: music21 v.9.9.1
!!!software: converter21
!!!XEN: Four-part Chorales
!!!SCT: BWV 266
!!!SCA: Thematisch-systematisches Verzeichnis der musikalischen Werke Johann Sebastian Bach: Bach-Werke-Verzeichnis (Schmieder)
!!!YOR: Bach Gesellschaft Edition xxxix
!!!ENC: Steven Rasmussen
!!!CDT: 1685/03-1750/07/28
!!!OCY: Deutschland
!!!YEM: Rights to all derivative editions reserved
!!!YEN: United States of America
!!!RWG: Key is interpreted using the Humdrum key tool.
!!!URL-pdf: https://s9.imslp.org/files/imglnks/usimg/6/6b/IMSLP24489-PMLP09471-CCARH_Bach_Chorales.pdf#page=8 Musedata edition
!!!URL-pdf: https://s9.imslp.org/files/imglnks/usimg/7/7a/IMSLP04155-Bach_-_BGA_253-438.pdf#page=8 \tBach-Gesellschaft Ausgabe, Band 39 (pp.175-276) Leipzig: Breitkopf und Härtel, 1892. Plate B.W. XXXIX.
!!!END: 1994/05/06
!!!EMD: Converted from MuseData to Humdrum Dec 19, 2000, by Andreas Kornstaedt using muse2kern.
!!!EED: Craig Stuart Sapp
!!!EEV: 2024/03/31
!!!filter-keyboard: extract -i kern | satb2gs
!!!title: @{OTL} (@{SCT})
!!!RDF**kern: > = above
!!!RDF**kern: < = below
"""

SEGMENTATION_FIXTURE_KERN = _SEGMENTATION_FIXTURE_KERN_RAW.replace("\\t", "\t")


@pytest.mark.skipif("OPENAI_API_KEY" not in os.environ, reason="OPENAI_API_KEY required")
def test_segmentation_live():
    score = kern_to_score(SEGMENTATION_FIXTURE_KERN)
    spec = ScoreSpec.build(score)
    context = SegmentationContext(
        spec=spec,
        kern_text=SEGMENTATION_FIXTURE_KERN,
        excerpt_label="bwv266_soprano",
    )
    model = os.environ.get("SEGMENTATION_TEST_MODEL", DEFAULT_SEGMENTATION_MODEL)
    result = analyze_segmentation(context=context, model=model)

    assert result.segments, "LLM must return at least one labeled span"

    annotated_score = kern_to_score(result.annotated_kern)
    comments = [
        (gc.comment or "").strip()
        for gc in annotated_score.recurse().getElementsByClass(GlobalComment)
        if (gc.comment or "").startswith("SEGMENT|")
    ]
    assert comments, "Annotated **kern must contain SEGMENT global comments"

    suffixes = {segment.suffix for segment in result.segments}
    for suffix in suffixes:
        assert any(comment.split("|", 2)[1] == suffix for comment in comments)

    assert result.overview.strip(), "Overview should summarize the segmentation"
