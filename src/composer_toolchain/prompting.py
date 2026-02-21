"""Shared prompt fragments for LLM interactions."""

from __future__ import annotations

from typing import Iterable

HUMDRUM_KERN_PRIMER = (
    "Humdrum **kern essentials (official reference):\n"
    "- Start each spine with an exclusive interpretation like `**kern`; each record\n"
    "  keeps tokens tab-aligned per spine.\n"
    "- Comments follow the `!!!` (reference), `!!` (global), `!` (local) convention;\n"
    "  global annotations such as `!! SEGMENT|…` or `!! SKETCH|…` must appear on\n"
    "  their own lines before the target barline so every spine inherits them.\n"
    "- Barlines are records beginning with `=` (e.g., `=12`); double barlines use\n"
    "  `==` and may include pause suffixes like `;`.\n"
    "- Pitch spelling relies on case and letter repetition for octaves (C/c/cc),\n"
    "  `#` for sharps, `-` for flats; durations use reciprocal values (`4` quarter,\n"
    "  `8.` dotted eighth) and rests use `r`.\n"
    "- Editorial flags mirror the reference: `x/X` for interpretations/interventions,\n"
    "  `y/Y` for invisible/sic symbols, and `?` for footnotes—emit them only when the\n"
    "  musical reasoning demands it."
)

TECHNIQUE_PROMPTS = {
    "variations": (
        "Theme-and-variations brief:\n"
        "- Present the source theme with its exact contour and cadence plan before "
        "altering it so listeners can track every transformation.\n"
        "- Each variation should spotlight a single device—displacement to another "
        "voice, ornamental filigree, meter or mode shifts, or contrapuntal treatments "
        "such as canon/fugue—while respecting the harmonic skeleton and phrase spans "
        "outlined in the theme.\n"
        "- Sequence the variations with a narrated arc (e.g., textural intensification "
        "toward a climactic penultimate variation followed by a reflective coda) so "
        "the set feels curated rather than random, à la Elgar’s Enigma portraits.\n"
        "- If you restate the theme at the end, make it feel earned—either as a "
        "recapitulation or as a triumphant transformation."
    ),
    "ostinato": (
        "Ostinato / basso-continuo brief:\n"
        "- Establish a bass or inner-voice pattern that spans a full harmonic loop "
        "before layering other material; the ostinato must recur verbatim (or with "
        "only registral color) every phrase so listeners can latch onto it.\n"
        "- Keep cadences clear: authentic cadences for stand-alone statements or "
        "half-cadence handoffs when dovetailing into the next variation, mirroring "
        "passacaglia and chaconne practice.\n"
        "- Rotate which voice carries the ostinato in select passages, but never lose "
        "the sense of ground—if it moves to an upper voice, articulate it with "
        "registration, articulation, or dynamics.\n"
        "- Exploit textural waves: start with transparent voicings, thicken toward "
        "the apex, then thin again before the coda so the repeating pattern feels "
        "like a narrative backbone rather than a loop."
    ),
}


def technique_guidance(names: Iterable[str]) -> str:
    """Return concatenated technique guidance strings for the chosen names."""

    blocks: list[str] = []
    for name in names:
        fragment = TECHNIQUE_PROMPTS.get(name)
        if fragment:
            blocks.append(fragment)
    return "\n\n".join(blocks)


__all__ = [
    "HUMDRUM_KERN_PRIMER",
    "TECHNIQUE_PROMPTS",
    "technique_guidance",
]
