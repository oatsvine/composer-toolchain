"""Shared prompt fragments for LLM interactions."""

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

__all__ = ["HUMDRUM_KERN_PRIMER"]

