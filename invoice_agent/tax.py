"""
GST-style tax slabs for the invoice.

The group-of-3 add-on asks for multiple tax slabs. We use four Indian GST
rates and classify each item from its name. The agent does not invent a rate;
it reads the slab this module assigned.
"""

from __future__ import annotations

# percent, not fraction — matches compute_total(tax_percent)
SLABS: dict[str, float] = {
    "nil": 0.0,
    "essential": 5.0,
    "standard_lower": 12.0,
    "standard": 18.0,
    "luxury": 28.0,
}

# Keywords are matched as whole-word substrings of the lowercased item name.
# First matching category wins, so keep more specific words above generic ones.
_CATEGORY_KEYWORDS: list[tuple[str, str]] = [
    # luxury / 28%
    ("air conditioner", "luxury"),
    ("refrigerator", "luxury"),
    ("television", "luxury"),
    ("perfume", "luxury"),
    ("car", "luxury"),
    ("ac", "luxury"),
    ("tv", "luxury"),
    ("phone", "luxury"),
    ("laptop", "luxury"),
    # essential / 5%
    ("textbook", "essential"),
    ("notebook", "essential"),  # stationery often 18%; books/education-ish at 5% for demo
    ("book", "essential"),
    ("medicine", "essential"),
    ("rice", "essential"),
    ("milk", "essential"),
    ("bread", "essential"),
    ("wheat", "essential"),
    # standard lower / 12%
    ("biscuit", "standard_lower"),
    ("juice", "standard_lower"),
    ("computer", "standard_lower"),
    # standard / 18% — clothing, bags, pens, default
    ("backpack", "standard"),
    ("bag", "standard"),
    ("pen", "standard"),
    ("shirt", "standard"),
    ("headphones", "standard"),
]


def classify_item(name: str) -> tuple[str, float]:
    """
    Return (category, slab_percent) for an item name.

    Unknown items fall through to the 18% standard slab so the invoice still
    has a legal-looking rate instead of silently charging 0%.
    """
    lowered = " ".join(name.lower().split())
    for keyword, category in _CATEGORY_KEYWORDS:
        if _contains_word(lowered, keyword):
            return category, SLABS[category]
    return "standard", SLABS["standard"]


def _contains_word(text: str, keyword: str) -> bool:
    """True if keyword appears as a whole phrase inside text (plurals ok)."""
    if " " in keyword:
        return keyword in text or (keyword + "s") in text
    tokens = set(text.replace("-", " ").split())
    if keyword in tokens:
        return True
    # Accept simple English plurals: biscuit↔biscuits, pen↔pens
    return any(
        token == keyword
        or token == keyword + "s"
        or (token.endswith("s") and token[:-1] == keyword)
        or (keyword.endswith("s") and keyword[:-1] == token)
        for token in tokens
    )


def slab_table() -> str:
    """Human-readable slab list for the system prompt and the README."""
    lines = ["GST slabs used by this agent:"]
    for category, percent in SLABS.items():
        lines.append(f"  {category:16} {percent:5.1f}%")
    return "\n".join(lines)
