"""
Discount thresholds.

The assignment says the agent must decide whether a discount threshold is met
and apply it. This module is the rulebook. check_discount reads it; the agent
loop is what chooses to call that tool before computing a total.
"""

from __future__ import annotations

# (minimum subtotal, discount rate as a fraction). Highest threshold first.
# Rs 10,000 → 15%, Rs 5,000 → 10%, Rs 1,000 → 5%, otherwise nothing.
THRESHOLDS: list[tuple[float, float]] = [
    (10_000.0, 0.15),
    (5_000.0, 0.10),
    (1_000.0, 0.05),
]


def lookup(subtotal: float) -> tuple[bool, float, float]:
    """
    Return (eligible, rate, threshold_that_fired).

    rate is a fraction (0.05 = 5%). threshold_that_fired is 0 when ineligible.
    """
    if subtotal < 0:
        return False, 0.0, 0.0
    for minimum, rate in THRESHOLDS:
        if subtotal >= minimum:
            return True, rate, minimum
    return False, 0.0, 0.0


def describe_rules() -> str:
    lines = ["Discount is decided on subtotal, before tax:"]
    for minimum, rate in THRESHOLDS:
        lines.append(f"  subtotal >= Rs {minimum:,.0f}  →  {rate * 100:.0f}% off")
    lines.append("  otherwise                 →  no discount")
    return "\n".join(lines)
