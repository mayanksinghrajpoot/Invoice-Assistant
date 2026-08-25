"""
Tools the invoice agent is allowed to call.

A tool is an ordinary Python function. The model never executes it; the
plan-act loop does. Names not in REGISTRY cannot run.

Required by T8:
    add_item(name, price, qty)
    compute_total(tax_percent)

Needed to make the discount decision visible, and for the group add-on:
    check_discount()
    format_invoice()
"""

from __future__ import annotations

import json
from typing import Any, Callable

from invoice_agent.discount import lookup
from invoice_agent.memory import DiscountDecision, LineItem, SessionMemory
from invoice_agent.tax import classify_item

# Bound by InvoiceAgent before each run so tools share one conversation.
MEMORY = SessionMemory()


def bind_memory(memory: SessionMemory) -> None:
    """Point every tool at this conversation's memory."""
    global MEMORY
    MEMORY = memory


def _as_float(value: Any) -> float:
    """LLM tool arguments often arrive as strings. Coerce, don't crash."""
    return float(value)


def _as_int(value: Any) -> int:
    return int(float(value))


def add_item(name: str, price: float, qty: int) -> str:
    """Add one line item to the invoice. Remembers it for later turns."""
    name = str(name).strip()
    if not name:
        return json.dumps({"ok": False, "error": "Item name is empty."})
    try:
        price_f = _as_float(price)
        qty_i = _as_int(qty)
    except (TypeError, ValueError):
        return json.dumps(
            {"ok": False, "error": f"price and qty must be numbers, got {price!r} and {qty!r}."}
        )
    if price_f < 0:
        return json.dumps({"ok": False, "error": "Price cannot be negative."})
    if qty_i <= 0:
        return json.dumps({"ok": False, "error": "Quantity must be at least 1."})

    category, slab_percent = classify_item(name)
    item = LineItem(
        name=name,
        price=round(price_f, 2),
        qty=qty_i,
        category=category,
        slab_percent=slab_percent,
    )
    MEMORY.add_item(item)
    return json.dumps(
        {
            "ok": True,
            "added": item.as_dict(),
            "item_count": len(MEMORY.items),
            "subtotal": MEMORY.subtotal(),
            "note": "Discount decision was cleared because the invoice changed. Call check_discount again before computing a total.",
        }
    )


def check_discount() -> str:
    """
    Decide whether the current subtotal meets a discount threshold.

    Does not apply the discount. compute_total / format_invoice apply it only
    after this tool has recorded a decision on the current items.
    """
    if not MEMORY.items:
        return json.dumps(
            {
                "ok": False,
                "error": "No items on the invoice yet. Call add_item first.",
            }
        )

    subtotal = MEMORY.subtotal()
    eligible, rate, threshold = lookup(subtotal)
    if eligible:
        reason = (
            f"Subtotal Rs {subtotal:.2f} meets the Rs {threshold:,.0f} threshold, "
            f"so {rate * 100:.0f}% off applies."
        )
    else:
        reason = (
            f"Subtotal Rs {subtotal:.2f} is below Rs 1,000, so no discount applies."
        )

    MEMORY.discount_decision = DiscountDecision(
        eligible=eligible,
        rate=rate,
        threshold=threshold,
        subtotal=subtotal,
        reason=reason,
    )
    return json.dumps(
        {
            "ok": True,
            "eligible": eligible,
            "rate_percent": round(rate * 100, 2),
            "threshold": threshold,
            "subtotal": subtotal,
            "reason": reason,
            "next": "Call compute_total or format_invoice. They will apply this decision.",
        }
    )


def _applied_discount() -> tuple[float, str]:
    """Return (discount_amount, note) using the last check_discount decision."""
    subtotal = MEMORY.subtotal()
    decision = MEMORY.discount_decision
    if decision is None:
        return 0.0, (
            "No check_discount call on the current items. "
            "Computed without a discount. Call check_discount first next time."
        )
    if not decision.eligible:
        return 0.0, decision.reason
    amount = round(subtotal * decision.rate, 2)
    return amount, decision.reason


def compute_total(tax_percent: float) -> str:
    """
    Tax the current invoice at one rate, after any decided discount.

    This is the T8 required tool. For mixed GST slabs use format_invoice instead.
    """
    if not MEMORY.items:
        return json.dumps(
            {"ok": False, "error": "No items on the invoice yet. Call add_item first."}
        )
    try:
        tax_p = _as_float(tax_percent)
    except (TypeError, ValueError):
        return json.dumps(
            {"ok": False, "error": f"tax_percent must be a number, got {tax_percent!r}."}
        )
    if tax_p < 0:
        return json.dumps({"ok": False, "error": "tax_percent cannot be negative."})

    subtotal = MEMORY.subtotal()
    discount_amount, discount_note = _applied_discount()
    taxable = round(subtotal - discount_amount, 2)
    tax_amount = round(taxable * (tax_p / 100.0), 2)
    grand_total = round(taxable + tax_amount, 2)

    return json.dumps(
        {
            "ok": True,
            "item_count": len(MEMORY.items),
            "subtotal": subtotal,
            "discount": discount_amount,
            "discount_note": discount_note,
            "taxable": taxable,
            "tax_percent": tax_p,
            "tax": tax_amount,
            "grand_total": grand_total,
        }
    )


REGISTRY: dict[str, Callable[..., str]] = {
    "add_item": add_item,
    "check_discount": check_discount,
    "compute_total": compute_total,
}

TOOL_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "add_item",
            "description": (
                "Add one product to the current invoice. Call once per distinct "
                "item. The item is remembered for the rest of this conversation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Product name, for example 'textbook' or 'backpack'.",
                    },
                    "price": {
                        "type": "number",
                        "description": "Unit price in rupees.",
                    },
                    "qty": {
                        "type": "integer",
                        "description": "How many units to add. Must be at least 1.",
                    },
                },
                "required": ["name", "price", "qty"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_discount",
            "description": (
                "Look at the current subtotal and decide whether a discount "
                "threshold is met. Call this after adding items and before "
                "compute_total or format_invoice. Does not apply the discount "
                "itself."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_total",
            "description": (
                "Compute the invoice total at a single tax rate after any "
                "discount already decided by check_discount. Use this when the "
                "user asks for one tax percent. For mixed GST slabs, use "
                "format_invoice instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tax_percent": {
                        "type": "number",
                        "description": "Tax rate to apply, for example 18 for 18%.",
                    }
                },
                "required": ["tax_percent"],
            },
        },
    },
]
