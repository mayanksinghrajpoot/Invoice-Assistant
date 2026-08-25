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

from invoice_agent.memory import LineItem, SessionMemory
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


REGISTRY: dict[str, Callable[..., str]] = {
    "add_item": add_item,
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
]
