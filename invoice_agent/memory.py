"""
Session memory for the invoice agent.

The assignment requires the agent to remember every line item across turns in
the same conversation. This module is that memory: a plain Python object the
tools write to and later tools read from. The LLM never mutates it directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LineItem:
    """One row on the invoice."""

    name: str
    price: float
    qty: int
    category: str
    slab_percent: float

    @property
    def amount(self) -> float:
        return round(self.price * self.qty, 2)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "price": self.price,
            "qty": self.qty,
            "amount": self.amount,
            "category": self.category,
            "slab_percent": self.slab_percent,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LineItem:
        return cls(
            name=str(data["name"]),
            price=float(data["price"]),
            qty=int(data["qty"]),
            category=str(data.get("category") or "standard"),
            slab_percent=float(data.get("slab_percent") or 18.0),
        )


@dataclass
class DiscountDecision:
    """What check_discount last concluded. compute_total reads this."""

    eligible: bool
    rate: float
    threshold: float
    subtotal: float
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "eligible": self.eligible,
            "rate": self.rate,
            "threshold": self.threshold,
            "subtotal": self.subtotal,
            "reason": self.reason,
        }


@dataclass
class SessionMemory:
    """
    Everything the agent must remember for this conversation.

    items                 line items added so far
    discount_decision     last check_discount result, or None if not checked yet
    turns                 user goals in this session (conversation memory)
    """

    items: list[LineItem] = field(default_factory=list)
    discount_decision: DiscountDecision | None = None
    turns: list[str] = field(default_factory=list)

    def add_item(self, item: LineItem) -> LineItem:
        self.items.append(item)
        self.discount_decision = None  # stale: totals changed
        return item

    def subtotal(self) -> float:
        return round(sum(item.amount for item in self.items), 2)

    def record_turn(self, goal: str) -> None:
        self.turns.append(goal)

    def restore_items(self, rows: list[dict[str, Any]]) -> None:
        """Rebuild the cart from a persisted snapshot without clearing turns."""
        self.items = [LineItem.from_dict(row) for row in rows]
        self.discount_decision = None

    def reset(self) -> None:
        self.items.clear()
        self.discount_decision = None
        self.turns.clear()

    def snapshot(self) -> dict[str, Any]:
        return {
            "item_count": len(self.items),
            "items": [item.as_dict() for item in self.items],
            "subtotal": self.subtotal(),
            "discount_decision": (
                self.discount_decision.as_dict() if self.discount_decision else None
            ),
            "turns": list(self.turns),
        }
