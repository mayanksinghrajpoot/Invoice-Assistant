"""
Human-readable explanations of the plan–act loop.

The UI and the GUIDE both use this so a tool JSON dump is never the only
thing a person sees. Every step answers: what happened, and why.
"""

from __future__ import annotations

import json
from typing import Any

TOOL_WHY: dict[str, str] = {
    "add_item": (
        "Store this product on the session invoice. Later totals and discounts "
        "must read remembered line items, not numbers the model invented."
    ),
    "check_discount": (
        "Decide whether the current subtotal meets a discount threshold. "
        "This is the agentic step — the agent checks a rule instead of guessing."
    ),
    "compute_total": (
        "Apply the decided discount, then tax the remainder at one rate, "
        "and return subtotal / tax / grand total."
    ),
    "format_invoice": (
        "Print a formatted invoice. Each line keeps its own GST slab "
        "(5 / 12 / 18 / 28%) after the discount is spread across items."
    ),
}

STAGE_LABELS = {
    "receive": "Receive goal",
    "think": "Think / plan",
    "act": "Act — call a tool",
    "observe": "Observe result",
    "remember": "Update memory",
    "answer": "Answer the user",
}


def why_tool(name: str) -> str:
    return TOOL_WHY.get(name, f"Call the registered tool `{name}`.")


def summarise_observation(tool: str, observation: str) -> str:
    """One-line English summary of a tool JSON result."""
    try:
        data = json.loads(observation)
    except json.JSONDecodeError:
        return observation[:180]

    if not data.get("ok", True):
        return f"Failed: {data.get('error', observation[:160])}"

    if tool == "add_item":
        added = data.get("added") or {}
        return (
            f"Remembered {added.get('qty')} × {added.get('name')} at "
            f"Rs {added.get('price')} (GST {added.get('slab_percent')}%). "
            f"Running subtotal Rs {data.get('subtotal')}."
        )
    if tool == "check_discount":
        return str(data.get("reason") or observation[:180])
    if tool == "compute_total":
        return (
            f"Subtotal Rs {data.get('subtotal')}, discount Rs {data.get('discount')}, "
            f"tax Rs {data.get('tax')} → grand total Rs {data.get('grand_total')}."
        )
    if tool == "format_invoice":
        slabs = data.get("tax_by_slab") or {}
        slab_txt = ", ".join(f"GST {k} Rs {v}" for k, v in slabs.items()) or "no tax rows"
        return (
            f"Formatted invoice. Discount Rs {data.get('discount')}, "
            f"{slab_txt}. Grand total Rs {data.get('grand_total')}."
        )
    return observation[:180]


def planner_label(mode: str) -> str:
    if mode == "llm":
        return (
            "The language model looked at the goal, the tool list, and session "
            "memory, then requested the next tool. The model never runs the tool itself."
        )
    return (
        "The offline planner parsed item names, prices, and quantities from the "
        "sentence, then walked the same tool list a live model would use."
    )


def build_process(
    goal: str,
    mode: str,
    trace: list[Any],
    stopped_because: str,
    memory_snapshot: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Ordered stages for the UI timeline.

    Shape of each stage:
        kind, title, detail, tool (optional), ok
    """
    stages: list[dict[str, Any]] = [
        {
            "kind": "receive",
            "title": STAGE_LABELS["receive"],
            "detail": goal.strip() or "(empty)",
            "ok": True,
        },
        {
            "kind": "think",
            "title": STAGE_LABELS["think"],
            "detail": planner_label(mode),
            "ok": True,
        },
    ]

    for step in trace:
        name = getattr(step, "tool", "")
        thought = getattr(step, "thought", "") or ""
        if thought:
            stages.append(
                {
                    "kind": "think",
                    "title": f"Thought before step {step.step}",
                    "detail": thought,
                    "ok": True,
                }
            )
        stages.append(
            {
                "kind": "act",
                "title": f"Act — {name}()",
                "detail": why_tool(name),
                "tool": name,
                "args": getattr(step, "args", {}),
                "ok": True,
            }
        )
        summary = getattr(step, "summary", "") or summarise_observation(
            name, getattr(step, "observation", "")
        )
        ok = True
        try:
            payload = json.loads(getattr(step, "observation", "") or "{}")
            if isinstance(payload, dict) and payload.get("ok") is False:
                ok = False
        except json.JSONDecodeError:
            pass
        stages.append(
            {
                "kind": "observe",
                "title": f"Observe — {name} result",
                "detail": summary,
                "tool": name,
                "ok": ok,
            }
        )

    mem_detail = "No items stored yet."
    if memory_snapshot:
        n = memory_snapshot.get("item_count", 0)
        sub = memory_snapshot.get("subtotal", 0)
        mem_detail = (
            f"{n} line item(s) in session memory, subtotal Rs {sub}. "
            "A later turn can print the invoice without adding items again."
        )
    stages.append(
        {
            "kind": "remember",
            "title": STAGE_LABELS["remember"],
            "detail": mem_detail,
            "ok": True,
        }
    )

    stop_note = {
        "goal met": "The loop stopped because the agent had a final answer (no more tool calls).",
        "no progress": "The loop stopped because the same tool call was repeating.",
        "budget": "The loop stopped because it hit the step budget.",
    }.get(stopped_because, f"Stopped: {stopped_because}.")
    stages.append(
        {
            "kind": "answer",
            "title": STAGE_LABELS["answer"],
            "detail": stop_note,
            "ok": stopped_because == "goal met",
        }
    )
    return stages
