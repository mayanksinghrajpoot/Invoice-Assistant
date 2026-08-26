"""Unit tests for T8 tools, discount decision, slabs, and the offline agent."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from invoice_agent.agent import InvoiceAgent, call_tool
from invoice_agent.discount import lookup
from invoice_agent.memory import SessionMemory
from invoice_agent.tax import classify_item
from invoice_agent.tools import bind_memory
from invoice_agent import tools


@pytest.fixture
def memory() -> SessionMemory:
    mem = SessionMemory()
    bind_memory(mem)
    return mem


def _add(name: str, price: float, qty: int) -> dict:
    return json.loads(tools.add_item(name, price, qty))


def test_add_item_remembers_line(memory: SessionMemory) -> None:
    result = _add("pen", 20, 2)
    assert result["ok"] is True
    assert result["subtotal"] == 40
    assert len(memory.items) == 1
    assert memory.items[0].qty == 2


def test_add_item_coerces_string_numbers(memory: SessionMemory) -> None:
    result = json.loads(tools.add_item("pen", "20.5", "3"))
    assert result["ok"] is True
    assert result["added"]["price"] == 20.5
    assert result["added"]["qty"] == 3


def test_classify_slabs() -> None:
    assert classify_item("textbook")[1] == 5.0
    assert classify_item("biscuits")[1] == 12.0
    assert classify_item("backpack")[1] == 18.0
    assert classify_item("television")[1] == 28.0
    assert classify_item("mystery gadget")[1] == 18.0


def test_discount_thresholds() -> None:
    assert lookup(999) == (False, 0.0, 0.0)
    assert lookup(1000)[1] == 0.05
    assert lookup(5000)[1] == 0.10
    assert lookup(10000)[1] == 0.15


def test_check_discount_below_threshold(memory: SessionMemory) -> None:
    _add("pen", 20, 2)
    decision = json.loads(tools.check_discount())
    assert decision["eligible"] is False
    assert memory.discount_decision is not None
    total = json.loads(tools.compute_total(18))
    assert total["discount"] == 0
    assert total["tax"] == round(40 * 0.18, 2)
    assert total["grand_total"] == round(40 + 40 * 0.18, 2)


def test_check_discount_applies_five_percent(memory: SessionMemory) -> None:
    _add("backpack", 800, 1)
    _add("textbook", 450, 1)
    decision = json.loads(tools.check_discount())
    assert decision["eligible"] is True
    assert decision["rate_percent"] == 5
    total = json.loads(tools.compute_total(18))
    assert total["subtotal"] == 1250
    assert total["discount"] == 62.5
    assert total["taxable"] == 1187.5
    assert total["tax"] == round(1187.5 * 0.18, 2)


def test_adding_item_clears_stale_discount(memory: SessionMemory) -> None:
    _add("backpack", 1200, 1)
    tools.check_discount()
    assert memory.discount_decision is not None
    _add("pen", 20, 1)
    assert memory.discount_decision is None


def test_format_invoice_multiple_slabs(memory: SessionMemory) -> None:
    _add("textbook", 450, 2)  # 5%
    _add("biscuits", 100, 1)  # 12%
    _add("backpack", 800, 1)  # 18%
    _add("television", 20000, 1)  # 28%
    tools.check_discount()
    payload = json.loads(tools.format_invoice())
    assert payload["ok"] is True
    assert "5%" in payload["tax_by_slab"]
    assert "12%" in payload["tax_by_slab"]
    assert "18%" in payload["tax_by_slab"]
    assert "28%" in payload["tax_by_slab"]
    assert "Grand total" in payload["formatted"]
    # 2*450 + 100 + 800 + 20000 = 21800 → 15% discount
    assert payload["discount"] == round(21800 * 0.15, 2)


def test_format_invoice_auto_decides_discount_if_skipped(memory: SessionMemory) -> None:
    """Even if the LLM forgets check_discount, totals must still decide."""
    _add("backpack", 1200, 1)
    assert memory.discount_decision is None
    payload = json.loads(tools.format_invoice())
    assert memory.discount_decision is not None
    assert payload["discount"] == 60.0
    assert payload["ok"] is True


def test_unknown_tool_is_rejected() -> None:
    result = json.loads(call_tool("delete_database", {}))
    assert result["ok"] is False
    assert "No tool named" in result["error"]


def test_offline_agent_goal_below_threshold() -> None:
    agent = InvoiceAgent(verbose=False)
    result = agent.run(
        "Add 2 notebooks at Rs 80 and 1 pen at Rs 20, then give me the total with 18% tax."
    )
    assert result.stopped_because == "goal met"
    tools_used = [step.tool for step in result.trace]
    assert tools_used.count("add_item") == 2
    assert "check_discount" in tools_used
    assert "compute_total" in tools_used
    assert any(
        step.args.get("tax_percent") == 18
        for step in result.trace
        if step.tool == "compute_total"
    )
    last = json.loads(result.trace[-1].observation)
    assert last["discount"] == 0
    assert last["subtotal"] == 180


def test_offline_agent_formatted_slabs() -> None:
    agent = InvoiceAgent(verbose=False)
    result = agent.run(
        "Invoice: 3 textbooks at 450 each and 1 backpack at 800. "
        "Apply tax slabs and print the formatted invoice."
    )
    tools_used = [step.tool for step in result.trace]
    assert "check_discount" in tools_used
    assert "format_invoice" in tools_used
    last = json.loads(result.trace[-1].observation)
    assert last["subtotal"] == 2150
    assert last["discount"] == 107.5  # 5% of 2150
    assert "5%" in last["tax_by_slab"]
    assert "18%" in last["tax_by_slab"]


def test_memory_survives_later_turn() -> None:
    agent = InvoiceAgent(verbose=False)
    first = agent.run("Add 2 pens at 40 and 1 bag at 200")
    assert "add_item" in [step.tool for step in first.trace]
    second = agent.run("Also add a television at 42000")
    assert agent.memory.subtotal() == 2 * 40 + 200 + 42000  # 42280
    third = agent.run("Print the formatted invoice with tax slabs")
    tools_used = [step.tool for step in third.trace]
    assert "add_item" not in tools_used
    assert "check_discount" in tools_used
    assert "format_invoice" in tools_used
    last = json.loads(third.trace[-1].observation)
    assert last["subtotal"] == 42280
    assert last["discount"] == round(42280 * 0.15, 2)


def test_trace_explains_why_each_tool_ran() -> None:
    from invoice_agent.explain import build_process

    agent = InvoiceAgent(verbose=False)
    result = agent.run(
        "Add 2 notebooks at Rs 80 and 1 pen at Rs 20, then give me the total with 18% tax."
    )
    assert result.trace
    assert all(step.why for step in result.trace)
    assert all(step.summary for step in result.trace)
    process = build_process(
        goal="demo",
        mode=result.mode,
        trace=result.trace,
        stopped_because=result.stopped_because,
        memory_snapshot=agent.memory.snapshot(),
    )
    kinds = {stage["kind"] for stage in process}
    assert {"receive", "think", "act", "observe", "remember", "answer"} <= kinds


def test_excel_csv_import() -> None:
    from invoice_agent.excel_import import from_csv_bytes, parse_rows

    csv = b"item,price,qty\npen,40,2\nbag,200,1\n"
    items = from_csv_bytes(csv)
    assert items == [("pen", 40.0, 2), ("bag", 200.0, 1)]
    rows = parse_rows(
        ["name", "rate", "quantity"],
        [["notebook", "80", "2"]],
    )
    assert rows == [("notebook", 80.0, 2)]


def test_json_import() -> None:
    from invoice_agent.excel_import import from_json_bytes, from_upload

    raw = b'[{"name": "pen", "price": 40, "qty": 2}, {"item": "bag", "rate": 200}]'
    assert from_json_bytes(raw) == [("pen", 40.0, 2), ("bag", 200.0, 1)]
    wrapped = b'{"items": [{"product": "textbook", "price": 450, "quantity": 3}]}'
    assert from_json_bytes(wrapped) == [("textbook", 450.0, 3)]
    assert from_upload("cart.json", raw)[0][0] == "pen"

