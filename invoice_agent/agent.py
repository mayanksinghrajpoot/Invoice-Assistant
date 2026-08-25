"""
Plan-act loop for the invoice agent.

Think (model or offline planner chooses a tool) → Act (this file runs it) →
Observe (result goes back into memory / transcript) → repeat until the goal
is met or the step budget runs out.

This is the same shape as the CSE476 tiny agent: the model never executes a
function. It only requests one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from invoice_agent.discount import describe_rules
from invoice_agent.memory import SessionMemory
from invoice_agent.tax import slab_table
from invoice_agent.tools import REGISTRY, TOOL_SCHEMA, bind_memory

SYSTEM = f"""You are an invoice assistant agent, not a chatbot.

You have tools. You must use them. Never invent line items, subtotals, discounts, or tax figures.

How to work:
1. For every product the user mentions, call add_item(name, price, qty). One call per product.
2. After the items for this request are on the invoice, ALWAYS call check_discount() before any total. That is how you decide whether a discount threshold is met. Do not guess a discount. Never skip this step.
3. If the user asked for a single tax percent, call compute_total(tax_percent).
4. If the user asked for a formatted invoice, mixed GST, or tax slabs, call format_invoice() AFTER check_discount().
5. Then answer using only numbers the tools returned.

Items added earlier in this conversation are still on the invoice. Do not add them again unless the user asks.

{describe_rules()}

{slab_table()}

If a tool returns an error, read it and recover. When you have the final numbers, stop calling tools and answer the user.
"""


@dataclass
class StepTrace:
    step: int
    tool: str
    args: dict[str, Any]
    observation: str

    def display(self) -> str:
        compact = self.observation if len(self.observation) < 240 else self.observation[:237] + "..."
        return f"[step {self.step}] {self.tool}({self.args}) -> {compact}"


@dataclass
class RunResult:
    answer: str
    steps: int
    trace: list[StepTrace] = field(default_factory=list)
    stopped_because: str = "goal met"
    mode: str = "offline"

    def print_trace(self) -> None:
        print(f"mode={self.mode}  steps={self.steps}  stopped={self.stopped_because}")
        for item in self.trace:
            print(item.display())
        print("--- answer ---")
        print(self.answer)


class NoProgress:
    """Stop if the agent repeats the same tool call without new information."""

    def __init__(self, repeat_limit: int = 3) -> None:
        self.repeat_limit = repeat_limit
        self._calls: list[str] = []

    def record(self, name: str, args: dict[str, Any]) -> None:
        self._calls.append(f"{name}({json.dumps(args, sort_keys=True)})")

    def verdict(self) -> str | None:
        if len(self._calls) < self.repeat_limit:
            return None
        recent = self._calls[-self.repeat_limit :]
        if len(set(recent)) == 1:
            return f"Repeated {recent[0]} {self.repeat_limit} times. Stopping."
        return None


def call_tool(name: str, args: dict[str, Any]) -> str:
    """The only place a tool actually runs. Unknown names never execute."""
    if name not in REGISTRY:
        return json.dumps(
            {
                "ok": False,
                "error": f"No tool named '{name}'. Available: {list(REGISTRY)}",
            }
        )
    try:
        return REGISTRY[name](**args)
    except TypeError as exc:
        return json.dumps({"ok": False, "error": f"Wrong arguments for {name}: {exc}"})
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"ok": False, "error": str(exc)})


class InvoiceAgent:
    """
    One conversation, one memory.

    Pass a client+model for Groq/Foundry/Ollama. Omit them to run the offline
    planner, which still calls the same tools so the notebook always has a trace.
    """

    def __init__(
        self,
        client: Any | None = None,
        model: str | None = None,
        max_steps: int = 12,
        verbose: bool = True,
    ) -> None:
        self.client = client
        self.model = model
        self.max_steps = max_steps
        self.verbose = verbose
        self.memory = SessionMemory()
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM}]

    def reset(self) -> None:
        self.memory.reset()
        self.messages = [{"role": "system", "content": SYSTEM}]
        bind_memory(self.memory)

    def run(self, goal: str) -> RunResult:
        """Handle one user goal, keeping memory from earlier turns."""
        bind_memory(self.memory)
        self.memory.record_turn(goal)
        self.messages.append({"role": "user", "content": goal})

        if self.client is None:
            result = self._run_offline(goal)
        else:
            result = self._run_llm()

        if self.verbose:
            result.print_trace()
        return result

    def _run_llm(self) -> RunResult:
        detector = NoProgress()
        trace: list[StepTrace] = []

        for step in range(1, self.max_steps + 1):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=TOOL_SCHEMA,
            )
            message = response.choices[0].message
            dumped = message.model_dump(exclude_none=True)
            self.messages.append(dumped)

            if not message.tool_calls:
                answer = message.content or ""
                return RunResult(
                    answer=answer,
                    steps=step,
                    trace=trace,
                    stopped_because="goal met",
                    mode="llm",
                )

            for call in message.tool_calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                observation = call_tool(name, args)
                trace.append(StepTrace(step=step, tool=name, args=args, observation=observation))
                detector.record(name, args)
                self.messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": observation}
                )

            stuck = detector.verdict()
            if stuck:
                return RunResult(
                    answer=stuck,
                    steps=step,
                    trace=trace,
                    stopped_because="no progress",
                    mode="llm",
                )

        return RunResult(
            answer=f"Stopped after {self.max_steps} steps without a final answer.",
            steps=self.max_steps,
            trace=trace,
            stopped_because="budget",
            mode="llm",
        )

    def _run_offline(self, goal: str) -> RunResult:
        """
        Deterministic planner used when no API key is configured.

        It still goes through call_tool, so the trace is a real plan-act loop,
        not a canned paragraph.
        """
        trace: list[StepTrace] = []
        step = 0

        def act(name: str, args: dict[str, Any]) -> str:
            nonlocal step
            step += 1
            observation = call_tool(name, args)
            trace.append(StepTrace(step=step, tool=name, args=args, observation=observation))
            return observation

        items = _parse_items(goal)
        if items:
            for name, price, qty in items:
                act("add_item", {"name": name, "price": price, "qty": qty})

        if not self.memory.items:
            answer = (
                "I need at least one item with a name, price, and quantity. "
                "Example: 'Add 2 notebooks at 80 and 1 pen at 20, then total with 18% tax'."
            )
            self.messages.append({"role": "assistant", "content": answer})
            return RunResult(
                answer=answer,
                steps=step,
                trace=trace,
                stopped_because="goal met",
                mode="offline",
            )

        if items and not _wants_total(goal):
            answer = (
                f"Added {len(items)} item(s). Running subtotal Rs {self.memory.subtotal():.2f}. "
                "Say when you want the discount check and the invoice."
            )
            self.messages.append({"role": "assistant", "content": answer})
            return RunResult(
                answer=answer,
                steps=step,
                trace=trace,
                stopped_because="goal met",
                mode="offline",
            )

        act("check_discount", {})

        wants_format = _wants_formatted_invoice(goal)
        tax_percent = _parse_tax_percent(goal)
        last = "{}"
        if wants_format or tax_percent is None:
            last = act("format_invoice", {})
        else:
            last = act("compute_total", {"tax_percent": tax_percent})

        payload = json.loads(last)
        answer = _offline_answer(payload, used_format=wants_format or tax_percent is None)
        self.messages.append({"role": "assistant", "content": answer})
        return RunResult(
            answer=answer,
            steps=step,
            trace=trace,
            stopped_because="goal met",
            mode="offline",
        )


def _parse_items(text: str) -> list[tuple[str, float, int]]:
    """Pull (name, price, qty) triples out of a student-style goal sentence."""
    found: list[tuple[str, float, int]] = []
    qty_first = re.compile(
        r"(?P<qty>\d+)\s+(?P<name>[A-Za-z][A-Za-z][A-Za-z\s]*?)\s+"
        r"(?:at|for|@)\s+(?:rs\.?\s*)?(?P<price>\d+(?:\.\d+)?)",
        re.I,
    )
    used: list[tuple[int, int]] = []
    for match in qty_first.finditer(text):
        name = re.sub(r"\s+", " ", match.group("name")).strip(" ,.")
        name = re.sub(r"\beach\b", "", name, flags=re.I).strip()
        found.append((name, float(match.group("price")), int(match.group("qty"))))
        used.append(match.span())

    bare = re.compile(
        r"(?:add|plus|also|and)\s+(?:a|an|one)\s+"
        r"(?P<name>[A-Za-z][A-Za-z\s]*?)\s+"
        r"(?:at|for|@)\s+(?:rs\.?\s*)?(?P<price>\d+(?:\.\d+)?)",
        re.I,
    )
    for match in bare.finditer(text):
        if any(start <= match.start() < end for start, end in used):
            continue
        name = re.sub(r"\s+", " ", match.group("name")).strip(" ,.")
        found.append((name, float(match.group("price")), 1))
    return found


def _parse_tax_percent(text: str) -> float | None:
    match = re.search(
        r"(?P<p>\d+(?:\.\d+)?)\s*%\s*tax|tax(?:\s*(?:of|at|with))?\s*(?P<p2>\d+(?:\.\d+)?)\s*%",
        text,
        re.I,
    )
    if not match:
        return None
    raw = match.group("p") or match.group("p2")
    return float(raw)


def _wants_formatted_invoice(text: str) -> bool:
    lowered = text.lower()
    return any(
        word in lowered
        for word in ("format", "formatted", "slab", "slabs", "print", "summary", "invoice")
    )


def _wants_total(text: str) -> bool:
    lowered = text.lower()
    return any(
        word in lowered
        for word in (
            "total",
            "invoice",
            "format",
            "tax",
            "discount",
            "print",
            "summary",
            "how much",
            "afford",
            "bill",
        )
    )


def _offline_answer(payload: dict[str, Any], used_format: bool) -> str:
    if not payload.get("ok"):
        return payload.get("error") or json.dumps(payload)
    if used_format and payload.get("formatted"):
        return payload["formatted"]
    return (
        f"Subtotal Rs {payload['subtotal']:.2f}. "
        f"Discount Rs {payload['discount']:.2f} ({payload.get('discount_note', '')}). "
        f"Tax Rs {payload['tax']:.2f}. "
        f"Grand total Rs {payload['grand_total']:.2f}."
    )


def run_goal(goal: str, verbose: bool = True) -> RunResult:
    """
    Run one goal on a fresh agent.

    Uses the configured lane when a key is present, otherwise the offline planner.
    """
    client = None
    model = None
    try:
        from invoice_agent.lanes import get_client, get_model, lane_is_configured

        if lane_is_configured():
            client = get_client()
            model = get_model()
    except Exception:  # noqa: BLE001
        client = None
        model = None
    agent = InvoiceAgent(client=client, model=model, verbose=verbose)
    return agent.run(goal)
