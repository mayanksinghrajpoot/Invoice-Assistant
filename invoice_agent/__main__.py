"""python -m invoice_agent "Add 2 pens at 40, then total with 18% tax" """

from __future__ import annotations

import sys

from invoice_agent.agent import InvoiceAgent
from invoice_agent.lanes import LaneError, get_client, get_model, lane_is_configured


def main() -> int:
    goal = " ".join(sys.argv[1:]).strip()
    if not goal:
        print('Usage: python -m invoice_agent "Add 2 notebooks at 80 and 1 pen at 20, then total with 18% tax"')
        return 2

    client = None
    model = None
    if lane_is_configured():
        try:
            client = get_client()
            model = get_model()
            print(f"Using LLM lane, model={model}")
        except LaneError as exc:
            print(f"Lane not usable ({exc}). Falling back to offline planner.")
    else:
        print("No lane configured. Using offline planner (same tools, no LLM).")

    agent = InvoiceAgent(client=client, model=model, verbose=True)
    agent.run(goal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
