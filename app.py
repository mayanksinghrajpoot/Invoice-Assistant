"""
Simple browser UI for the T8 Invoice Assistant.

Run from the InvoiceAssistant folder:
    streamlit run app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from invoice_agent.agent import InvoiceAgent
from invoice_agent.discount import describe_rules
from invoice_agent.lanes import LaneError, describe, get_client, get_model, lane_is_configured
from invoice_agent.tax import slab_table

EXAMPLES = [
    (
        "Small bill + 18% tax",
        "Add 2 notebooks at 80 and 1 pen at 20, then total with 18% tax",
    ),
    (
        "Discount + GST slabs",
        "Invoice: 3 textbooks at 450 each and 1 backpack at 800. Print formatted invoice with tax slabs",
    ),
    (
        "Add a TV (memory)",
        "Also add a television at 42000",
    ),
    (
        "Print invoice (memory)",
        "Print the formatted invoice with tax slabs",
    ),
]


def _build_agent() -> tuple[InvoiceAgent, str]:
    client = None
    model = None
    mode = "offline"
    if lane_is_configured():
        try:
            client = get_client()
            model = get_model()
            mode = "llm"
        except LaneError:
            client, model, mode = None, None, "offline"
    return InvoiceAgent(client=client, model=model, verbose=False), mode


def _ensure_state() -> None:
    if "agent" not in st.session_state:
        agent, mode = _build_agent()
        st.session_state.agent = agent
        st.session_state.mode = mode
        st.session_state.chat = []
        st.session_state.last_trace = []
        st.session_state.last_error = None


def _render_message(role: str, content: str) -> None:
    with st.chat_message(role):
        # Keep invoice tables monospace so columns stay aligned.
        if role == "assistant" and ("Grand total" in content or "----" in content):
            st.code(content, language=None)
        else:
            st.markdown(content)


st.set_page_config(page_title="Invoice Assistant", page_icon=None, layout="wide")
_ensure_state()

st.title("Simple Invoice Assistant")
st.caption("CSE476 T8 — agent with tools, discount decision, GST slabs, and session memory")

with st.sidebar:
    st.subheader("Session")
    st.write(f"Mode: **{st.session_state.mode}**")
    try:
        if st.session_state.mode == "llm":
            st.write(describe())
        else:
            st.info("No usable API key — offline planner (same tools).")
    except Exception:
        st.info("Offline planner.")

    mem = st.session_state.agent.memory.snapshot()
    st.metric("Items", mem["item_count"])
    st.metric("Subtotal (Rs)", f"{mem['subtotal']:.2f}")

    if mem["items"]:
        st.markdown("**Line items**")
        for item in mem["items"]:
            st.write(
                f"- {item['name']} × {item['qty']} @ Rs {item['price']} "
                f"(GST {item['slab_percent']:g}%)"
            )

    if st.button("Clear invoice / new chat", use_container_width=True):
        st.session_state.agent.reset()
        st.session_state.chat = []
        st.session_state.last_trace = []
        st.session_state.last_error = None
        st.rerun()

    with st.expander("Discount rules"):
        st.text(describe_rules())
    with st.expander("GST slabs"):
        st.text(slab_table())

    st.markdown("**Try these**")
    for i, (label, prompt) in enumerate(EXAMPLES):
        if st.button(label, key=f"example_{i}", use_container_width=True):
            st.session_state.pending = prompt
            st.rerun()

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Chat")
    for turn in st.session_state.chat:
        _render_message(turn["role"], turn["content"])

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    prompt = st.chat_input("Add items, ask for total, or print the invoice…")
    if "pending" in st.session_state:
        prompt = st.session_state.pop("pending")

    if prompt:
        st.session_state.chat.append({"role": "user", "content": prompt})
        st.session_state.last_error = None
        try:
            with st.spinner("Agent running plan–act loop…"):
                result = st.session_state.agent.run(prompt)
            st.session_state.last_trace = list(result.trace)
            answer = result.answer or "(no answer)"
            if result.stopped_because == "budget":
                answer = f"{answer}\n\n_(stopped: step budget)_"
            st.session_state.chat.append({"role": "assistant", "content": answer})
        except Exception as exc:  # noqa: BLE001
            msg = f"Agent failed: {exc}"
            st.session_state.last_error = msg
            st.session_state.chat.append(
                {
                    "role": "assistant",
                    "content": (
                        "Something went wrong talking to the model or running a tool. "
                        f"Details: `{exc}`. Try Clear, or switch to offline by clearing "
                        "GROQ_API_KEY in `.env`."
                    ),
                }
            )
        st.rerun()

with right:
    st.subheader("Tool trace")
    st.caption("Proof it is an agent: real tool calls, not just chat text.")
    trace = st.session_state.last_trace
    if not trace:
        st.write("Run a goal to see steps here.")
    else:
        for step in trace:
            with st.expander(f"Step {step.step}: `{step.tool}`", expanded=True):
                st.code(json.dumps(step.args, indent=2), language="json")
                try:
                    pretty = json.dumps(json.loads(step.observation), indent=2)
                    st.code(pretty, language="json")
                except Exception:
                    st.code(step.observation)
