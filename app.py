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


def _build_agent() -> InvoiceAgent:
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


st.set_page_config(page_title="Invoice Assistant", page_icon="🧾", layout="wide")
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
    st.metric("Subtotal (₹)", f"{mem['subtotal']:.2f}")

    if mem["items"]:
        st.markdown("**Line items**")
        for item in mem["items"]:
            st.write(
                f"- {item['name']} × {item['qty']} @ ₹{item['price']} "
                f"(GST {item['slab_percent']:g}%)"
            )

    if st.button("Clear invoice / new chat", use_container_width=True):
        st.session_state.agent.reset()
        st.session_state.chat = []
        st.session_state.last_trace = []
        st.rerun()

    with st.expander("Discount rules"):
        st.text(describe_rules())
    with st.expander("GST slabs"):
        st.text(slab_table())

    st.markdown("**Try these**")
    examples = [
        "Add 2 notebooks at 80 and 1 pen at 20, then total with 18% tax",
        "Invoice: 3 textbooks at 450 each and 1 backpack at 800. Print formatted invoice with tax slabs",
        "Also add a television at 42000",
        "Print the formatted invoice with tax slabs",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{hash(ex)}", use_container_width=True):
            st.session_state.pending = ex
            st.rerun()

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Chat")
    for turn in st.session_state.chat:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    prompt = st.chat_input("Add items, ask for total, or print the invoice…")
    if "pending" in st.session_state:
        prompt = st.session_state.pop("pending")

    if prompt:
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.spinner("Agent running plan–act loop…"):
            result = st.session_state.agent.run(prompt)
        st.session_state.last_trace = result.trace
        answer = result.answer or "(no answer)"
        st.session_state.chat.append({"role": "assistant", "content": answer})
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
