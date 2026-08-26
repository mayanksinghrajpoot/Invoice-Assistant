"""
Chat-first UI for the T8 Invoice Assistant.

Live plan–act detail prints in the terminal that launched Streamlit.
The page is chat plus a short how-to. Excel/CSV can be attached in chat.

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
from invoice_agent.excel_import import SheetError, from_upload
from invoice_agent.lanes import LaneError, describe, get_client, get_model, lane_is_configured
from invoice_agent.tools import add_item, bind_memory

EXAMPLES = [
    ("Small bill", "Add 2 notebooks at 80 and 1 pen at 20, then total with 18% tax"),
    ("Discount + slabs", "Invoice: 3 textbooks at 450 each and 1 backpack at 800. Print formatted invoice with tax slabs"),
    ("Add a TV", "Also add a television at 42000"),
    ("Print invoice", "Print the formatted invoice with tax slabs"),
]

CSS = """
<style>
.stApp {
  background:
    radial-gradient(1200px 500px at 8% -10%, #dceee6 0%, transparent 55%),
    radial-gradient(900px 400px at 110% 0%, #efe6d4 0%, transparent 50%),
    #f4f1ea;
}
.block-container { padding-top: 1.4rem; padding-bottom: 6rem; max-width: 740px; }
h1 { font-size: 1.45rem !important; letter-spacing: -0.02em; margin-bottom: 0.15rem !important; color: #16382f !important; }
.guide {
  background: #fffdf8;
  border: 1px solid #ddd4c3;
  border-radius: 14px;
  padding: 0.85rem 1rem;
  color: #2b2723;
  font-size: 0.92rem;
  line-height: 1.45;
  margin: 0.6rem 0 1.1rem;
}
.guide ol { margin: 0.35rem 0 0 1.1rem; padding: 0; }
.guide li { margin: 0.2rem 0; }
.badge {
  display: inline-block;
  background: #c9f07a;
  color: #16382f;
  font-weight: 600;
  font-size: 0.72rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  padding: 0.18rem 0.5rem;
  border-radius: 999px;
  margin-right: 0.35rem;
}
</style>
"""


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
    return InvoiceAgent(client=client, model=model, verbose=True), mode


def _ensure_state() -> None:
    if "agent" not in st.session_state:
        agent, mode = _build_agent()
        st.session_state.agent = agent
        st.session_state.mode = mode
        st.session_state.chat = []
        st.session_state.last_error = None


def _render_message(turn: dict) -> None:
    with st.chat_message(turn["role"]):
        content = turn.get("content") or ""
        if turn["role"] == "assistant" and ("Grand total" in content or "----" in content):
            st.code(content, language=None)
        else:
            st.markdown(content)


def _split_chat_input(raw):
    """Streamlit 1.39+ returns ChatInputValue when accept_file is on."""
    if raw is None:
        return "", []
    if hasattr(raw, "text") or hasattr(raw, "files"):
        return (raw.text or "").strip(), list(getattr(raw, "files", None) or [])
    return str(raw).strip(), []


def _ingest_sheet(filename: str, data: bytes) -> str:
    """Add spreadsheet rows through add_item so memory and terminal logs stay honest."""
    agent = st.session_state.agent
    items = from_upload(filename, data)
    bind_memory(agent.memory)
    agent._log("=" * 56)
    agent._log(f"GOAL   upload {filename} ({len(items)} row(s))")
    lines = []
    for name, price, qty in items:
        agent._log(f"ACT    add_item({{'name': {name!r}, 'price': {price}, 'qty': {qty}}})")
        payload = json.loads(add_item(name, price, qty))
        if not payload.get("ok"):
            agent._log(f"OBS    failed: {payload.get('error')}")
            lines.append(f"- skipped {name}: {payload.get('error')}")
            continue
        added = payload["added"]
        agent._log(
            f"OBS    remembered {added['qty']} × {added['name']} at Rs {added['price']}"
        )
        lines.append(
            f"- {added['qty']} × {added['name']} @ Rs {added['price']} "
            f"(GST {added['slab_percent']:g}%)"
        )
    snap = agent.memory.snapshot()
    agent._log(f"STOP   sheet imported  ({snap['item_count']} item(s), Rs {snap['subtotal']:.2f})")
    agent._log("=" * 56)
    body = "\n".join(lines) if lines else "No rows added."
    return (
        f"Loaded **{filename}**.\n\n{body}\n\n"
        f"Session subtotal **Rs {snap['subtotal']:.2f}**. "
        "Type `Print the formatted invoice with tax slabs` or `total with 18% tax` when you want the bill."
    )


st.set_page_config(
    page_title="Invoice Assistant",
    page_icon=":receipt:",
    layout="centered",
    initial_sidebar_state="collapsed",
)
_ensure_state()
st.markdown(CSS, unsafe_allow_html=True)

mode = st.session_state.mode
try:
    lane = describe() if mode == "llm" else "Offline planner (no API key)"
except Exception:
    lane = "Offline planner"

st.markdown('<span class="badge">CSE476 T8</span>', unsafe_allow_html=True)
st.title("Invoice Assistant")
st.caption(lane)

st.markdown(
    """
<div class="guide">
<strong>How to use</strong>
<ol>
<li>Type a goal — name, price, and quantity. Example: <em>Add 2 pens at 40, then total with 18% tax</em>.</li>
<li>Or attach <strong>Excel (.xlsx), CSV, or JSON</strong> in the chat box. Columns/keys: <code>name</code> (or item), <code>price</code>, <code>qty</code>.</li>
<li>The agent adds items, checks discount, then tax. Live THINK → ACT → OBS is in the terminal.</li>
</ol>
</div>
""",
    unsafe_allow_html=True,
)

if not st.session_state.chat:
    st.info("Say what to put on the invoice, or attach a spreadsheet in the chat box.")

for turn in st.session_state.chat:
    _render_message(turn)

if st.session_state.last_error:
    st.error(st.session_state.last_error)

with st.sidebar:
    st.markdown("**Examples**")
    for i, (label, prompt) in enumerate(EXAMPLES):
        if st.button(label, key=f"ex_{i}", use_container_width=True):
            st.session_state.pending = prompt
            st.rerun()

    mem = st.session_state.agent.memory.snapshot()
    st.caption(f"{mem['item_count']} item(s) · Rs {mem['subtotal']:.2f} this session")

    if st.button("New chat", use_container_width=True):
        st.session_state.agent.reset()
        st.session_state.chat = []
        st.session_state.last_error = None
        st.rerun()

    st.divider()
    st.markdown(
        "Discount: 5% off at Rs 1,000, 10% at 5,000, 15% at 10,000. "
        "GST slab is picked from the item name."
    )

raw = st.chat_input(
    "Add 2 pens at 40, then total with 18% tax — or attach .xlsx / .csv / .json",
    accept_file=True,
    file_type=["xlsx", "csv", "json"],
)
text, files = _split_chat_input(raw)
if "pending" in st.session_state and not text and not files:
    text = st.session_state.pop("pending")

if text or files:
    st.session_state.last_error = None
    label = text if text else f"(uploaded {files[0].name})" if files else ""
    st.session_state.chat.append({"role": "user", "content": label})
    try:
        replies: list[str] = []
        if files:
            with st.spinner("Reading file…"):
                for uploaded in files:
                    data = uploaded.getvalue()
                    replies.append(_ingest_sheet(uploaded.name, data))
        if text:
            with st.spinner("Working… check the terminal for live steps"):
                result = st.session_state.agent.run(text)
            answer = result.answer or "(no answer)"
            if result.stopped_because == "budget":
                answer = f"{answer}\n\n_(stopped: step budget)_"
            replies.append(answer)
        st.session_state.chat.append({"role": "assistant", "content": "\n\n".join(replies)})
    except SheetError as exc:
        st.session_state.last_error = str(exc)
        st.session_state.chat.append(
            {"role": "assistant", "content": f"Could not read that file. {exc}"}
        )
    except Exception as exc:  # noqa: BLE001
        st.session_state.last_error = str(exc)
        st.session_state.chat.append(
            {"role": "assistant", "content": f"Something went wrong. `{exc}`"}
        )
    st.rerun()
