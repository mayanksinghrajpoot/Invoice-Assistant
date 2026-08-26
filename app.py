"""
Invoice Assistant UI — chat plus live invoice sidebar.

Cart memory lives in st.session_state until Clear invoice / new chat.
Live THINK/ACT/OBS still prints in the Streamlit terminal.

    streamlit run app.py
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from invoice_agent.agent import InvoiceAgent, StepTrace
from invoice_agent.discount import describe_rules
from invoice_agent.excel_import import SheetError, from_upload
from invoice_agent.explain import STAGE_LABELS, build_process, why_tool
from invoice_agent.lanes import LaneError, describe, get_client, get_model, lane_is_configured
from invoice_agent.tax import slab_table
from invoice_agent.tools import add_item, bind_memory

EXAMPLES = [
    (
        "Small bill + 18% tax",
        "Add 2 notebooks at 80 and 1 pen at 20, then total with 18% tax",
        "Below the discount threshold. add_item → check_discount → compute_total.",
    ),
    (
        "Discount + GST slabs",
        "Invoice: 3 textbooks at 450 each and 1 backpack at 800. Print formatted invoice with tax slabs",
        "Crosses Rs 1,000 so 5% off. Mixed GST on the printed invoice.",
    ),
    (
        "Add a TV (memory)",
        "Also add a television at 42000",
        "Adds to the same session. Does not wipe earlier items.",
    ),
    (
        "Print invoice (memory)",
        "Print the formatted invoice with tax slabs",
        "No re-add. Reads the cart, decides discount, prints slabs.",
    ),
]

CSS = """
<style>
.stApp {
  background:
    radial-gradient(1200px 500px at 8% -10%, #dceee6 0%, transparent 55%),
    radial-gradient(900px 400px at 110% 0%, #efe6d4 0%, transparent 50%),
    #f4f1ea;
}
.block-container { padding-top: 1.1rem; max-width: 1280px; }
.hero {
  background: #16382f;
  color: #f4f1ea;
  border-radius: 18px;
  padding: 1.1rem 1.3rem 1rem;
  margin-bottom: 0.85rem;
  box-shadow: 0 18px 40px rgba(22, 56, 47, 0.18);
}
.hero h1 { font-size: 1.5rem; margin: 0 0 0.25rem 0; color: #f4f1ea !important; }
.hero p { margin: 0; color: #c9d9d2; font-size: 0.95rem; }
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
  margin-right: 0.4rem;
}
.pipeline { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.85rem 0 0.2rem; }
.pipe {
  background: #21483d; color: #e6efe9; border: 1px solid #2f5d50;
  border-radius: 999px; padding: 0.28rem 0.7rem; font-size: 0.78rem; font-weight: 500;
}
.pipe em { color: #c9f07a; font-style: normal; }
.stage {
  border-radius: 14px; padding: 0.75rem 0.85rem; margin-bottom: 0.55rem;
  border: 1px solid #ddd4c3; background: #fffdf8;
}
.stage.receive { border-left: 5px solid #3b6d9b; }
.stage.think { border-left: 5px solid #b5812a; }
.stage.act { border-left: 5px solid #1f6f5b; }
.stage.observe { border-left: 5px solid #5b4db1; }
.stage.remember { border-left: 5px solid #8a4b2a; }
.stage.answer { border-left: 5px solid #16382f; }
.stage.bad { background: #fdecea; border-color: #e8b4ae; }
.kicker {
  font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em;
  text-transform: uppercase; color: #6b6256; margin-bottom: 0.18rem;
}
.stage p { margin: 0; font-size: 0.92rem; color: #2b2723; }
.invoice-card, .empty {
  background: #fffdf8; border: 1px dashed #cbbd9f; border-radius: 14px;
  padding: 0.8rem; color: #3a3530;
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
    return InvoiceAgent(client=client, model=model, verbose=True, max_steps=48), mode


def _persist_bill() -> None:
    agent: InvoiceAgent = st.session_state.agent
    st.session_state.bill_items = [item.as_dict() for item in agent.memory.items]
    st.session_state.bill_turns = list(agent.memory.turns)


def _hydrate_bill() -> None:
    """Streamlit reruns reload tools.py; pin the cart in session_state until New chat."""
    agent: InvoiceAgent = st.session_state.agent
    saved = st.session_state.get("bill_items")
    if saved:
        agent.memory.restore_items(saved)
    turns = st.session_state.get("bill_turns")
    if turns:
        agent.memory.turns = list(turns)
    bind_memory(agent.memory)


def _ensure_state() -> None:
    if "agent" not in st.session_state:
        agent, mode = _build_agent()
        st.session_state.agent = agent
        st.session_state.mode = mode
        st.session_state.chat = []
        st.session_state.selected = None
        st.session_state.last_error = None
        st.session_state.bill_items = []
        st.session_state.bill_turns = []
    _hydrate_bill()


def _selected_turn() -> dict | None:
    idx = st.session_state.selected
    chat = st.session_state.chat
    if idx is None:
        for turn in reversed(chat):
            if turn.get("role") == "assistant" and turn.get("process"):
                return turn
        return None
    if 0 <= idx < len(chat):
        return chat[idx]
    return None


def _render_message(i: int, turn: dict) -> None:
    with st.chat_message(turn["role"]):
        content = turn.get("content") or ""
        if turn["role"] == "assistant" and ("Grand total" in content or "----" in content):
            st.code(content, language=None)
        else:
            st.markdown(content)
        if turn["role"] == "assistant" and turn.get("process"):
            st.caption(
                f"{turn.get('mode', '?')} · {turn.get('steps', 0)} step(s) · "
                f"{turn.get('stopped_because', '?')}"
            )
            if st.button("Show how this reply was built", key=f"show_process_{i}"):
                st.session_state.selected = i
                st.rerun()


def _render_process(turn: dict) -> None:
    process = turn.get("process") or []
    if not process:
        st.markdown(
            '<div class="empty">Send a goal or attach a file. This panel shows '
            "receive → think → tool → observe → memory → answer.</div>",
            unsafe_allow_html=True,
        )
        return
    chips = []
    seen = []
    for kind in ("receive", "think", "act", "observe", "remember", "answer"):
        if any(s.get("kind") == kind for s in process) and kind not in seen:
            seen.append(kind)
            chips.append(f'<span class="pipe"><em>{STAGE_LABELS.get(kind, kind)}</em></span>')
    st.markdown('<div class="pipeline">' + "".join(chips) + "</div>", unsafe_allow_html=True)
    for stage in process:
        kind = stage.get("kind", "act")
        ok = stage.get("ok", True)
        extra = " bad" if not ok else ""
        st.markdown(
            f'<div class="stage {kind}{extra}">'
            f'<div class="kicker">{html.escape(kind)}</div>'
            f"<strong>{html.escape(str(stage.get('title', '')))}</strong>"
            f"<p>{html.escape(str(stage.get('detail', '')))}</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        if stage.get("args"):
            st.code(json.dumps(stage["args"], indent=2), language="json")
    trace: list[StepTrace] = turn.get("trace") or []
    if trace:
        with st.expander("Raw tool JSON (viva evidence)", expanded=False):
            for step in trace:
                st.markdown(f"**Step {step.step} — `{step.tool}`**")
                st.caption(step.why)
                st.code(json.dumps(step.args, indent=2), language="json")
                try:
                    st.code(json.dumps(json.loads(step.observation), indent=2), language="json")
                except Exception:
                    st.code(step.observation)


def _split_chat_input(raw):
    if raw is None:
        return "", []
    if hasattr(raw, "text") or hasattr(raw, "files"):
        return (raw.text or "").strip(), list(getattr(raw, "files", None) or [])
    return str(raw).strip(), []


def _ingest_sheet(filename: str, data: bytes) -> str:
    if not data:
        raise SheetError(f"{filename} was empty. Attach the file again.")
    agent: InvoiceAgent = st.session_state.agent
    bind_memory(agent.memory)
    before = len(agent.memory.items)
    items = from_upload(filename, data)
    agent._log("=" * 56)
    agent._log(f"GOAL   upload {filename} ({len(items)} row(s)) — APPEND to cart")
    agent._log(f"CART   already had {before} line(s)")
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
    _persist_bill()
    snap = agent.memory.snapshot()
    added_n = len(agent.memory.items) - before
    agent._log(
        f"STOP   appended {added_n}; cart now {snap['item_count']} item(s), "
        f"Rs {snap['subtotal']:.2f}"
    )
    agent._log("=" * 56)
    body = "\n".join(lines) if lines else "No new rows added."
    return (
        f"Added **{added_n}** row(s) from **{filename}** (did not replace the cart).\n\n"
        f"{body}\n\n"
        f"Bill now **{snap['item_count']}** item(s), subtotal **Rs {snap['subtotal']:.2f}**. "
        "Upload another file to add more, or tap **Print invoice (memory)**."
    )


def _assistant_from_run(prompt: str, result) -> dict:
    process = build_process(
        goal=prompt,
        mode=result.mode,
        trace=result.trace,
        stopped_because=result.stopped_because,
        memory_snapshot=st.session_state.agent.memory.snapshot(),
    )
    answer = result.answer or "(no answer)"
    if result.stopped_because == "budget":
        answer = f"{answer}\n\n_(stopped: step budget)_"
    return {
        "role": "assistant",
        "content": answer,
        "trace": list(result.trace),
        "process": process,
        "mode": result.mode,
        "steps": result.steps,
        "stopped_because": result.stopped_because,
        "goal": prompt,
    }


st.set_page_config(
    page_title="Invoice Assistant",
    page_icon=":receipt:",
    layout="wide",
    initial_sidebar_state="expanded",
)
_ensure_state()
st.markdown(CSS, unsafe_allow_html=True)

mode = st.session_state.mode
try:
    lane_line = describe() if mode == "llm" else "Offline planner (no API key)"
except Exception:
    lane_line = "Offline planner"
mode_label = "LLM lane" if mode == "llm" else "Offline planner"

st.markdown(
    f"""
<div class="hero">
  <span class="badge">CSE476 T8</span>
  <span class="badge">{mode_label}</span>
  <h1>Simple Invoice Assistant</h1>
  <p>An agent, not a chatbot. It calls tools, remembers the cart, then answers.
  {html.escape(lane_line)}</p>
  <div class="pipeline">
    <span class="pipe">1. Receive</span>
    <span class="pipe">2. Think</span>
    <span class="pipe">3. Act (tool)</span>
    <span class="pipe">4. Observe</span>
    <span class="pipe">5. Remember</span>
    <span class="pipe">6. Answer</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Live invoice")
    mem = st.session_state.agent.memory.snapshot()
    c1, c2 = st.columns(2)
    c1.metric("Items", mem["item_count"])
    c2.metric("Subtotal", f"Rs {mem['subtotal']:.2f}")

    if mem["items"]:
        st.dataframe(
            [
                {
                    "Item": it["name"],
                    "Qty": it["qty"],
                    "Price": it["price"],
                    "GST %": it["slab_percent"],
                    "Amount": it["amount"],
                }
                for it in mem["items"]
            ],
            hide_index=True,
            use_container_width=True,
        )
        decision = mem.get("discount_decision")
        if decision:
            if decision.get("eligible"):
                st.success(decision.get("reason", "Discount applies."))
            else:
                st.info(decision.get("reason", "No discount."))
        st.caption("Uploads and chat add to this bill until you clear it.")
    else:
        st.markdown(
            '<div class="invoice-card">Empty cart. Add items in chat, attach '
            "Excel/CSV/JSON, or tap an example below.</div>",
            unsafe_allow_html=True,
        )

    if st.button("Clear invoice / new chat", use_container_width=True, type="primary"):
        st.session_state.agent.reset()
        st.session_state.chat = []
        st.session_state.selected = None
        st.session_state.last_error = None
        st.session_state.bill_items = []
        st.session_state.bill_turns = []
        st.rerun()

    st.divider()
    st.markdown("**Try a goal**")
    for i, (label, prompt, hint) in enumerate(EXAMPLES):
        if st.button(label, key=f"example_{i}", use_container_width=True):
            st.session_state.pending = prompt
            st.rerun()
        st.caption(hint)

    st.divider()
    with st.expander("Discount rules"):
        st.text(describe_rules())
    with st.expander("GST slabs"):
        st.text(slab_table())
    with st.expander("Tools this agent can call"):
        for name in ("add_item", "check_discount", "compute_total", "format_invoice"):
            st.markdown(f"**`{name}`** — {why_tool(name)}")

left, right = st.columns([1.15, 1], gap="large")

with left:
    st.subheader("Chat")
    if not st.session_state.chat:
        st.markdown(
            """
<div class="empty">
<strong>How to use</strong>
<ol>
<li>Type a goal, tap a sidebar example, or attach <code>.xlsx</code> / <code>.csv</code> / <code>.json</code>.</li>
<li>Each file <em>adds</em> to the live invoice. Nothing is wiped until <strong>Clear invoice / new chat</strong>.</li>
<li>Live THINK → ACT → OBS is in the terminal. This page shows the process on the right.</li>
</ol>
</div>
""",
            unsafe_allow_html=True,
        )
    for i, turn in enumerate(st.session_state.chat):
        _render_message(i, turn)
    if st.session_state.last_error:
        st.error(st.session_state.last_error)

with right:
    st.subheader("Inside this reply")
    selected = _selected_turn()
    if selected is None:
        st.markdown(
            '<div class="empty">When you send a goal, this panel shows the exact process: '
            "receive → think → each tool call → observe → memory → answer. "
            "That is the proof this is an agent, not a one-shot chatbot.</div>",
            unsafe_allow_html=True,
        )
    else:
        if selected.get("goal"):
            st.markdown(f"**Goal:** {selected['goal']}")
        _render_process(selected)

raw = st.chat_input(
    "Add items, print the invoice, or attach .xlsx / .csv / .json",
    accept_file=True,
    file_type=["xlsx", "csv", "json"],
)
text, files = _split_chat_input(raw)
if "pending" in st.session_state and not text and not files:
    text = st.session_state.pop("pending")

if text or files:
    st.session_state.last_error = None
    label = text if text else f"(uploaded {', '.join(f.name for f in files)})"
    st.session_state.chat.append({"role": "user", "content": label})
    try:
        replies: list[str] = []
        last_turn = None
        if files:
            with st.spinner("Adding file rows to the live invoice…"):
                for uploaded in files:
                    replies.append(_ingest_sheet(uploaded.name, uploaded.getvalue()))
        # Do not treat a filename as a chat goal after an upload.
        looks_like_filename = bool(
            text and Path(text).suffix.lower() in {".xlsx", ".csv", ".json"}
        )
        if text and not looks_like_filename:
            with st.spinner("Working… check the terminal for live steps"):
                result = st.session_state.agent.run(text)
            _persist_bill()
            last_turn = _assistant_from_run(text, result)
            replies.append(last_turn["content"])
        else:
            _persist_bill()
        content = "\n\n".join(replies)
        if last_turn:
            last_turn["content"] = content
            st.session_state.chat.append(last_turn)
            st.session_state.selected = len(st.session_state.chat) - 1
        else:
            st.session_state.chat.append({"role": "assistant", "content": content, "process": []})
    except SheetError as exc:
        _persist_bill()
        st.session_state.last_error = str(exc)
        st.session_state.chat.append(
            {
                "role": "assistant",
                "content": (
                    f"Could not read that file. {exc}\n\n"
                    f"The existing bill is unchanged "
                    f"({len(st.session_state.agent.memory.items)} item(s) still on it)."
                ),
            }
        )
    except Exception as exc:  # noqa: BLE001
        _persist_bill()
        st.session_state.last_error = str(exc)
        st.session_state.chat.append(
            {"role": "assistant", "content": f"Something went wrong. `{exc}`"}
        )
    st.rerun()
