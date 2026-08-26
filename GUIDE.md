# Invoice Assistant — full guide

CSE476 CA1 Project 1, topic **T8 Simple Invoice Assistant**.

This file lives in the **InvoiceAssistant** project folder. Read it to **run the app**, **use it**, **demo it**, or **explain it in a viva**. The short README is the submission write-up (tools, memory, honest failure). This guide covers everything around that.

---

## 1. What this project is

You give the agent a **goal** (build an invoice, add items, ask whether a discount applies, print a bill). The agent does **not** invent the numbers in one chat reply.

It runs a **plan–act loop**:

1. **Receive** the goal.
2. **Think** — a language model (or an offline planner) chooses the next tool.
3. **Act** — Python runs a **whitelisted** function. The model never executes code.
4. **Observe** — the JSON result goes back into the transcript.
5. **Remember** — line items stay in session memory for later turns.
6. **Answer** — only after the tools have produced the figures.

That is the course rule: a chatbot answers; an agent acts.

---

## 2. What you get when you run it

| Surface | What it is for |
|---|---|
| **Browser UI** (`streamlit run app.py`) | Chat + live invoice + **visible process** for each reply |
| **CLI** (`python -m invoice_agent "…"`) | Fast terminal demo with a printed tool trace |
| **Notebook** (`demo_invoice_agent.ipynb`) | Submission proof: 3 goals, multi-step traces |
| **Tests** (`pytest tests/ -v`) | Locks tools, discount, slabs, memory, explanations |

The UI is the easiest way to use it. The notebook is what the rubric asks you to submit as the demo.

---

## 3. Run it (from zero)

Open a terminal:

```bash
cd InvoiceAssistant
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3.1 Environment file

The app reads `.env` (gitignored). Copy the example if you need a fresh one:

```bash
cp .env.example .env
```

Useful keys:

```
PROVIDER=groq
GROQ_API_KEY=gsk_...your key...
```

| `PROVIDER` | When to use it |
|---|---|
| `groq` | Default. Free OpenAI-compatible API. Needs `GROQ_API_KEY`. |
| `foundry` | Microsoft Foundry. Needs `AZURE_OPENAI_ENDPOINT` (ending in `/openai/v1/`) and `AZURE_OPENAI_API_KEY`. |
| `local` | Ollama on this machine. No key. Run `ollama serve` and `ollama pull llama3.2`. |

If no key is configured, the **same tools still run** through the offline planner. The UI will say **Offline planner**. With a Groq key it says **LLM lane**.

### 3.2 Start the UI

```bash
source .venv/bin/activate
streamlit run app.py
```

Then open **http://127.0.0.1:8501**.

If Streamlit asks for an email on first run, leave it blank and press Enter. The project already has `.streamlit/config.toml` so later starts skip that.

### 3.3 CLI (optional)

```bash
python -m invoice_agent "Add 2 notebooks at 80 and 1 pen at 20, then total with 18% tax"
```

### 3.4 Tests and notebook

```bash
pytest tests/ -v
jupyter notebook demo_invoice_agent.ipynb
```

---

## 4. How to use the UI

The page is **chat first**. A short how-to sits above the conversation. Live THINK → ACT → OBS lines print in the **terminal** where you ran Streamlit — not as a dashboard on the page.

### Layout

- **Main** — chat. Type a goal, or attach `.xlsx` / `.csv` in the chat box (columns: name, price, qty).
- **Sidebar** — example prompts, item count, **New chat**.
- **Terminal** — live process log (`[invoice] GOAL / THINK / ACT / OBS / ANSWER`).

### First goal (below discount)

Type, or tap **Small bill** in the sidebar:

```
Add 2 notebooks at 80 and 1 pen at 20, then total with 18% tax
```

You should see in chat: grand total **Rs 212.40**. In the terminal: two `add_item` calls, `check_discount` (not eligible), then `compute_total(18)`.

### Second goal (discount + slabs)

Tap **New chat**, then **Discount + slabs**:

```
Invoice: 3 textbooks at 450 each and 1 backpack at 800. Print formatted invoice with tax slabs
```

Subtotal Rs 2,150 → **5% off**. Textbooks at GST 5%, backpack at GST 18%.

### Third goal (memory across turns)

Stay in the same chat (do not clear):

1. `Add 2 pens at 40 and 1 bag at 200`
2. `Also add a television at 42000`
3. `Print the formatted invoice with tax slabs`

Turn 3 must **not** call `add_item` again. The terminal log for that turn should jump to `check_discount` + `format_invoice`.

### Reading the terminal log

| Line | Meaning |
|---|---|
| **GOAL** | Your exact prompt |
| **THINK** | Planner/model choosing the next tool |
| **ACT** | Tool name and arguments |
| **OBS** | Plain-English result |
| **STOP / ANSWER** | Why the loop ended, and the reply |

That log is the proof it is an agent, not a chatbot.

---

## 5. How to talk to it (prompt style)

The agent understands goals like:

- `Add 2 pens at 40`
- `Add a backpack for 800`
- `… then total with 18% tax`
- `Print the formatted invoice with tax slabs`

Be specific about **name, price, quantity**. Vague “make me a bill” with no products will ask you for items.

GST slab is inferred from the **name** (textbook → 5%, biscuits → 12%, bag/pen → 18%, television → 28%, unknown → 18%). You do not pass the slab yourself on `add_item`.

---

## 6. Business rules

### Discount (before tax)

| Subtotal | Discount |
|---|---|
| Rs 10,000 or more | 15% |
| Rs 5,000 or more | 10% |
| Rs 1,000 or more | 5% |
| Below Rs 1,000 | none |

`check_discount` **decides**. `compute_total` / `format_invoice` **apply** that decision. If the model forgets to call `check_discount`, the total tools still decide from the current subtotal so the bill is never silently undiscounted.

Order of money:

1. Line amounts (`price × qty`)
2. Discount on subtotal
3. GST on the discounted amount

Never tax first and then take discount (that would overcharge GST).

### GST slabs (group add-on)

| Category | Rate | Examples |
|---|---|---|
| essential | 5% | textbook, notebook, book, milk, rice |
| standard_lower | 12% | biscuit, juice |
| standard | 18% | bag, pen, backpack, unknown items |
| luxury | 28% | television, phone, laptop, AC |

`format_invoice` taxes each line at its own slab and shows a **tax-by-slab** breakdown. `compute_total(tax_percent)` is the T8 required single-rate path (for example 18% on the whole taxable amount).

---

## 7. Architecture (what lives where)

```
InvoiceAssistant/
  app.py                      # Streamlit UI + process panel
  GUIDE.md                    # this file
  README.md                   # 3-paragraph submission write-up
  demo_invoice_agent.ipynb    # 2–3 goals with traces
  requirements.txt
  .env / .env.example
  invoice_agent/
    agent.py      # plan–act loop (LLM + offline)
    tools.py      # add_item, check_discount, compute_total, format_invoice
    memory.py     # session line items + discount decision
    discount.py   # threshold table
    tax.py        # GST classification
    lanes.py      # Groq / Foundry / Ollama
    explain.py    # human “why” + process stages for the UI
    __main__.py   # CLI
  tests/test_invoice_agent.py
```

### Tools (whitelist)

Only names in `REGISTRY` can run. If the model invents a tool name, `call_tool` returns an error string and the loop continues.

| Tool | Signature | Role |
|---|---|---|
| `add_item` | `(name, price, qty)` | Required T8 tool. Writes memory, assigns GST slab. |
| `compute_total` | `(tax_percent)` | Required T8 tool. One tax rate after discount. |
| `check_discount` | `()` | Makes the threshold **decision** visible. |
| `format_invoice` | `()` | Group add-on: slabs + printable summary. |

### Memory

`SessionMemory` holds:

- `items` — every line this conversation
- `discount_decision` — last threshold check (cleared when a new item is added)
- `turns` — user goals so far

A later “print the invoice” turn **must** reuse `items`. That is how you show memory in the demo.

### Loop exits

1. **Goal met** — the model returns text and no more tool calls.
2. **No progress** — the same tool + args repeated three times.
3. **Budget** — `max_steps` (default 12) so the loop cannot run forever.

---

## 8. LLM vs offline (same tools)

| | LLM lane | Offline planner |
|---|---|---|
| Needs | Groq / Foundry / Ollama | Nothing |
| Who chooses tools | The model, via tool schema | Regex + a fixed recipe |
| Who runs tools | `call_tool` → `REGISTRY` | Same |
| Trace | Yes | Yes |

The assignment does not mark you on which lane you use. What it marks is: tools actually called, more than one step, memory used later.

---

## 9. Honest failure (also in the README)

LLM tool arguments often arrive as strings (`"20"` instead of `20`). Multiplying crashed the loop. Tools now coerce numbers and return JSON errors, so the agent can recover.

A second failure: the model sometimes called `format_invoice` without `check_discount`. Totals now apply the threshold themselves if no decision is on file, so the user is not undercharged.

---

## 10. Viva cheat sheet

**One line:** It is a plan–act invoice agent that calls Python tools to add items, decide a discount, apply GST (including multiple slabs), and remember the cart.

**Point in code:**

- Loop — `invoice_agent/agent.py` (`_run_llm` / `_run_offline`)
- Tools actually run — `call_tool` + `REGISTRY` in `tools.py`
- Discount decision — `check_discount` then `compute_total` / `format_invoice`
- Memory — `SessionMemory` and demo goal 3 (print without re-add)
- Process on screen — chat UI; live loop in the Streamlit **terminal** (`[invoice] ACT / OBS`)

**Sample numbers:**

- 2×80 + 20 = Rs 180 → no discount → 18% tax Rs 32.40 → **Rs 212.40**
- 3×450 + 800 = Rs 2,150 → 5% off Rs 107.50 → mixed 5% and 18% GST

**If they ask “is this just ChatGPT?”:** the model only *requests* `add_item`. Python stores the row. The next step uses that stored subtotal. Watch that chain in the terminal log.

---

## 11. Submit checklist

Zip (or GitHub) should include:

- [ ] Code: `invoice_agent/`, `app.py`, tests, notebook
- [ ] `README.md` — names the two required tools, what memory does, one honest failure
- [ ] `GUIDE.md` — this walkthrough
- [ ] `demo_invoice_agent.ipynb` — 2–3 goals with traces, **runs**
- [ ] Do **not** commit `.env` (secrets). `.env.example` is enough
- [ ] Group of 3: one-line who-did-what in the README; slabs + formatted invoice already implemented

Marks: Implementation 10, Presentation (running notebook) 10, Viva 10. If the notebook does not run, implementation cannot be verified.

---

## 12. Troubleshooting

| Symptom | Fix |
|---|---|
| UI asks for email and sits there | Press Enter on a blank email, or use the project `.streamlit/config.toml` |
| Port 8501 in use | `streamlit run app.py --server.port 8502` |
| `No module named invoice_agent` | Run commands from the `InvoiceAssistant` folder with the venv active |
| Groq errors / 429 | Wait, or set `PROVIDER=local` with Ollama, or drop the key to use offline |
| Offline parser misses an item | Use `N name at PRICE` (example: `2 pens at 40`) |
| Discount looks wrong | Confirm subtotal vs the table in section 6; new items clear a stale decision |

That is the whole product: run it, talk to it in chat, watch the terminal log, then explain the loop in the viva.
