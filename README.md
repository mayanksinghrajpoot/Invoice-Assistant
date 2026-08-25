# T8 Simple Invoice Assistant

CSE476 CA1 Project 1 — an **agent** that builds a small invoice with tax and discounts.

## Tools

1. **`add_item(name, price, qty)`** — adds a line item and remembers it for the session. Classifies GST slab from the name (5 / 12 / 18 / 28%).
2. **`compute_total(tax_percent)`** — applies the decided discount, then one tax rate, and returns the grand total.

Also used so the agent stays honest about discounts and the group add-on:

- **`check_discount()`** — decides whether the subtotal meets a threshold (5% off at ₹1,000, 10% at ₹5,000, 15% at ₹10,000).
- **`format_invoice()`** — printable summary with **multiple tax slabs** per line.

## Memory

`SessionMemory` keeps every line item, the last discount decision, and earlier user turns in the same conversation. Later goals (for example “print the invoice”) reuse those items instead of starting over.

## Honest failure

LLM tool arguments sometimes arrived as strings (`"20"` instead of `20`), which crashed `qty * price`. The tools now coerce numbers and return a clear JSON error if the value is not numeric, so the agent can recover instead of dying mid-loop.

## Setup

```bash
cd InvoiceAssistant
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional: set PROVIDER=groq and GROQ_API_KEY
```

## Run

```bash
# Browser UI (easiest way to use it)
streamlit run app.py

# Offline / LLM CLI — still shows a real multi-step tool trace
python -m invoice_agent "Add 2 notebooks at 80 and 1 pen at 20, then total with 18% tax"

# Tests
pytest tests/ -v

# Demo notebook (2–3 goals with traces)
jupyter notebook demo_invoice_agent.ipynb
```

## Project layout

```
InvoiceAssistant/
  invoice_agent/
    agent.py      # plan–act loop (LLM or offline)
    tools.py      # add_item, check_discount, compute_total, format_invoice
    memory.py     # session line items + discount decision
    discount.py   # threshold rules
    tax.py        # GST slabs
    lanes.py      # Groq / Foundry / local Ollama
  demo_invoice_agent.ipynb
  tests/
  README.md
```
