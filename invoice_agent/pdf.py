"""Render a format_invoice payload as a downloadable A4 PDF."""

from __future__ import annotations

from datetime import date
from typing import Any

from fpdf import FPDF


def _t(value: Any) -> str:
    return str(value).encode("latin-1", "replace").decode("latin-1")


def invoice_pdf(payload: dict[str, Any]) -> bytes:
    """Return PDF bytes for a successful format_invoice payload."""
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()

    pdf.set_text_color(22, 56, 47)
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "Invoice", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(
        0,
        6,
        _t(f"Simple Invoice Assistant  |  {date.today().isoformat()}"),
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(5)

    cols = (46, 16, 26, 26, 18, 24, 26)
    headers = ("Item", "Qty", "Price", "Amount", "GST%", "Tax", "Line total")
    pdf.set_fill_color(232, 243, 234)
    pdf.set_text_color(22, 56, 47)
    pdf.set_font("Helvetica", "B", 8)
    for width, title in zip(cols, headers):
        pdf.cell(width, 8, title, border=1, fill=True, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(43, 39, 35)
    for row in payload.get("lines") or []:
        vals = (
            _t(row.get("name", ""))[:26],
            str(row.get("qty", "")),
            f"{float(row.get('price', 0)):.2f}",
            f"{float(row.get('amount', 0)):.2f}",
            f"{float(row.get('gst_percent', 0)):g}",
            f"{float(row.get('tax', 0)):.2f}",
            f"{float(row.get('line_total', 0)):.2f}",
        )
        aligns = ("L", "R", "R", "R", "R", "R", "R")
        for width, value, align in zip(cols, vals, aligns):
            pdf.cell(width, 7, value, border=1, align=align)
        pdf.ln()

    pdf.ln(6)

    def money_row(label: str, value: float, bold: bool = False) -> None:
        pdf.set_font("Helvetica", "B" if bold else "", 10)
        pdf.cell(110, 7, "")
        pdf.cell(36, 7, _t(label), align="R")
        pdf.cell(36, 7, _t(f"Rs {float(value):.2f}"), align="R", new_x="LMARGIN", new_y="NEXT")

    money_row("Subtotal", payload.get("subtotal", 0))
    money_row("Discount", payload.get("discount", 0))
    money_row("Taxable", payload.get("taxable", 0))
    slabs = payload.get("tax_by_slab") or {}
    for slab in sorted(slabs, key=lambda s: float(str(s).rstrip("%") or 0)):
        money_row(f"GST {slab}", slabs[slab])
    money_row("Grand total", payload.get("grand_total", 0), bold=True)

    note = payload.get("discount_note")
    if note:
        pdf.ln(4)
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(90, 90, 90)
        pdf.multi_cell(0, 5, _t(note))

    return bytes(pdf.output())
