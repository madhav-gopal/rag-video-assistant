"""
pdf_exporter.py – Generate a formatted PDF meeting report using fpdf2.
"""

from fpdf import FPDF


def generate_pdf(result: dict) -> bytes:
    """
    Build a PDF from the pipeline result dict and return it as bytes
    so Streamlit's st.download_button can serve it directly.
    """
    pdf = FPDF()
    pdf.set_margins(15, 15, 15)
    pdf.add_page()

    # ── Title ────────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 18)
    title = result.get("title", "Meeting Report")
    pdf.multi_cell(0, 10, title, align="L")
    pdf.ln(4)

    # ── Divider line ─────────────────────────────────────────────────────────
    pdf.set_draw_color(124, 58, 237)
    pdf.set_line_width(0.8)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(6)

    # ── Stats row (if present) ───────────────────────────────────────────────
    stats = result.get("stats")
    if stats:
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(100, 100, 130)
        pdf.cell(0, 6,
                 f"Words: {stats['word_count']:,}   |   "
                 f"Est. duration: ~{stats['est_minutes']} min   |   "
                 f"Audio chunks: {stats['chunk_count']}",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(4)

    # ── Sections ─────────────────────────────────────────────────────────────
    sections = [
        ("Summary",         result.get("summary", "")),
        ("Action Items",    result.get("action_items", "")),
        ("Key Decisions",   result.get("key_decisions", "")),
        ("Open Questions",  result.get("open_questions", "")),
        ("Full Transcript", result.get("transcript", "")),
    ]

    for heading, content in sections:
        # Section heading bar
        pdf.set_fill_color(240, 237, 255)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(60, 20, 120)
        pdf.cell(0, 8, heading, fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        # Section body
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(30, 30, 30)
        # fpdf2 multi_cell handles long text with automatic line breaks
        safe_content = content.encode("latin-1", errors="replace").decode("latin-1")
        pdf.multi_cell(0, 6, safe_content)
        pdf.ln(6)

    return bytes(pdf.output())
