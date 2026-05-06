"""
handbook_export.py
──────────────────
PDF export and generation-history management for the Handbook Generator.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from fpdf import FPDF

# ─────────────────────────────────────────────────────────────────────────────
# Directories  (created lazily so module import has no side-effects)
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR   = Path("output")
PDF_DIR      = OUTPUT_DIR / "pdfs"
HISTORY_DIR  = OUTPUT_DIR / "history"
HISTORY_FILE = HISTORY_DIR / "handbooks.json"


def _ensure_dirs() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def safe_name(text: str) -> str:
    """Return a filesystem-safe slug (max 80 chars) derived from *text*."""
    slug = re.sub(r"[^\w\s-]", "", text.strip().lower())
    slug = re.sub(r"[\s-]+", "_", slug).strip("_")
    return slug[:80] or "handbook"


def _strip_markdown(text: str) -> str:
    """
    Remove common Markdown syntax so raw symbols don't appear in the PDF.

    FIX: The original code wrote lines verbatim, so **bold**, *italic*,
         `code`, etc. all appeared as literal characters in the output.
    """
    # Bold / italic
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}(.+?)_{1,3}", r"\1", text)
    # Inline code
    text = re.sub(r"`(.+?)`", r"\1", text)
    # Bullet points — preserve the dash as a leading character
    text = re.sub(r"^\s*[-*+]\s+", "  • ", text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# PDF export
# ─────────────────────────────────────────────────────────────────────────────

class _HandbookPDF(FPDF):
    """FPDF subclass with Unicode support via DejaVu fonts."""

    def __init__(self) -> None:
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        # FIX: built-in 'Arial' does not support Unicode (accented/non-Latin
        #      chars raise UnicodeEncodeError at runtime).  DejaVu is bundled
        #      with fpdf2 and covers the full Basic Multilingual Plane.
        self.add_font("DejaVu", style="",  fname="fonts/DejaVuSansCondensed.ttf",      uni=True)
        self.add_font("DejaVu", style="B", fname="fonts/DejaVuSansCondensed-Bold.ttf", uni=True)

    # ── Semantic write helpers ────────────────────────────────────────────────

    def write_h1(self, text: str) -> None:
        self.set_font("DejaVu", style="B", size=18)
        self.multi_cell(0, 12, _strip_markdown(text))
        self.ln(3)
        self.set_font("DejaVu", size=11)

    def write_h2(self, text: str) -> None:
        self.ln(4)
        self.set_font("DejaVu", style="B", size=14)
        self.multi_cell(0, 10, _strip_markdown(text))
        self.ln(2)
        self.set_font("DejaVu", size=11)

    def write_body(self, text: str) -> None:
        self.set_font("DejaVu", size=11)
        self.multi_cell(0, 7, _strip_markdown(text))

    def write_separator(self) -> None:
        self.ln(5)
        self.set_draw_color(180, 180, 180)
        self.line(self.get_x(), self.get_y(), self.get_x() + 190, self.get_y())
        self.ln(5)


def save_handbook_as_pdf(final_doc: str, topic: str) -> str:
    """
    Render *final_doc* (Markdown) as a PDF and return the file path as a string.

    Args:
        final_doc:  Full handbook text (Markdown format).
        topic:      Used for the cover title and the output filename.
    """
    _ensure_dirs()

    pdf = _HandbookPDF()
    pdf.add_page()

    for line in final_doc.splitlines():
        stripped = line.strip()

        if stripped.startswith("# "):
            pdf.write_h1(stripped[2:])
        elif stripped.startswith("## "):
            pdf.write_h2(stripped[3:])
        elif stripped == "---":
            pdf.write_separator()
        elif stripped:
            pdf.write_body(line)
        else:
            pdf.ln(3)   # blank line → small vertical gap

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"{safe_name(topic)}_{timestamp}.pdf"
    path      = PDF_DIR / filename
    pdf.output(str(path))
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# History
# ─────────────────────────────────────────────────────────────────────────────

def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        with HISTORY_FILE.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return []


def save_history_entry(
    topic: str,
    final_doc: str,
    pdf_path: str,
    sections: int,
) -> dict:
    """Append a new entry to the JSON history file and return it."""
    _ensure_dirs()
    history = load_history()
    entry = {
        "id":         len(history) + 1,
        "topic":      topic,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pdf_path":   pdf_path,
        "word_count": len(final_doc.split()),
        "sections":   sections,
    }
    history.append(entry)
    with HISTORY_FILE.open("w", encoding="utf-8") as fh:
        json.dump(history, fh, ensure_ascii=False, indent=2)
    return entry


def get_history() -> list[dict]:
    return load_history()
