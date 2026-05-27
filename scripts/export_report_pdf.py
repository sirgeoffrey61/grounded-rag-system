#!/usr/bin/env python3
"""Export PROJECT_REPORT.md to PDF (markdown -> HTML -> PDF)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MD_PATH = ROOT / "PROJECT_REPORT.md"
PDF_PATH = ROOT / "PROJECT_REPORT.pdf"
HTML_PATH = ROOT / "PROJECT_REPORT.html"

CSS = """
body { font-family: Georgia, "Times New Roman", serif; font-size: 11pt;
       line-height: 1.45; margin: 2cm; color: #111; }
h1 { font-size: 22pt; border-bottom: 2px solid #333; padding-bottom: 6px;
     margin-top: 0; page-break-before: always; }
h1:first-of-type { page-break-before: avoid; }
h2 { font-size: 16pt; margin-top: 1.2em; color: #1a365d; }
h3 { font-size: 13pt; margin-top: 1em; color: #2d3748; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 10pt; }
th, td { border: 1px solid #ccc; padding: 6px 8px; text-align: left; vertical-align: top; }
th { background: #f0f4f8; font-weight: bold; }
code { background: #f5f5f5; padding: 1px 4px; font-size: 9pt; font-family: Consolas, monospace; }
pre { background: #f5f5f5; padding: 10px; font-size: 8pt; white-space: pre-wrap;
      font-family: Consolas, monospace; border: 1px solid #e2e8f0; }
hr { border: none; border-top: 1px solid #ddd; margin: 2em 0; }
ul, ol { margin: 0.5em 0 0.5em 1.2em; }
p { margin: 0.6em 0; }
strong { color: #1a202c; }
"""


def main() -> int:
    if not MD_PATH.is_file():
        print(f"Missing: {MD_PATH}", file=sys.stderr)
        return 1

    try:
        import markdown
        from xhtml2pdf import pisa
    except ImportError:
        print("Install: pip install markdown xhtml2pdf", file=sys.stderr)
        return 1

    md_text = MD_PATH.read_text(encoding="utf-8")
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "nl2br"],
    )
    html_doc = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'/>"
        f"<style>{CSS}</style></head><body>{html_body}</body></html>"
    )

    HTML_PATH.write_text(html_doc, encoding="utf-8")

    with PDF_PATH.open("wb") as pdf_file:
        status = pisa.CreatePDF(html_doc, dest=pdf_file, encoding="utf-8")

    if status.err:
        print(f"PDF generation reported errors (see {HTML_PATH})", file=sys.stderr)
        return 1

    size_kb = PDF_PATH.stat().st_size / 1024
    print(f"Wrote {PDF_PATH} ({size_kb:.1f} KB)")
    print(f"HTML fallback: {HTML_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
