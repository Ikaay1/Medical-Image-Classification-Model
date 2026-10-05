#!/bin/bash
# Build the midterm report as PDF and DOCX from Midterm_Project_Report.html.
#
# Figures are pulled from ../outputs/figures via the FIG:<name> placeholders in
# the HTML. Requires Google Chrome (PDF) and the docx npm module (Word).
set -euo pipefail
cd "$(dirname "$0")"
SRC=Midterm_Project_Report.html

# ---- PDF: inline the figures as base64, then print with headless Chrome ----
../.venv/bin/python - <<PY
import base64, pathlib, re
html = pathlib.Path("$SRC").read_text()
figdir = pathlib.Path("../outputs/figures")
missing = []
def sub(m):
    p = figdir / m.group(1)
    if not p.exists():
        missing.append(m.group(1)); return m.group(0)
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()
out = re.sub(r'FIG:([A-Za-z0-9_.-]+)', sub, html)
if missing: raise SystemExit(f"missing figures: {missing}")
pathlib.Path("_embedded.html").write_text(out)
print(f"inlined {out.count('data:image/png')} figures")
PY

"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="Midterm_Project_Report.pdf" \
  "file://$(pwd)/_embedded.html" 2>/dev/null
rm -f _embedded.html
echo "built Midterm_Project_Report.pdf"

# ---- DOCX: parse the HTML to JSON, then build with docx-js ----
[ -d node_modules/docx ] || npm install docx --silent
../.venv/bin/python parse_html.py
node build_docx.js
rm -f doc.json
