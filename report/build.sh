#!/bin/bash
# Build a report as PDF and DOCX from its HTML source.
#
#   ./build.sh                               # defaults to the midterm report
#   ./build.sh Week8_9_Progress_Report.html  # any report
#
# Figures are pulled from ../outputs/figures via FIG:<name> placeholders.
# PDF needs Google Chrome; DOCX needs the docx npm module.
set -euo pipefail
cd "$(dirname "$0")"
SRC="${1:-Midterm_Project_Report.html}"
BASE="${SRC%.html}"

# ---- PDF: inline figures as base64, print with headless Chrome ----
../.venv/bin/python - "$SRC" <<'PY'
import base64, pathlib, re, sys
html = pathlib.Path(sys.argv[1]).read_text()
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
  --print-to-pdf="$BASE.pdf" "file://$(pwd)/_embedded.html" 2>/dev/null
rm -f _embedded.html
echo "built $BASE.pdf"

# ---- DOCX: parse the HTML to JSON, build with docx-js ----
[ -d node_modules/docx ] || npm install docx --silent
../.venv/bin/python parse_html.py "$SRC" _blocks.json
node build_docx.js _blocks.json "$BASE.docx"
rm -f _blocks.json
