#!/bin/bash
# Rebuild Midterm_Report_Mgbemele.pdf from midterm.html.
# Figures are pulled from ../outputs/figures and inlined as base64, so the PDF
# is self-contained. Requires Google Chrome.
set -euo pipefail
cd "$(dirname "$0")"

../.venv/bin/python - <<'PY'
import base64, pathlib, re
html = pathlib.Path("midterm.html").read_text()
figdir = pathlib.Path("../outputs/figures")
missing = []
def sub(m):
    p = figdir / m.group(1)
    if not p.exists():
        missing.append(m.group(1)); return m.group(0)
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()
out = re.sub(r'FIG:([A-Za-z0-9_.-]+)', sub, html)
if missing:
    raise SystemExit(f"missing figures: {missing}")
pathlib.Path("midterm_embedded.html").write_text(out)
print(f"inlined {out.count('data:image/png')} figures")
PY

"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="Midterm_Report_Mgbemele.pdf" \
  "file://$(pwd)/midterm_embedded.html" 2>/dev/null

rm -f midterm_embedded.html
echo "built Midterm_Report_Mgbemele.pdf"
