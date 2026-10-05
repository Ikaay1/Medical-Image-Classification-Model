"""Parse the midterm HTML into a JSON block list for the docx builder."""
import json, pathlib, re
from html.parser import HTMLParser
from PIL import Image

FIGDIR = pathlib.Path("../outputs/figures")

class R(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks = []
        self.stack = []          # inline formatting
        self.runs = []           # current inline runs
        self.mode = None         # p | h1 | h2 | figcaption | td | th
        self.table = None
        self.row = None
        self.fig = None

    # ---- inline text -------------------------------------------------
    def handle_data(self, d):
        if self.mode is None or not d:
            return
        if not d.strip() and not self.runs:
            return
        self.runs.append({
            "text": d,
            "bold": "strong" in self.stack or "b" in self.stack,
            "italic": "em" in self.stack or "i" in self.stack,
        })

    def _flush(self):
        runs, self.runs = self.runs, []
        # collapse whitespace, drop empties
        out = []
        for r in runs:
            # Line-break runs carry no text and must survive the whitespace
            # cleanup below, which would otherwise discard them as empty.
            if r.get("break"):
                out.append(r); continue
            t = re.sub(r"\s+", " ", r["text"])
            if t.strip() or (out and not t.isspace()):
                out.append({**r, "text": t})
        def droppable(r):
            return not r.get("break") and not r["text"].strip()
        while out and droppable(out[0]): out.pop(0)
        while out and droppable(out[-1]): out.pop()
        for i in (0, -1):
            if out and not out[i].get("break"):
                out[i]["text"] = out[i]["text"].lstrip() if i == 0 else out[i]["text"].rstrip()
        return out

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("strong","b","em","i"):
            self.stack.append(tag)
        elif tag in ("h1","h2","p","figcaption"):
            self.mode = tag; self.runs = []
        elif tag == "table":
            self.table = []
        elif tag == "tr":
            self.row = []
        elif tag in ("td","th"):
            self.mode = tag; self.runs = []
        elif tag == "figure":
            self.fig = {"type":"figure"}
        elif tag == "img":
            src = a.get("src","")
            m = re.match(r"FIG:(.+)", src)
            if m and self.fig is not None:
                name = m.group(1)
                p = FIGDIR / name
                w, h = Image.open(p).size
                self.fig["image"] = str(p.resolve())
                self.fig["aspect"] = h / w
        elif tag == "br":
            if self.mode:
                self.runs.append({"text":"\n","bold":False,"italic":False,"break":True})

    def handle_endtag(self, tag):
        if tag in ("strong","b","em","i"):
            if tag in self.stack: self.stack.remove(tag)
        elif tag in ("h1","h2","p"):
            runs = self._flush()
            if runs: self.blocks.append({"type":tag,"runs":runs})
            self.mode = None
        elif tag == "figcaption":
            if self.fig is not None:
                self.fig["caption"] = self._flush()
            self.mode = None
        elif tag == "figure":
            if self.fig and "image" in self.fig:
                self.blocks.append(self.fig)
            self.fig = None
        elif tag in ("td","th"):
            if self.row is not None:
                self.row.append({"runs": self._flush(), "header": tag == "th"})
            self.mode = None
        elif tag == "tr":
            if self.table is not None and self.row:
                self.table.append(self.row)
            self.row = None
        elif tag == "table":
            if self.table:
                self.blocks.append({"type":"table","rows":self.table})
            self.table = None

src = pathlib.Path("Midterm_Project_Report.html").read_text()
body = src.split("<body>",1)[1].rsplit("</body>",1)[0]
r = R(); r.feed(body)
pathlib.Path("doc.json").write_text(json.dumps(r.blocks, indent=1))

from collections import Counter
c = Counter(b["type"] for b in r.blocks)
print("parsed blocks:", dict(c))
for b in r.blocks:
    if b["type"] == "table":
        print(f"  table: {len(b['rows'])} rows x {len(b['rows'][0])} cols")
