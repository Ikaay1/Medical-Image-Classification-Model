const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell,
  WidthType, ShadingType, HeadingLevel, AlignmentType, BorderStyle,
} = require("docx");

const blocks = JSON.parse(fs.readFileSync("doc.json", "utf8"));

// US Letter, 0.75in margins -> content width in DXA (1440 per inch)
const PAGE_W = 12240, MARGIN = 1080;
const CONTENT = PAGE_W - 2 * MARGIN;            // 10080
const IMG_PX = 660;                              // ~6.9in at 96dpi

const runs = (rs, opts = {}) =>
  (rs || []).map(r =>
    r.break
      ? new TextRun({ break: 1 })
      : new TextRun({ text: r.text, bold: r.bold || opts.bold, italics: r.italic, ...opts })
  );

function table(rows) {
  const nCols = Math.max(...rows.map(r => r.length));
  // first column wider; remainder split evenly
  const first = nCols > 2 ? Math.round(CONTENT * 0.34) : Math.round(CONTENT * 0.5);
  const rest = Math.floor((CONTENT - first) / (nCols - 1));
  const widths = [first, ...Array(nCols - 1).fill(rest)];
  widths[nCols - 1] += CONTENT - widths.reduce((a, b) => a + b, 0); // make it sum exactly

  return new Table({
    columnWidths: widths,
    width: { size: CONTENT, type: WidthType.DXA },
    rows: rows.map((cells, ri) =>
      new TableRow({
        tableHeader: ri === 0 && cells.every(c => c.header),
        children: cells.map((c, ci) =>
          new TableCell({
            width: { size: widths[ci], type: WidthType.DXA },
            shading: c.header
              ? { type: ShadingType.CLEAR, fill: "EEEEEE", color: "auto" }
              : undefined,
            margins: { top: 60, bottom: 60, left: 110, right: 110 },
            children: [new Paragraph({
              spacing: { before: 0, after: 0 },
              children: runs(c.runs, { bold: c.header, size: 19 }),
            })],
          })
        ),
      })
    ),
  });
}

const children = [];
for (const b of blocks) {
  if (b.type === "h1") {
    children.push(new Paragraph({
      heading: HeadingLevel.TITLE,
      spacing: { after: 160 },
      children: runs(b.runs, { size: 40, bold: true }),
    }));
  } else if (b.type === "h2") {
    children.push(new Paragraph({
      heading: HeadingLevel.HEADING_1,
      spacing: { before: 320, after: 120 },
      children: runs(b.runs, { size: 26, bold: true }),
    }));
  } else if (b.type === "p") {
    children.push(new Paragraph({
      spacing: { before: 60, after: 120, line: 300 },
      children: runs(b.runs, { size: 22 }),
    }));
  } else if (b.type === "table") {
    children.push(table(b.rows));
    children.push(new Paragraph({ spacing: { after: 160 }, children: [] }));
  } else if (b.type === "figure") {
    children.push(new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 200, after: 60 },
      children: [new ImageRun({
        type: "png",
        data: fs.readFileSync(b.image),
        transformation: { width: IMG_PX, height: Math.round(IMG_PX * b.aspect) },
      })],
    }));
    children.push(new Paragraph({
      spacing: { after: 220 },
      border: { top: { style: BorderStyle.SINGLE, size: 2, color: "CCCCCC", space: 6 } },
      children: runs(b.caption, { size: 18 }),
    }));
  }
}

const doc = new Document({
  styles: { default: { document: { run: { font: "Arial", size: 22 }, paragraph: {} } } },
  sections: [{
    properties: {
      page: { size: { width: PAGE_W, height: 15840 },
              margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } },
    },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("Midterm_Project_Report.docx", buf);
  console.log(`wrote Midterm_Project_Report.docx (${(buf.length / 1e6).toFixed(2)} MB, ${children.length} blocks)`);
});
