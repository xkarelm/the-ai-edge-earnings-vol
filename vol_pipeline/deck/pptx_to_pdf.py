"""Render deck/pitch_deck.pptx to PDF without PowerPoint/Keynote.

Every shape, picture, table and text run is placed at the exact position and size stored in the .pptx,
written as HTML, and printed to PDF with headless Chrome. Calibri is used if installed, otherwise
Arial (slightly wider, so text that fits here also fits in PowerPoint).

    python deck/pptx_to_pdf.py            # -> deck/pitch_deck.pdf
"""
import base64
import html
import subprocess
import sys
from pathlib import Path

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN

HERE = Path(__file__).resolve().parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "pitch_deck.pptx"
OUT_HTML = SRC.with_name(SRC.stem + "_render.html")
OUT_PDF = SRC.with_suffix(".pdf")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"

inch = lambda emu: emu / 914400


def rgb(fill):
    try:
        if fill.type == 1:
            return "#" + str(fill.fore_color.rgb)
    except Exception:
        pass
    return None


def para_html(p):
    pPr = p._p.pPr
    marL = inch(int(pPr.get("marL"))) if pPr is not None and pPr.get("marL") else 0
    buChar = pPr.find(A + "buChar") if pPr is not None else None
    bu_col = None
    if buChar is not None:
        c = pPr.find(A + "buClr/" + A + "srgbClr")
        bu_col = "#" + c.get("val") if c is not None else "inherit"
    align = {PP_ALIGN.CENTER: "center", PP_ALIGN.RIGHT: "right"}.get(p.alignment, "left")
    ls = p.line_spacing if isinstance(p.line_spacing, float) else 1.0
    sa = p.space_after.pt if p.space_after is not None else 0
    runs = []
    for r in p.runs:
        f = r.font
        style = [f"font-size:{f.size.pt if f.size else 18}pt"]
        if f.bold: style.append("font-weight:700")
        if f.italic: style.append("font-style:italic")
        try:
            if f.color.type is not None: style.append(f"color:#{f.color.rgb}")
        except Exception:
            pass
        runs.append(f'<span style="{";".join(style)}">{html.escape(r.text)}</span>')
    first_size = p.runs[0].font.size.pt if p.runs and p.runs[0].font.size else 18
    bullet = (f'<span class="bu" style="color:{bu_col};font-size:{first_size}pt;left:{-marL}in">•</span>'
              if buChar is not None else "")
    return (f'<p style="margin:0 0 {sa}pt {marL}in;text-align:{align};line-height:{1.2 * ls:.3f}">'
            f"{bullet}{''.join(runs) or '&nbsp;'}</p>")


def tf_html(tf, extra=""):
    v = {MSO_ANCHOR.MIDDLE: "center", MSO_ANCHOR.BOTTOM: "flex-end"}.get(tf.vertical_anchor, "flex-start")
    pad = [inch(m or 0) for m in (tf.margin_top, tf.margin_right, tf.margin_bottom, tf.margin_left)]
    return (f'<div class="tf" style="justify-content:{v};padding:{pad[0]}in {pad[1]}in {pad[2]}in {pad[3]}in;{extra}">'
            + "".join(para_html(p) for p in tf.paragraphs) + "</div>")


def shape_html(sh):
    x, y, w, h = (inch(v) for v in (sh.left, sh.top, sh.width, sh.height))
    pos = f"left:{x}in;top:{y}in;width:{w}in;height:{h}in"
    if sh.shape_type == MSO_SHAPE_TYPE.PICTURE:
        b64 = base64.b64encode(sh.image.blob).decode()
        return f'<img style="position:absolute;{pos}" src="data:{sh.image.content_type};base64,{b64}">'
    if getattr(sh, "has_table", False) and sh.has_table:
        rows = []
        for r_i, row in enumerate(sh.table.rows):
            cells = []
            for c_i, col in enumerate(sh.table.columns):
                cell = sh.table.cell(r_i, c_i)
                bg = rgb(cell.fill) or "transparent"
                pad = [inch(m) for m in (cell.margin_top, cell.margin_right, cell.margin_bottom, cell.margin_left)]
                inner = "".join(para_html(p) for p in cell.text_frame.paragraphs)
                cells.append(f'<td style="width:{inch(col.width)}in;height:{inch(row.height)}in;background:{bg};'
                             f'padding:{pad[0]}in {pad[1]}in {pad[2]}in {pad[3]}in">{inner}</td>')
            rows.append("<tr>" + "".join(cells) + "</tr>")
        return f'<table style="position:absolute;left:{x}in;top:{y}in;width:{w}in">' + "".join(rows) + "</table>"
    style = [pos, "position:absolute"]
    fill = rgb(sh.fill) if hasattr(sh, "fill") else None
    if fill: style.append(f"background:{fill}")
    try:
        if sh.line.fill.type == 1:
            dash = sh.line._ln.find(A + "prstDash")
            kind = "dashed" if dash is not None and dash.get("val") != "solid" else "solid"
            style.append(f"border:{sh.line.width.pt or 1}pt {kind} #{sh.line.color.rgb}")
    except Exception:
        pass
    try:
        kind = str(sh.auto_shape_type)
        if "OVAL" in kind: style.append("border-radius:50%")
        elif "ROUNDED" in kind: style.append(f"border-radius:{0.08 * min(w, h):.3f}in")
    except Exception:
        pass
    inner = tf_html(sh.text_frame) if sh.has_text_frame and sh.text_frame.text.strip() else ""
    return f'<div style="{";".join(style)};box-sizing:border-box">{inner}</div>'


prs = Presentation(SRC)
W, H = inch(prs.slide_width), inch(prs.slide_height)
pages = []
for slide in prs.slides:
    bg = "#FFFFFF"
    try:
        bg = rgb(slide.background.fill) or bg
    except Exception:
        pass
    pages.append(f'<section style="background:{bg}">' + "".join(shape_html(sh) for sh in slide.shapes) + "</section>")

css = f"""
@page {{ size: {W}in {H}in; margin: 0; }}
html, body {{ margin: 0; padding: 0; }}
body {{ font-family: Calibri, Carlito, Arial, sans-serif; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
section {{ position: relative; width: {W}in; height: {H}in; overflow: hidden; page-break-after: always; }}
.tf {{ position: absolute; inset: 0; display: flex; flex-direction: column; box-sizing: border-box; }}
p {{ position: relative; }}
.bu {{ position: absolute; font-family: Arial, sans-serif; }}
table {{ border-collapse: collapse; table-layout: fixed; }}
td {{ border-bottom: 1pt solid #D0D3D9; vertical-align: middle; box-sizing: border-box; }}
"""
OUT_HTML.write_text(f'<!doctype html><html><head><meta charset="utf-8"><title>Pitch deck</title><style>{css}</style></head>'
                    f"<body>{''.join(pages)}</body></html>")
subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={OUT_PDF}",
                OUT_HTML.as_uri()], check=True, capture_output=True)
OUT_HTML.unlink()
print(f"{len(pages)} slides -> {OUT_PDF}")
