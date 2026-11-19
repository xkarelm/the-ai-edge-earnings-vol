"""Render notebooks/figures/data_pipeline.png with graphviz (dot). Labels only; numbers from build_log.txt.

    python deck/data_pipeline.dot.py
"""
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "notebooks" / "figures" / "data_pipeline.png"
BLUE, INK, BODY, MUTED, LINE, TINT = "#1E4B8F", "#1F1F1F", "#404040", "#7A7A7A", "#C9CDD4", "#F2F4F7"
FONT = "Arial"

def src(name, title, sub):
    return (f'{name} [label=<<table border="0" cellpadding="1"><tr><td align="left" width="215"><font point-size="15" color="{INK}">'
            f'<b>{title}</b></font></td></tr><tr><td align="left" width="215"><font point-size="12.5" color="{BODY}">{sub}</font>'
            f'<br align="left"/></td></tr></table>>, width=2.75, height=0.74];')

steps = [
    ("s1", "Recompute reaction session from the exact release time", "(NYSE calendar, early closes)"),
    ("s2", "Link filing → stock", "(CIK → PERMNO, valid on the release date)"),
    ("s3", "Build pre / post volatility windows", "from the exact release time"),
    ("s4", "Compute earnings surprise (SUE)", "estimates and price must predate the release"),
    ("s5", "Keep events where 8-K and I/B/E/S", "agree on the release session"),
]
rows = []
for i, (port, a, b) in enumerate(steps, 1):
    if i > 1:
        rows.append(f'<tr><td height="16" align="center"><font point-size="13" color="{MUTED}">↓</font></td></tr>')
    rows.append(f'<tr><td port="{port}" bgcolor="{TINT}" align="left" cellpadding="8" width="320">'
                f'<font point-size="13.5" color="{INK}"><b>{i}  </b>{a}</font><br align="left"/>'
                f'<font point-size="12" color="{BODY}">     {b}</font><br align="left"/></td></tr>')
proc = f'proc [shape=plaintext, label=<<table border="0" cellspacing="0" cellpadding="0">{"".join(rows)}</table>>];'

COLW = 270
def bar(n, frac, label, color):
    w = max(int(COLW * frac), 4)
    rest = f'<td width="{COLW - w}" height="5"></td>' if COLW - w > 0 else ""
    return (f'<td align="left" valign="bottom" cellpadding="6"><table border="0" cellspacing="0" cellpadding="0">'
            f'<tr><td align="left" colspan="2" width="{COLW}"><font point-size="14" color="{color}"><b>{n}</b></font>'
            f'<font point-size="12" color="{BODY}">  {label}</font></td></tr>'
            f'<tr><td height="5" width="{w}" bgcolor="{color if color == BLUE else LINE}"></td>{rest}</tr></table></td>')

arrow = f'<td valign="middle" cellpadding="6"><font point-size="14" color="{MUTED}">→</font></td>'
funnel = (f'<<table border="0" cellspacing="0" cellpadding="0"><tr>'
          f'{bar("383,153", 1.0, "8-K earnings filings", INK)}{arrow}'
          f'{bar("299,379", 299379 / 383153, "linked to CRSP", INK)}{arrow}'
          f'{bar("168,906", 168906 / 383153, "surprise + timing agreement", BLUE)}</tr></table>>')

dot = f"""
digraph G {{
  graph [rankdir=LR, bgcolor="white", fontname="{FONT}", nodesep=0.2, ranksep=0.42, pad=0.12,
         label={funnel}, labelloc=b, labeljust=c];
  node  [fontname="{FONT}", shape=box, style="rounded,filled", fillcolor="white", color="{LINE}", penwidth=1.2, margin="0.16,0.08"];
  edge  [color="{MUTED}", penwidth=1.1, arrowsize=0.6, arrowhead=vee];

  subgraph cluster_main {{
    label=<<font point-size="20" color="{INK}"><b>From raw filings to a clean, leakage-safe dataset</b></font>>;
    labelloc=t; labeljust=l; color="white"; margin=10;

    subgraph cluster_src {{
      label=<<font point-size="12" color="{MUTED}"><b>SOURCES</b></font>>; labeljust=l; color="white";
      {src("edgar", "SEC EDGAR", "8-K earnings filings")}
      {src("links", "Link tables", "CIK → PERMNO (CRSP–Compustat);<br align='left'/>I/B/E/S ↔ CRSP")}
      {src("crsp", "CRSP", "daily returns &amp; market cap")}
      {src("ibes", "I/B/E/S", "analyst estimates &amp; actuals")}
    }}
    subgraph cluster_proc {{
      label=<<font point-size="12" color="{MUTED}"><b>PROCESSING</b></font>>; labeljust=l; color="white";
      {proc}
    }}
    subgraph cluster_out {{
      label=<<font point-size="12" color="{MUTED}"><b>OUTPUT</b></font>>; labeljust=l; color="white";
      out [fillcolor="{BLUE}", color="{BLUE}", width=2.6, label=<<table border="0" cellpadding="3">
        <tr><td align="left"><font point-size="16" color="white"><b>Clean event-level dataset</b></font></td></tr>
        <tr><td align="left"><font point-size="22" color="white"><b>168,906</b></font><font point-size="13" color="white"> events</font></td></tr>
        <tr><td align="left"><font point-size="13" color="white">2006–2025 · leakage-safe</font></td></tr></table>>];
    }}
    edgar -> proc:s1:w;
    links -> proc:s2:w;
    crsp  -> proc:s3:w;
    ibes  -> proc:s4:w;
    proc:s5:e -> out:w;
  }}
}}
"""
OUT.parent.mkdir(exist_ok=True)
subprocess.run(["dot", "-Tpng", "-Gdpi=220", "-o", str(OUT)], input=dot, text=True, check=True)
from PIL import Image
im = Image.open(OUT)
print(f"wrote {OUT}  {im.size[0]}x{im.size[1]} px  = {im.size[0]/220:.2f} x {im.size[1]/220:.2f} in at 220 dpi")
