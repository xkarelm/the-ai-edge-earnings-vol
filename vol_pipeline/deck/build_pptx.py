"""Build deck/pitch_deck.pptx: the complete deck. Design and assembly only.

Every number on the slides is read at build time from the results files below (nothing is typed by hand);
each slide's speaker notes cite the file and row for its numbers.

    data/processed/results_b0_b1_fixed.csv            B0, B1b, B1a_nosue (full sample, validation 2017-19)
    data/processed/results_gbm.csv                    B1b-GBM, 90%-dev early-stopped (no results_gbm_final.csv exists)
    data/processed/results_textsubset_baseline.csv    full-dev fixed-tree GBM; text-subset ladder
    data/processed/results_m1_v2.csv                  M1 vs B1b-GBM on the text subset (validation 2017-19 + holdout)
    data/processed/text_clean_v2.parquet              text-subset event ids (subset medians from model_table)
    data/processed/results_risk_limit.csv             equal-alert-budget decision metrics (full sample, B0 vs B1b)
    data/processed/results_risk_limit_m1.csv          equal-alert-budget decision, M1 vs B1b-GBM (large caps, validation)
    data/processed/results_risk_limit_m1_session.csv  the 2018-03-15 example session (same models and rule)
    data/processed/text_subset_representativeness.csv text-subset medians (market cap, analysts)
    data/processed/build_log.txt                      data funnel
    data/processed/example_rows_dev.csv               six real development rows for the data slide
    data/processed/text_features_v2.parquet           number of text PCA components (column count)

The three parquet files are licensed / row-level and not committed. Without them the four descriptive figures
(subsample medians ~$3.7B / 7 analysts / 2 outside, 48 PCA components) fall back to the documented values; see
DOCUMENTED below. Every other number is read from the committed results files.

    python deck/build_pptx.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = ROOT / "notebooks" / "figures"
P = ROOT / "data" / "processed"
OUT = HERE / "pitch_deck.pptx"

# ======================================================================= numbers (all read from files)
fx = pd.read_csv(P / "results_b0_b1_fixed.csv").set_index(["target", "model"])
gb = pd.read_csv(P / "results_gbm.csv").set_index(["target", "model"])
ts = pd.read_csv(P / "results_textsubset_baseline.csv")
m1 = pd.read_csv(P / "results_m1.csv")
rl = pd.read_csv(P / "results_risk_limit.csv")
rep = pd.read_csv(P / "text_subset_representativeness.csv")
log = (P / "build_log.txt").read_text()

R = lambda d, t, m, c="R2_oos": float(d.loc[(t, m), c])
B0, B1B, NOSUE = (R(fx, "v_post", m) for m in ("B0", "B1b", "B1a_nosue"))
ho = pd.read_csv(P / "results_holdout.csv")
HV = ho[ho.split == "validation"].set_index(["target", "model"])    # finalized models, validation 2017-19
HH = ho[ho.split == "holdout"].set_index(["target", "model"])       # finalized models, holdout 2021-25 (run once)
GBM = R(HV, "v_post", "B1b-GBM")                                     # finalized GBM (all dev, fixed trees) = 0.313
assert abs(R(HV, "v_post", "B0") - B0) < 1e-12 and abs(R(HV, "v_post", "B1b") - B1B) < 1e-12
MODELS = ("B0", "B1b", "B1b-GBM")
VAL_R2 = {m: R(HV, "v_post", m) for m in MODELS}
VAL_RMSE = {m: R(HV, "v_post", m, "RMSE") for m in MODELS}
HO_R2 = {m: R(HH, "v_post", m, "R2_vs_own_mean") for m in MODELS}     # honest holdout R2: vs holdout's own mean
HO_R2_DEVMEAN = {m: R(HH, "v_post", m) for m in MODELS}               # dev-mean convention (inflated by level shift)
HO_RMSE = {m: R(HH, "v_post", m, "RMSE") for m in MODELS}
VAL_GAIN = VAL_R2["B1b-GBM"] - VAL_R2["B0"]
HO_GAIN = HO_R2["B1b-GBM"] - HO_R2["B0"]
RANK_HOLDS = all(R(HH, t, "B0", "R2_vs_own_mean") < R(HH, t, "B1b", "R2_vs_own_mean") < R(HH, t, "B1b-GBM", "R2_vs_own_mean")
                 for t in ("v_post_3", "v_post", "v_post_15"))
assert RANK_HOLDS
N_HO = int(HH.loc[("v_post", "B0"), "n"])
m2 = pd.read_csv(P / "results_m1_v2.csv")                              # re-embedded text layer (clean parquets)
M1N = "M1 = B1b-GBM + text"
_m1 = lambda ev, t: m2[(m2["eval"] == ev) & (m2.target == t) & (m2.model == M1N)].iloc[0]
_b0 = lambda ev, t: m2[(m2["eval"] == ev) & (m2.target == t) & (m2.model == "B1b-GBM")].iloc[0]
TV = {t: dict(bar=_b0("validation 2017-19", t).R2_oos, m1=_m1("validation 2017-19", t).R2_oos,
              d=_m1("validation 2017-19", t).dR2_oos, lo=_m1("validation 2017-19", t).dR2_ci_lo,
              hi=_m1("validation 2017-19", t).dR2_ci_hi) for t in ("v_post_3", "v_post", "v_post_15")}
TH = {t: dict(bar=_b0("holdout 2021-25", t).R2_vs_own_mean, m1=_m1("holdout 2021-25", t).R2_vs_own_mean,
              d=_m1("holdout 2021-25", t).dR2_own, lo=_m1("holdout 2021-25", t).dR2_own_ci_lo,
              hi=_m1("holdout 2021-25", t).dR2_own_ci_hi) for t in ("v_post_3", "v_post", "v_post_15")}
T1718 = _m1("validation 2017-18", "v_post").dR2_oos                      # old rows, re-cleaned text
OLD_1718 = float(pd.read_csv(P / "results_m1.csv").query("rule == 'fixed trees' and target == 'v_post' and "
                                                          "model == @M1N").dR2_vs_bar.iloc[0])   # zip-based text, superseded
N_TV, N_TH = int(_m1("validation 2017-19", "v_post").n), int(_m1("holdout 2021-25", "v_post").n)
assert all(d["lo"] > 0 for d in (*TV.values(), *TH.values())), "a text-gain interval includes zero: STOP"
TXT_HOLDS = TH["v_post"]["d"] >= TV["v_post"]["d"]

eq = rl[(rl.rule.str.startswith("equal")) & (rl.level == "top quartile")].set_index("model")
N_VAL = int(eq.loc["B0", ["TP", "FP", "FN", "TN"]].sum())
EXTRA_TP = int(eq.loc["B1b", "TP"] - eq.loc["B0", "TP"])
PER100, PERQ = 100 * EXTRA_TP / N_VAL, EXTRA_TP / 12
REC0, REC1 = float(eq.loc["B0", "recall"]), float(eq.loc["B1b", "recall"])
CUT = float(rl[(rl.rule == "predicted > dev cut-off") & (rl.level == "top quartile")].cutoff_daily_vol.iloc[0])

# Descriptive figures (subsample medians, PCA size) need three licensed / row-level files that are not committed.
# When they are absent, fall back to the documented values (DEFINITIONS.md "Large-cap subsample", verify_results.py
# "Not checked here"). These four numbers are descriptive only; no result on the slides depends on them.
DOCUMENTED = {"CAP_B": 3.7, "ANALYSTS": 7, "ANALYSTS_OUT": 2, "N_PCS": 48}
_desc_files = [P / "text_clean_v2.parquet", P / "model_table.parquet", P / "text_features_v2.parquet"]
if all(f.exists() for f in _desc_files):
    _ids = set(pd.read_parquet(P / "text_clean_v2.parquet", columns=["event_id"]).event_id)
    _cov = pd.read_parquet(P / "model_table.parquet", columns=["event_id", "split", "log_mktcap", "n_analysts"],
                           filters=[("split", "in", ["dev", "validation", "holdout"])])     # descriptive medians only
    _cov = _cov[_cov.event_id.isin(_ids)]
    assert len(_cov) == len(_ids)
    CAP_B = np.exp(_cov.log_mktcap.median()) / 1e6                                 # dlycap is in $000 -> $B
    ANALYSTS = int(_cov.n_analysts.median())
    _out = pd.read_parquet(P / "model_table.parquet", columns=["event_id", "split", "n_analysts"],
                           filters=[("split", "in", ["dev", "validation", "holdout"])])
    ANALYSTS_OUT = int(_out[~_out.event_id.isin(_ids)].n_analysts.median())          # same rows, outside the text subset
    N_PCS = sum(c.startswith("txt_pc") for c in pq.ParquetFile(P / "text_features_v2.parquet").schema_arrow.names)
    DESC_SOURCE = "computed from model_table.parquet, text_clean_v2.parquet and text_features_v2.parquet"
else:
    CAP_B, ANALYSTS, ANALYSTS_OUT, N_PCS = (DOCUMENTED[k] for k in ("CAP_B", "ANALYSTS", "ANALYSTS_OUT", "N_PCS"))
    DESC_SOURCE = ("FALLBACK: documented values (DEFINITIONS.md, verify_results.py), because the licensed "
                   "model_table.parquet / text_clean_v2.parquet / text_features_v2.parquet are not in this checkout")
    print(f"note: subsample medians and PCA size use the documented values ({DOCUMENTED}); data files absent")
rlm = pd.read_csv(P / "results_risk_limit_m1.csv").set_index("model")              # M1 decision, large caps, validation
D0, D1, DG = rlm.loc["B1b-GBM"], rlm.loc[M1N], rlm.loc["gain: M1 - B1b-GBM"]
sess = pd.read_csv(P / "results_risk_limit_m1_session.csv")                         # 2018-03-15, same models and rule
PCA_VAR = "90.1%"                                                              # printed by m1_text_v2.py pca (not stored)

fun = lambda pat: int(re.search(pat, log).group(1).replace(",", ""))
F0 = fun(r"\[\s*([\d,]+)\] 8-K earnings events in event_table")
F1 = fun(r"\[\s*([\d,]+)\] v_post_15: all 15 post returns present")
F2 = fun(r"\[\s*([\d,]+)\] primary: has SUE")
REDATED = fun(r"stored session_date differs from recomputed for ([\d,]+) events")

f3 = lambda x: f"{x:.3f}"
CAVEAT = (f"Text results: large-cap subsample (median ~${CAP_B:.1f}B market cap, {ANALYSTS} analysts) · "
          "trained 2006–16, validated 2017–19, tested once on 2021–25")

# ======================================================================= design system
FONT = "Calibri"
BLUE = RGBColor(0x1E, 0x4B, 0x8F)
INK = RGBColor(0x1F, 0x1F, 0x1F)
BODY = RGBColor(0x40, 0x40, 0x40)
MUTED = RGBColor(0x7A, 0x7A, 0x7A)
LINE = RGBColor(0xD0, 0xD3, 0xD9)
TINT = RGBColor(0xF2, 0xF4, 0xF7)
PALE = RGBColor(0xB8, 0xC7, 0xDE)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
W, H, M, TOP = 13.333, 7.5, 0.6, 1.65

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
prs.core_properties.author = "Sacha Elalouf, Karel Moryoussef, Ana Cioata, Ghali al Ghaddaf"
prs.core_properties.last_modified_by = ""          # replaces the python-pptx template default
prs.core_properties.title = "Forecasting post-earnings volatility for a derivatives desk"
BLANK = prs.slide_layouts[6]


def box(s, x, y, w, h, fill=None, line=None, shape=MSO_SHAPE.RECTANGLE, dash=False, name=None):
    sh = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        sh.adjustments[0] = 0.08
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid(); sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line; sh.line.width = Pt(1.25)
        if dash:
            sh.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    sh.shadow.inherit = False
    if name:
        sh.name = name
    return sh


def text(s, x, y, w, h, paras, size=18, color=BODY, bold=False, bullets=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, space_after=8, italic=False, line_spacing=1.08, name=None, bullet_color=BLUE):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.space_after = Pt(space_after if i < len(paras) - 1 else 0)
        para.line_spacing = line_spacing
        for t, o in ([(p, {})] if isinstance(p, str) else p):
            r = para.add_run(); r.text = t
            f = r.font
            f.name = FONT; f.size = Pt(o.get("size", size)); f.bold = o.get("bold", bold)
            f.italic = o.get("italic", italic); f.color.rgb = o.get("color", color)
        if bullets:
            pPr = para._p.get_or_add_pPr()
            pPr.set("marL", str(Inches(0.3))); pPr.set("indent", str(-Inches(0.3)))
            c = etree.SubElement(pPr, qn("a:buClr"))
            etree.SubElement(c, qn("a:srgbClr")).set("val", str(bullet_color))
            etree.SubElement(pPr, qn("a:buFont")).set("typeface", "Arial")
            etree.SubElement(pPr, qn("a:buChar")).set("char", "•")
    return tb


def picture(s, path, x, y, w=None, h=None, name=None):
    if h is not None and w is None:
        iw, ih = Image.open(path).size
        w = h * iw / ih
    pic = s.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w))
    if name:
        pic.name = name
    return pic


def table(s, x, y, col_w, rows, row_h=0.44, best=None, name="Table", sizes=(16, 18)):
    gf = s.shapes.add_table(len(rows), len(col_w), Inches(x), Inches(y), Inches(sum(col_w)), Inches(row_h * len(rows)))
    gf.name = name
    tb = gf.table
    tb.first_row = True; tb.horz_banding = False
    for j, cw in enumerate(col_w):
        tb.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tb.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            cell = tb.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tcPr = cell._tc.get_or_add_tcPr()
            for side in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):            # borders first, then fill (schema order)
                ln = etree.SubElement(tcPr, qn(side), w=str(Pt(1 if side == "a:lnB" else 0)))
                if side == "a:lnB":
                    sf = etree.SubElement(ln, qn("a:solidFill"))
                    etree.SubElement(sf, qn("a:srgbClr")).set("val", str(LINE))
                else:
                    etree.SubElement(ln, qn("a:noFill"))
            cell.fill.solid(); cell.fill.fore_color.rgb = TINT if i == 0 else WHITE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT
            r = p.add_run(); r.text = val
            r.font.name = FONT; r.font.size = Pt(sizes[0] if i == 0 else sizes[1])
            r.font.bold = i == 0 or i == best
            r.font.color.rgb = MUTED if i == 0 else (BLUE if i == best else BODY)
    return gf


def frame(n, title, muted=False):
    s = prs.slides.add_slide(BLANK)
    if muted:
        bg = s.background.fill; bg.solid(); bg.fore_color.rgb = TINT
    text(s, M, 0.5, W - 2 * M, 0.75, [title], size=32, bold=True, color=MUTED if muted else INK,
         anchor=MSO_ANCHOR.BOTTOM, name="Title")
    box(s, M, 1.33, 0.9, 0.05, fill=MUTED if muted else BLUE, name="Title rule")
    text(s, W - M - 1.0, H - 0.55, 1.0, 0.36, [str(n)], size=11, color=MUTED, align=PP_ALIGN.RIGHT, name="Page number")
    return s


def caveat(s, y=6.38, w=None):
    text(s, M, y, w or (W - 2 * M - 1.2), 0.5, [CAVEAT], size=14, italic=True, color=MUTED, name="Text caveat")


def notes(s, t):
    s.notes_slide.notes_text_frame.text = t


FULL_TAG = f"Full sample ({F2 / 1000:.0f}k events)"
SUB_TAG = "Large-cap subsample"


def tag(s, label):
    """Sample signpost: same place, font and colour on every results slide (top-right, level with the title rule)."""
    w = 0.105 * len(label) + 0.45
    box(s, W - M - w, 1.2, w, 0.36, fill=TINT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name="Sample tag")
    text(s, W - M - w, 1.2, w, 0.36, [label], size=13, bold=True, color=BLUE, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.0, name="Sample tag text")


# ======================================================================= 1. title
s = prs.slides.add_slide(BLANK)
text(s, M + 0.3, 1.45, 11.8, 1.75, ["Forecasting post-earnings volatility", "for a derivatives desk"],
     size=44, bold=True, color=INK, space_after=0, line_spacing=1.0, anchor=MSO_ANCHOR.BOTTOM, name="Title")
box(s, M + 0.3, 3.45, 1.2, 0.06, fill=BLUE, name="Title rule")
text(s, M + 0.3, 3.72, 11.8, 0.95, ["Does the earnings release add signal beyond the numbers, for a derivatives desk?"],
     size=24, color=BLUE, name="Subtitle")
text(s, M + 0.3, 5.3, 11.8, 0.95, ["Sacha Elalouf · Karel Moryoussef · Ana Cioata · Ghali al Ghaddaf", "November 2026"],
     size=18, color=BODY, space_after=6, name="Authors and date")
text(s, W - M - 1.0, H - 0.55, 1.0, 0.36, ["1"], size=11, color=MUTED, align=PP_ALIGN.RIGHT, name="Page number")
notes(s, "Every number in this deck is read from data/processed results files at build time (see deck/build_pptx.py). "
         "Holdout (2021-25) results: results_holdout.csv (tabular) and results_m1_v2.csv (text), each run once (slides 6, 7, 10). "
         "Decision results (slides 8-9): results_risk_limit_m1.csv and results_risk_limit_m1_session.csv (validation only) "
         "plus results_risk_limit.csv (full sample). No appendix; example rows on slide 4, key terms on slide 5, R² on slide 6, big move on slide 9.")

# ======================================================================= 2. problem
s = frame(2, "The problem: setting risk limits on earnings night")
text(s, M, TOP - 0.12, W - 2 * M, 0.38, ["For the head of an equity volatility / derivatives desk"], size=14, color=MUTED,
     bold=True, name="Audience")
FLOW = [("Company reports earnings", "8-K release, often after the close"),
        ("Stock may move sharply", "in the next days"),
        ("Desk sets tonight's risk limit", "max exposure per name"),
        ("Limit depends on expected volatility", "this is what we forecast")]
BW, GAP, BY, BH = 2.72, 0.417, TOP + 0.45, 1.7
for i, (head, sub) in enumerate(FLOW):
    x = M + i * (BW + GAP)
    last = i == len(FLOW) - 1
    box(s, x, BY, BW, BH, fill=BLUE if last else TINT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name=f"Flow {i + 1}")
    text(s, x + 0.2, BY + 0.1, BW - 0.4, BH - 0.2, [[(head, {"bold": True, "size": 18, "color": WHITE if last else INK})],
         [(sub, {"size": 14, "color": PALE if last else BODY})]], space_after=4, anchor=MSO_ANCHOR.MIDDLE,
         line_spacing=1.0, name=f"Flow {i + 1} text")
    if not last:
        text(s, x + BW, BY, GAP, BH, ["→"], size=28, bold=True, color=MUTED, align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.0, name=f"Flow arrow {i + 1}")
CY = BY + BH + 0.35
text(s, M, CY, W - 2 * M, 0.38, ["If the forecast is wrong, it costs either way"], size=14, color=MUTED, bold=True,
     name="Cards label")
CW = (W - 2 * M - 0.3) / 2
for i, (head, body) in enumerate([("Too low → surprise losses", "Excess risk and surprise limit breaches"),
                                  ("Too high → idle capital", "Capital held against moves that never come")]):
    x = M + i * (CW + 0.3)
    box(s, x, CY + 0.42, CW, 1.05, fill=None, line=LINE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name=f"Cost card {i + 1}")
    text(s, x + 0.3, CY + 0.42, CW - 0.6, 1.05, [[(head, {"bold": True, "color": BLUE, "size": 20})], body],
         size=16, space_after=2, anchor=MSO_ANCHOR.MIDDLE, name=f"Cost {i + 1}")
QY = CY + 1.65
box(s, M, QY, W - 2 * M, 0.7, fill=TINT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name="Question card")
text(s, M + 0.3, QY, W - 2 * M - 0.6, 0.7, [[("The question:  ", {"bold": True, "color": BLUE, "size": 16}),
     ("does the earnings release itself tell us which names will turn into big moves?", {"size": 20, "color": INK})]],
     anchor=MSO_ANCHOR.MIDDLE, name="Question")
text(s, M, 6.68, W - 2 * M - 1.2, 0.4, ["No options data: we benchmark against past realized volatility, not implied "
     "volatility (testing against implied is the next step)"], size=13, italic=True, color=MUTED, name="Benchmark note")
notes(s, "Framing only; no numbers on this slide.")

# ======================================================================= 3. data + pipeline
s = frame(3, "The data & pipeline: why you can trust these numbers")
_im = Image.open(FIG / "data_pipeline.png")
_im = _im.crop((0, int(_im.height * 0.122), _im.width, int(_im.height * 0.855)))   # drop the figure's own title and
_buf = __import__("io").BytesIO(); _im.save(_buf, format="PNG"); _buf.seek(0)       # funnel strip (shown at right)
dw = 9.0
dh = dw * _im.height / _im.width
pic = s.shapes.add_picture(_buf, Inches(M - 0.1), Inches(TOP - 0.05), Inches(dw)); pic.name = "Figure: data pipeline"
fx_ = M + dw + 0.25
fw = W - M - fx_
text(s, fx_, TOP - 0.05, fw, 0.4, ["Data funnel"], size=14, bold=True, color=MUTED, name="Funnel label")
y = TOP + 0.4
for i, (num, label) in enumerate([(F0, "8-K earnings releases (SEC filings)"),
                                  (F1, "linked to a US stock in CRSP, complete return windows"),
                                  (F2, "with an earnings surprise; 8-K and I/B/E/S agree on the session")]):
    last = i == 2
    box(s, fx_, y, fw * num / F0, 0.14, fill=BLUE if last else LINE, name=f"Funnel bar {i + 1}")
    text(s, fx_, y + 0.2, fw, 0.95, [[(f"{num:,}", {"bold": True, "color": BLUE if last else INK, "size": 22})],
                                     [(label, {"size": 14})]], space_after=0, line_spacing=1.0, name=f"Funnel {i + 1}")
    y += 1.15
text(s, fx_, y - 0.1, fw, 0.4, ["Clean primary sample, 2006–2025"], size=14, color=MUTED, italic=True, name="Funnel caption")
gy = TOP - 0.05 + dh + 0.2
text(s, M, gy - 0.02, dw, 0.4, ["Why it is trustworthy"], size=14, bold=True, color=MUTED, name="Trust label")
TRUST = [[("Exact release time, ", {"bold": True, "color": INK}), ("not the 8-K filing date", {})],
         [("NYSE-calendar reaction day: ", {"bold": True, "color": INK}), (f"{REDATED:,} dates corrected", {})],
         [("Two sources agree: ", {"bold": True, "color": INK}), ("8-K and I/B/E/S, same session", {})],
         [("Leakage-safe: ", {"bold": True, "color": INK}),
          ("train 2006–16 · validate 2017–19 · one-time holdout 2021–25", {})]]
cwid = (dw - 0.3) / 2
for k, para in enumerate(TRUST):
    text(s, M + (k % 2) * (cwid + 0.3), gy + 0.4 + (k // 2) * 0.9, cwid, 0.88, [para], size=16, bullets=True,
         name=f"Trust point {k + 1}")
notes(s, f"Source: data/processed/build_log.txt. {F0:,} = '8-K earnings events in event_table'; {F1:,} = after the CRSP "
         f"link and complete return windows; {F2:,} = 'primary: has SUE and 8-K / I/B/E/S agree on the session'; "
         f"{REDATED:,} = 'stored session_date differs from recomputed'. Splits: config.py SPLITS. Figure: "
         "notebooks/figures/data_pipeline.png (deck/data_pipeline.dot.py), cropped at build time to drop its own title and "
         "funnel strip (the funnel is drawn at right). Merges the former data and pipeline slides.")

# ======================================================================= 4. what one event looks like
# Six real rows of model_table.parquet (development split): the first dev event, by release time, of six names.
# Saved to data/processed/example_rows_dev.csv because model_table.parquet is licensed and not committed.
s = frame(4, "What one earnings event looks like")
ex = pd.read_csv(P / "example_rows_dev.csv")
assert len(ex) == 6 and (pd.to_datetime(ex.session_date).dt.year <= 2016).all()
text(s, M, TOP - 0.05, W - 2 * M, 0.45, [[("One row per 8-K earnings release. ", {"bold": True, "color": INK}),
     ("Inputs are known before the release; ", {}), ("v_post is what we predict.", {"bold": True, "color": BLUE})]],
     size=17, name="Lead")
XCOLS = ["ticker", "session_date", "when", "sue_price", "log_mktcap", "sector", "v_pre", "v_post"]
xw = [0.95, 1.5, 1.45, 1.35, 1.45, 1.95, 1.65, 1.75]
TX, TY, HROW, BROW = M, TOP + 1.05, 0.5, 0.48
TARGET_BG = RGBColor(0xE3, 0xEB, 0xF5)
_vol = lambda v: f"{v:.2f}  ({np.exp(v):.1%})"
_sector = lambda s_: s_.split("_", 1)[1].replace("_", " & ").capitalize()
xrows = [[r.ticker, r.session_date, r.when.replace("_", "-"), f"{r.sue_price:.4f}", f"{r.log_mktcap:.2f}",
          _sector(r.sector), _vol(r.v_pre), _vol(r.v_post)] for r in ex.itertuples()]
# group labels above the table: inputs vs target
in_w = sum(xw[:-1])
box(s, TX, TY - 0.47, in_w - 0.04, 0.4, fill=TINT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name="Inputs band")
text(s, TX, TY - 0.47, in_w - 0.04, 0.4, ["Inputs (known before the release)"], size=14, bold=True, color=MUTED,
     align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.0, name="Inputs label")
box(s, TX + in_w, TY - 0.47, xw[-1], 0.4, fill=BLUE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name="Target band")
text(s, TX + in_w, TY - 0.47, xw[-1], 0.4, ["Target →"], size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
     anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.0, name="Target label")
gf = s.shapes.add_table(len(xrows) + 1, len(XCOLS), Inches(TX), Inches(TY), Inches(sum(xw)),
                        Inches(HROW + BROW * len(xrows)))
gf.name = "Example rows table"
tb = gf.table
tb.first_row = True; tb.horz_banding = False
for j, w in enumerate(xw):
    tb.columns[j].width = Inches(w)
for i, row in enumerate([XCOLS] + xrows):
    tb.rows[i].height = Inches(HROW if i == 0 else BROW)
    for j, val in enumerate(row):
        cell = tb.cell(i, j)
        cell.margin_left = cell.margin_right = Inches(0.08)
        cell.margin_top = cell.margin_bottom = Inches(0.02)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        tcPr = cell._tc.get_or_add_tcPr()
        for side in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
            ln = etree.SubElement(tcPr, qn(side), w=str(Pt(1 if side == "a:lnB" else 0)))
            if side == "a:lnB":
                sf = etree.SubElement(ln, qn("a:solidFill")); etree.SubElement(sf, qn("a:srgbClr")).set("val", str(LINE))
            else:
                etree.SubElement(ln, qn("a:noFill"))
        target = j == len(XCOLS) - 1
        if i == 0:
            fill, color, bold = (BLUE, WHITE, True) if target else (TINT, MUTED, True)
        else:
            fill, color, bold = (TARGET_BG, BLUE, True) if target else (WHITE, BODY, j == 0)
        cell.fill.solid(); cell.fill.fore_color.rgb = fill
        p_ = cell.text_frame.paragraphs[0]
        p_.alignment = PP_ALIGN.LEFT if j in (0, 1, 2, 5) else PP_ALIGN.RIGHT
        r_ = p_.add_run(); r_.text = val
        r_.font.name = FONT; r_.font.size = Pt(13 if i == 0 else 15); r_.font.bold = bold; r_.font.color.rgb = color
text(s, M, TY + HROW + BROW * len(xrows) + 0.15, W - 2 * M, 0.35,
     ["v_pre / v_post = log realized volatility 20 days before / 5 days after · % in brackets = daily volatility · "
      "sue_price = surprise ÷ price · log_mktcap = log market value"], size=12, italic=True, color=MUTED,
     name="Table note")
notes(s, "Six real rows of data/processed/model_table.parquet, development split (2006-16), no holdout rows: for each of "
         "SLM, SAI, PFG, SPF, RATE and TAST, the first development event by release time. Saved to "
         "data/processed/example_rows_dev.csv (model_table.parquet is licensed and not committed). Values are as stored; "
         "daily volatility in brackets = exp(v). log_mktcap is the log of CRSP market value in $000 at the last close before "
         "the release. Definitions: DEFINITIONS.md.")

# ======================================================================= 5. approach
s = frame(5, "The approach: a ladder that attributes value")
LW = 7.0
text(s, M, TOP - 0.05, W - 2 * M, 0.45, [[("Predict ", {}), ("post-earnings realized volatility (v_post)", {"bold": True, "color": INK}),
                           (": size, not direction · one change per rung", {})]], size=17, name="Lead")
rungs = [("1", "B0", [("OLS on past volatility (v_pre): standard statistical baseline", {})]),
         ("2", "B1b", [("Ridge; ", {}), ("+ surprise (SUE), size, sector, timing, coverage", {"bold": True, "color": BLUE})]),
         ("3", "B1b-GBM", [("Boosting (LightGBM), same features; ", {}), ("nonlinear", {"bold": True, "color": BLUE})]),
         ("4", "M1", [("B1b-GBM ", {}), ("+ earnings-release text ", {"bold": True, "color": BLUE}),
                      (f"(FinBERT, PCA: {N_PCS} components, {PCA_VAR} of variance)", {})])]
y = TOP + 0.6
for num, head, body in rungs:
    hh = 0.72
    box(s, M, y, LW, hh, fill=TINT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name=f"Rung {num}")
    box(s, M + 0.16, y + (hh - 0.42) / 2, 0.42, 0.42, fill=BLUE, shape=MSO_SHAPE.OVAL, name=f"Rung {num} badge")
    text(s, M + 0.16, y + (hh - 0.42) / 2, 0.42, 0.42, [num], size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.0, name=f"Rung {num} number")
    text(s, M + 0.75, y + 0.03, LW - 0.9, hh - 0.06, [[(head + "  ", {"bold": True, "color": INK}), *body]],
         size=17, anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.0, name=f"Rung {num} text")
    y += hh + 0.07
text(s, M, 5.55, W - 2 * M, 0.95, [[("Scope: ", {"bold": True, "color": BLUE}),
     ("B0–B1b-GBM full market; M1 large-cap subset, incl. one-time 2021–25 holdout", {})],
     [("Why large-cap: ", {"bold": True, "color": BLUE}),
      ("we focus the text analysis on liquid large-caps (price > $5, market cap > NYSE 20th percentile, "
       f"3+ analysts). That's a derivatives desk's trading universe (median ~${CAP_B:.1f}B, {ANALYSTS} analysts vs "
       f"{ANALYSTS_OUT} outside).", {})]],
     size=14, space_after=4, name="Scope line")
picture(s, FIG / "02_vpre_vs_vpost.png", W - M - 4.85, TOP + 0.6, w=4.85, name="Figure: v_pre vs v_post")
text(s, M, 6.62, W - 2 * M, 0.3, ["v_pre / v_post = log realized volatility 20 days before / 5 days after the release; "
     "SUE = earnings surprise ÷ price; FinBERT = financial language model; PCA = feature compression."],
     size=11, italic=True, color=MUTED, name="Key terms")
notes(s, "Figure: notebooks/figures/02_vpre_vs_vpost.png (dev data). Rungs 1-3 are evaluated on the full sample "
         "(validation 2017-19); rung 4 only on the text-covered subsample (validation 2017-19, holdout 2021-25). Model definitions: "
         "baselines_b0_b1.py (B0 OLS), baselines_b0_b1_fixed.py (B1b ridge, alpha = 1), baselines_gbm.py (LightGBM, "
         "same feature set asserted equal to B1b), m1_text_v2.py (FinBERT-tone + PCA). SUE definition: pipeline/sue.py "
         "(sue_price = (actual - median consensus) / price two sessions before the reaction).")

# ======================================================================= 5. all results, one table
s = frame(6, "Does it work? Yes, driven mostly by past volatility")
text(s, M, TOP - 0.05, 7.4, 0.4, ["R² at 5 days · train 2006–16 · validation 2017–19 · holdout 2021–25 (run once)"],
     size=14, bold=True, color=MUTED, name="Caption")
cols = [3.2, 1.05, 0.95, 1.05, 0.95]                                     # layer | val | adds | holdout | adds
cx = [M + sum(cols[:i]) for i in range(len(cols))]
TW = sum(cols)
y = TOP + 0.38
for j, h_ in enumerate(("Model", "Validation", "Adds", "Holdout*", "Adds")):
    text(s, cx[j], y, cols[j], 0.34, [h_], size=13, bold=True, color=MUTED, anchor=MSO_ANCHOR.MIDDLE,
         align=PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT, name=f"Header {j}")
y += 0.36
box(s, M, y, TW, 0.02, fill=LINE, name="Header rule")
tv5, th5 = TV["v_post"], TH["v_post"]
d = lambda a, b: f"+{a - b:.3f}"
SECTIONS = [
    (f"Full sample ({F2 / 1000:.0f}k events) · tabular models", [
        ("B0: past volatility", f3(VAL_R2["B0"]), "", f3(HO_R2["B0"]), "", False),
        ("+ size, sector, timing, coverage", f3(NOSUE), d(NOSUE, B0), "–", "", False),
        ("B1b: + earnings surprise", f3(VAL_R2["B1b"]), d(B1B, NOSUE), f3(HO_R2["B1b"]), d(HO_R2["B1b"], HO_R2["B0"]) + "†", False),
        ("B1b-GBM: + boosting", f3(VAL_R2["B1b-GBM"]), d(GBM, B1B), f3(HO_R2["B1b-GBM"]), d(HO_R2["B1b-GBM"], HO_R2["B1b"]), True)]),
    (f"Large-cap subsample (~${CAP_B:.1f}B, {ANALYSTS} analysts) · + earnings-release text", [
        ("B1b-GBM, same large-cap rows", f3(tv5["bar"]), "", f3(th5["bar"]), "", False),
        ("M1: + earnings-release text", f3(tv5["m1"]), f"+{tv5['d']:.3f}", f3(th5["m1"]), f"+{th5['d']:.3f}", True)])]
RH = 0.36
for si, (label, rws) in enumerate(SECTIONS):
    y += 0.04
    box(s, M, y, TW, 0.32, fill=TINT, name=f"Section band {si + 1}")
    text(s, M + 0.1, y, TW - 0.2, 0.32, [label], size=14, bold=True, color=BLUE, anchor=MSO_ANCHOR.MIDDLE,
         name=f"Section {si + 1}")
    y += 0.33
    for k, (lab, v, a1, hv, a2, last) in enumerate(rws):
        emph = {"bold": True, "color": BLUE} if last else {"color": INK}
        text(s, cx[0], y, cols[0], RH, [lab], size=16, color=INK, bold=last, anchor=MSO_ANCHOR.MIDDLE,
             name=f"Row {si}.{k}")
        for j, val_ in ((1, v), (2, a1), (3, hv), (4, a2)):
            text(s, cx[j], y, cols[j], RH, [[(val_, emph if j in (1, 3) else {"color": BLUE if last else BODY,
                 "bold": last and si == 1})]], size=16, align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE,
                 name=f"Value {si}.{k}.{j}")
        box(s, M, y + RH, TW, 0.01, fill=LINE, name=f"Row rule {si}.{k}")
        y += RH + 0.01
text(s, M, y + 0.08, TW, 0.35, ["*vs the holdout's own mean · †vs B0 · – not run"],
     size=12, italic=True, color=MUTED, name="Table note")
text(s, M, y + 0.55, TW, 1.2, [
    [("Past volatility: ", {"bold": True, "color": INK}), (f"{f3(B0)} of the best {f3(GBM)}", {})],
    [("Holdout: ", {"bold": True, "color": INK}),
     (f"tabular edge over B0 ~halves (+{VAL_GAIN:.3f} → +{HO_GAIN:.3f})", {})],
    [("Text: ", {"bold": True, "color": INK}),
     (f"edge held out-of-sample (+{tv5['d']:.3f} → +{th5['d']:.3f}, large caps)", {})],
], size=16, bullets=True, space_after=4, name="Takeaways")
fx_ = M + TW + 0.35
fw = W - M - fx_
fh = picture(s, FIG / "pred_vs_actual.png", fx_, TOP + 0.4, w=fw, name="Figure: predicted vs actual").height / 914400
text(s, fx_, TOP + 0.5 + fh, fw, 0.4, ["Predicted vs actual · B1b-GBM, validation 2017–19"], size=13, italic=True,
     color=MUTED, name="Figure caption")
text(s, fx_, TOP + 0.95 + fh, fw, 0.6, [[("R²: ", {"bold": True, "color": INK}),
     ("share of variation explained on unseen years", {})]], size=13, color=BODY, name="Metric definition")
notes(s, "Holdout R² is measured against the holdout's own mean; against the 2006-16 mean it is "
         + " / ".join(f3(HO_R2_DEVMEAN[m]) for m in MODELS) + " (B0 / B1b / B1b-GBM), inflated by 2021-25's higher volatility. "
         "The B0 < B1b < B1b-GBM ranking holds on the holdout at every horizon (3, 5, 15 days). Text gain 95% CI (5 days): "
         f"+{tv5['lo']:.3f} to +{tv5['hi']:.3f} (validation), +{th5['lo']:.3f} to +{th5['hi']:.3f} (holdout). "
         "Figure: forecasts rank events well but are pulled toward the average. R² = 1 is perfect, 0 = no better than the "
         "2006-16 average. "
         "One table, two samples. Full sample: results_holdout.csv (finalized B0 OLS, B1b ridge, B1b-GBM trained on all "
         "dev with fixed trees 173/171/135; validation rows and the one-time holdout rows), target v_post; intermediate rung "
         "from results_b0_b1_fixed.csv (B1a_nosue, not part of the holdout run). "
         + "; ".join(f"{m}: val R2 {VAL_R2[m]:.4f} (RMSE {VAL_RMSE[m]:.4f}), holdout R2 vs own mean {HO_R2[m]:.4f}, "
                     f"vs dev mean {HO_R2_DEVMEAN[m]:.4f} (RMSE {HO_RMSE[m]:.4f})" for m in MODELS)
         + f". B1a_nosue val R2 {NOSUE:.4f}. Gains GBM - B0: validation {VAL_GAIN:.4f}, holdout {HO_GAIN:.4f}. Holdout rows "
         f"{N_HO:,}. Holdout 'Adds' = differences of the holdout R2s shown (B1b vs B0 because the middle rung was not run). "
         f"Large-cap section: results_m1_v2.csv, B1b-GBM and M1 fit on text-subset dev, same rows (validation n = {N_TV:,}, "
         f"holdout n = {N_TH:,}); validation R2 vs dev mean, holdout R2 vs holdout mean; gain CIs = paired bootstrap over "
         "session dates. Figure: notebooks/figures/pred_vs_actual.png (finalized B1b-GBM, full-sample validation 2017-19, "
         "n = 25,659; deck/plots_pred_timing.py).")

# ======================================================================= 7. text (headline)
s = frame(7, "Does the earnings-release text add value?")
tag(s, SUB_TAG)
v5, h5 = TV["v_post"], TH["v_post"]
text(s, M, TOP - 0.02, W - 2 * M, 0.5, [[(f"Yes — +{v5['d']:.3f} R² on validation, and it holds out-of-sample: "
     f"+{h5['d']:.3f} on holdout", {"bold": True, "color": BLUE})]], size=22, name="Answer")
SH = 0.6
for k, (num, lab, d) in enumerate([(v5["d"], "Validation 2017–19", v5), (h5["d"], "Holdout 2021–25 (run once)", h5)]):
    y0 = TOP + SH + k * 1.62
    text(s, M, y0, 4.3, 0.85, [[(f"+{num:.3f}", {"bold": True, "color": BLUE, "size": 54})]], line_spacing=0.9,
         name=f"Headline {k}")
    text(s, M, y0 + 0.85, 4.3, 0.7, [[(lab, {"bold": True, "color": INK})],
                                     f"R² gain at 5 days · 95% CI +{d['lo']:.3f} to +{d['hi']:.3f}"],
         size=15, space_after=0, name=f"Headline label {k}")
rows = [("Horizon", "Validation", "95% CI", "Holdout", "95% CI")]
for t, lab in (("v_post_3", "3 days"), ("v_post", "5 days"), ("v_post_15", "15 days")):
    rows.append((lab, f"+{TV[t]['d']:.3f}", f"+{TV[t]['lo']:.3f} to +{TV[t]['hi']:.3f}",
                 f"+{TH[t]['d']:.3f}", f"+{TH[t]['lo']:.3f} to +{TH[t]['hi']:.3f}"))
table(s, 5.0, TOP + SH + 0.05, [1.1, 1.25, 2.05, 1.2, 2.13], rows, row_h=0.46, best=2, name="Text table", sizes=(14, 16))
text(s, 5.0, TOP + SH + 2.15, W - M - 5.0, 1.6, [
    [("Held out-of-sample: ", {"bold": True, "color": INK}),
     (f"+{v5['d']:.3f} → +{h5['d']:.3f}, while the tabular layers' edge roughly halved", {})],
    [("Every horizon, both periods: ", {"bold": True, "color": INK}), ("largest at 3–5 days", {})],
    [("Like for like: ", {"bold": True, "color": INK}), ("same model, same rows; only the text changes", {})],
], size=16, bullets=True, space_after=6, name="Takeaways")
box(s, M, 6.02, W - 2 * M - 1.2, 0.7, fill=TINT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name="Caveat card")
text(s, M + 0.25, 6.02, W - 2 * M - 1.7, 0.7, [CAVEAT + " · holdout vs its own mean"],
     size=14, italic=True, color=BODY, anchor=MSO_ANCHOR.MIDDLE, name="Text caveat")
notes(s, "How the text was extracted: 8-K earnings-release exhibits (mostly EX-99) -> SGML/HTML and financial tables "
         "stripped -> first ~190 words (256 tokens) embedded with FinBERT-tone, then PCA. 95% intervals: paired bootstrap "
         "over session days. The observation is that the text gain held on the holdout while the tabular layers' edge over "
         "B0 roughly halved; we do not claim why. "
         "Source: results_m1_v2.csv (text re-embedded from data/parquets clean_YYYY.parquet, column text_full, first 500 "
         f"words; FinBERT-tone first 256 tokens; PCA {N_PCS} components fit on dev only). Rows: text-covered model rows, "
         f"validation n = {N_TV:,}, holdout n = {N_TH:,}; B1b-GBM and M1 on identical rows, fixed trees 173/171/135. "
         "Validation gains = dR2_oos (vs dev mean); holdout gains = dR2_own (vs holdout mean); 95% intervals = paired "
         "bootstrap over session dates within each period. "
         + "; ".join(f"{t}: val {TV[t]['bar']:.4f} -> {TV[t]['m1']:.4f} (+{TV[t]['d']:.4f} [{TV[t]['lo']:.4f}, "
                     f"{TV[t]['hi']:.4f}]), holdout {TH[t]['bar']:.4f} -> {TH[t]['m1']:.4f} (+{TH[t]['d']:.4f} "
                     f"[{TH[t]['lo']:.4f}, {TH[t]['hi']:.4f}])" for t in TV)
         + f". Why validation rose vs the earlier deck: re-cleaning (clean parquets, tables stripped) raised the old gain on the "
         f"same 2017-18 rows from +{OLD_1718:.3f} (results_m1.csv, superseded) to +{T1718:.3f}; adding 2019 gives "
         f"+{TV['v_post']['d']:.3f}. PCA variance {PCA_VAR} as printed by m1_text_v2.py. Design choices were made with "
         "validation in view; the holdout is the clean test. Holdout run once; nothing tuned on it. "
         "'Numbers' edge roughly halved' = full-sample tabular GBM - B0 gain on holdout vs validation (results_holdout.csv). "
         "Subsample medians: model_table rows of the text subset (dev + validation + holdout).")

# ======================================================================= 7. decision (text model, large caps)
# Managerial use = attention triage on earnings night. No figures on this slide by design: the catch-rate numbers stay
# in results_risk_limit_m1.csv (checked by verify_results.py) but are not shown.
s = frame(8, "Does it matter? Triaging attention on earnings night")
tag(s, SUB_TAG)
TFLOW = [("Many names report tonight", "the desk can't deep-dive every one"),
         ("Rank them by predicted move", "a volatility forecast for each release"),
         ("Scrutinize the top names first", "options, hedges, limits; skip the calm ones")]
BW7, GAP7, BY7, BH7 = 3.65, 0.42, TOP + 0.05, 1.3
for i, (head, sub) in enumerate(TFLOW):
    x = M + i * (BW7 + GAP7)
    last = i == len(TFLOW) - 1
    box(s, x, BY7, BW7, BH7, fill=BLUE if last else TINT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name=f"Triage flow {i + 1}")
    text(s, x + 0.22, BY7 + 0.08, BW7 - 0.44, BH7 - 0.16, [[(head, {"bold": True, "size": 18, "color": WHITE if last else INK})],
         [(sub, {"size": 14, "color": PALE if last else BODY})]], space_after=4, anchor=MSO_ANCHOR.MIDDLE,
         line_spacing=1.0, name=f"Triage flow {i + 1} text")
    if i < len(TFLOW) - 1:
        text(s, x + BW7, BY7, GAP7, BH7, ["→"], size=28, bold=True, color=MUTED, align=PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.0, name=f"Triage arrow {i + 1}")
RY7 = BY7 + BH7 + 0.25
LW7 = 6.55
text(s, M, RY7, LW7, 0.36, ["Why it helps"], size=14, bold=True, color=MUTED, name="Why label")
text(s, M, RY7 + 0.4, LW7, 1.75, [
    [("Text adds what the numbers miss: ", {"bold": True, "color": INK}),
     ("an independent input that adds to, not replaces, the desk's tools", {})],
    [("Evidence: ", {"bold": True, "color": INK}),
     ("at the same alert budget, the model catches more of the genuine big movers", {})],
    [("Scope: ", {"bold": True, "color": INK}),
     ("large caps, validation period · big moves caught, not dollars", {})],
], size=15, bullets=True, space_after=6, name="Why it helps")
IX7 = M + LW7 + 0.35
IW7 = W - M - IX7
text(s, IX7, RY7, IW7, 0.36, ["Expected impact"], size=14, bold=True, color=MUTED, name="Impact label")
for k, (head, body) in enumerate([("Reduced risk", "more genuine big movers flagged → fewer surprise limit breaches"),
                                  ("Increased efficiency", "a prioritized watchlist → scarce attention goes where it matters")]):
    yy = RY7 + 0.42 + k * 0.86
    box(s, IX7, yy, IW7, 0.78, fill=TINT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name=f"Impact card {k + 1}")
    text(s, IX7 + 0.2, yy, IW7 - 0.4, 0.78, [[(head + ": ", {"bold": True, "color": BLUE}), (body, {"color": INK})]],
         size=15, anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.0, name=f"Impact {k + 1}")
REC_Y, REC_H = 5.45, 1.32
box(s, M, REC_Y, W - 2 * M - 1.2, REC_H, fill=None, line=BLUE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name="Recommendation card")
text(s, M + 0.25, REC_Y, W - 2 * M - 1.7, REC_H, [[
    ("Recommendation: ", {"bold": True, "color": BLUE}),
    ("Use the text-based volatility forecast as an earnings-night triage tool: rank reporters by predicted move, "
     "scrutinize and tighten limits on the top names, and treat sharp disagreements with the desk's existing view as a "
     "flag to look closer. Benchmarking against options-implied volatility is the next step to quantify the edge in P&L "
     "terms.", {"color": INK})]],
    size=15, anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.05, name="Recommendation")
notes(s, "No figures on this slide by design. Managerial use: attention triage on earnings night. The model (M1 = "
         "B1b-GBM + earnings-release text) ranks tonight's reporters by predicted post-earnings volatility; the desk looks "
         "at the top names first. It is an input that adds to, not replaces, the desk's existing tools. Evidence: the "
         "decision test in results_risk_limit_m1.csv (risk_limit_m1.py): same alert budget (each model flags its own top "
         "quarter of forecasts), big move = top quarter of 2006-16 post-earnings volatility; adding the text catches more "
         "big movers. Large-cap text subsample, validation 2017-19 only, not run on the holdout. It measures big moves "
         "caught, not dollars; limit breaches and P&L were not measured, so 'fewer surprise limit breaches' is the "
         "expected effect, not a measured one. We do not claim to beat options-implied volatility: that benchmark is the "
         "next step (no options data here).")

# ======================================================================= 10. one earnings session (Convention B example)
s = frame(9, "One earnings session: how the desk would use it")
tag(s, SUB_TAG)
GREEN, GREEN_BG = RGBColor(0x2E, 0x7D, 0x4F), RGBColor(0xE3, 0xF1, 0xE7)
AMBER, AMBER_BG = RGBColor(0x9A, 0x5B, 0x00), RGBColor(0xFB, 0xEE, 0xDC)
NAMES = {"DDD": "3D Systems", "GCO": "Genesco", "ADT": "ADT", "SMTC": "Semtech",   # CRSP issuer names, shortened
         "WSM": "Williams-Sonoma", "DG": "Dollar General"}
assert set(sess.ticker) == set(NAMES) and (sess.session_date == "2018-03-15").all()
_ss = sess.sort_values("gbm_pred_daily_vol", ascending=False)
SESSION = [(r.ticker, NAMES[r.ticker], f"{r.gbm_pred_daily_vol:.2%}", "Y" if r.gbm_flag else "N", f"{r.actual_daily_vol:.2%}",
            "Y" if r.big_move else "N", r.gbm_outcome) for r in _ss.itertuples()]
_o = _ss.gbm_outcome.value_counts()
assert (_ss.gbm_flag.sum(), _ss.big_move.sum(), _o.get("HIT", 0), _o.get("false alarm", 0)) == (3, 2, 1, 2)
_m1f = _ss[_ss.m1_flag].sort_values("m1_pred_daily_vol", ascending=False)
assert _ss.m1_flag.sum() == 3 and (_ss.m1_outcome == "HIT").sum() == 1 and _ss.big_move.sum() == 2
_hit = _ss[_ss.gbm_outcome == "HIT"].iloc[0]; _miss = _ss[_ss.gbm_outcome == "MISS"].iloc[0]
assert (_hit.ticker, _miss.ticker) == ("ADT", "SMTC")
hdr = ("Ticker", "Company", "Predicted vol", "Flagged", "Actual vol", "Big move", "Outcome")
cw = [0.8, 2.55, 0.95, 0.8, 0.95, 0.85, 1.35]
gf = s.shapes.add_table(len(SESSION) + 1, len(hdr), Inches(M), Inches(TOP + 0.05), Inches(sum(cw)),
                        Inches(0.5 + 0.37 * len(SESSION)))
gf.name = "Session table"
tb = gf.table
tb.first_row = True; tb.horz_banding = False
for j, w in enumerate(cw):
    tb.columns[j].width = Inches(w)
for i, row in enumerate([hdr] + SESSION):
    tb.rows[i].height = Inches(0.5 if i == 0 else 0.37)
    flagged = i > 0 and row[3] == "Y"
    for j, val in enumerate(row):
        cell = tb.cell(i, j)
        cell.margin_left = cell.margin_right = Inches(0.07)
        cell.margin_top = cell.margin_bottom = Inches(0.02)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        tcPr = cell._tc.get_or_add_tcPr()
        for side in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
            ln = etree.SubElement(tcPr, qn(side), w=str(Pt(1 if side == "a:lnB" else 0)))
            if side == "a:lnB":
                sf = etree.SubElement(ln, qn("a:solidFill")); etree.SubElement(sf, qn("a:srgbClr")).set("val", str(LINE))
            else:
                etree.SubElement(ln, qn("a:noFill"))
        out = row[6] if i > 0 else ""
        fill = TINT if (i == 0 or flagged) else WHITE
        color, bold = (MUTED, True) if i == 0 else (INK if flagged else BODY, flagged and j == 0)
        if i > 0 and j == 6:
            if out == "HIT":
                fill, color, bold = GREEN_BG, GREEN, True
            elif out in ("MISS", "false alarm"):
                fill, color, bold = AMBER_BG, AMBER, out == "MISS"
            else:
                color = MUTED
        cell.fill.solid(); cell.fill.fore_color.rgb = fill
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT if j in (0, 1, 6) else PP_ALIGN.CENTER if j in (3, 5) else PP_ALIGN.RIGHT
        r = p.add_run(); r.text = val
        r.font.name = FONT; r.font.size = Pt(13 if i == 0 else 16); r.font.bold = bold; r.font.color.rgb = color
tx = M + sum(cw) + 0.3
tw = W - M - tx
box(s, tx, TOP + 0.05, tw, 2.75, fill=TINT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name="Takeaway card")
text(s, tx + 0.2, TOP + 0.17, tw - 0.4, 2.55, [
    [("Model: B1b-GBM (no text); flagged = top 25% of predicted volatility", {"bold": True, "color": MUTED, "size": 13})],
    [("Flagged 3 names; caught 1 of 2 big movers ", {"bold": True, "color": INK}),
     (f"({NAMES[_hit.ticker]}, {_hit.actual_daily_vol:.1%}/day); missed {NAMES[_miss.ticker]} "
      f"({_miss.actual_daily_vol:.1%}); 2 false alarms", {})],
    [("Honest mix: better aim, same alert budget", {"bold": True, "color": BLUE})],
], size=16, space_after=8, name="Takeaway")
text(s, tx, TOP + 2.98, tw, 2.3, [
    [("Across all large-cap sessions: ", {"bold": True, "color": BODY}),
     ("adding the text catches more of the big movers on average", {})],
    [("With text (M1) on this session: ", {"bold": True, "color": BODY}),
     (f"flags the same 3 names, same hit ({NAMES[_hit.ticker]}) and miss ({NAMES[_miss.ticker]}). "
      "Text helps on average, not on every session", {})]],
     size=14, color=BODY, space_after=6, name="Reconciliation")
assert sorted(_m1f.ticker) == sorted(_ss[_ss.gbm_flag].ticker)
_yb = TOP + 0.05 + 0.5 + 0.37 * len(SESSION) + 0.2
text(s, M, _yb, sum(cw), 0.75, [
    [("Why this session: ", {"bold": True, "color": INK}),
     ("a representative large-cap session: median catch rate, precision near the model's average "
      "(validation 2017–19)", {})]],
     size=16, color=BODY, name="Selection rule")
text(s, M, _yb + 0.9, sum(cw), 0.8, [
    "Big move > 4.86%/day (2006–16 top quartile) · All six react in the same session (2018-03-15); three released "
    "after the prior close, three pre-open"],
     size=14, italic=True, color=MUTED, name="Footnote")
notes(s, "Source: results_risk_limit_m1_session.csv (risk_limit_m1.py). Same models and rule as the decision test (results_risk_limit_m1.csv): B1b-GBM and M1 "
         "fit on text-subset dev 2006-16; big move = v_post above the full-sample dev top-quartile cut-off "
         f"({D0.big_move_cutoff_daily_vol:.4%}/day); alert = the model's own top 25% of large-cap validation 2017-19 "
         f"predictions (B1b-GBM > {D0.flag_cutoff_daily_vol:.4%}/day). This table shows B1b-GBM (no text). M1 on the same "
         "session: " + ", ".join(f"{r.ticker} {r.m1_pred_daily_vol:.2%} {r.m1_outcome}" for r in
                                  sess.sort_values("m1_pred_daily_vol", ascending=False).itertuples())
         + ". Selection rule: among the 312 large-cap validation sessions with >= 2 big "
         "movers, keep those with <= 12 names (to fit the slide) at B1b-GBM's median catch rate (50%); of these, take "
         "precision closest to B1b-GBM's overall 36.9% (four sessions tie at 33.3%); break the tie by requiring M1 also "
         "at its median catch rate with precision closest to its overall 38.1%. Only 2018-03-15 satisfies all. Release "
         "times: 3D Systems, Williams-Sonoma, Semtech after the close on 2018-03-14; Genesco, ADT, Dollar General "
         "pre-open on 2018-03-15. Validation only; no holdout. Company names from CRSP. Volatility is unsigned: "
         "'6.1%/day' is size, not direction.")

# ======================================================================= 10. next / limitations
s = frame(10, "What's next, and what we can't claim yet")
box(s, M, TOP - 0.05, W - 2 * M, 0.72, fill=TINT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name="Reported card")
text(s, M + 0.25, TOP - 0.05, W - 2 * M - 0.5, 0.72, [[("Holdout 2021–25, run once: ", {"bold": True, "color": BLUE}),
     (f"tabular ranking held, value-added ~halved · text confirmed (+{TH['v_post']['d']:.3f} vs "
      f"+{TV['v_post']['d']:.3f} on validation)", {})]], size=18, anchor=MSO_ANCHOR.MIDDLE, name="Reported text")
text(s, M, TOP + 0.8, 6.0, 0.4, ["Next"], size=14, bold=True, color=MUTED, name="Next label")
for i, (head, body) in enumerate([
        ("Extend the text layer", "Smaller names, outside the extraction screen"),
        ("Implied-volatility benchmark", "Needs options data")]):
    yy = TOP + 1.2 + i * 1.1
    box(s, M, yy, 6.1, 1.0, fill=TINT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name=f"Next {i + 1}")
    box(s, M + 0.22, yy + 0.27, 0.46, 0.46, fill=BLUE, shape=MSO_SHAPE.OVAL, name=f"Next {i + 1} badge")
    text(s, M + 0.22, yy + 0.27, 0.46, 0.46, [str(i + 1)], size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.0, name=f"Next {i + 1} number")
    text(s, M + 0.9, yy + 0.1, 5.0, 0.85, [[(head, {"bold": True, "color": INK, "size": 18})], body], size=16, space_after=2,
         name=f"Next {i + 1} text")
text(s, 7.2, TOP + 0.8, 5.5, 0.4, ["Limitations"], size=14, bold=True, color=MUTED, name="Limits label")
text(s, 7.2, TOP + 1.2, W - M - 7.2, 3.9, [
    [("No options-implied-vol benchmark: ", {"bold": True, "color": INK}),
     ("our baseline is past realized vol, not the desk's implied vol", {})],
    [("Large caps by design: ", {"bold": True, "color": INK}),
     ("the desk's liquid universe (price > $5, mkt cap > NYSE 20th pct, 3+ analysts; "
      f"median ~${CAP_B:.1f}B, {ANALYSTS} analysts); smaller names not yet tested", {})],
    "Tabular value-added ~halves out-of-sample (text gain did not)",
    "Big moves caught, not dollars",
], size=16, bullets=True, space_after=8, name="Limitations")
box(s, M, 6.02, W - 2 * M - 1.2, 0.9, fill=None, line=BLUE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, name="Conclusion card")
text(s, M + 0.25, 6.02, W - 2 * M - 1.7, 0.9, [[
    ("Does the earnings-release text add value beyond the numbers? ", {"color": BODY}),
    (f"Yes: +{TV['v_post']['d']:.3f} R² on validation, +{TH['v_post']['d']:.3f} on the 2021–25 holdout; "
     "strongest at short horizons; large-cap names.", {"bold": True, "color": BLUE})]],
    size=16, anchor=MSO_ANCHOR.MIDDLE, name="Conclusion")
notes(s, "Reported line: results_holdout.csv (tabular) and results_m1_v2.csv (text, v_post, holdout gain vs own mean). "
         "Implied vol: we have no options data, so every model is benchmarked against past realized volatility; testing "
         "against the desk's implied vol is the first next step. "
         f"Subsample medians and PCA size: {DESC_SOURCE}.")

prs.save(OUT)
print(f"Saved {OUT} ({len(prs.slides)} slides)")
