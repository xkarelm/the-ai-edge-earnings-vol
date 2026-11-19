"""Check every headline number in the pitch deck against the committed results files. No WRDS, no downloads.

    python verify_results.py

Each line shows a deck claim exactly as printed on the slide, the value recomputed from data/processed/, and
OK or MISMATCH. The script exits with status 1 if any claim does not match. It reads only the small CSVs and build
log committed in data/processed/ (pandas is the only dependency).

Not checkable here, because they need the licensed model table or the text features (see deck/build_pptx.py):
the subsample medians (~$3.7B market cap, 7 analysts vs 2) and the PCA size (48 components, 90.1% of variance).
"""
import re
import sys
from pathlib import Path

import pandas as pd

P = Path(__file__).resolve().parent / "data" / "processed"
M1N = "M1 = B1b-GBM + text"
fails = []


def check(slide, claim, shown, value):
    ok = shown == value
    if not ok:
        fails.append(claim)
    print(f"  {'OK ' if ok else 'MISMATCH'}  slide {slide:<2} {claim:<62} deck {shown:<22} files {value}")


f3 = lambda x: f"{x:.3f}"
pct = lambda x: f"{x:.1%}"

# ------------------------------------------------------------------ data funnel (build_log.txt)
print("\nData funnel   (build_log.txt)")
log = (P / "build_log.txt").read_text()
fun = lambda pat: f"{int(re.search(pat, log).group(1).replace(',', '')):,}"
check(3, "8-K earnings releases", "383,153", fun(r"\[\s*([\d,]+)\] 8-K earnings events in event_table"))
check(3, "linked to CRSP, complete return windows", "299,379", fun(r"\[\s*([\d,]+)\] v_post_15: all 15 post returns present"))
check(3, "primary sample (SUE, 8-K and I/B/E/S agree)", "168,906", fun(r"\[\s*([\d,]+)\] primary: has SUE"))
check(3, "reaction dates corrected", "75,359", fun(r"stored session_date differs from recomputed for ([\d,]+) events"))

# ------------------------------------------------------------------ tabular ladder (results_holdout.csv)
print("\nTabular models, full sample, 5-day R²   (results_holdout.csv, results_b0_b1_fixed.csv)")
ho = pd.read_csv(P / "results_holdout.csv")
V = ho[ho.split == "validation"].set_index(["target", "model"])
H = ho[ho.split == "holdout"].set_index(["target", "model"])
fx = pd.read_csv(P / "results_b0_b1_fixed.csv").set_index(["target", "model"])
v = {m: V.loc[("v_post", m), "R2_oos"] for m in ("B0", "B1b", "B1b-GBM")}
h = {m: H.loc[("v_post", m), "R2_vs_own_mean"] for m in ("B0", "B1b", "B1b-GBM")}
hd = {m: H.loc[("v_post", m), "R2_oos"] for m in ("B0", "B1b", "B1b-GBM")}
nosue = fx.loc[("v_post", "B1a_nosue"), "R2_oos"]
check(5, "B0 validation R²", "0.226", f3(v["B0"]))
check(5, "B0 holdout R² (vs holdout mean)", "0.268", f3(h["B0"]))
check(5, "+ size, sector, timing, coverage: validation R²", "0.265", f3(nosue))
check(5, "  ... adds", "+0.039", f"+{nosue - v['B0']:.3f}")
check(5, "B1b validation R²", "0.278", f3(v["B1b"]))
check(5, "  ... adds (surprise)", "+0.013", f"+{v['B1b'] - nosue:.3f}")
check(5, "B1b holdout R²", "0.293", f3(h["B1b"]))
check(5, "  ... holdout adds vs B0", "+0.025", f"+{h['B1b'] - h['B0']:.3f}")
check(5, "B1b-GBM validation R²", "0.313", f3(v["B1b-GBM"]))
check(5, "  ... adds (boosting)", "+0.035", f"+{v['B1b-GBM'] - v['B1b']:.3f}")
check(5, "B1b-GBM holdout R²", "0.311", f3(h["B1b-GBM"]))
check(5, "  ... holdout adds vs B1b", "+0.018", f"+{h['B1b-GBM'] - h['B1b']:.3f}")
check(5, "holdout R² vs 2006-16 mean (B0 / B1b / B1b-GBM)", "0.323 / 0.345 / 0.362",
      " / ".join(f3(hd[m]) for m in ("B0", "B1b", "B1b-GBM")))
check(5, "GBM edge over B0, validation", "+0.087", f"+{v['B1b-GBM'] - v['B0']:.3f}")
check(5, "GBM edge over B0, holdout (roughly halves)", "+0.043", f"+{h['B1b-GBM'] - h['B0']:.3f}")
rank = all(H.loc[(t, "B0"), "R2_vs_own_mean"] < H.loc[(t, "B1b"), "R2_vs_own_mean"] < H.loc[(t, "B1b-GBM"), "R2_vs_own_mean"]
           for t in ("v_post_3", "v_post", "v_post_15"))
check(5, "ranking B0 < B1b < B1b-GBM holds on holdout, every horizon", "yes", "yes" if rank else "no")

# ------------------------------------------------------------------ text model (results_m1_v2.csv)
print("\nText model M1 vs B1b-GBM, large-cap subsample   (results_m1_v2.csv)")
m = pd.read_csv(P / "results_m1_v2.csv")
row = lambda ev, t, mod: m[(m["eval"] == ev) & (m.target == t) & (m.model == mod)].iloc[0]
VAL, HOL = "validation 2017-19", "holdout 2021-25"
check(5, "large-cap B1b-GBM validation / holdout R²", "0.263 / 0.257",
      f"{f3(row(VAL, 'v_post', 'B1b-GBM').R2_oos)} / {f3(row(HOL, 'v_post', 'B1b-GBM').R2_vs_own_mean)}")
check(5, "M1 validation / holdout R²", "0.287 / 0.295",
      f"{f3(row(VAL, 'v_post', M1N).R2_oos)} / {f3(row(HOL, 'v_post', M1N).R2_vs_own_mean)}")
deck_gain = {"v_post_3": ("3 days", "+0.025", "+0.020 to +0.031", "+0.037", "+0.033 to +0.041"),
             "v_post": ("5 days", "+0.024", "+0.019 to +0.029", "+0.038", "+0.034 to +0.042"),
             "v_post_15": ("15 days", "+0.020", "+0.015 to +0.025", "+0.022", "+0.019 to +0.026")}
for t, (lab, gv, civ, gh, cih) in deck_gain.items():
    a, b = row(VAL, t, M1N), row(HOL, t, M1N)
    check(6, f"{lab}: text gain, validation", gv, f"+{a.dR2_oos:.3f}")
    check(6, f"{lab}: 95% CI, validation", civ, f"+{a.dR2_ci_lo:.3f} to +{a.dR2_ci_hi:.3f}")
    check(6, f"{lab}: text gain, holdout (vs holdout mean)", gh, f"+{b.dR2_own:.3f}")
    check(6, f"{lab}: 95% CI, holdout", cih, f"+{b.dR2_own_ci_lo:.3f} to +{b.dR2_own_ci_hi:.3f}")
lo_pos = all(r.dR2_ci_lo > 0 and r.dR2_own_ci_lo > 0 for r in m[m.model == M1N].itertuples() if r.eval != "validation 2017-18")
check(6, "every text-gain interval above zero (both periods, 3 horizons)", "yes", "yes" if lo_pos else "no")
b5 = row(HOL, "v_post", M1N)
check(6, f"(extra) holdout gain vs 2006-16 mean also > 0: +{b5.dR2_oos:.3f}", "yes", "yes" if b5.dR2_ci_lo > 0 else "no")

# ------------------------------------------------------------------ decision (results_risk_limit*.csv)
print("\nRisk-limit decision, validation 2017-19   (results_risk_limit_m1.csv, results_risk_limit.csv)")
d = pd.read_csv(P / "results_risk_limit_m1.csv").set_index("model")
g0, g1, gg = d.loc["B1b-GBM"], d.loc[M1N], d.loc["gain: M1 - B1b-GBM"]
check(7, "big move cut-off (2006-16 top quartile)", "4.86%", f"{g0.big_move_cutoff_daily_vol:.2%}")
check(7, "large-cap events scored", "14,614", f"{int(g0.n):,}")
check(7, "big moves caught: B1b-GBM (no text)", "53.7%", pct(g0.recall))
check(7, "big moves caught: M1 (+ text)", "55.3%", pct(g1.recall))
check(7, "gain from text", "+1.6 pts", f"+{100 * gg.recall:.1f} pts")
check(7, "gain 95% CI", "+0.4 to +2.9 pts", f"+{100 * gg.recall_ci_lo:.1f} to +{100 * gg.recall_ci_hi:.1f} pts")
check(7, "extra big moves caught per 100 events", "+0.28", f"+{gg.extra_TP_per_100_events:.2f}")
check(7, "precision B1b-GBM -> M1", "36.9% -> 38.1%", f"{pct(g0.precision)} -> {pct(g1.precision)}")
rl = pd.read_csv(P / "results_risk_limit.csv")
eq = rl[rl.rule.str.startswith("equal") & (rl.level == "top quartile")].set_index("model")
n_val = int(eq.loc["B0", ["TP", "FP", "FN", "TN"]].sum())
extra = int(eq.loc["B1b", "TP"] - eq.loc["B0", "TP"])
check(7, "full sample, no text: big moves caught B0 -> B1b", "47.5% -> 49.4%",
      f"{pct(eq.loc['B0', 'recall'])} -> {pct(eq.loc['B1b', 'recall'])}")
check(7, "  ... extra per 100 events", "+0.46", f"+{100 * extra / n_val:.2f}")
check(7, "  ... extra per quarter (12 validation quarters)", "~10", f"~{extra / 12:.0f}")

# ------------------------------------------------------------------ one session (results_risk_limit_m1_session.csv)
print("\nOne session, 2017-07-24   (results_risk_limit_m1_session.csv)")
s = pd.read_csv(P / "results_risk_limit_m1_session.csv")
o = s.gbm_outcome.value_counts()
check(8, "B1b-GBM: flagged / big movers / caught / false alarms", "3 / 2 / 1 / 2",
      f"{int(s.gbm_flag.sum())} / {int(s.big_move.sum())} / {o.get('HIT', 0)} / {o.get('false alarm', 0)}")
check(8, "Scientific Games realized vol (the hit)", "13.29%", f"{s.set_index('ticker').loc['SGMS', 'actual_daily_vol']:.2%}")
check(8, "M1 on the same session: flagged / caught", "2 / 0",
      f"{int(s.m1_flag.sum())} / {int((s.m1_outcome == 'HIT').sum())}")

print(f"\n{'ALL CLAIMS MATCH' if not fails else f'{len(fails)} MISMATCH(ES): ' + '; '.join(fails)}")
print("Not checked here (need licensed data): subsample medians ~$3.7B / 7 analysts / 2 outside; PCA 48 components, 90.1%.")
sys.exit(1 if fails else 0)
