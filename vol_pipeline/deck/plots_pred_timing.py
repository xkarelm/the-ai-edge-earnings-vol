"""Two deck figures. Plotting only: no tuning, no holdout.

  notebooks/figures/pred_vs_actual.png         finalized B1b-GBM, full-sample VALIDATION (2017-19)
  notebooks/figures/vol_by_release_timing.png  mean realized v_post by release timing, DEV (2006-16)

Predictions are not cached, so the finalized B1b-GBM (all dev, fixed trees) is refit with baselines_holdout.fit_finalized
and must reproduce the stored validation R2 in results_holdout.csv before anything is plotted. Only dev and validation
are loaded.

    python deck/plots_pred_timing.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from baselines_b0_b1 import metrics
from baselines_b0_b1_fixed import add_surprise_features
from baselines_gbm import CATS, FEATURES
from baselines_holdout import fit_finalized, load

FIG = C.ROOT / "notebooks" / "figures"
BLUE, INK, BODY, MUTED, GRID = "#1E4B8F", "#1F1F1F", "#404040", "#7A7A7A", "#E3E5E9"
plt.rcParams.update({
    "font.family": "Arial", "font.size": 13, "axes.titlesize": 18, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 30, "axes.labelsize": 13, "axes.labelcolor": BODY, "axes.edgecolor": MUTED, "text.color": INK,
    "xtick.color": BODY, "ytick.color": BODY, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "savefig.dpi": 220, "savefig.bbox": "tight",
    "figure.facecolor": "white", "text.parse_math": False,
})


def subtitle(ax, s):
    ax.text(0, 1.015, s, transform=ax.transAxes, color=BODY, fontsize=12.5, va="bottom")


def pct(x):
    return f"{np.exp(x):.1%}"


# ------------------------------------------------------------------ data: dev + validation only
df, _ = load(["dev", "validation"])
assert set(df["split"]) == {"dev", "validation"}
(dev, val), _ = add_surprise_features(df[df.split == "dev"], df[df.split == "validation"])
models, cats, trees = fit_finalized(dev)
gv = val.copy()
for c in CATS:
    gv[c] = pd.Categorical(gv[c], categories=cats[c])
pred = models["v_post"]["B1b-GBM"].predict(gv[FEATURES])
y = val["v_post"].to_numpy()
r = metrics(y, pred, dev["v_post"].mean())
stored = pd.read_csv(C.OUT / "results_holdout.csv").query("split == 'validation' and target == 'v_post' and model == 'B1b-GBM'")
assert abs(r["R2_oos"] - stored.R2_oos.iloc[0]) < 1e-9, "refit does not reproduce the finalized validation R2: STOP"

# ------------------------------------------------------------------ plot 1: predicted vs actual
fig, ax = plt.subplots(figsize=(10, 5.5))
lims = (-6.2, -1.2)
hb = ax.hexbin(pred, y, gridsize=55, extent=(*lims, *lims), cmap="Blues", mincnt=1, bins="log", linewidths=0)
ax.plot(lims, lims, color=INK, lw=1.2, ls="--")
ax.annotate("perfect prediction", xy=(-1.75, -1.75), xytext=(-6, -16), textcoords="offset points", ha="right",
            color=BODY, fontsize=11.5)
b = pd.DataFrame({"p": pred, "y": y}).assign(bin=lambda d: pd.qcut(d.p, 20)).groupby("bin", observed=True).mean()
ax.plot(b.p, b.y, color="#E0782F", lw=2.2, marker="o", ms=5, label="average actual, by prediction bin (20 bins)")
ax.set_xlim(lims); ax.set_ylim(lims)
ticks = [-6, -5, -4, -3, -2]
ax.set_xticks(ticks, [f"{t}\n({pct(t)})" for t in ticks])
ax.set_yticks(ticks, [f"{t} ({pct(t)})" for t in ticks])
ax.set_xlabel("Predicted 5-day volatility: log realized vol (daily vol in brackets)")
ax.set_ylabel("Actual 5-day volatility")
ax.legend(loc="upper left", frameon=False, fontsize=11.5)
cb = fig.colorbar(hb, ax=ax, pad=0.02, fraction=0.04)
cb.set_label("events per cell (log scale)", color=BODY); cb.outline.set_visible(False)
cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
cb.ax.yaxis.set_minor_formatter(plt.NullFormatter())
ax.set_title(f"The model tracks realized volatility (R² = {r['R2_oos']:.3f})")
subtitle(ax, f"B1b-GBM (gradient boosting, tabular features) · FULL SAMPLE validation 2017–19 · n = {len(y):,} events")
fig.savefig(FIG / "pred_vs_actual.png")
plt.close(fig)

# ------------------------------------------------------------------ plot 2: realized vol by release timing (dev)
d_all = pd.read_parquet(C.OUT_MODEL, columns=["split", "session_date", "when", "v_post"], filters=[("split", "==", "dev")])
assert set(d_all.split) == {"dev"} and d_all.session_date.dt.year.between(2006, 2016).all()
order = ["pre_open", "intraday", "after_close"]
g = d_all.groupby("when")["v_post"].agg(["mean", "sem", "size"]).reindex(order)
g["daily_vol"] = np.exp(g["mean"])
g["lo"], g["hi"] = np.exp(g["mean"] - 1.96 * g["sem"]), np.exp(g["mean"] + 1.96 * g["sem"])
top = g["daily_vol"].idxmax()
names = {"pre_open": "Before the open", "intraday": "During trading", "after_close": "After the close"}
title = f"{names[top].replace('the ', '').capitalize()} releases are followed by the largest moves"
title = {"after_close": "After-close releases are followed by larger moves",
         "pre_open": "Pre-open releases are followed by larger moves",
         "intraday": "Intraday releases are followed by larger moves"}[top]

fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(order))
colors = [BLUE if w == top else "#9AA3B2" for w in order]
ax.bar(x, g["daily_vol"], width=0.58, color=colors, edgecolor="white", linewidth=2,
       yerr=[g["daily_vol"] - g["lo"], g["hi"] - g["daily_vol"]], error_kw={"ecolor": BODY, "capsize": 5, "lw": 1.2})
for xi, w in zip(x, order):
    ax.text(xi, g.loc[w, "hi"] + 0.0006, f"{g.loc[w, 'daily_vol']:.2%}", ha="center", va="bottom", fontsize=16,
            fontweight="bold", color=INK)
ax.set_xticks(x, [f"{names[w]}\n({w.replace('_', '-')}) · n = {g.loc[w, 'size']:,}" for w in order])
ax.yaxis.set_major_locator(plt.MultipleLocator(0.01))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_ylim(0, g["hi"].max() * 1.22)
ax.grid(axis="x", visible=False)
ax.set_ylabel("Typical daily volatility over the 5 days after the release")
ax.set_title(title)
subtitle(ax, f"Dev split 2006–16 · typical = geometric mean (exp of mean log realized vol) · 95% intervals · "
             f"intraday is small (n = {g.loc['intraday', 'size']:,})")
fig.savefig(FIG / "vol_by_release_timing.png")
plt.close(fig)

# ------------------------------------------------------------------ numbers behind the plots
print("PLOT 1  pred_vs_actual.png")
print(f"  model: finalized B1b-GBM (trained on all {len(dev):,} dev rows, fixed {trees['v_post']} trees)")
print(f"  sample: FULL-SAMPLE validation 2017-19, n = {len(y):,} (no holdout, no subset)")
print(f"  R2 (vs dev mean) {r['R2_oos']:.4f} | stored {stored.R2_oos.iloc[0]:.4f} | MAE {r['MAE']:.4f} | RMSE {r['RMSE']:.4f}")
print(f"  corr(pred, actual) {np.corrcoef(pred, y)[0, 1]:.4f}")
print(f"  predicted v_post: min {pred.min():.3f}, p5 {np.percentile(pred, 5):.3f}, median {np.median(pred):.3f}, "
      f"p95 {np.percentile(pred, 95):.3f}, max {pred.max():.3f}")
print(f"  actual v_post:    min {y.min():.3f}, p5 {np.percentile(y, 5):.3f}, median {np.median(y):.3f}, "
      f"p95 {np.percentile(y, 95):.3f}, max {y.max():.3f}")
print(f"  points outside the plotted window {(-6.2, -1.2)}: {int(((pred < -6.2) | (pred > -1.2) | (y < -6.2) | (y > -1.2)).sum())}")
print("  binned means (first/last 3 of 20):", [(round(a, 3), round(c, 3)) for a, c in zip(b.p, b.y)][:3], "...",
      [(round(a, 3), round(c, 3)) for a, c in zip(b.p, b.y)][-3:])
print("\nPLOT 2  vol_by_release_timing.png")
print(f"  sample: DEV 2006-16, all {len(d_all):,} primary-sample dev events (target only; no features needed)")
print(g.assign(mean_log=g["mean"].round(4), daily_vol=g["daily_vol"].map("{:.3%}".format),
               ci95=[f"{a:.2%}-{c:.2%}" for a, c in zip(g.lo, g.hi)])[["size", "mean_log", "daily_vol", "ci95"]].to_string())
print(f"  highest: {top} -> title: '{title}'")
