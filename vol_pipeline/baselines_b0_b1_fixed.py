"""B1 fix: surprise measured as magnitude, fixed ridge penalty (no tuning on validation).

B0  : v_post ~ v_pre                                                     (OLS)
B1a : v_pre + sue_price_w + abs_sue_w    + log_mktcap + n_analysts + sector + when   (ridge)
B1b : v_pre + sue_price_w + abs_sue_rank + log_mktcap + n_analysts + sector + when   (ridge)
B1a_nosue (ablation): B1a without sue_price_w and abs_sue_w

Surprise transforms are fit on dev only and applied to validation:
  sue_price_w   sue_price clipped at dev 1st/99th percentiles
  abs_sue_w     |sue_price| clipped at the dev 99th percentile of |sue_price|
  abs_sue_rank  share of dev |sue_price| values <= x (dev empirical CDF), in [0, 1]

Same holdout guard and rows as baselines_b0_b1.py. Writes only
data/processed/results_b0_b1_fixed.csv.

    python baselines_b0_b1_fixed.py
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config as C
from baselines_b0_b1 import CAT_B1, NUM_B1, TARGETS, b0_model, load_locked, metrics

ALPHA = 1.0  # fixed; not tuned on validation
BASE_NUM = ["v_pre", "log_mktcap", "n_analysts"]
SPECS = {
    "B1a": BASE_NUM + ["sue_price_w", "abs_sue_w"],
    "B1b": BASE_NUM + ["sue_price_w", "abs_sue_rank"],
    "B1a_nosue": BASE_NUM,
}
OUT_RESULTS = C.OUT / "results_b0_b1_fixed.csv"


def add_surprise_features(dev, val):
    """Fit clipping points and rank mapping on dev; apply the same mapping to both."""
    lo, hi = dev["sue_price"].quantile([0.01, 0.99])
    abs_dev = dev["sue_price"].abs()
    abs_hi = abs_dev.quantile(0.99)
    dev_sorted = np.sort(abs_dev.to_numpy())
    out = []
    for d in (dev, val):
        d = d.copy()
        a = d["sue_price"].abs()
        d["sue_price_w"] = d["sue_price"].clip(lo, hi)
        d["abs_sue_w"] = a.clip(upper=abs_hi)
        d["abs_sue_rank"] = np.searchsorted(dev_sorted, a.to_numpy(), side="right") / len(dev_sorted)
        out.append(d)
    return out, {"sue_p01": lo, "sue_p99": hi, "abs_sue_p99": abs_hi}


def ridge_model(num):
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_B1),
    ])
    return Pipeline([("pre", pre), ("reg", Ridge(alpha=ALPHA))])


def main():
    df = load_locked()
    df = df.dropna(subset=[*TARGETS, *NUM_B1, *CAT_B1]).copy()  # same rows as the last run
    for c in CAT_B1:
        df[c] = df[c].astype(str)
    (dev, val), cuts = add_surprise_features(df[df["split"] == "dev"], df[df["split"] == "validation"])
    assert (len(dev), len(val)) == (84_940, 25_659), f"row count changed: {len(dev)}, {len(val)}"
    print(f"Rows: dev {len(dev):,}, validation {len(val):,} (identical across models and targets)")
    print("Dev cut-points: " + ", ".join(f"{k} {v:.5f}" for k, v in cuts.items()))
    print(f"Ridge alpha fixed at {ALPHA}\n")

    rows, coef_v_post = [], None
    for t in TARGETS:
        dev_mean = dev[t].mean()
        b0 = b0_model().fit(dev, dev[t])
        rows.append({"target": t, "model": "B0", "alpha": np.nan,
                     **metrics(val[t].to_numpy(), b0.predict(val), dev_mean)})
        for name, num in SPECS.items():
            m = ridge_model(num).fit(dev, dev[t])
            rows.append({"target": t, "model": name, "alpha": ALPHA,
                         **metrics(val[t].to_numpy(), m.predict(val), dev_mean)})
            if t == "v_post" and name == "B1a":
                names = [n.split("__", 1)[1] for n in m.named_steps["pre"].get_feature_names_out()]
                coef_v_post = pd.Series(m.named_steps["reg"].coef_, index=names)
                coef_v_post["(intercept)"] = m.named_steps["reg"].intercept_

    res = pd.DataFrame(rows)
    res.to_csv(OUT_RESULTS, index=False)
    pd.set_option("display.width", 140)

    print("Validation (2017-19) metrics")
    print(res.round(4).to_string(index=False))

    print("\nB1a coefficients, target v_post (numeric per 1 dev-sd; categoricals one-hot, no dropped level)")
    print(coef_v_post.round(4).to_string())
    sd = dev[["sue_price_w", "abs_sue_w"]].std(ddof=0)
    for f in ["abs_sue_w", "sue_price_w"]:
        print(f"  -> {f}: {coef_v_post[f]:+.4f} per dev-sd  ({coef_v_post[f] / sd[f]:+.3f} per unit)")

    r = res.set_index(["target", "model"])
    print("\nSurprise contribution: R2_oos(B1a) - R2_oos(B1a without both surprise features)")
    for t in TARGETS:
        d = r.loc[(t, "B1a"), "R2_oos"] - r.loc[(t, "B1a_nosue"), "R2_oos"]
        drmse = r.loc[(t, "B1a"), "RMSE"] - r.loc[(t, "B1a_nosue"), "RMSE"]
        print(f"  {t:10s} dR2 {d:+.4f}   dRMSE {drmse:+.4f}")
    print(f"\nSaved {OUT_RESULTS}")


if __name__ == "__main__":
    main()
