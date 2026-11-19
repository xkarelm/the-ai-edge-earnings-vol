"""B0 and B1 baselines: post-earnings log realized volatility, linear models.

B0: v_post ~ v_pre                                                  (OLS)
B1: v_post ~ v_pre + sue_price + log_mktcap + n_analysts
             + one-hot(sector) + one-hot(when)                     (ridge)

Train on dev (2006-16), tune and evaluate on validation (2017-19). Holdout (2021-25)
and excluded_2020 are never loaded: the parquet read filters on `split` and the
guard below fails if anything else gets through. Reads model_table.parquet, writes
only data/processed/results_b0_b1.csv.

    python baselines_b0_b1.py
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config as C

ALLOWED = ("dev", "validation")
TARGETS = ["v_post_3", "v_post", "v_post_15"]
NUM_B1 = ["v_pre", "sue_price", "log_mktcap", "n_analysts"]
CAT_B1 = ["sector", "when"]
ALPHAS = [0.01, 0.1, 1, 10, 100, 1_000, 10_000]
OUT_RESULTS = C.OUT / "results_b0_b1.csv"


def load_locked() -> pd.DataFrame:
    """Load dev + validation only. Holdout and excluded_2020 rows are filtered at read time."""
    cols = ["event_id", "session_date", "split", *TARGETS, *NUM_B1, *CAT_B1]
    df = pd.read_parquet(C.OUT_MODEL, columns=cols, filters=[("split", "in", list(ALLOWED))])
    # Holdout guard: fail loudly if any locked row got through.
    assert set(df["split"].unique()) <= set(ALLOWED), "holdout guard: unexpected split loaded"
    years = df["session_date"].dt.year
    for s in ALLOWED:
        lo, hi = C.SPLITS[s]
        y = years[df["split"] == s]
        assert y.between(lo, hi).all(), f"holdout guard: {s} has years outside {lo}-{hi}"
    assert years.max() <= C.SPLITS["validation"][1], "holdout guard: post-2019 row loaded"
    return df


def metrics(y, pred, dev_mean):
    err = y - pred
    sse = np.sum(err ** 2)
    return {
        "MAE": np.mean(np.abs(err)),
        "RMSE": np.sqrt(np.mean(err ** 2)),
        # OOS R2 against the dev-mean forecast (the only benchmark known at forecast time)
        "R2_oos": 1 - sse / np.sum((y - dev_mean) ** 2),
        # Conventional R2 against the validation mean, for reference
        "R2_val": 1 - sse / np.sum((y - y.mean()) ** 2),
    }


def b0_model():
    return Pipeline([("pre", ColumnTransformer([("num", StandardScaler(), ["v_pre"])])),
                     ("reg", LinearRegression())])


def b1_model(alpha):
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM_B1),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_B1),
    ])
    return Pipeline([("pre", pre), ("reg", Ridge(alpha=alpha))])


def main():
    df = load_locked()
    n0 = len(df)
    # Same rows for B0 and B1: drop rows missing any B1 feature or any target.
    df = df.dropna(subset=[*TARGETS, *NUM_B1, *CAT_B1]).copy()
    for c in CAT_B1:
        df[c] = df[c].astype(str)
    dev, val = df[df["split"] == "dev"], df[df["split"] == "validation"]
    print(f"Loaded {n0:,} dev+validation rows; {n0 - len(df)} dropped for missing values.")
    print(f"Common sample: dev {len(dev):,}, validation {len(val):,} (same rows for B0 and B1, all targets)\n")

    rows, coefs = [], {}
    for t in TARGETS:
        dev_mean = dev[t].mean()

        b0 = b0_model().fit(dev, dev[t])
        rows.append({"target": t, "model": "B0", "alpha": np.nan,
                     **metrics(val[t].to_numpy(), b0.predict(val), dev_mean)})

        # Choose ridge alpha on validation RMSE (scaler/encoder are fit on dev inside the pipeline).
        fits = {a: b1_model(a).fit(dev, dev[t]) for a in ALPHAS}
        rmse = {a: np.sqrt(np.mean((val[t] - m.predict(val)) ** 2)) for a, m in fits.items()}
        best = min(rmse, key=rmse.get)
        b1 = fits[best]
        rows.append({"target": t, "model": "B1", "alpha": best,
                     **metrics(val[t].to_numpy(), b1.predict(val), dev_mean)})

        names = [n.split("__", 1)[1] for n in b1.named_steps["pre"].get_feature_names_out()]
        coefs[t] = pd.Series(b1.named_steps["reg"].coef_, index=names)
        coefs[t]["(intercept)"] = b1.named_steps["reg"].intercept_
        if t == "v_post":
            b0_slope = b0.named_steps["reg"].coef_[0]
            print(f"B0 (v_post): intercept {b0.named_steps['reg'].intercept_:.4f}, "
                  f"v_pre {b0_slope:.4f} per dev-sd = {b0_slope / dev['v_pre'].std(ddof=0):.4f} per unit\n")

    res = pd.DataFrame(rows)
    res.to_csv(OUT_RESULTS, index=False)

    pd.set_option("display.width", 140)
    print("Validation (2017-19) metrics")
    print(res.round(4).to_string(index=False))

    print("\nB1 - B0 on validation (negative MAE/RMSE = B1 better)")
    w = res.pivot(index="target", columns="model", values=["MAE", "RMSE", "R2_oos"]).reindex(TARGETS)
    print(pd.DataFrame({m: w[(m, "B1")] - w[(m, "B0")] for m in ["MAE", "RMSE", "R2_oos"]}).round(4).to_string())

    sd = dev[NUM_B1].std(ddof=0)
    cf = pd.DataFrame(coefs)
    print("\nB1 ridge coefficients (fit on dev). Numeric: per 1 dev-sd; categoricals: one-hot, no dropped level")
    print(cf.round(4).to_string())
    print("\nB1 numeric coefficients in raw units (per 1 unit of the feature)")
    print(cf.loc[NUM_B1].div(sd, axis=0).round(5).to_string())
    print(f"\nSaved {OUT_RESULTS}")


if __name__ == "__main__":
    main()
