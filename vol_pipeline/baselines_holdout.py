"""ONE-TIME final holdout evaluation of the finalized tabular ladder (B0, B1b, B1b-GBM). No text (no holdout text).

    python baselines_holdout.py check   # dev + validation only: refit and confirm the stored validation numbers
    python baselines_holdout.py final   # the single holdout run; refuses to run twice

Finalized models (no re-tuning, no re-selection):
  B0      OLS on v_pre                                                   (baselines_b0_b1.b0_model)
  B1b     ridge, alpha = 1, B1b features, surprise transforms fit on dev (baselines_b0_b1_fixed)
  B1b-GBM LightGBM, GBM_PARAMS, fixed trees 173 / 171 / 135, trained on ALL dev, categories fixed from dev
          (= the full-sample final fit in results_textsubset_baseline.csv, validation R2 0.3127 for v_post)
excluded_2020 is never loaded. OOS R2 is against the dev mean (same convention as before); R2 against the holdout
mean is also reported to separate the level shift from ranking skill.
"""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

import config as C
from baselines_b0_b1 import CAT_B1, NUM_B1, TARGETS, b0_model, metrics
from baselines_b0_b1_fixed import SPECS, add_surprise_features, ridge_model
from baselines_gbm import CATS, FEATURES, GBM_PARAMS

OUT = C.OUT / "results_holdout.csv"
COLS = ["event_id", "session_date", "split", *TARGETS, *NUM_B1, *CAT_B1]


def load(splits):
    assert "excluded_2020" not in splits
    df = pd.read_parquet(C.OUT_MODEL, columns=COLS, filters=[("split", "in", list(splits))])
    assert set(df["split"].unique()) <= set(splits), "unexpected split loaded"
    assert not (df["session_date"].dt.year == 2020).any(), "2020 row loaded"
    for s in splits:
        lo, hi = C.SPLITS[s]
        assert df.loc[df.split == s, "session_date"].dt.year.between(lo, hi).all(), f"{s} outside {lo}-{hi}"
    n_raw = df.groupby("split").size().to_dict()
    df = df.dropna(subset=[*TARGETS, *NUM_B1, *CAT_B1]).copy()                 # same row rule as every baseline
    for c in CAT_B1:
        df[c] = df[c].astype(str)
    return df, n_raw


def fit_finalized(dev):
    trees = pd.read_csv(C.OUT / "results_gbm.csv").query("model == 'B1b-GBM'").set_index("target")["best_iteration"]
    cats = {c: sorted(dev[c].unique()) for c in CATS}
    gdev = dev.copy()
    for c in CATS:
        gdev[c] = pd.Categorical(gdev[c], categories=cats[c])
    models = {}
    for t in TARGETS:
        g = lgb.LGBMRegressor(**{**GBM_PARAMS, "n_estimators": int(trees[t])})
        g.fit(gdev[FEATURES], gdev[t], categorical_feature=CATS)
        models[t] = {"B0": b0_model().fit(dev, dev[t]), "B1b": ridge_model(SPECS["B1b"]).fit(dev, dev[t]), "B1b-GBM": g}
    return models, cats, {t: int(trees[t]) for t in TARGETS}


def score(models, cats, dev, ev):
    ge = ev.copy()
    for c in CATS:
        ge[c] = pd.Categorical(ge[c], categories=cats[c])
    rows = []
    for t in TARGETS:
        y = ev[t].to_numpy()
        for name, m in models[t].items():
            p = m.predict(ge[FEATURES]) if name == "B1b-GBM" else m.predict(ev)
            r = metrics(y, p, dev[t].mean())
            rows.append({"target": t, "model": name, "MAE": r["MAE"], "RMSE": r["RMSE"], "R2_oos": r["R2_oos"],
                         "R2_vs_own_mean": r["R2_val"], "n": len(ev)})
    return pd.DataFrame(rows)


def stored_validation():
    fx = pd.read_csv(C.OUT / "results_b0_b1_fixed.csv").query("model in ['B0', 'B1b']")
    ts = pd.read_csv(C.OUT / "results_textsubset_baseline.csv")
    g = ts[ts["sample"].str.startswith("full") & (ts.model == "B1b-GBM")]
    return pd.concat([fx, g])[["target", "model", "MAE", "RMSE", "R2_oos"]].set_index(["target", "model"])


def main(mode):
    if mode == "check":
        df, _ = load(["dev", "validation"])
        dev = df[df.split == "dev"]
        (dev, val), _ = add_surprise_features(dev, df[df.split == "validation"])
        assert (len(dev), len(val)) == (84_940, 25_659)
        models, cats, trees = fit_finalized(dev)
        got = score(models, cats, dev, val).set_index(["target", "model"])
        ref = stored_validation()
        diff = (got[["MAE", "RMSE", "R2_oos"]] - ref.loc[got.index]).abs().max().max()
        print(f"fixed trees {trees}; dev {len(dev):,}, validation {len(val):,}")
        print(got[["MAE", "RMSE", "R2_oos"]].round(4).to_string())
        print(f"max |refit - stored validation| = {diff:.2e}")
        assert diff < 1e-9, "refit does not reproduce the finalized validation numbers: STOP"
        print("CHECK PASSED: refit models reproduce the finalized validation results exactly.")
        return

    assert mode == "final"
    assert not OUT.exists(), f"{OUT} exists: the holdout has already been run once. Not re-running."
    df, n_raw = load(["dev", "validation", "holdout"])
    dev = df[df.split == "dev"]
    (dev, val, hold) = (dev, df[df.split == "validation"], df[df.split == "holdout"])
    (dev, val), _ = add_surprise_features(dev, val)
    (_, hold), cuts = add_surprise_features(df[df.split == "dev"], hold)       # dev-fit transforms applied to holdout
    models, cats, trees = fit_finalized(dev)
    vcheck = score(models, cats, dev, val).set_index(["target", "model"])
    assert (vcheck[["MAE", "RMSE", "R2_oos"]] - stored_validation().loc[vcheck.index]).abs().max().max() < 1e-9
    unseen = {c: sorted(set(hold[c]) - set(cats[c])) for c in CATS}

    res = score(models, cats, dev, hold)                                       # <- the one holdout scoring
    res.insert(0, "split", "holdout")
    out = pd.concat([vcheck.reset_index().assign(split="validation"), res], ignore_index=True)
    out["gbm_trees"] = out.apply(lambda r: trees[r.target] if r.model == "B1b-GBM" else np.nan, axis=1)
    out.to_csv(OUT, index=False)                                               # saved before anything else

    pd.set_option("display.width", 200)
    print(f"Holdout rows: {len(hold):,} of {n_raw['holdout']:,} loaded ({n_raw['holdout'] - len(hold)} dropped: missing "
          f"feature, same rule as every baseline); sessions {hold.session_date.min().date()} .. {hold.session_date.max().date()}")
    print(f"Training rows (dev): {len(dev):,}; fixed GBM trees {trees}; categories unseen in dev: {unseen}")
    print("Finalized-model check: refit reproduces stored validation results exactly (max diff < 1e-9).\n")
    print("HOLDOUT (2021-25) - R2_oos vs dev mean; R2_vs_own_mean vs the holdout mean")
    print(res[["target", "model", "MAE", "RMSE", "R2_oos", "R2_vs_own_mean"]].round(4).to_string(index=False))
    side = pd.DataFrame({
        "val R2": vcheck["R2_oos"], "holdout R2": res.set_index(["target", "model"])["R2_oos"],
        "val RMSE": vcheck["RMSE"], "holdout RMSE": res.set_index(["target", "model"])["RMSE"]})
    side["R2 change"] = side["holdout R2"] - side["val R2"]
    print("\nValidation (2017-19) vs holdout (2021-25)")
    print(side.reindex(pd.MultiIndex.from_product([TARGETS, ["B0", "B1b", "B1b-GBM"]])).round(4).to_string())
    print(f"\nTarget level: dev mean v_post {dev['v_post'].mean():.3f}, validation {val['v_post'].mean():.3f}, "
          f"holdout {hold['v_post'].mean():.3f}")
    print("\nCaveats:")
    print("  - Tabular ladder only: no text model on holdout (no 2021-25 text on disk).")
    print("  - Holdout spans 2021-25, including volatile periods; excluded_2020 was not loaded.")
    print("  - One-time final evaluation: run once, not tuned or re-run; the script refuses a second run.")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main(sys.argv[1])
