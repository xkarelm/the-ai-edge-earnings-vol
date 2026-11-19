"""B1b-GBM: the B1b feature set with LightGBM instead of ridge. Only the model class changes.

B0      : v_post ~ v_pre                                                             (OLS, full dev)
B1b     : v_pre + sue_price_w + abs_sue_rank + log_mktcap + n_analysts + sector + when (ridge, full dev)
B1b-GBM : same seven features, LightGBM; sector and when as native categoricals (no one-hot)

Train on dev (2006-16), score on validation (2017-19). Holdout and excluded_2020 are never loaded
(same load_locked() guard). Early stopping uses the most recent ~10% of DEV by session_date; the GBM
trains on the earlier ~90% of dev. Validation is used only for final scoring. Writes only
data/processed/results_gbm.csv.

    python baselines_gbm.py
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

import config as C
from baselines_b0_b1 import CAT_B1, NUM_B1, TARGETS, b0_model, load_locked, metrics
from baselines_b0_b1_fixed import SPECS, add_surprise_features, ridge_model

FEATURES = ["v_pre", "sue_price_w", "abs_sue_rank", "log_mktcap", "n_analysts", "sector", "when"]  # = B1b set
CATS = ["sector", "when"]
ES_SHARE = 0.10
GBM_PARAMS = dict(num_leaves=31, learning_rate=0.05, n_estimators=2000, random_state=0,
                  deterministic=True, force_col_wise=True, n_jobs=4, verbose=-1)
PATIENCE = 50
OUT_RESULTS = C.OUT / "results_gbm.csv"
PRIOR = C.OUT / "results_b0_b1_fixed.csv"


def main():
    df = load_locked()                                              # holdout guard runs here
    df = df.dropna(subset=[*TARGETS, *NUM_B1, *CAT_B1]).copy()      # same rows as B0 / B1b
    for c in CAT_B1:
        df[c] = df[c].astype(str)
    (dev, val), _ = add_surprise_features(df[df["split"] == "dev"], df[df["split"] == "validation"])
    assert (len(dev), len(val)) == (84_940, 25_659), f"row count changed: {len(dev)}, {len(val)}"
    assert set(FEATURES) == set(SPECS["B1b"]) | set(CAT_B1), "GBM features must equal the B1b set"

    # categories fixed from dev, applied unchanged to validation (unseen levels -> NaN)
    for c in CATS:
        cats = sorted(dev[c].unique())
        dev[c] = pd.Categorical(dev[c], categories=cats)
        val[c] = pd.Categorical(val[c], categories=cats)

    # dev-internal early-stopping slice: most recent ~10% of dev, cut on a whole session date
    cut = dev["session_date"].sort_values().iloc[int(len(dev) * (1 - ES_SHARE))]
    fit_part, es_part = dev[dev["session_date"] < cut], dev[dev["session_date"] >= cut]
    assert fit_part["session_date"].max() < es_part["session_date"].min()
    print(f"Rows: dev {len(dev):,} | validation {len(val):,} (identical to B0 / B1b)")
    print(f"GBM fit slice:            {fit_part['session_date'].min().date()} .. {fit_part['session_date'].max().date()}  "
          f"({len(fit_part):,} rows)")
    print(f"GBM early-stopping slice: {es_part['session_date'].min().date()} .. {es_part['session_date'].max().date()}  "
          f"({len(es_part):,} rows, {len(es_part) / len(dev):.1%} of dev)")
    print(f"Validation (final scoring only): {val['session_date'].min().date()} .. {val['session_date'].max().date()}\n")

    rows, importance = [], None
    for t in TARGETS:
        dev_mean = dev[t].mean()                                    # OOS R2 benchmark: full-dev mean
        y_val = val[t].to_numpy()

        b0 = b0_model().fit(dev, dev[t])
        rows.append({"target": t, "model": "B0", **metrics(y_val, b0.predict(val), dev_mean)})
        b1 = ridge_model(SPECS["B1b"]).fit(dev, dev[t])
        rows.append({"target": t, "model": "B1b", **metrics(y_val, b1.predict(val), dev_mean)})

        gbm = lgb.LGBMRegressor(**GBM_PARAMS)
        gbm.fit(fit_part[FEATURES], fit_part[t], eval_set=[(es_part[FEATURES], es_part[t])], eval_metric="l2",
                categorical_feature=CATS, callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
        best = gbm.best_iteration_
        pred = gbm.predict(val[FEATURES], num_iteration=best)
        rows.append({"target": t, "model": "B1b-GBM", **metrics(y_val, pred, dev_mean), "best_iteration": best,
                     "es_rmse": float(np.sqrt(gbm.best_score_["valid_0"]["l2"]))})
        if t == "v_post":
            g = gbm.booster_.feature_importance(importance_type="gain", iteration=best)
            importance = pd.Series(g, index=FEATURES).sort_values(ascending=False)

    res = pd.DataFrame(rows)
    res["es_slice"] = f"{es_part['session_date'].min().date()}..{es_part['session_date'].max().date()}"
    res.loc[res["model"] != "B1b-GBM", "es_slice"] = ""

    # B0 / B1b must reproduce the earlier run exactly (same rows, same models)
    prior = pd.read_csv(PRIOR).set_index(["target", "model"])
    for _, r in res[res["model"].isin(["B0", "B1b"])].iterrows():
        assert np.isclose(r["R2_oos"], prior.loc[(r["target"], r["model"]), "R2_oos"], atol=1e-10), \
            f"{r['target']} {r['model']} does not match results_b0_b1_fixed.csv"
    print("B0 and B1b reproduce results_b0_b1_fixed.csv exactly.\n")

    res.to_csv(OUT_RESULTS, index=False)
    pd.set_option("display.width", 160)
    print("Validation (2017-19) metrics; R2_oos is against the dev-mean forecast")
    print(res[["target", "model", "MAE", "RMSE", "R2_oos", "best_iteration"]].round(4).to_string(index=False))

    w = res.pivot(index="target", columns="model", values=["MAE", "RMSE", "R2_oos"]).reindex(TARGETS)
    diff = pd.DataFrame({m: w[(m, "B1b-GBM")] - w[(m, "B1b")] for m in ["MAE", "RMSE", "R2_oos"]})
    print("\nB1b-GBM minus B1b (negative MAE/RMSE, positive R2 = GBM better)")
    print(diff.round(4).to_string())

    print("\nLightGBM feature importance, v_post (total gain, share of total)")
    print((importance / importance.sum()).map("{:.1%}".format).to_string())

    print("\nCaveats:")
    print("  - Validation (2017-19) is not the final test; holdout (2021-25) is reserved and was not loaded.")
    print(f"  - Early stopping used a dev-internal time slice ({res['es_slice'].iloc[2]}), never validation; "
          "no other hyperparameter was tuned.")
    print("  - The GBM trains on ~90% of dev (the rest is its early-stopping slice); B0 and B1b train on all of dev.")
    print("  - Row counts and event sets match B0 / B1b (84,940 dev / 25,659 validation).")
    print(f"\nSaved {OUT_RESULTS}")


if __name__ == "__main__":
    main()
