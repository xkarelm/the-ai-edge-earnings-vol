"""Risk-limit decision for the text model: M1 (B1b-GBM + text) vs B1b-GBM (no text), large-cap text subsample,
VALIDATION 2017-19 only. No holdout rows are loaded.

Models are exactly those of m1_text_v2.py stage_model (fit on text-subset dev 2006-16, surprise transforms and
categories fit on dev, fixed tree counts from results_gbm.csv, PCA text features from text_features_v2.parquet); their
validation R2 must reproduce results_m1_v2.csv before any decision metric is computed.

Decision rule = Convention B (as on the one-session example slide):
  big move = realized v_post above the top-quartile cut-off of ALL dev 2006-16 events (full sample, = 4.86%/day)
  alert    = each model flags its own top 25% of predicted v_post on the evaluated rows (equal alert budget)
Gain from text: M1 - B1b-GBM, with a 95% interval from a paired bootstrap over session dates.
Also reports the 2017-07-24 session under M1 (the example slide).

    python risk_limit_m1.py   -> data/processed/results_risk_limit_m1.csv
                                 data/processed/results_risk_limit_m1_session.csv   (the 2017-07-24 session, both models)
"""
import numpy as np
import pandas as pd

import config as C
from baselines_b0_b1 import metrics
from baselines_b0_b1_fixed import add_surprise_features
from baselines_gbm import CATS, FEATURES, GBM_PARAMS
from baselines_holdout import load
from m1_text_v2 import FEAT_FILE, accession_map

OUT = C.OUT / "results_risk_limit_m1.csv"
OUT_SESSION = C.OUT / "results_risk_limit_m1_session.csv"
M1N = "M1 = B1b-GBM + text"
EXPECTED = {"dev": 44_739, "validation": 14_614}
SESSION = pd.Timestamp("2017-07-24")
B, SEED = 2000, 0


def main():
    import lightgbm as lgb
    full, _ = load(["dev", "validation"])                                   # asserts: no holdout, no 2020
    assert set(full.split) == {"dev", "validation"}
    cut_big = full.loc[full.split == "dev", "v_post"].quantile(0.75)       # full-sample dev top quartile (Convention B)
    df = full[full.event_id.isin(set(accession_map().event_id))].copy()
    assert df.groupby("split").size().to_dict() == EXPECTED

    dev0 = df[df.split == "dev"]
    (dev, val), _ = add_surprise_features(dev0, df[df.split == "validation"])
    cats = {c: sorted(dev[c].unique()) for c in CATS}
    feats = pd.read_parquet(FEAT_FILE)
    feats = feats[feats.event_id.isin(set(df.event_id))]
    pcs = [c for c in feats.columns if c.startswith("txt_pc")]
    dev, val = (d.merge(feats, on="event_id", how="left") for d in (dev, val))
    for d in (dev, val):
        for c in CATS:
            d[c] = pd.Categorical(d[c], categories=cats[c])
    trees = int(pd.read_csv(C.OUT / "results_gbm.csv").query("model == 'B1b-GBM'").set_index("target")
                .loc["v_post", "best_iteration"])

    ref = pd.read_csv(C.OUT / "results_m1_v2.csv").query("eval == 'validation 2017-19' and target == 'v_post'") \
        .set_index("model")
    pred = {}
    for name, f in (("B1b-GBM", FEATURES), (M1N, FEATURES + pcs)):
        m = lgb.LGBMRegressor(**{**GBM_PARAMS, "n_estimators": trees}).fit(dev[f], dev["v_post"], categorical_feature=CATS)
        pred[name] = m.predict(val[f])
        r2 = metrics(val.v_post.to_numpy(), pred[name], dev.v_post.mean())["R2_oos"]
        assert abs(r2 - ref.loc[name, "R2_oos"]) < 1e-9, f"{name}: R2 {r2} does not reproduce results_m1_v2.csv: STOP"

    y = val.v_post.to_numpy()
    big = y > cut_big
    days = val.session_date.to_numpy()

    def decide(p):
        cut = np.quantile(p, 0.75)
        return p > cut, cut

    def counts(flag, big_):
        tp, fp, fn = int((flag & big_).sum()), int((flag & ~big_).sum()), int((~flag & big_).sum())
        return tp, fp, fn

    rows, flags = [], {}
    for name in pred:
        flag, cut = decide(pred[name])
        flags[name] = flag
        tp, fp, fn = counts(flag, big)
        rows.append({"eval": "validation 2017-19", "sample": "large-cap text subsample", "target": "v_post",
                     "rule": "own top 25% of predictions (Convention B)", "model": name, "n": len(y),
                     "big_move_cutoff_daily_vol": np.exp(cut_big), "flag_cutoff_daily_vol": np.exp(cut),
                     "base_rate": big.mean(), "flag_rate": flag.mean(), "TP": tp, "FP": fp, "FN": fn,
                     "TN": int(len(y) - tp - fp - fn), "recall": tp / (tp + fn), "precision": tp / (tp + fp)})

    # paired bootstrap over session dates for the recall / precision gain (thresholds held fixed)
    rng = np.random.default_rng(SEED)
    ud, inv = np.unique(days, return_inverse=True)
    f0, f1 = flags["B1b-GBM"], flags[M1N]
    per = lambda a: np.bincount(inv, weights=a.astype(float), minlength=len(ud))
    tp0, tp1, fl0, fl1, bg = per(f0 & big), per(f1 & big), per(f0), per(f1), per(big)
    dr, dp = [], []
    for _ in range(B):
        w = np.bincount(rng.integers(0, len(ud), len(ud)), minlength=len(ud))
        dr.append((w @ tp1 - w @ tp0) / (w @ bg))
        dp.append(w @ tp1 / (w @ fl1) - w @ tp0 / (w @ fl0))
    res = pd.DataFrame(rows)
    gain = {"eval": "validation 2017-19", "sample": "large-cap text subsample", "target": "v_post",
            "rule": "own top 25% of predictions (Convention B)", "model": "gain: M1 - B1b-GBM", "n": len(y),
            "TP": res.TP[1] - res.TP[0], "FP": res.FP[1] - res.FP[0], "FN": res.FN[1] - res.FN[0],
            "recall": res.recall[1] - res.recall[0], "precision": res.precision[1] - res.precision[0],
            "recall_ci_lo": np.percentile(dr, 2.5), "recall_ci_hi": np.percentile(dr, 97.5),
            "precision_ci_lo": np.percentile(dp, 2.5), "precision_ci_hi": np.percentile(dp, 97.5),
            "extra_TP_per_100_events": 100 * (res.TP[1] - res.TP[0]) / len(y)}
    res = pd.concat([res, pd.DataFrame([gain])], ignore_index=True)
    res.to_csv(OUT, index=False)
    pd.set_option("display.width", 250)
    print(res.drop(columns=["sample", "rule", "target"]).round(4).T.to_string())
    print(f"Saved {OUT}")

    # the example session under M1 (same rule, M1's own validation-wide flag cut-off)
    meta = pd.read_parquet(C.OUT_MODEL, columns=["event_id", "ticker"], filters=[("split", "==", "validation")])
    t = val.assign(p0=pred["B1b-GBM"], p1=pred[M1N], f0=f0, f1=f1, big=big).merge(meta, on="event_id")
    t = t[t.session_date == SESSION].sort_values("p1", ascending=False)
    out = lambda f, b: np.select([f & b, ~f & b, f & ~b], ["HIT", "MISS", "false alarm"], "quiet")
    pd.DataFrame({"session_date": SESSION.date(), "ticker": t.ticker, "actual_daily_vol": np.exp(t.v_post),
                  "big_move": t.big, "gbm_pred_daily_vol": np.exp(t.p0), "gbm_flag": t.f0, "gbm_outcome": out(t.f0, t.big),
                  "m1_pred_daily_vol": np.exp(t.p1), "m1_flag": t.f1, "m1_outcome": out(t.f1, t.big)}).to_csv(OUT_SESSION, index=False)
    print(f"\nSession {SESSION.date()} ({len(t)} names):")
    print(pd.DataFrame({"ticker": t.ticker, "M1 pred": np.exp(t.p1).map("{:.2%}".format), "M1 flag": np.where(t.f1, "Y", "N"),
                        "actual": np.exp(t.v_post).map("{:.2%}".format), "big": np.where(t.big, "Y", "N"),
                        "M1 outcome": out(t.f1, t.big), "GBM pred": np.exp(t.p0).map("{:.2%}".format),
                        "GBM outcome": out(t.f0, t.big)}).to_string(index=False))


if __name__ == "__main__":
    main()
