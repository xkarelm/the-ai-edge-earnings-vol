"""Text-subset baseline: (A) is the text-covered subset representative? (B) baseline ladder on text-covered rows only.

Text-subset = primary-sample events whose event_id (cik_accession) is in data/raw_manifest.parquet AND whose file
exists in data/rawTexts/raw_2006..2018.zip. Only zip member NAMES are read; no text is opened, embedded or featurized.
Dev + validation only (same load_locked() holdout guard).

Choices confirmed with the user:
  - B1b-GBM: fixed tree counts from the full-sample early-stopped run (results_gbm.csv: 173 / 171 / 135), trained on ALL
    text-subset dev rows, no early stopping. A full-sample GBM with the same fixed counts on all of dev is added so the
    full vs subset comparison is like for like (the existing 90%-dev 0.307 is shown too, unchanged).
  - Surprise transforms (winsor cut-points, abs_sue_rank map) for the subset ladder are fit on text-subset dev rows.

Writes data/processed/text_subset_representativeness.csv and data/processed/results_textsubset_baseline.csv.

    python baselines_textsubset.py
"""
import zipfile

import lightgbm as lgb
import numpy as np
import pandas as pd

import config as C
from baselines_b0_b1 import CAT_B1, NUM_B1, TARGETS, b0_model, load_locked, metrics
from baselines_b0_b1_fixed import SPECS, add_surprise_features, ridge_model
from baselines_gbm import CATS, FEATURES, GBM_PARAMS

MANIFEST = C.ROOT / "data" / "raw_manifest.parquet"
TEXT_DIR = C.ROOT / "data" / "rawTexts"
TEXT_YEARS = range(2006, 2019)
PRIOR_GBM = C.OUT / "results_gbm.csv"
OUT_REP = C.OUT / "text_subset_representativeness.csv"
OUT_RES = C.OUT / "results_textsubset_baseline.csv"
NUMERIC = ["v_pre", "v_post", "abs_sue_rank", "log_mktcap", "n_analysts"]


def text_covered_ids() -> set:
    m = pd.read_parquet(MANIFEST, columns=["cik", "accession", "filename"])
    names = set()
    for y in TEXT_YEARS:
        names |= {n for n in zipfile.ZipFile(TEXT_DIR / f"raw_{y}.zip").namelist() if not n.endswith("/")}
    on_disk = (m["accession"] + "__" + m["filename"]).isin(names)
    return set((m["cik"] + "_" + m["accession"])[on_disk])


def as_categories(dev, val):
    dev, val = dev.copy(), val.copy()
    for c in CATS:
        cats = sorted(dev[c].unique())
        dev[c] = pd.Categorical(dev[c], categories=cats)
        val[c] = pd.Categorical(val[c], categories=cats)
    return dev, val


def ladder(dev, val, trees, label):
    """B0 / B1b / B1b-GBM(fixed trees) fit on dev, scored on val. OOS R2 vs this dev's mean."""
    (dev, val), _ = add_surprise_features(dev, val)
    gdev, gval = as_categories(dev, val)
    rows = []
    for t in TARGETS:
        dm, y = dev[t].mean(), val[t].to_numpy()
        common = {"sample": label, "target": t, "n_dev": len(dev), "n_val": len(val)}
        rows.append({**common, "model": "B0", **metrics(y, b0_model().fit(dev, dev[t]).predict(val), dm)})
        rows.append({**common, "model": "B1b", **metrics(y, ridge_model(SPECS["B1b"]).fit(dev, dev[t]).predict(val), dm)})
        gbm = lgb.LGBMRegressor(**{**GBM_PARAMS, "n_estimators": int(trees[t])})
        gbm.fit(gdev[FEATURES], gdev[t], categorical_feature=CATS)
        rows.append({**common, "model": "B1b-GBM", "trees": int(trees[t]), **metrics(y, gbm.predict(gval[FEATURES]), dm)})
    return rows


def main():
    df = load_locked()                                              # holdout guard (dev/validation only)
    df = df.dropna(subset=[*TARGETS, *NUM_B1, *CAT_B1]).copy()      # same rows as B0 / B1b / B1b-GBM
    for c in CAT_B1:
        df[c] = df[c].astype(str)
    assert df.groupby("split").size().to_dict() == {"dev": 84_940, "validation": 25_659}
    covered = text_covered_ids()
    df["text"] = df["event_id"].isin(covered)
    n = df.groupby("split")["text"].sum()
    print(f"Text-subset: dev {n['dev']:,} | validation {n['validation']:,}  "
          f"(discovery: 44,739 / 9,651 on the 84,941-row dev before the 1 missing-market-cap row was dropped)")
    print("Text-covered validation by session year:",
          df[df.split == "validation"].groupby(df["session_date"].dt.year)["text"].sum().to_dict(), "\n")

    # ------------------------------------------------------------- Task A: representativeness
    (fd, fv), _ = add_surprise_features(df[df.split == "dev"], df[df.split == "validation"])   # full-dev rank map
    a = pd.concat([fd, fv])
    a["year"] = a["session_date"].dt.year
    rep = []
    for scope, d in [("dev+validation 2006-2019", a), ("dev+validation 2006-2018 (years with text)", a[a.year <= 2018])]:
        cov, non = d[d.text], d[~d.text]
        for f in NUMERIC:
            sd = np.sqrt((cov[f].var() + non[f].var()) / 2)
            rep.append({"scope": scope, "feature": f, "level": "", "covered_n": len(cov), "noncovered_n": len(non),
                        "covered_mean": cov[f].mean(), "noncovered_mean": non[f].mean(),
                        "covered_median": cov[f].median(), "noncovered_median": non[f].median(),
                        "covered_std": cov[f].std(), "noncovered_std": non[f].std(),
                        "std_diff": (cov[f].mean() - non[f].mean()) / sd})
        for f in ["sector", "when"]:
            sc, sn = cov[f].value_counts(normalize=True), non[f].value_counts(normalize=True)
            for lvl in sorted(set(sc.index) | set(sn.index)):
                p, q = sc.get(lvl, 0.0), sn.get(lvl, 0.0)
                pooled = (p * (1 - p) + q * (1 - q)) / 2
                rep.append({"scope": scope, "feature": f, "level": lvl, "covered_n": len(cov), "noncovered_n": len(non),
                            "covered_share": p, "noncovered_share": q,
                            "std_diff": (p - q) / np.sqrt(pooled) if pooled > 0 else np.nan})
    rep = pd.DataFrame(rep)
    rep.to_csv(OUT_REP, index=False)
    pd.set_option("display.width", 200)
    for scope in rep["scope"].unique():
        r = rep[rep.scope == scope]
        print(f"[Task A] {scope}: covered {r.covered_n.iloc[0]:,} vs non-covered {r.noncovered_n.iloc[0]:,}")
        print(r[r.level == ""][["feature", "covered_mean", "noncovered_mean", "covered_median", "noncovered_median",
                                "covered_std", "noncovered_std", "std_diff"]].round(3).to_string(index=False))
        print(r[r.level != ""][["feature", "level", "covered_share", "noncovered_share", "std_diff"]]
              .round(3).to_string(index=False), "\n")

    # ------------------------------------------------------------- Task B: ladder on the text-subset
    prior = pd.read_csv(PRIOR_GBM)
    trees = prior[prior.model == "B1b-GBM"].set_index("target")["best_iteration"].to_dict()
    print("Fixed GBM tree counts (from results_gbm.csv):", {k: int(v) for k, v in trees.items()}, "\n")
    full = ladder(df[df.split == "dev"], df[df.split == "validation"], trees, "full (fixed-tree GBM on all dev)")
    sub = ladder(df[(df.split == "dev") & df.text], df[(df.split == "validation") & df.text], trees, "text-subset")
    res = pd.DataFrame(full + sub)
    res.to_csv(OUT_RES, index=False)

    s = res[res["sample"] == "text-subset"]
    print(f"[Task B] Text-subset ladder: fit on {s.n_dev.iloc[0]:,} dev rows, scored on {s.n_val.iloc[0]:,} validation rows "
          "(R2_oos vs text-subset dev mean)")
    print(s[["target", "model", "MAE", "RMSE", "R2_oos"]].round(4).to_string(index=False))

    # side by side: full-sample (existing 90%-dev GBM + new fixed-tree full-dev GBM) vs text-subset
    old = prior.assign(model=prior.model.replace({"B1b-GBM": "B1b-GBM"})).set_index(["target", "model"])["R2_oos"]
    f = res[res["sample"].str.startswith("full")].set_index(["target", "model"])
    t = s.set_index(["target", "model"])
    side = pd.DataFrame({
        "full R2 (published)": old,
        "full R2 (fixed-tree GBM)": f["R2_oos"],
        "text-subset R2": t["R2_oos"],
        "full RMSE": f["RMSE"], "text-subset RMSE": t["RMSE"],
    }).reindex(pd.MultiIndex.from_product([TARGETS, ["B0", "B1b", "B1b-GBM"]]))
    side["shift R2 (subset - full)"] = side["text-subset R2"] - side["full R2 (fixed-tree GBM)"]
    print("\nFull sample vs text-subset (validation; full-sample B0/B1b are identical in both 'full' columns)")
    print(side.round(4).to_string())
    print(f"\nSaved {OUT_REP}\nSaved {OUT_RES}")


if __name__ == "__main__":
    main()
