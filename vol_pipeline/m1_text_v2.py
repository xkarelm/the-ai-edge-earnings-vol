"""M1 v2: earnings-release text layer rebuilt from the clean parquets, tested on validation AND the one-time holdout.

Replaces the zip-based text pipeline (m1_text.py, kept for reproducibility of results_m1.csv).

Stages (each cached; rerun only what is missing):
  python m1_text_v2.py prep    -> data/processed/text_clean_v2.parquet       first ~500 words of text_full per event
  python m1_text_v2.py embed   -> data/processed/text_embeddings_v2.parquet  FinBERT-tone mean-pooled 768-d (resumable)
  python m1_text_v2.py pca     -> data/processed/text_features_v2.parquet    PCA fit on DEV embeddings only
  python m1_text_v2.py model   -> data/processed/results_m1_v2.csv           B1b-GBM vs M1, validation + holdout

Inputs: data/parquets/clean_YYYY.parquet, column text_full only (never text_body). clean_2020.parquet is never opened.
Join: accession -> cik from data/raw_manifest.parquet -> event_id = cik + "_" + accession (matches model_table).
Rows: model rows (same tabular-feature drop as every baseline) in dev / validation / holdout that have a clean-parquet
document. Both models use identical rows.

Holdout: loaded because testing M1 out-of-sample is the explicit goal; it is a ONE-TIME test. Nothing is fit, tuned or
selected on validation or holdout: surprise transforms, categories and PCA are fit on dev; trees are the fixed counts
from results_gbm.csv (173 / 171 / 135); GBM hyperparameters are unchanged (baselines_gbm.GBM_PARAMS).
"""
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config as C
from baselines_b0_b1 import TARGETS, metrics
from baselines_b0_b1_fixed import add_surprise_features
from baselines_gbm import CATS, FEATURES, GBM_PARAMS
from baselines_holdout import load
from m1_text import TRANSCRIPT_RE, day_bootstrap

PARQ = C.ROOT / "data" / "parquets"
MANIFEST = C.ROOT / "data" / "raw_manifest.parquet"
YEARS = [y for y in range(2006, 2026) if y != 2020]                       # 2020 excluded: file never opened
SPLITS = ["dev", "validation", "holdout"]
CLEAN_FILE = C.OUT / "text_clean_v2.parquet"
EMB_PARTS = C.OUT / "text_embeddings_v2_parts"
EMB_FILE = C.OUT / "text_embeddings_v2.parquet"
FEAT_FILE = C.OUT / "text_features_v2.parquet"
OUT_RES = C.OUT / "results_m1_v2.csv"
MODEL_NAME = "yiyanghkust/finbert-tone"
KEEP_WORDS, MAX_TOKENS, BATCH, CHUNK = 500, 256, 8, 2000
CHUNKS_PER_RUN = 4                                                        # embed exits(3) after this many chunks
PCA_TARGET, PCA_CAP = 0.90, 50
EXPECTED = {"dev": 44_739, "validation": 14_614, "holdout": 27_629}       # expected row counts


def accession_map() -> pd.DataFrame:
    parts = [pq.read_table(PARQ / f"clean_{y}.parquet", columns=["accession"]).to_pandas().assign(file_year=y)
             for y in YEARS]
    cp = pd.concat(parts, ignore_index=True)
    assert cp.accession.is_unique
    man = pd.read_parquet(MANIFEST, columns=["accession", "cik"])
    cp = cp.merge(man, on="accession", how="left")
    assert cp.cik.notna().all()
    cp["event_id"] = cp.cik + "_" + cp.accession
    return cp


def subset_rows():
    df, _ = load(SPLITS)                                                  # asserts no 2020; same row rule as baselines
    cp = accession_map()
    df = df[df.event_id.isin(set(cp.event_id))].copy()
    n = df.groupby("split").size().to_dict()
    assert n == EXPECTED, f"text subset changed: {n}"
    return df, cp


# ------------------------------------------------------------------------------------------------ step 2
def stage_prep():
    if CLEAN_FILE.exists():
        print(f"cached: {CLEAN_FILE}"); return
    df, cp = subset_rows()
    cp = cp[cp.event_id.isin(set(df.event_id))]
    out = []
    for y, g in cp.groupby("file_year"):
        t = pq.read_table(PARQ / f"clean_{y}.parquet", columns=["accession", "text_full"],
                          filters=[("accession", "in", g.accession.tolist())]).to_pandas()
        words = t.text_full.fillna("").str.split()
        t["n_words_full"] = words.str.len()
        t["text500"] = words.str[:KEEP_WORDS].str.join(" ")              # light prep: whitespace normalised only
        t["is_transcript"] = t.text_full.fillna("").str.contains(TRANSCRIPT_RE)
        out.append(t.drop(columns="text_full").merge(g[["accession", "event_id"]], on="accession"))
        print(f"  {y}: {len(t):,} documents", flush=True)
    res = pd.concat(out, ignore_index=True).merge(df[["event_id", "split"]], on="event_id")
    assert len(res) == len(df) and res.event_id.is_unique
    res.to_parquet(CLEAN_FILE, index=False)
    w, k = res.n_words_full, res.text500.str.split().str.len()
    print(f"\nprepared {len(res):,} documents -> {CLEAN_FILE}  {res.groupby('split').size().to_dict()}")
    print("text_full words per document: " + ", ".join(
        f"{a} {b:,.0f}" for a, b in w.describe(percentiles=[.05, .25, .5, .75, .95]).items()))
    print(f"words kept (first {KEEP_WORDS}): median {k.median():.0f}; shorter than {KEEP_WORDS}: {(w < KEEP_WORDS).mean():.1%}; "
          f"empty: {(w == 0).sum()}; transcripts flagged (text set to missing): {res.is_transcript.sum()}")


# ------------------------------------------------------------------------------------------------ step 3
def stage_embed():
    import os
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.6")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")
    import torch
    from transformers import AutoModel, AutoTokenizer
    if EMB_FILE.exists():
        print(f"cached: {EMB_FILE}"); return
    txt = pd.read_parquet(CLEAN_FILE)
    todo = txt[~txt.is_transcript & txt.text500.str.len().gt(0)].reset_index(drop=True)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).half().to(dev).eval()
    EMB_PARTS.mkdir(parents=True, exist_ok=True)
    order = todo.text500.str.len().sort_values(kind="stable").index
    chunks = np.array_split(order.to_numpy(), int(np.ceil(len(order) / CHUNK)))
    t0, done_now = time.time(), 0
    for i, idx in enumerate(chunks):
        part = EMB_PARTS / f"part_{i:03d}.parquet"
        if part.exists():
            continue
        if done_now >= CHUNKS_PER_RUN:
            print(f"  pausing after {done_now} chunks to reset memory", flush=True)
            sys.exit(3)
        done_now += 1
        vecs = []
        for b in range(0, len(idx), BATCH):
            enc = tok(todo.loc[idx[b:b + BATCH], "text500"].tolist(), padding=True, truncation=True,
                      max_length=MAX_TOKENS, return_tensors="pt").to(dev)
            with torch.no_grad():
                h = model(**enc).last_hidden_state.float()
            mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            vecs.append(((h * mask).sum(1) / mask.sum(1)).cpu().numpy())
            del h, enc
            if dev.type == "mps":
                torch.mps.empty_cache()
        e = pd.DataFrame(np.vstack(vecs), columns=[f"emb_{j}" for j in range(768)])
        e.insert(0, "event_id", todo.loc[idx, "event_id"].to_numpy())
        e.to_parquet(part, index=False)
        print(f"  chunk {i + 1}/{len(chunks)}  ({time.time() - t0:.0f}s)", flush=True)
    emb = pd.concat([pd.read_parquet(p) for p in sorted(EMB_PARTS.glob("part_*.parquet"))], ignore_index=True)
    assert emb.event_id.is_unique and len(emb) == len(todo), "embedding chunks incomplete"
    emb = txt[["event_id"]].merge(emb, on="event_id", how="left")         # empty / transcript -> NaN vector
    emb.to_parquet(EMB_FILE, index=False)
    print(f"embedded {emb.emb_0.notna().sum():,}; NaN vectors {emb.emb_0.isna().sum()} -> {EMB_FILE}")


# ------------------------------------------------------------------------------------------------ step 4
def stage_pca():
    from sklearn.decomposition import PCA
    if FEAT_FILE.exists():
        print(f"cached: {FEAT_FILE}"); return
    emb = pd.read_parquet(EMB_FILE).merge(pd.read_parquet(CLEAN_FILE, columns=["event_id", "split"]), on="event_id")
    cols = [f"emb_{j}" for j in range(768)]
    have = emb[cols[0]].notna()
    fit = emb.loc[(emb.split == "dev") & have, cols].to_numpy()          # DEV ONLY
    pca = PCA(n_components=PCA_CAP, random_state=0).fit(fit)
    cum = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cum, PCA_TARGET) + 1) if cum[-1] >= PCA_TARGET else PCA_CAP
    z = np.full((len(emb), k), np.nan)
    z[have.to_numpy()] = pca.transform(emb.loc[have, cols].to_numpy())[:, :k]
    feats = pd.DataFrame(z, columns=[f"txt_pc{j:02d}" for j in range(k)])
    feats.insert(0, "event_id", emb.event_id.to_numpy())
    feats.to_parquet(FEAT_FILE, index=False)
    print(f"PCA fit on {len(fit):,} dev embeddings only; kept {k} components = {cum[k - 1]:.1%} of dev variance "
          f"({'reached' if cum[-1] >= PCA_TARGET else 'NOT reached; capped'}); applied to validation and holdout -> {FEAT_FILE}")


# ------------------------------------------------------------------------------------------------ step 5
def stage_model():
    import lightgbm as lgb
    df, _ = subset_rows()
    dev0 = df[df.split == "dev"]
    (dev, val), _ = add_surprise_features(dev0, df[df.split == "validation"])   # transforms fit on subset dev
    (_, hold), _ = add_surprise_features(dev0, df[df.split == "holdout"])
    cats = {c: sorted(dev[c].unique()) for c in CATS}                            # categories fixed from dev
    feats = pd.read_parquet(FEAT_FILE)
    pcs = [c for c in feats.columns if c.startswith("txt_pc")]
    sets = {}
    for name, d in (("dev", dev), ("validation", val), ("holdout", hold)):
        d = d.merge(feats, on="event_id", how="left")
        for c in CATS:
            d[c] = pd.Categorical(d[c], categories=cats[c])
        sets[name] = d
    print("Rows: " + ", ".join(f"{k} {len(v):,} (text missing {v[pcs[0]].isna().sum()})" for k, v in sets.items()))
    trees = pd.read_csv(C.OUT / "results_gbm.csv").query("model == 'B1b-GBM'").set_index("target")["best_iteration"]
    M1 = FEATURES + pcs
    dv = sets["dev"]
    evals = {"validation 2017-19": sets["validation"],
             "validation 2017-18": sets["validation"][sets["validation"].session_date.dt.year <= 2018],
             "holdout 2021-25": sets["holdout"]}
    rows, imp = [], {}
    for t in TARGETS:
        ms = {}
        for name, f in (("B1b-GBM", FEATURES), ("M1 = B1b-GBM + text", M1)):
            m = lgb.LGBMRegressor(**{**GBM_PARAMS, "n_estimators": int(trees[t])})
            m.fit(dv[f], dv[t], categorical_feature=CATS)
            ms[name] = (m, f)
        g = pd.Series(ms["M1 = B1b-GBM + text"][0].booster_.feature_importance(importance_type="gain"), index=M1)
        imp[t] = (g[pcs].sum() / g.sum(), g)
        for ev_name, ev in evals.items():
            y, dm, om = ev[t].to_numpy(), dv[t].mean(), ev[t].mean()
            p = {k: m.predict(ev[f]) for k, (m, f) in ms.items()}
            for k in ms:
                r = metrics(y, p[k], dm)
                rows.append({"eval": ev_name, "target": t, "model": k, "n": len(ev), "trees": int(trees[t]),
                             "MAE": r["MAE"], "RMSE": r["RMSE"], "R2_oos": r["R2_oos"], "R2_vs_own_mean": r["R2_val"]})
            days = ev.session_date.to_numpy()
            p0, p1 = p["B1b-GBM"], p["M1 = B1b-GBM + text"]
            lo, hi = day_bootstrap(y, p0, p1, days, dm)
            lo_o, hi_o = day_bootstrap(y, p0, p1, days, om)
            rows[-1].update({"dR2_oos": rows[-1]["R2_oos"] - rows[-2]["R2_oos"], "dR2_ci_lo": lo, "dR2_ci_hi": hi,
                             "dR2_own": rows[-1]["R2_vs_own_mean"] - rows[-2]["R2_vs_own_mean"],
                             "dR2_own_ci_lo": lo_o, "dR2_own_ci_hi": hi_o})
    res = pd.DataFrame(rows)
    res.to_csv(OUT_RES, index=False)

    pd.set_option("display.width", 250)
    for ev_name in evals:
        r = res[res["eval"] == ev_name]
        print(f"\n=== {ev_name} (n = {r.n.iloc[0]:,}) ===")
        print(r[["target", "model", "MAE", "RMSE", "R2_oos", "R2_vs_own_mean", "dR2_oos", "dR2_ci_lo", "dR2_ci_hi",
                 "dR2_own", "dR2_own_ci_lo", "dR2_own_ci_hi"]].round(4).to_string(index=False))
    print("\nM1 feature importance (gain share): " + ", ".join(f"{t}: text {s:.1%} / tabular {1 - s:.1%}"
                                                              for t, (s, _) in imp.items()))
    g = imp["v_post"][1]; sh = g / g.sum()
    print("v_post tabular gain shares: " + ", ".join(f"{k} {v:.1%}" for k, v in sh[FEATURES].sort_values(ascending=False).items()))
    print(f"\nPCA components: {len(pcs)}. Saved {OUT_RES}")


if __name__ == "__main__":
    {"prep": stage_prep, "embed": stage_embed, "pca": stage_pca, "model": stage_model}[sys.argv[1]]()
