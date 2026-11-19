"""M1: does earnings-release text add predictive value over the tabular bar (B1b-GBM) on the text subset?

SUPERSEDED (v1) by m1_text_v2.py, which gives the reported text results. This file is kept for two reasons:
m1_text_v2.py imports TRANSCRIPT_RE and day_bootstrap from it, and the deck cites its 2017-18 result (results_m1.csv).
Its sample is the v1 text subset (2006-18 only: dev and validation 2017-18, no holdout). The medians it prints
(~$3.1B market cap) are therefore not the ~$3.7B of the current subset (dev, validation and holdout; DEFINITIONS.md).

Stages (each cached; rerun only what is missing):
  python m1_text.py clean   -> data/processed/text_clean/text_clean.parquet   (event_id, cleaned first ~500 words)
  python m1_text.py embed   -> data/processed/text_features.parquet           (FinBERT mean-pooled 768-d + PCA)
  python m1_text.py model   -> data/processed/results_m1.csv                  (B1b-GBM vs M1)

Rows: the text subset from baselines_textsubset.py (44,739 dev / 9,651 validation), dev + validation only via the same
load_locked() guard. Holdout and 2020 are never loaded. Nothing is fit on validation.

Choices confirmed with the user:
  - Both models use the fixed tree counts (173 / 171 / 135) on all subset dev; an early-stopped run of both
    (latest 10% of subset dev by session date) is reported as a sensitivity check.
  - Call transcripts are flagged and their text features set to missing; their rows stay, so both models use
    identical rows.
  - Encoder: yiyanghkust/finbert-tone, frozen, fp16 on the M2 GPU, mean-pooled last hidden state, first 256 tokens
    (~190 words: headline + highlights), batch 8, GPU cache released per batch, GPU memory capped.
"""
import re
import sys
import time
import zipfile

import numpy as np
import pandas as pd

import config as C
from baselines_b0_b1 import CAT_B1, NUM_B1, TARGETS, load_locked, metrics
from baselines_b0_b1_fixed import add_surprise_features
from baselines_gbm import CATS, FEATURES, GBM_PARAMS
from baselines_textsubset import MANIFEST, TEXT_DIR, PRIOR_GBM, as_categories, text_covered_ids

CLEAN_DIR = C.OUT / "text_clean"
CLEAN_FILE = CLEAN_DIR / "text_clean.parquet"
EMB_PARTS = C.OUT / "text_embeddings_parts"          # resumable chunks
EMB_FILE = C.OUT / "text_embeddings.parquet"         # 768-d per event; NaN for empty documents
FEAT_FILE = C.OUT / "text_features.parquet"          # dev-fit PCA components
OUT_RES = C.OUT / "results_m1.csv"
MODEL_NAME = "yiyanghkust/finbert-tone"
KEEP_WORDS, MAX_TOKENS, BATCH = 500, 256, 8
CHUNKS_PER_RUN = 4                                  # embed stage exits(3) after this many new chunks
PCA_TARGET, PCA_CAP = 0.90, 50

TRANSCRIPT_PATTERNS = [
    r"please (?:go ahead and )?begin the question[- ]and[- ]answer session",
    r"(?:that )?concludes (?:our|the|today'?s) question[- ]and[- ]answer session",
    r"all participants are (?:now )?in (?:a )?listen[- ]only mode",
    r"we will (?:now )?(?:begin|conduct|open) (?:the|a) question[- ]and[- ]answer session",
    r"\boperator\s+instructions\b",
]
TRANSCRIPT_RE = re.compile("|".join(TRANSCRIPT_PATTERNS), re.I)
NUMLIKE = re.compile(r"^[\(\)\$€£%\-–—.,:;/*xX0-9]+$")


def subset_rows() -> pd.DataFrame:
    df = load_locked()                                              # holdout guard
    df = df.dropna(subset=[*TARGETS, *NUM_B1, *CAT_B1]).copy()
    for c in CAT_B1:
        df[c] = df[c].astype(str)
    df = df[df["event_id"].isin(text_covered_ids())].copy()
    n = df.groupby("split").size().to_dict()
    assert n == {"dev": 44_739, "validation": 9_651}, f"text subset changed: {n}"
    return df


# ------------------------------------------------------------------------------------------------ step 1
def clean_doc(raw: str, is_html: bool) -> str:
    from bs4 import BeautifulSoup
    m = re.search(r"<TEXT>(.*?)(?:</TEXT>|\Z)", raw, re.S | re.I)          # strip SGML <DOCUMENT> wrapper
    body = m.group(1) if m else raw
    if is_html or re.search(r"<(html|body|p|div|table|font)\b", body[:5000], re.I):
        soup = BeautifulSoup(body, "lxml")
        for t in soup(["script", "style"]):
            t.decompose()
        body = soup.get_text("\n")
    body = body.replace("\xa0", " ")
    keep = []
    for line in body.split("\n"):
        toks = line.split()
        if not toks:
            continue
        numeric = sum(bool(NUMLIKE.match(t)) for t in toks)
        if numeric / len(toks) >= 0.4 and numeric >= 2:                    # table row: mostly numbers
            continue
        if re.fullmatch(r"(page \d+( of \d+)?|\d+|exhibit 99\.?\d*|ex-99\.?\d*)", line.strip(), re.I):
            continue                                                       # page numbers, exhibit labels
        keep.append(" ".join(toks))
    return re.sub(r"\s+", " ", " ".join(keep)).strip()


def _clean_job(job):
    """Worker: clean one chunk of one year's documents (runs in a separate process)."""
    import warnings
    warnings.filterwarnings("ignore")
    y, rows = job
    z = zipfile.ZipFile(TEXT_DIR / f"raw_{y}.zip")
    out = []
    for event_id, accession, filename, source in rows:
        raw = z.read(f"{accession}__{filename}").decode("utf-8", errors="replace")
        text = clean_doc(raw, filename.lower().endswith((".htm", ".html")))
        words = text.split(" ") if text else []
        out.append({"event_id": event_id, "source": source, "n_words_clean": len(words),
                    "is_transcript": bool(TRANSCRIPT_RE.search(text)), "text500": " ".join(words[:KEEP_WORDS])})
    return out


def stage_clean():
    if CLEAN_FILE.exists():
        print(f"cached: {CLEAN_FILE}"); return
    df = subset_rows()
    man = pd.read_parquet(MANIFEST, columns=["cik", "accession", "filename", "source"])
    man["event_id"] = man["cik"] + "_" + man["accession"]
    man = man[man["event_id"].isin(set(df["event_id"]))]
    man["year"] = 2000 + man["accession"].str.split("-").str[1].astype(int)
    from multiprocessing import Pool
    jobs = [(y, rows) for y, g in man.groupby("year")
            for rows in np.array_split(g[["event_id", "accession", "filename", "source"]].to_numpy(), 4)]
    out, t0 = [], time.time()
    with Pool(7) as pool:
        for k, part in enumerate(pool.imap_unordered(_clean_job, jobs), 1):
            out.extend(part)
            if k % 7 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} chunks, {len(out):,} docs  ({time.time() - t0:.0f}s)", flush=True)
    res = pd.DataFrame(out)
    res = res.merge(df[["event_id", "split"]], on="event_id")
    assert len(res) == len(df), "every subset event must have a cleaned document"
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    res.to_parquet(CLEAN_FILE, index=False)
    w = res["n_words_clean"]
    print(f"\ncleaned {len(res):,} documents -> {CLEAN_FILE}")
    print(f"transcripts flagged (text set to missing): {res.is_transcript.sum():,} "
          f"({res.is_transcript.mean():.2%}); by split {res.groupby('split').is_transcript.sum().to_dict()}")
    print("cleaned words per document (full cleaned text): "
          + ", ".join(f"{k} {v:,.0f}" for k, v in w.describe(percentiles=[.05, .25, .5, .75, .95]).items()))
    kept = res["text500"].str.split(" ").str.len()
    print(f"words kept (first {KEEP_WORDS}): median {kept.median():.0f}; documents shorter than {KEEP_WORDS} words: "
          f"{(w < KEEP_WORDS).mean():.1%}; empty after cleaning: {(w == 0).sum()}")


# ------------------------------------------------------------------------------------------------ step 2
def stage_embed():
    import os
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.6")   # cap GPU memory (set before torch loads)
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")    # must sit below the high watermark
    import torch
    from sklearn.decomposition import PCA
    from transformers import AutoModel, AutoTokenizer
    if FEAT_FILE.exists():
        print(f"cached: {FEAT_FILE}"); return
    txt = pd.read_parquet(CLEAN_FILE)
    todo = txt[~txt["is_transcript"] & txt["text500"].str.len().gt(0)].reset_index(drop=True)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).half().to(dev).eval()
    EMB_PARTS.mkdir(parents=True, exist_ok=True)
    order = todo["text500"].str.len().sort_values().index                    # length-sorted batches = less padding
    chunks = np.array_split(order.to_numpy(), int(np.ceil(len(order) / 2000)))
    t0, done_now = time.time(), 0
    for i, idx in enumerate(chunks):
        part = EMB_PARTS / f"part_{i:03d}.parquet"
        if part.exists():
            continue
        if done_now >= CHUNKS_PER_RUN:          # restart the process to release GPU/host memory; caller loops
            print(f"  pausing after {done_now} chunks to reset memory", flush=True)
            sys.exit(3)
        done_now += 1
        vecs = []
        for b in range(0, len(idx), BATCH):
            batch = todo.loc[idx[b:b + BATCH], "text500"].tolist()
            enc = tok(batch, padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt").to(dev)
            with torch.no_grad():
                h = model(**enc).last_hidden_state.float()          # pool in fp32
            mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            vecs.append(((h * mask).sum(1) / mask.sum(1)).cpu().numpy())
            del h, enc
            if dev.type == "mps":
                torch.mps.empty_cache()                               # release cached GPU memory every batch
        e = pd.DataFrame(np.vstack(vecs), columns=[f"emb_{j}" for j in range(768)])
        e.insert(0, "event_id", todo.loc[idx, "event_id"].to_numpy())
        e.to_parquet(part, index=False)
        print(f"  chunk {i + 1}/{len(chunks)}  {sum(len(c) for c in chunks[:i + 1]):,} docs  ({time.time() - t0:.0f}s)",
              flush=True)
    cols = [f"emb_{j}" for j in range(768)]
    emb = pd.concat([pd.read_parquet(p) for p in sorted(EMB_PARTS.glob("part_*.parquet"))], ignore_index=True)
    assert emb["event_id"].is_unique and len(emb) == len(todo), "embedding chunks incomplete"
    emb = txt[["event_id", "split"]].merge(emb, on="event_id", how="left")   # empty docs -> NaN vector (not zeros)
    print(f"\nembedded {emb[cols[0]].notna().sum():,} documents; NaN vectors for {emb[cols[0]].isna().sum()} "
          f"empty/excluded documents")
    emb.drop(columns="split").to_parquet(EMB_FILE, index=False)

    fit = emb.loc[(emb.split == "dev") & emb[cols[0]].notna(), cols].to_numpy()       # DEV ONLY
    pca = PCA(n_components=PCA_CAP, random_state=0).fit(fit)
    cum = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cum, PCA_TARGET) + 1) if cum[-1] >= PCA_TARGET else PCA_CAP
    have = emb[cols[0]].notna()
    z = np.full((len(emb), k), np.nan)
    z[have.to_numpy()] = pca.transform(emb.loc[have, cols].to_numpy())[:, :k]
    feats = pd.DataFrame(z, columns=[f"txt_pc{j:02d}" for j in range(k)])
    feats.insert(0, "event_id", emb["event_id"].to_numpy())
    feats.to_parquet(FEAT_FILE, index=False)
    print(f"PCA fit on {len(fit):,} dev embeddings only; kept {k} components = {cum[k - 1]:.1%} of dev variance "
          f"({'reached' if cum[-1] >= PCA_TARGET else 'NOT reached; capped at ' + str(PCA_CAP)} {PCA_TARGET:.0%} target; "
          f"first 5: {np.round(pca.explained_variance_ratio_[:5], 3).tolist()})")
    print(f"-> {EMB_FILE}\n-> {FEAT_FILE}")


# ------------------------------------------------------------------------------------------------ step 3-4
def fit_predict(dev, val, t, feats, trees=None, es=None):
    import lightgbm as lgb
    cat = [c for c in CATS if c in feats]
    if es is None:
        m = lgb.LGBMRegressor(**{**GBM_PARAMS, "n_estimators": int(trees)})
        m.fit(dev[feats], dev[t], categorical_feature=cat)
        return m, m.predict(val[feats]), int(trees)
    fit, stop = es
    m = lgb.LGBMRegressor(**GBM_PARAMS)
    m.fit(fit[feats], fit[t], eval_set=[(stop[feats], stop[t])], categorical_feature=cat,
          callbacks=[lgb.early_stopping(50, verbose=False)])
    return m, m.predict(val[feats], num_iteration=m.best_iteration_), int(m.best_iteration_)


def day_bootstrap(y, p0, p1, days, dev_mean, n=1000, seed=0):
    """Paired bootstrap of the R2 difference, resampling whole session dates (events on a day are correlated)."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(days)
    groups = [np.flatnonzero(codes == k) for k in range(len(uniq))]
    out = []
    for _ in range(n):
        idx = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        sst = np.sum((y[idx] - dev_mean) ** 2)
        out.append((np.sum((y[idx] - p0[idx]) ** 2) - np.sum((y[idx] - p1[idx]) ** 2)) / sst)
    return np.percentile(out, [2.5, 97.5])


def stage_model():
    df = subset_rows()
    (dev, val), _ = add_surprise_features(df[df.split == "dev"], df[df.split == "validation"])  # fit on subset dev
    dev, val = as_categories(dev, val)
    feats = pd.read_parquet(FEAT_FILE)
    pcs = [c for c in feats.columns if c.startswith("txt_pc")]
    dev = dev.merge(feats, on="event_id", how="left")                        # transcripts -> NaN text features
    val = val.merge(feats, on="event_id", how="left")
    assert (len(dev), len(val)) == (44_739, 9_651)
    print(f"Rows: dev {len(dev):,} | validation {len(val):,}; text features missing (transcripts / empty) "
          f"dev {dev[pcs[0]].isna().sum()}, validation {val[pcs[0]].isna().sum()}")
    prior = pd.read_csv(PRIOR_GBM)
    trees = prior[prior.model == "B1b-GBM"].set_index("target")["best_iteration"].to_dict()
    cut = dev["session_date"].sort_values().iloc[int(len(dev) * 0.9)]
    es = (dev[dev.session_date < cut], dev[dev.session_date >= cut])
    print(f"Sensitivity early-stopping slice (subset dev): {es[1].session_date.min().date()} .. "
          f"{es[1].session_date.max().date()} ({len(es[1]):,} rows)\n")

    rows, imp = [], None
    M1 = FEATURES + pcs
    for t in TARGETS:
        y, dm = val[t].to_numpy(), dev[t].mean()
        for rule, kw in [("fixed trees", {"trees": trees[t]}), ("early stopping (sensitivity)", {"es": es})]:
            m0, p0, n0 = fit_predict(dev, val, t, FEATURES, **kw)
            m1, p1, n1 = fit_predict(dev, val, t, M1, **kw)
            lo, hi = day_bootstrap(y, p0, p1, val["session_date"].to_numpy(), dm)
            for name, p, n in [("B1b-GBM", p0, n0), ("M1 = B1b-GBM + text", p1, n1)]:
                rows.append({"rule": rule, "target": t, "model": name, "trees": n, **metrics(y, p, dm)})
            rows[-1].update({"dR2_vs_bar": rows[-1]["R2_oos"] - rows[-2]["R2_oos"], "dR2_ci_lo": lo, "dR2_ci_hi": hi})
            if t == "v_post" and rule == "fixed trees":
                g = pd.Series(m1.booster_.feature_importance(importance_type="gain"), index=M1)
                imp = g
    res = pd.DataFrame(rows)
    res.to_csv(OUT_RES, index=False)

    pd.set_option("display.width", 200)
    for rule in res.rule.unique():
        r = res[res.rule == rule]
        print(f"=== {rule} ===")
        print(r[["target", "model", "trees", "MAE", "RMSE", "R2_oos", "dR2_vs_bar", "dR2_ci_lo", "dR2_ci_hi"]]
              .round(4).to_string(index=False), "\n")
    bar = res[(res.rule == "fixed trees") & (res.target == "v_post") & (res.model == "B1b-GBM")].R2_oos.iloc[0]
    print(f"Bar check: B1b-GBM v_post R2 = {bar:.4f} (expected 0.2679 from results_textsubset_baseline.csv)")
    share = imp / imp.sum()
    print(f"\nM1 feature importance, v_post (gain share): text PCs total {share[pcs].sum():.1%}, "
          f"tabular total {share[FEATURES].sum():.1%}")
    print(share[FEATURES].sort_values(ascending=False).map("{:.1%}".format).to_string())
    print("top 5 text components:", share[pcs].sort_values(ascending=False).head(5).map("{:.1%}".format).to_dict())
    print("\nCaveats:")
    print("  - Large-cap, well-covered subsample (median market cap ~$3.1B, 7 analysts); not representative of small caps.")
    print("  - Validation is 2017-18 only (no 2019 text); holdout (2021-25) not run; 2020 not loaded.")
    print(f"  - Text: frozen yiyanghkust/finbert-tone (fp16), first 256 tokens (~190 words), mean-pooled; PCA ({len(pcs)} components) "
          "fit on dev embeddings only.")
    print("  - 20 documents empty after cleaning kept with missing (NaN) text features; 0 call transcripts in the subset.")
    print("  - 95% intervals: paired bootstrap over validation session dates (models fixed, no refitting).")
    print(f"\nSaved {OUT_RES}")


if __name__ == "__main__":
    {"clean": stage_clean, "embed": stage_embed, "model": stage_model}[sys.argv[1]]()
