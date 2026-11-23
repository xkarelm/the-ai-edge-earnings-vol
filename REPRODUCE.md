# Reproducing the results from raw data

For the fast check of every reported number, which needs no WRDS access, run `cd vol_pipeline && python verify_results.py`
(see the [README](README.md)). This file covers the full rebuild from WRDS and EDGAR.

**Status:** every input can be obtained with the credentials below, but some steps are manual, and a fresh WRDS pull
matches the results only approximately (see Data versions below).

**Data sources**
- **WRDS tables:** `crsp.dsf_v2`, `crsp.stocknames_v2`, `crsp.ccmxpf_lnkhist`, `wrdssec.wciklink_gvkey`, `wrdsapps.ibcrsphist`,
  `ibes.actu_epsus`, `ibes.detu_epsus`, `ibes.stopu_epsus`.
- **Entitlements:** besides CRSP and I/B/E/S, you need **CRSP/Compustat Merged** (`ccmxpf_lnkhist`) and the **WRDS SEC Analytics
  Suite** (`wrdssec`). No OptionMetrics.
- **RavenPack:** only step 1's optional wire-time check (`wire_times.py`) uses it; the results do not need it.
- **SEC EDGAR:** public and free. Set `SEC_USER_AGENT` to `"Your Name your@email"`; SEC refuses requests without a real contact.

**Credentials and paths**

| Variable | Used by |
|---|---|
| `WRDS_USERNAME` | `1_events/code/` WRDS scripts (`wrds_match_and_funnel.py` takes `--user`). The step 3–4 notebooks prompt for the username. The password and the Duo (two-factor) approval are always asked for interactively |
| `SEC_USER_AGENT` | Every EDGAR download (`1_events/code/`, `5_data_cleaning/code/cache_raw.py`, `2_text_extraction`) |
| `TEAM_DIR`, `PROJECT_DIR`, `STEP5_DIR` | Point the extraction code at its data folders. See `pipeline_extraction/README.md` |
| `HF_TOKEN` | Optional: only for the gated Llama tokenizer in step 5's token counts. FinBERT-tone needs no token, but the embedding step downloads it, so it needs internet access |

**Environments:** the steps need different Python and numpy versions, so use a separate environment for each.
- **Modeling:** `vol_pipeline/requirements.txt` (Python 3.12, numpy 1.26).
- **Steps 3–4:** `pipeline_extraction/3_4_surprise_returns/requirements.txt` (Colab, Python 3.13, numpy 2.x).
- **Steps 1 and 5:** have no requirements file. They need `wrds`, `pandas`, `pyarrow`; step 5 also needs `nltk` and `datasketch`.
- **Deck:** Graphviz (`dot`), plus Google Chrome at the macOS path in `deck/pptx_to_pdf.py`.

**1. Extraction** (`pipeline_extraction/`). The step 1 notebook's automatic runner does not work, so run these by hand, in this
order, inside `1_events/code/`:

```sh
python funnel_counts.py --zip submissions.zip                 # downloads EDGAR's bulk submissions.zip (several GB)
python build_event_table.py --zip submissions.zip             # pass 1: no header times yet (falls back to the faulty JSON)
python fetch_acceptance.py --events event_table.csv           # one EDGAR header per filing -> acceptance_times.csv
python build_event_table.py --zip submissions.zip             # pass 2: rerun, now reads acceptance_times.csv -> final event_table.csv
python wrds_match_and_funnel.py --user $WRDS_USERNAME         # needs ccm_link.csv (manual export, below) -> matched_sample.csv
python fetch_releases.py --events matched_sample.csv          # chooses each release document -> extraction_report_full.csv
```

Then:
- **Steps 3–4:** run notebooks `00`–`04` in `3_4_surprise_returns/`, which pull WRDS into `cache/` and write
  `data/trading_calendar.csv`.
- **Step 5:** run `5_data_cleaning/code/cache_raw.py --report <step 1's extraction_report_full.csv>`, then `clean_release.py`.
  Called directly, they need no `project_paths.py`. They write `cache/raw_manifest.parquet` and `data/clean/clean_YYYY.parquet`.

**2. Modeling** (`cd vol_pipeline`). Place the inputs first. `setup_inputs.py` skips files already in `data/raw/`:
- `data/raw/event_table.csv`: the **pass-2** event table, which has `release_et`;
- `data/raw/trading_calendar.csv`: from step 3–4's `data/`;
- the WRDS cache: `python setup_inputs.py --from-cache <3_4 cache>`;
- `data/parquets/clean_YYYY.parquet` and `data/raw_manifest.parquet`: from step 5;
- `data/rawTexts/raw_YYYY.zip` for 2006–2018: step 5's raw archives (`cache/raw/raw_YYYY.zip`). `baselines_textsubset.py`
  needs them; it reads only the member names, never the text.

Then:

```sh
python run.py                          # model_table.parquet (168,906 events) + build_log.txt
python baselines_b0_b1.py && python baselines_b0_b1_fixed.py && python baselines_gbm.py && python baselines_textsubset.py
# notebooks/03_risk_limit_impact.ipynb -> results_risk_limit.csv (B0 vs B1b, full sample)
python baselines_holdout.py check      # then `final` (one-time holdout; see notes)
python m1_text_v2.py prep
until python m1_text_v2.py embed; do :; done   # exits by design every 4 chunks to free memory; MPS or CPU, several hours
python m1_text_v2.py pca && python m1_text_v2.py model
python risk_limit_m1.py                # M1 vs B1b-GBM decision metrics (validation only)
python verify_results.py && python -m unittest discover -s tests
```

**Known manual steps and gaps**
- **`ccm_link.csv` is a manual WRDS export, not produced by any script.** It is the CRSP/Compustat Merged link history with
  each gvkey's CIK, with columns `cik, gvkey, LPERMNO, LINKPRIM, LINKDT, LINKENDDT`, for example `crsp.ccmxpf_lnkhist`
  joined to the Compustat company header on gvkey. The script keeps `LINKPRIM` P and C.
- **`project_paths.py`** is not in this repository. Only the step 5 notebooks need it (the step 5 scripts above do not).
  To run them, supply a `scripts/project_paths.py` that reads the repository root from the **`DATA_ROOT`** environment variable and defines `REPO_ROOT`,
  `TEAM_DIR`, `NOV2_DATA`, `STEP5_DIR` and `STEP5_CODE` (see `pipeline_extraction/README.md`).
- **Holdout guard:** `baselines_holdout.py final` refuses to run while `results_holdout.csv` exists, and that file is committed.
  Move it aside to rerun the one-time holdout.
- **Data paths in the notebooks:** the step 1–4 notebooks (originally run in Google Colab) use the placeholder `path/to/data/...`. Replace it with your local data folder.
- **Interrupted text embedding:** if the embedding loop is killed mid-run, it resumes from saved chunks.

**Data versions**
- The event table the results use is the **pass-2** output: it has `release_et` and `acceptance_source`, with 383,153 rows.
  Older copies of `event_table.csv` lack `release_et`, and `run.py` fails on them; use the pass-2 table.
- CRSP and I/B/E/S are revised over time. A fresh pull reproduces the results **approximately**, not to the digit, and
  `m1_text_v2.py` checks exact row counts that a newer pull may not match. `verify_results.py` is the exact check against
  the committed results.

**Runtime** (measured on the team's machines)

| Step | Time |
|---|---|
| WRDS pulls (steps 3–4) | ~20 min |
| Raw EDGAR cache (`cache_raw.py`) | ~3.75 h |
| Text cleaning (`clean_release.py`) | ~12.5 h (4-core laptop) |
| FinBERT embedding (`m1_text_v2.py embed`) | several hours (Apple MPS or CPU; no CUDA path) |
| Dataset build and models | minutes |

**Superseded but kept**
- `results_b0_b1.csv` is the first B0/B1 run; `results_b0_b1_fixed.csv` replaces it. `baselines_b0_b1.py` still provides
  shared helpers.
- `m1_text.py` and `results_m1.csv` are the earlier text layer, replaced by `m1_text_v2.py`. `m1_text_v2.py` still imports
  from `m1_text.py`, and the deck cites the old result.
