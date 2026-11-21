# vol_pipeline — post-earnings volatility dataset

Builds the modeling table for predicting **realized volatility after 8-K earnings releases** (magnitude, not returns), then
runs the model ladder and writes the results (see [Models](#models)). Everything runs from this folder: code, inputs, outputs
and tests.

## Run

```sh
cd vol_pipeline
pip install -r requirements.txt
# first place event_table.csv (pass 2) and trading_calendar.csv in data/raw/ (see ../REPRODUCE.md, step 2), then:
python setup_inputs.py --from-cache /path/to/3_4_surprise_returns/cache   # copies the WRDS cache into data/raw/wrds_cache/
python run.py                   # about 1.5 minutes; add --rebuild-sue to recompute SUE
python -m unittest discover -s tests
```

`setup_inputs.py` does not run standalone in this repository. Its default sources (`SETUP_SOURCES` in `config.py`) point to a
data-folder layout that is not part of this repo, so a plain `python setup_inputs.py` stops with "not found". Follow
[REPRODUCE.md](../REPRODUCE.md), step 2: put the two CSVs in `data/raw/` yourself, then pass `--from-cache` with the step 3–4
WRDS cache. The script skips files already in `data/raw/`, so only the cache is copied.

On macOS the copies are APFS clones and take no extra disk space. `data/` is git-ignored because it holds licensed WRDS data.

## Layout

| Path | What it is |
|---|---|
| `config.py` | Every path and parameter, in one place |
| `run.py` | Entry point; writes the outputs and `build_log.txt` |
| `setup_inputs.py` | Fills `data/raw/` |
| `pipeline/trading_calendar.py` | NYSE trading days with close times, including early closes |
| `pipeline/windows.py` | Pre and post windows from `release_et` |
| `pipeline/linking.py` | CIK → PERMNO, ticker, 8-K → SUE link |
| `pipeline/volatility.py` | `ln(sqrt(mean(r²)))` |
| `pipeline/sue.py` | The part 3 SUE method, copied from `3_4_surprise_returns/code/` |
| `pipeline/build.py` | Runs every step in order and prints the report |
| `tests/` | Synthetic tests of the windows and the volatility formula |
| `data/raw/` | Inputs: `event_table.csv`, `trading_calendar.csv`, `wrds_cache/` |
| `data/processed/` | Outputs |

## Method

**Windows.** Windows are computed from `release_et` and the trading calendar, never from calendar days.

- **Post window:** starts at the first session whose 09:30 open is after `release_et`. That is D for a pre_open release, D+1 for intraday or after_close, and the next trading day for a weekend or holiday release.
- **Pre window:** the 20 sessions ending at the last close before `release_et`, with early closes respected.
- **Intraday releases:** day D is in neither window.
- **Checks:** the build stops with an error if the windows overlap or fall on the wrong side of the release.

**Targets.** `v = ln(sqrt(mean(r_t²)))` on CRSP `dlyret`.

- `v_post_3`, `v_post` (5 days) and `v_post_15` require every return in the window.
- `v_pre` uses the 20 days before the release and requires at least 15 of them.

**Features (B1).** All are known before the release.

- `sue_price`: (actual − median point-in-time consensus) / price at t−2. This is part 3's primary definition. `sue_disp` is the alternative.
- `log_mktcap` and `sector` (SIC division): taken on the last trading day before the release.
- `when`: pre_open, intraday or after_close.

**Linking.**

- **8-K → stock:** CIK → gvkey → PERMNO through the CRSP–Compustat link tables, valid on the release date.
- **8-K → SUE:** same PERMNO, with the filing 1 day before to 4 days after the IBES release.
- **SUE guard:** every analyst estimate and the t−2 price must predate the 8-K.

**Primary sample.** Only events where the 8-K and IBES put the release in the same session. Otherwise the press release came out before the 8-K, and `release_et` would not be when the news became public.

**Splits.** These are the team's, by year of `session_date`: dev 2006–16, validation 2017–19, excluded_2020, holdout 2021–25. Apply the holdout lock before any analysis.

## Outputs (`data/processed/`)

| File | Rows | Contents |
|---|---|---|
| `model_table.parquet` | 168,906 | Primary sample, sorted by `release_et` |
| `model_table_all.parquet` | 299,379 | Before the session-agreement filter, for robustness checks |
| `event_windows.parquet` | 168,906 | Every window date, plus the original stored `session_date` |
| `sue_ibes_events.parquet` | 217,609 | Rebuilt SUE, one row per IBES event |
| `trading_calendar_closes.csv` | | Calendar with close times |
| `build_log.txt` | | Full printout: column mapping, rows dropped per step, statistics |

`model_table` columns:
- **Keys:** event_id, cik, accession, permno, ticker, release_et, when, session_date, split
- **Targets:** v_post_3, v_post, v_post_15
- **B0:** v_pre
- **B1:** sue_price, sue_disp, n_analysts, log_mktcap, sector, sic2
- **Audit:** has_sue, sue_same_session, sue_event_id

## Models

The model ladder runs on `model_table.parquet`. Run order and inputs: [REPRODUCE.md](../REPRODUCE.md), step 2. Exact
definitions: [DEFINITIONS.md](../DEFINITIONS.md). Results go to `data/processed/` and are committed (small, no row-level data).

| Script | What it does | Writes |
|---|---|---|
| `baselines_b0_b1.py` | First B0/B1 run; still provides shared helpers (`metrics`, `load_locked`) | `results_b0_b1.csv` (superseded) |
| `baselines_b0_b1_fixed.py` | **B0** (OLS on `v_pre`) and **B1b** (ridge + surprise, size, sector, timing, coverage), validation | `results_b0_b1_fixed.csv` |
| `baselines_gbm.py` | **B1b-GBM** (LightGBM, same inputs); sets the tree counts the later scripts reuse | `results_gbm.csv` |
| `baselines_textsubset.py` | Is the text subsample representative? The same ladder on text-covered rows only (dev and validation) | `text_subset_representativeness.csv`, `results_textsubset_baseline.csv` |
| `baselines_holdout.py` | The one-time holdout (2021–25) for B0, B1b and B1b-GBM (`check`, then `final`) | `results_holdout.csv` |
| `notebooks/03_risk_limit_impact.ipynb` | Decision test, B0 vs B1b, full sample | `results_risk_limit.csv` |
| `m1_text_v2.py` | **M1** = B1b-GBM + release text (FinBERT → PCA), validation and holdout, with bootstrap intervals | `results_m1_v2.csv` |
| `m1_text.py` | Earlier text layer (2006–18 only); kept for shared helpers and the result the deck cites | `results_m1.csv` (superseded) |
| `risk_limit_m1.py` | Decision test, M1 vs B1b-GBM (large-cap subsample, validation) and the one-session example | `results_risk_limit_m1.csv`, `results_risk_limit_m1_session.csv` |
| `verify_results.py` | Checks 50 reported numbers (37 on the deck slides, 2 in the speaker notes, 11 from the results files) against the files above; needs only pandas | — |
| `deck/build_pptx.py`, `deck/pptx_to_pdf.py` | Build the deck (`deck/pitch_deck.pptx`, then PDF). The sample medians (~$3.7B, 7 analysts) are read from `model_table.parquet`, which is licensed and not committed | `deck/pitch_deck.pptx` |
| `deck/data_pipeline.dot.py`, `deck/plots_pred_timing.py` | Deck figures | `notebooks/figures/*.png` |

## Known limits

- **No wire clock:** IBES announcement times stand in for the Dow Jones wire clock in SUE, because the wire file is not available.
- **No winsorizing:** the lowest targets come from illiquid stocks.
- **Intraday is nearly absent** from the primary sample (343 rows), because most intraday 8-Ks follow an earlier press release.
- **The holdout is more volatile:** mean `v_post` is −3.31 in 2021–25, against −3.51 in dev.
