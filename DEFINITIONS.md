# Definitions

The exact definition of every key quantity, taken from the code (source file in brackets). Paths are relative to
`vol_pipeline/` unless they start with `pipeline_extraction/`.

## Volatility

**Realized volatility**: how much the stock swung per day, on a log scale.

    v = ln( √( mean of r_t² ) )        r_t = CRSP daily total return (dlyret)

Reading the log scale: −3.5 ≈ 3% a day, −3.0 ≈ 5%, −2.5 ≈ 8% (daily vol = e^v). `[pipeline/volatility.py]`

**v_post (the prediction target)**: v over the **5 trading sessions** starting at the first session whose 09:30 open
comes *after* the release time. All 5 returns must be present.
- Pre-open release: the window starts that day. Intraday or after-close release: it starts the next session.
- `v_post_3` and `v_post_15` are the 3- and 15-session versions.
- `[pipeline/windows.py, pipeline/build.py, config.py: POST_HORIZONS]`

**v_pre (past volatility)**: v over the **20 trading sessions** ending at the last close *before* the release; at least
15 returns required. The two windows never overlap, and an intraday release day is in neither.
`[pipeline/windows.py, config.py: PRE_DAYS=20, MIN_PRE_OBS=15]`

## Earnings surprise (SUE)

    sue_price = (actual EPS − median analyst estimate) ÷ share price two sessions before the reaction day

- **Which estimates count:** each analyst's latest I/B/E/S estimate for the same fiscal quarter, both issued and recorded
  *before* the release date and within 90 days of it, from a broker still covering the firm.
- **Share basis:** estimates are rescaled to the actual's share basis, and the price is adjusted with CRSP's share factor.
- **Sources:** `[pipeline/sue.py: add_sue, build_consensus]`, copied from
  `[pipeline_extraction/3_4_surprise_returns/code/surprise.py]`. The formula is identical in both files.
- **In the models** (`[baselines_b0_b1_fixed.py: add_surprise_features]`):
  - `sue_price_w`: signed SUE, clipped at the 2006–16 1st and 99th percentiles;
  - `abs_sue_rank`: size of the surprise as a rank, i.e. the share of 2006–16 |SUE| values at or below it.

  Both cut-offs are fit on 2006–16 data only.

## Other features

All of these are known before the release. `[pipeline/build.py, pipeline/sue.py]`

| Feature | Definition |
|---|---|
| **log_mktcap** | Natural log of CRSP market value (`dlycap`, in $000s) on the last trading day before the release |
| **sector** | SIC division (agriculture, mining, construction, manufacturing, transport and utilities, wholesale, retail, finance, services, public administration), from the CRSP SIC code |
| **when** | Release timing from the filing's acceptance time: `pre_open` (before 09:30 ET), `intraday`, `after_close` (16:00 ET or later) `[pipeline_extraction/1_events/code/build_event_table.py]` |
| **n_analysts** | Number of analysts in the point-in-time consensus used for SUE |

## Decision metrics

- **Big move:** realized 5-day volatility (`v_post`) above the **top-quartile cut-off of 2006–16 events**, which is
  **4.86% a day** (v > −3.024).
- **Alert:** an event whose forecast is in the **model's own top 25% of forecasts**. Every model gets the same alert budget.
- **Catch rate:** share of big moves that were alerted. **Precision:** share of alerts that were big moves.
- **Sources:** `[risk_limit_m1.py]`; `[notebooks/03_risk_limit_impact.ipynb]` for the full-sample B0 vs B1b comparison.

## Accuracy (R²)

    R² = 1 − Σ(actual − forecast)² ÷ Σ(actual − benchmark mean)²

This is the share of variation explained on years the model never saw: 1 is perfect, 0 is no better than the benchmark.
`[baselines_b0_b1.py: metrics]`

There are two benchmark means:
- **Validation:** the 2006–16 average, the only one known at forecast time.
- **Holdout:** the holdout's own average (`R2_vs_own_mean`). 2021–25 is more volatile than 2006–16, so measuring against
  the old average would inflate R². The 2006–16-mean versions are reported alongside.

Text-gain 95% intervals come from a paired bootstrap that resamples whole trading days.

## Models

| Model | Type | What it adds |
|---|---|---|
| **B0** | Linear regression (OLS) | Past volatility (`v_pre`) only: the standard statistical baseline `[baselines_b0_b1.py]` |
| **B1b** | Ridge regression (alpha = 1, fixed) | + SUE (`sue_price_w`, `abs_sue_rank`), `log_mktcap`, `n_analysts`, `sector`, `when` `[baselines_b0_b1_fixed.py]` |
| **B1b-GBM** | Gradient boosting (LightGBM: 31 leaves, learning rate 0.05, fixed tree counts) | Same inputs as B1b; captures nonlinear effects `[baselines_gbm.py]` |
| **M1** | B1b-GBM + text | The release text: first 500 words → FinBERT-tone (first 256 tokens, mean-pooled to 768 numbers) → PCA to 48 components, fit on 2006–16 `[m1_text_v2.py]` |

**Splits** are by year of the reaction session: dev 2006–16, validation 2017–19, 2020 excluded, holdout 2021–25 (run
once). `[config.py, pipeline/sue.py: split_of]`

## Large-cap subsample

Every text result is measured on this subsample: M1, the text-gain intervals, and the M1 decision test.

- **What it is:** the primary-sample events whose release text was extracted and cached in step 5. In other words, the
  event's accession is in `raw_manifest.parquet` and has a cleaned text in `clean_YYYY.parquet`. Rows: 44,739 dev,
  14,614 validation, 27,629 holdout. `[m1_text_v2.py: accession_map, subset_rows]`
- **What it is not:** a size cut-off. No market-cap, price or analyst filter was applied to select it.
- **Why "large-cap":** the events with text happen to be large, well-covered firms: median market cap ~$3.7B and 7
  analysts, against a median of 2 analysts for primary-sample events without text. This is an observed property of text
  coverage, not a filter we applied. These medians are computed from `model_table.parquet` (licensed, not committed), so
  `verify_results.py` does not check them. `[deck/build_pptx.py]`
- **Earlier version:** `baselines_textsubset.py` (dev and validation only) uses the same idea, with step 5's raw archives
  for 2006–18.
