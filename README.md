# Forecasting post-earnings volatility

Forecasting how much a stock will move in the days after an earnings release, and testing whether the **text of the
earnings release** adds anything beyond the numbers.

**Team:** Sacha Elalouf, Karel Moryoussef, Ana Cioata, Ghadi Ghaddaf

## The finding

- **Past volatility does most of the work.** How much the stock moved in the 20 trading days before the release explains
  most of what the models capture.
- **The release text adds a small but statistically clear gain.** For large, well-covered companies, adding the text raises
  R² (the share of the variation in volatility the model explains) by **+0.024 on validation (2017–19)** and **+0.038 on a
  one-time holdout (2021–25)**, years never used to build or tune the model. The holdout gain is measured against the
  holdout's own mean (+0.035 against the 2006–16 mean). Both 95% confidence intervals are above zero.
- **Scope:** the text result covers large caps only (median market cap ~$3.7B, 7 analysts), because release text was
  collected only for filings passing the extraction screen (price > $5, market cap above the NYSE 20th percentile, 3+
  analysts; see [DEFINITIONS.md](DEFINITIONS.md#large-cap-subsample)).

## What's in this repo

| Path | What it is |
|---|---|
| `vol_pipeline/` | The modeling work: builds the dataset, runs the model ladder (simple baseline → gradient boosting → text model), and writes the results |
| `vol_pipeline/deck/` | The pitch deck, **[`pitch_deck.pdf`](vol_pipeline/deck/pitch_deck.pdf)** (and `.pptx`), and the scripts that build it from the committed results files. The sample medians (~$3.7B, 7 analysts) need the licensed `model_table.parquet`, which is not committed; without it the build uses the documented values |
| `vol_pipeline/verify_results.py` | **Run this first.** Re-checks the reported numbers against the results files in about 2 seconds, with no data download |
| `vol_pipeline/data/processed/` | The small results files the deck and the check read. Raw and licensed data are not in the repo |
| `vol_pipeline/notebooks/` | Exploratory analysis and the risk-limit notebook; saved figures in `figures/` |
| `pipeline_extraction/` | The team's data-extraction pipeline, stages 1–5: SEC filings → events and timing → earnings surprise and returns → cleaned text. Code only |
| `REPRODUCE.md` | Full step-by-step instructions to rebuild everything from the raw data |

Please don't rename folders: the code refers to some of them by name (`pipeline_extraction/README.md` explains the naming).

## Check the results fast (no data needed)

```sh
cd vol_pipeline && python verify_results.py      # needs only pandas
```

The script recomputes 50 reported numbers from the committed results files (the 37 shown on the deck slides, 2 from the
speaker notes and 11 from the results files) and prints each next to the expected figure (OK or MISMATCH).

## The data

- **Sources:** WRDS (CRSP stock returns, I/B/E/S analyst forecasts) and SEC EDGAR (8-K filings, the form companies use to publish their earnings releases).
- **Why it isn't here:** the WRDS data is licensed, and the files are too large for GitHub (several exceed 100 MB).
- **To rebuild:** you need a WRDS account with CRSP, I/B/E/S, CRSP/Compustat Merged and the WRDS SEC Analytics Suite. EDGAR is
  free. **[REPRODUCE.md](REPRODUCE.md)** has the full steps, the manual parts and the runtimes.
- **Timing check:** every earnings-release text used was public before the volatility window it predicts (0 of 86,982 events
  failed). In `event_table.csv`, use `acceptance_et`, not `acceptance_utc`, which is known to be wrong.

## Method in brief

Precise formulas: see **[DEFINITIONS.md](DEFINITIONS.md)**.

- **Target:** realized volatility after the release, i.e. the typical size of the stock's daily move over the 5 trading
  days after it (log scale). We predict the size of the move, not its direction.
- **Model ladder:** each rung adds one thing, so its gain shows what that thing is worth.
  - **B0:** linear regression on past volatility (the last 20 trading days). The standard statistical baseline.
  - **B1b:** ridge regression (linear regression with a penalty that keeps coefficients small) that adds the earnings
    surprise (actual earnings vs the analyst forecast, scaled by price), plus company size, sector, release timing and
    analyst coverage.
  - **B1b-GBM:** gradient boosting (LightGBM), i.e. many small decision trees on the same inputs, which captures nonlinear
    effects.
  - **M1:** B1b-GBM plus the earnings-release text, read by FinBERT (a language model trained on financial text) and
    compressed with PCA (a standard way to reduce many features to a few).
- **Splits, by year and never shuffled:**
  - train on 2006–16 (the development, or "dev", period);
  - tune and choose models on validation, 2017–19;
  - 2020 is excluded;
  - holdout, 2021–25: left untouched during development and run once at the end.
