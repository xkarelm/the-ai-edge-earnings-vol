# earnings-call-volatility

Forecasting how much a stock will move in the days after an earnings release, and testing whether the **text of the
earnings release** adds anything beyond the numbers.

## The finding

- **Past volatility does most of the work.** A stock's recent volatility alone explains most of what the models capture.
- **The text adds a small but statistically clear improvement.** For large, well-covered companies, adding the
  earnings-release text raises R² by **+0.024 on validation (2017–19)** and **+0.038 on a one-time holdout (2021–25)** (against the
  holdout's own mean; +0.035 against the dev mean). R² is the share of variation explained. Both 95% intervals are above
  zero, so the gain holds on years the model never saw.
- **Scope:** large caps, median ~$3.7B market cap, 7 analysts. These are the events with extracted release text, which was
  collected only for filings passing the extraction screen (price > $5, market cap above the NYSE 20th percentile, 3+
  analysts; see [DEFINITIONS.md](DEFINITIONS.md#large-cap-subsample)).

## What's in this repo

| Path | What it is |
|---|---|
| `vol_pipeline/` | The modeling work: builds the dataset, runs the model ladder (simple baseline → gradient boosting → text model), and writes the results |
| `vol_pipeline/deck/` | The pitch deck, **[`pitch_deck.pdf`](vol_pipeline/deck/pitch_deck.pdf)** (and `.pptx`), and the scripts that build it from the committed results files. The sample medians (~$3.7B, 7 analysts) need the licensed `model_table.parquet`, which is not committed; without it the build uses the documented values |
| `vol_pipeline/verify_results.py` | **Run this first.** Confirms every number in the deck in about 2 seconds, with no data download |
| `vol_pipeline/data/processed/` | The small results files the deck and the check read. Raw and licensed data are not in the repo |
| `vol_pipeline/notebooks/` | Exploratory analysis and the risk-limit notebook; saved figures in `figures/` |
| `pipeline_extraction/` | The team's data-extraction pipeline, stages 1–5: SEC filings → events and timing → earnings surprise and returns → cleaned text. Code only |
| `REPRODUCE.md` | Full step-by-step instructions to rebuild everything from the raw data |

Some folder names are referenced directly by the code, so please don't rename folders. `pipeline_extraction/README.md`
explains the naming.

## Check the results fast (no data needed)

```sh
cd vol_pipeline && python verify_results.py      # needs only pandas
```

The script recomputes 50 reported numbers from the committed results files (the 37 shown on the deck slides, 2 from the
speaker notes and 11 from the results files) and prints each next to the expected figure (OK or MISMATCH). It is the
quickest way to confirm the claims.

## The data

- **Sources:** WRDS (CRSP stock returns, I/B/E/S analyst forecasts) and SEC EDGAR (the 8-K earnings-release filings).
- **Why it isn't here:** the WRDS data is licensed, and the files are too large for GitHub (several exceed 100 MB).
- **To rebuild:** you need a WRDS account with CRSP, I/B/E/S, CRSP/Compustat Merged and the WRDS SEC Analytics Suite. EDGAR is
  free. **[REPRODUCE.md](REPRODUCE.md)** has the full steps, the manual parts and the runtimes.
- **Timing check:** every earnings-release text used was public before the volatility window it predicts (0 of 86,982 events
  failed). In `event_table.csv`, use `acceptance_et`, not `acceptance_utc`, which is known to be wrong.

## Method in brief

Precise formulas: see **[DEFINITIONS.md](DEFINITIONS.md)**.

- **Target:** post-earnings realized volatility, i.e. how much the stock actually swung each day over the 5 trading days
  after the release (log scale). We predict the size of the move, not its direction.
- **Model ladder:** each rung adds one thing, so its gain shows what that thing is worth.
  - **B0:** linear regression on past volatility (the last 20 trading days). The standard statistical baseline.
  - **B1b:** ridge regression (penalized linear) that adds the earnings surprise (actual earnings vs the analyst forecast,
    scaled by price), plus company size, sector, release timing and analyst coverage.
  - **B1b-GBM:** gradient boosting (LightGBM), i.e. many small decision trees on the same inputs, which captures nonlinear
    effects.
  - **M1:** B1b-GBM plus the earnings-release text, read by FinBERT (a language model trained on financial text) and
    compressed with PCA (a standard way to reduce many features to a few).
- **Splits, by year and never shuffled:**
  - train on 2006–16;
  - tune and choose on validation, 2017–19;
  - 2020 is excluded;
  - a one-time holdout, 2021–25, run once at the end.
