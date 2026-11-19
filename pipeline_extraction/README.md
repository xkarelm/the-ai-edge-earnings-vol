# pipeline_extraction: building the event data (parts 1–5)

This folder holds the **code and notebooks** that built the event table, the release timing, the earnings surprise and
returns, and the cleaned 8-K text that `vol_pipeline/` models. It is a code-only copy of the team's pipeline folders. No
data is included. WRDS (CRSP, I/B/E/S) data is licensed and stays outside git, under `data/` (git-ignored).

Notebook outputs were cleared in these copies, because several had printed WRDS rows. Each subfolder's own `README.md`
gives the full details of its step.

## Pipeline order

| Step | Folder | What it does | Main outputs (not in git) |
|---|---|---|---|
| 1 | `1_events/` | Builds the 8-K earnings event table and EDGAR acceptance times, matches to WRDS, fetches the release document (`fetch_releases.py`), three-source release timing | `nov2_data/event_table.csv`, `extraction_report.csv`, timing files |
| 2 | `2_text_extraction/` | Checks that the extracted text is the right document (manual validation of the extraction) | hand-check sample |
| 3–4 | `3_4_surprise_returns/` | WRDS pulls (`01_download`, the only notebook that connects to WRDS); surprise (SUE); returns; sample flags | `cache/` (WRDS pulls), `data/events_final.parquet`, `data/trading_calendar.csv` |
| 5 | `5_data_cleaning/` | Caches the raw EDGAR documents, then cleans them (SGML/HTML and financial tables stripped), with duplicate flags and splits | `cache/raw/raw_YYYY.zip`, `data/clean/clean_YYYY.parquet`, `manifest.json` |
| 6+ | `../vol_pipeline/` | Builds the model table, baselines (B0, B1b, B1b-GBM) and the text model (M1); also the deck | `vol_pipeline/data/processed/results_*.csv` (tracked) |

`vol_pipeline/setup_inputs.py` copies the step 1–4 inputs it needs into `vol_pipeline/data/raw/`. The step 5 clean parquets go in
`vol_pipeline/data/parquets/` and the raw zips in `vol_pipeline/data/rawTexts/`. See `vol_pipeline/README.md`.

## Pointing the code at the data

The code was written to run inside the team's shared data folder, with each step's `cache/`, `data/` and `results/` next to its code.
In this repo the data lives elsewhere (any local folder; `path/to/data/` below), so set these environment variables rather
than editing paths.

**Folder names.** The code folders here were renamed for clarity (`1_events`, `2_text_extraction`, `5_data_cleaning`). The data
folders keep the pipeline's original names (`1_2nov`, `2_EDGAR text`, `5_data cleaning`). Data-path
strings in the code use those original names and carry the comment `# original data-folder name`.

| Variable | Read by | Set it to |
|---|---|---|
| `PROJECT_DIR` | `3_4_surprise_returns/code/common.py` (defaults to the current directory) | the folder holding the step 3–4 `cache/`, `data/` and `results/`, e.g. `path/to/data/3_4_surprise_returns` |
| `TEAM_DIR` | `5_data_cleaning/code/settings.py` (defaults to this folder) | the data folder holding `1_2nov/`, `3_4_surprise_returns/` and `5_data cleaning/` (original data folder names), e.g. `path/to/data` |
| `STEP5_DIR` | `5_data_cleaning/code/settings.py` (defaults to `$TEAM_DIR/5_data cleaning`, the data folder name) | only if step 5's `cache/` and `data/` live elsewhere |
| `HF_TOKEN` | `5_data_cleaning/code/token_stats.py` | a Hugging Face token (the Llama 3.1 tokenizer is gated) |
| `WRDS_USERNAME` | `1_events/code/check_access.py`, `wire_times.py`, `validate_timing.py` (required for these WRDS steps) | your own WRDS username. The password is never stored; WRDS prompts for it (with a Duo push) |
| `SEC_USER_AGENT` | `1_events/code/fetch_acceptance.py`, `fetch_releases.py`, `funnel_counts.py`, the matching notebook cells, `2_text_extraction/2_EDGAR.ipynb` | `"Your Name your@email"`. SEC EDGAR requires a real contact in the User-Agent; the placeholder default will be refused |

Example:

```bash
export WRDS_USERNAME="your_wrds_username"
export SEC_USER_AGENT="Your Name your@email"
export TEAM_DIR="/path/to/data"
export PROJECT_DIR="$TEAM_DIR/3_4_surprise_returns"
```

**WRDS access.** In steps 3–4, only `3_4_surprise_returns/01_download.ipynb` (with `code/download.py`) connects to WRDS. In
step 1, `1_events/code/check_access.py`, `wire_times.py`, `validate_timing.py` and `wrds_match_and_funnel.py` do (set
`WRDS_USERNAME`, or pass `--user` to `wrds_match_and_funnel.py`). Use your own WRDS account (username, password, Duo push). Never save the password; answer `n` to the `.pgpass` prompt. Pulls are cached per table
and year in `cache/` and are never committed.

**Data paths in the notebooks.** The notebooks in `1_events/`, `2_text_extraction/` and `3_4_surprise_returns/` originally ran in Google Colab.
Their data paths are now the placeholder `path/to/data/...` (in steps 3–4, `TEAM_DIR = Path("path/to/data")`).
Replace it with your local data folder before running. The `.py` code reads the environment variables above.

**Missing `project_paths.py`.** The three `5_data_cleaning` notebooks find their folders by
importing `scripts/project_paths.py` from the data root, and that file is not included in this repository. `5_data_cleaning/code/settings.py`
applies the same path rules, so the step 5 code works with `TEAM_DIR`/`STEP5_DIR` set. To run the notebooks, supply
`project_paths.py` (interface below; see also `REPRODUCE.md`), or replace their path cell with `from settings import TEAM_DIR, NOV2_DATA, STEP5_DIR`.
**Interface for a supplied `scripts/project_paths.py`:** it must read the repository root from the **`DATA_ROOT`**
environment variable (the notebooks search upward from `DATA_ROOT`, or the current folder, for `scripts/project_paths.py`,
then set `DATA_ROOT` for the scripts they start). It must define `REPO_ROOT`, `TEAM_DIR`, `NOV2_DATA`, `STEP5_DIR` and `STEP5_CODE`.

## Part 1 scripts (`1_events/code/`)

`5_data_cleaning/code/settings.py` expects the Part 1 scripts in `1_events/code/`. They are now there: the team's final
`scripts/` folder, copied as is:

`check_access.py`, `build_event_table.py`, `fetch_acceptance.py`, `wrds_match_and_funnel.py`, `fetch_releases.py`,
`wire_times.py`, `three_source_timing.py`, `validate_timing.py`, `funnel_counts.py`.

`1_events/code/` is the canonical version. `1_events/1_Nov_2_DATA.ipynb` no longer carries `%%writefile` copies of these
scripts; a markdown cell points here instead. With this folder in place, every step of the run order above has its code.

The only edits to the team's code remove personal details:
- a hard-coded WRDS username became `os.environ.get("WRDS_USERNAME")` (`check_access.py`, `wire_times.py`, `validate_timing.py`);
- a personal email in the SEC User-Agent became `os.environ.get("SEC_USER_AGENT", ...)` (`fetch_acceptance.py`,
  `fetch_releases.py`, `funnel_counts.py`, the `fetch_acceptance.py` cell in `creation_of_acceptance_times.ipynb`, and
  `2_EDGAR.ipynb`).

## Not included

Everything under each step's `cache/`, `data/`, `clean/`, `raw/`, `logs/` and `results/` folders: WRDS pulls, EDGAR documents,
cleaned texts, row-level CSVs. Also the model-output SQLite cache, `__pycache__/` and the pytest cache.
`3_4_surprise_returns/requirements.txt` lists the Python packages; step 5's `5_data_cleaning/code/dedup.py` also needs `datasketch`,
which is not in it. `5_data_cleaning/manifest.json` records every step 5
output (path, rows, bytes, sha256) so a rebuild can be checked against it.
