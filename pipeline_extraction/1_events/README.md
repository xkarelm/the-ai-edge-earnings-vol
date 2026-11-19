# Part 1: earnings events and release timing

Builds the list of 8-K earnings releases (Item 2.02, 2006–2025), each release's exact EDGAR acceptance time, the link to a
CRSP stock, the release documents, and a cross-check of release timing across sources.

**Scripts (`code/`, the canonical version; run order in `../../REPRODUCE.md`)**

| Script | What it does | Writes |
|---|---|---|
| `funnel_counts.py` | Counts all 8-K Item 2.02 filings in EDGAR's bulk `submissions.zip` (downloads it if missing) | `raw_counts.csv` |
| `build_event_table.py` | One row per filing, with acceptance time, reaction session and timing (`when`). Run it twice: before and after `acceptance_times.csv` exists | `event_table.csv` |
| `fetch_acceptance.py` | Reads `ACCEPTANCE-DATETIME` from each filing header on EDGAR | `acceptance_times.csv` |
| `wrds_match_and_funnel.py` | Links filings to CRSP (needs the manual `ccm_link.csv` export) and applies the price, size and coverage filters | `matched_sample.csv`, `attrition_funnel.csv` |
| `fetch_releases.py` | Chooses and downloads each release document (mostly EX-99) | `extraction_report_full.csv`, `releases/` |
| `validate_timing.py`, `wire_times.py`, `three_source_timing.py` | Cross-check timing against I/B/E/S and Dow Jones wire times (RavenPack); optional for the results | `timing_validation.csv`, `wire_times.csv`, `three_source_timing.csv` |
| `check_access.py` | Confirms the WRDS tables are accessible | — |

**Notebooks**
- `1_Nov_2_DATA.ipynb`: loads the files above and checks the counts and timing summaries. Its acquisition cell is off by
  default (`RUN_FROM_SCRATCH`).
- `creation_of_acceptance_times.ipynb`: runs `fetch_acceptance.py` to build `acceptance_times.csv`.

**Needs:** `SEC_USER_AGENT`, plus `WRDS_USERNAME` (or `--user`) for the WRDS scripts. Also two hand-supplied files:
`ccm_link.csv`, and the `identity.csv` read by `validate_timing.py`.
