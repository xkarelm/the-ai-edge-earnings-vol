"""Every path and parameter of the pipeline. Edit here, nowhere else."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- inputs ----
RAW = ROOT / "data" / "raw"                     # filled by setup_inputs.py; never written by the build
EVENT_FILE = RAW / "event_table.csv"            # 8-K earnings filings (part 1)
CALENDAR_FILE = RAW / "trading_calendar.csv"    # NYSE trading days (CRSP)
CACHE = RAW / "wrds_cache"                      # WRDS pulls (data folder: 3_4_surprise_returns/cache)
CACHE_ITEMS = ["crsp_daily", "crsp_names.parquet", "link_cik_gvkey.parquet", "link_ccm.parquet",
               "link_ibes_crsp.parquet", "ibes_actuals", "ibes_detail", "ibes_stopped"]

# Where setup_inputs.py copies the inputs from (the team's data checkout around this folder).
SETUP_SOURCES = {
    EVENT_FILE: ROOT.parent / "files csv" / "event_table.csv",
    CALENDAR_FILE: ROOT.parent / "files csv" / "trading_calendar.csv",
    **{CACHE / n: ROOT.parent / "3_4_surprise_returns" / "cache" / n for n in CACHE_ITEMS},
}

# --------------------------------------------------------------- outputs ----
OUT = ROOT / "data" / "processed"
OUT_MODEL = OUT / "model_table.parquet"          # primary sample
OUT_MODEL_ALL = OUT / "model_table_all.parquet"  # before the session-agreement filter
OUT_WINDOWS = OUT / "event_windows.parquet"      # every window date, for audit
OUT_SUE = OUT / "sue_ibes_events.parquet"        # rebuilt SUE (slow step, reused)
OUT_CALENDAR = OUT / "trading_calendar_closes.csv"
OUT_LOG = OUT / "build_log.txt"

# ------------------------------------------------------------ parameters ----
TZ = "America/New_York"
OPEN = "09:30"
CALENDAR_EXTEND_TO = "2026-06-30"               # window dates only; no returns past CRSP's end
POST_HORIZONS = {"v_post_3": 3, "v_post": 5, "v_post_15": 15}
PRE_DAYS, MIN_PRE_OBS = 20, 15
SUE_LINK_DAYS = (-1, 4)                         # 8-K date minus I/B/E/S release date (part 4 rule)

# Part 3 SUE settings (3_4_surprise_returns/code/surprise.py)
SUE_FIRST, SUE_LAST = "2006-01-01", "2025-12-31"
SUE_WINDOW_DAYS = 90
SUE_LINK_MAX_SCORE = 3

# Team splits by year of the reaction session (3_4_surprise_returns/code/common.py)
SPLITS = {"dev": (2006, 2016), "validation": (2017, 2019),
          "excluded_2020": (2020, 2020), "holdout": (2021, 2025)}

SIC_DIVISIONS = ([0, 999, 1499, 1799, 3999, 4999, 5199, 5999, 6799, 8999, 9999],
                 ["A_agri", "B_mining", "C_construction", "D_manufacturing", "E_transport_utilities",
                  "F_wholesale", "G_retail", "H_finance", "I_services", "J_public_admin"])

COLUMN_MAP = {
    "event id": "event_table: cik + '_' + accession",
    "release timestamp": "event_table: release_et",
    "session": "session_date recomputed from release_et + calendar (stored value kept as session_date_orig); when = event_table.when",
    "daily returns": "crsp_daily: dlyret (CRSP v2 total return, incl. delisting)",
    "PERMNO": "cik -> gvkey (link_cik_gvkey, valid links) -> permno (link_ccm LC/LU, P/C), as of release date",
    "ticker": "crsp_names: ticker as of release date",
    "SUE": "part 3 sue_price = (actual - median consensus) / price_t2 (primary); sue_disp alternative",
    "market cap": "crsp_daily: dlycap ($000) at pre_end -> log_mktcap",
    "sector": "crsp_daily: siccd at pre_end -> sic2, SIC division",
}
