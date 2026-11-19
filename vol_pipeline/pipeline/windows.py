"""Leakage-safe pre/post windows, from release_et and the calendar only.

post_start  first session whose 09:30 open is strictly after release_et
            (pre_open -> D, intraday -> D+1, after_close -> D+1, weekend -> next day)
pre_end     last session whose close (early closes included) is strictly before release_et
For an intraday release, day D is in neither window.
"""
import numpy as np
import pandas as pd

import config as C
from pipeline.trading_calendar import stamps


def window_index(release_et: pd.Series, cal: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Calendar positions (i_post, i_pre_end) for tz-aware release times."""
    opens, closes = stamps(cal, C.OPEN), stamps(cal, cal["close_et"])
    t = release_et.dt.tz_convert("UTC").astype("int64").to_numpy()
    i_post = np.searchsorted(opens, t, side="right")
    i_pre_end = np.searchsorted(closes, t, side="left") - 1
    return i_post, i_pre_end


def add_windows(ev: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    """Keep events whose windows fit in CRSP's calendar; add window positions and dates."""
    ev = ev.copy()
    ev["i_post"], ev["i_pre_end"] = window_index(ev["release_et"], cal)
    n_crsp = int(cal["in_crsp"].sum())
    ok = (ev["i_pre_end"] - C.PRE_DAYS + 1 >= 0) & (ev["i_post"] + max(C.POST_HORIZONS.values()) <= n_crsp)
    ev = ev[ok].copy()

    # leakage guards
    opens, closes = stamps(cal, C.OPEN), stamps(cal, cal["close_et"])
    t = ev["release_et"].dt.tz_convert("UTC").astype("int64").to_numpy()
    assert (ev["i_pre_end"] < ev["i_post"]).all(), "pre/post windows overlap"
    assert (closes[ev["i_pre_end"]] < t).all(), "pre window ends after release"
    assert (opens[ev["i_post"]] > t).all(), "post window starts before release"

    days = cal["date"].to_numpy("datetime64[ns]")
    ev["session_date_orig"] = pd.to_datetime(ev["session_date"], errors="coerce")
    ev["session_date"] = days[ev["i_post"]]
    ev["pre_start"] = days[ev["i_pre_end"] - C.PRE_DAYS + 1]
    ev["pre_end"] = days[ev["i_pre_end"]]
    for k in C.POST_HORIZONS.values():
        ev[f"post_end_{k}"] = days[ev["i_post"] + k - 1]
    return ev
