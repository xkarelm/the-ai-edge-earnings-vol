"""NYSE trading calendar with close times (early closes included)."""
import numpy as np
import pandas as pd

import config as C


def load_calendar(path=C.CALENDAR_FILE, extend_to=C.CALENDAR_EXTEND_TO) -> pd.DataFrame:
    """Trading days from the project calendar, close times from the NYSE schedule.

    Sessions after the calendar's last day are appended (in_crsp=False) so that
    window dates can be computed; no returns exist for them.
    """
    import pandas_market_calendars as mcal
    dates = pd.to_datetime(pd.read_csv(path)["date"])
    sched = mcal.get_calendar("NYSE").schedule(start_date=str(dates.min().date()), end_date=extend_to)
    close = sched["market_close"].dt.tz_convert(C.TZ).dt.strftime("%H:%M")
    close.index = pd.to_datetime(close.index)
    nyse = set(close.index)
    print(f"calendar: {len(dates):,} days {dates.min():%Y-%m-%d}..{dates.max():%Y-%m-%d}; "
          f"not in NYSE schedule: {len(set(dates) - nyse)}; NYSE days missing: "
          f"{len({d for d in nyse if d <= dates.max()} - set(dates))}")
    ext = close.index[close.index > dates.max()]
    cal = pd.DataFrame({"date": pd.DatetimeIndex(list(dates) + list(ext))})
    cal["close_et"] = cal["date"].map(close).fillna("16:00")
    cal["in_crsp"] = cal["date"] <= dates.max()
    print(f"early closes: {(cal['close_et'] != '16:00').sum()}; extended {len(ext)} sessions past "
          f"{dates.max():%Y-%m-%d} (window dates only)")
    return cal


def stamps(cal: pd.DataFrame, hhmm) -> np.ndarray:
    """UTC nanoseconds of a clock time (str or per-day Series) on every calendar day."""
    txt = cal["date"].dt.strftime("%Y-%m-%d") + " " + hhmm
    return pd.DatetimeIndex(pd.to_datetime(txt)).tz_localize(C.TZ).tz_convert("UTC").asi8
