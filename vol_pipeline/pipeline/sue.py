"""SUE (standardized unexpected earnings); the exact definition is in DEFINITIONS.md.

The functions below are copied from pipeline_extraction/3_4_surprise_returns/code/
(surprise.py, clock.py and common.py) so this folder runs on its own.
Changes from the original: paths and settings come from config.py, and the
Dow Jones wire clock (apply_wire_clock) is not used because its input file
is not available; I/B/E/S announcement times are the release clock, which is
also what the original code uses when no wire file exists.

SUE, primary:     (actual - median consensus) / price at t-2
SUE, alternative: (actual - median consensus) / dispersion, if 3+ analysts
Point-in-time consensus: estimates issued and recorded before R's date, for
the same fiscal quarter, within 90 days, broker still covering, each
analyst's latest, rescaled to the actual's share basis with CRSP factors.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import config as C

NS = "datetime64[ns]"
OPEN_MIN = 9 * 60 + 30
FIRST, LAST = pd.Timestamp(C.SUE_FIRST), pd.Timestamp(C.SUE_LAST)


# ----------------------------------------------------------- helpers ----
def load_cache(name: str, columns: list[str] | None = None) -> pd.DataFrame | None:
    """One cached pull, all years. None if it has not been pulled."""
    folder = C.CACHE / name
    if folder.is_dir():
        files = [f for f in sorted(folder.glob("*.parquet")) if ".partial" not in f.name]
        if not files:
            return None
        return pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)
    single = C.CACHE / f"{name}.parquet"
    return pd.read_parquet(single, columns=columns) if single.exists() else None


def _dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_datetime(df[c]).astype(NS)
    return df


def _strip(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip()


class Funnel:
    def __init__(self):
        self.rows: list[tuple[str, int]] = []

    def step(self, name: str, n: int) -> None:
        self.rows.append((name, int(n)))

    def frame(self) -> pd.DataFrame:
        f = pd.DataFrame(self.rows, columns=["step", "remaining"])
        f["removed"] = (-f["remaining"].diff()).fillna(0).astype(int)
        f["removed_pct"] = (100 * f["removed"] / f["remaining"].shift(1)).round(1).fillna(0)
        return f


def split_of(dates: pd.Series) -> pd.Series:
    """Label each date with its split, by calendar year of the reaction date."""
    y = pd.to_datetime(dates).dt.year
    out = pd.Series(pd.NA, index=dates.index, dtype="string")
    for name, (lo, hi) in C.SPLITS.items():
        out = out.mask(y.between(lo, hi), name)
    return out


def value_asof(keys: pd.DataFrame, table: pd.DataFrame, col: str) -> pd.Series:
    """For each (permno, date) in keys, `col` on the last CRSP day at or before that date."""
    left = keys[["permno", "date"]].copy()
    left["_row"] = np.arange(len(left))
    left = left.dropna(subset=["permno", "date"]).astype({"permno": "int64", "date": NS})
    right = table[["permno", "date", col]].dropna().astype({"permno": "int64", "date": NS})
    m = pd.merge_asof(left.sort_values("date"), right.sort_values("date"),
                      on="date", by="permno", direction="backward")
    out = pd.Series(np.nan, index=range(len(keys)))
    out.loc[m["_row"].to_numpy()] = m[col].to_numpy()
    return pd.Series(out.to_numpy(), index=keys.index)


def assign_session(r_et: pd.Series, cal: pd.DataFrame) -> pd.DataFrame:
    """Release time -> first reacting session (clock.py).

    On a trading day d: R < 09:30 -> d; 09:30 <= R < close(d) -> d; R >= close(d) -> next day.
    On a non-trading day: next trading day.
    """
    out = pd.DataFrame(index=r_et.index, columns=["session_type", "reaction_date", "entry_field"])
    ok = r_et.notna()
    if not ok.any():
        return out
    r = r_et[ok]
    wall = r.dt.tz_convert(C.TZ).dt.tz_localize(None)
    d = wall.dt.normalize()
    minutes = (wall.dt.hour * 60 + wall.dt.minute + wall.dt.second / 60).to_numpy()
    days = cal["date"].to_numpy(dtype=NS)
    close_min = (cal["close_et"].str.slice(0, 2).astype(int) * 60
                 + cal["close_et"].str.slice(3, 5).astype(int)).to_numpy()
    pos = np.searchsorted(days, d.to_numpy(dtype=NS), side="left")
    if (pos >= len(days) - 1).any():
        raise ValueError("trading calendar ends too early for some release times")
    is_trading = days[pos] == d.to_numpy(dtype=NS)
    cmin = close_min[pos]
    pre = is_trading & (minutes < OPEN_MIN)
    intra = is_trading & (minutes >= OPEN_MIN) & (minutes < cmin)
    after = is_trading & (minutes >= cmin)
    react_pos = np.where(after, pos + 1, pos)
    out.loc[ok, "reaction_date"] = pd.to_datetime(days[react_pos])
    out.loc[ok, "session_type"] = np.select([pre, intra, after], ["pre_open", "intraday", "after_close"],
                                            "non_trading_day")
    out.loc[ok, "entry_field"] = np.where(intra, "close", "open")
    out["reaction_date"] = pd.to_datetime(out["reaction_date"])
    return out


# ------------------------------------------------------ step 1: events ----
def build_events(act: pd.DataFrame, link: pd.DataFrame, cal: pd.DataFrame, funnel: Funnel) -> pd.DataFrame:
    a = act.copy()
    a["ticker"] = _strip(a["ticker"])
    a = _dates(a, ["pends", "anndats", "actdats"])
    a = a[a["anndats"].between(FIRST, LAST)]
    funnel.step("I/B/E/S quarterly EPS actuals announced 2006-2025", len(a))
    a = (a.sort_values(["ticker", "pends", "anndats", "actdats"])
          .drop_duplicates(["ticker", "pends"], keep="first"))
    funnel.step("one announcement per firm-quarter", len(a))
    a = a[a["value"].notna()]
    funnel.step("actual EPS value present", len(a))
    a = a[a["curr_act"].astype("string").eq("USD")]
    funnel.step("actual reported in US dollars", len(a))

    lk = link.copy()
    lk["ticker"] = _strip(lk["ticker"])
    lk = _dates(lk, ["sdate", "edate"])
    lk["edate"] = lk["edate"].fillna(pd.Timestamp("2099-12-31"))
    lk = lk[lk["score"] <= C.SUE_LINK_MAX_SCORE]
    m = a.merge(lk[["ticker", "permno", "sdate", "edate", "score"]], on="ticker", how="inner")
    m = m[(m["anndats"] >= m["sdate"]) & (m["anndats"] <= m["edate"])]
    m = m.sort_values(["score", "sdate"]).drop_duplicates(["ticker", "pends"], keep="first")
    m["permno"] = m["permno"].astype("int64")
    funnel.step(f"linked to a CRSP permno on the announcement date (score <= {C.SUE_LINK_MAX_SCORE})", len(m))

    tims = m["anntims"].astype("string").str.strip()
    unknown = tims.isna() | tims.eq("00:00:00")      # 00:00:00 means unknown, not midnight
    txt = (m["anndats"].dt.strftime("%Y-%m-%d") + " " + tims).where(~unknown)
    naive = pd.to_datetime(txt, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    m["R_et"] = naive.dt.tz_localize(C.TZ, ambiguous="NaT", nonexistent="NaT")
    m = m[m["R_et"].notna()].copy()
    funnel.step("announcement time known (I/B/E/S)", len(m))

    sess = assign_session(m["R_et"], cal)
    m = pd.concat([m, sess], axis=1)
    m["reaction_date"] = pd.to_datetime(m["reaction_date"]).astype(NS)
    m["R_date"] = m["R_et"].dt.tz_localize(None).dt.normalize().astype(NS)
    m["R_source"] = "ibes_standin"
    days = cal["date"].to_numpy(dtype=NS)
    pos = np.searchsorted(days, m["reaction_date"].to_numpy(dtype=NS))
    m["t2_date"] = pd.to_datetime(days[pos - 2])     # two sessions before the reaction
    m["event_id"] = m["ticker"] + "|" + m["pends"].dt.strftime("%Y-%m-%d")
    m = m.rename(columns={"value": "actual"})
    keep = ["event_id", "ticker", "permno", "cname", "pends", "anndats", "anntims",
            "actual", "R_et", "R_date", "R_source", "session_type", "reaction_date",
            "entry_field", "t2_date", "score"]
    return m[keep].reset_index(drop=True)


# -------------------------------------------- step 2: point-in-time consensus ----
def build_consensus(ev: pd.DataFrame, det: pd.DataFrame, fac: pd.DataFrame,
                    stops: pd.DataFrame | None, window_days: int = C.SUE_WINDOW_DAYS):
    d = det.copy()
    d["ticker"] = _strip(d["ticker"])
    d = _dates(d, ["fpedats", "anndats", "actdats"])
    d = d.rename(columns={"fpedats": "pends", "anndats": "est_date", "actdats": "est_actdats", "value": "est"})
    e = ev[["event_id", "ticker", "pends", "permno", "R_date"]]
    x = e.merge(d, on=["ticker", "pends"], how="inner")
    counts = {"estimates for the event's quarter": len(x)}

    before = (x["est_date"] < x["R_date"]) & (x["est_actdats"] < x["R_date"])
    counts["dropped: issued or recorded on or after R's date"] = int((~before).sum())
    x = x[before]
    fresh = x["est_date"] >= x["R_date"] - pd.Timedelta(days=window_days)
    counts[f"dropped: older than {window_days} days"] = int((~fresh).sum())
    x = x[fresh & x["est"].notna()]

    cur, rep = x["curr"].astype("string"), x["report_curr"].astype("string")
    usd = (cur.isna() | cur.eq("USD")) & (rep.isna() | rep.eq("USD"))
    counts["dropped: estimate not in US dollars"] = int((~usd).sum())
    x = x[usd].copy()

    x["estimator"] = _strip(x["estimator"])
    if stops is not None and len(stops):
        s = stops.copy()
        s["ticker"] = _strip(s["ticker"])
        s["estimator"] = _strip(s["estimator"])
        s = _dates(s, ["fpedats", "astpdats"]).rename(columns={"fpedats": "pends"})
        k = (x[["ticker", "estimator", "pends", "est_date", "R_date"]].reset_index()
             .merge(s[["ticker", "estimator", "pends", "astpdats"]], on=["ticker", "estimator", "pends"], how="inner"))
        stale = k.loc[(k["astpdats"] >= k["est_date"]) & (k["astpdats"] < k["R_date"]), "index"].unique()
        counts["dropped: broker stopped covering before R"] = len(stale)
        x = x.drop(index=stale)
    else:
        counts["dropped: broker stopped covering before R"] = "not checked (no stopped-estimates file)"

    x["analyst"] = x["estimator"] + "/" + x["analys"].astype("string")
    x = (x.sort_values(["event_id", "analyst", "est_date", "est_actdats"])
          .drop_duplicates(["event_id", "analyst"], keep="last"))
    counts["estimates used (latest per analyst)"] = len(x)

    x["f_est"] = value_asof(x.rename(columns={"est_date": "date"}), fac, "dlycumfacshr")
    x["f_R"] = value_asof(x.rename(columns={"R_date": "date"}), fac, "dlycumfacshr")
    ratio = (x["f_R"] / x["f_est"]).where(x["f_est"].gt(0) & x["f_R"].gt(0))
    x["share_adj"] = ratio.fillna(1.0)
    x["est_adj"] = x["est"] * x["share_adj"]
    counts["estimates rescaled for a split or distribution"] = int((x["share_adj"].sub(1).abs() > 1e-9).sum())
    counts["estimates with no CRSP factor (left unscaled)"] = int(ratio.isna().sum())

    g = x.groupby("event_id")["est_adj"]
    cons = pd.DataFrame({
        "consensus_median": g.median(), "consensus_mean": g.mean(), "dispersion": g.std(),
        "n_analysts": g.size(), "est_latest_date": x.groupby("event_id")["est_date"].max(),
    }).reset_index()
    return cons, x, counts


# ---------------------------------------------------- step 3: price and SUE ----
def add_sue(ev: pd.DataFrame, cons: pd.DataFrame, px: pd.DataFrame, funnel: Funnel) -> pd.DataFrame:
    s = ev.merge(cons, on="event_id", how="inner")
    funnel.step(f"at least one point-in-time estimate within {C.SUE_WINDOW_DAYS} days", len(s))
    p = px[["permno", "date", "dlyprc", "dlycumfacshr"]].copy().astype({"permno": "int64", "date": NS})
    s = s[s["permno"].isin(p["permno"].unique())]        # CRSP pull = US common shares only
    funnel.step("stock is a US common share in CRSP", len(s))
    s = s.merge(p.rename(columns={"date": "t2_date", "dlyprc": "prc_t2", "dlycumfacshr": "f_t2"}),
                on=["permno", "t2_date"], how="left")
    s["f_R"] = value_asof(s.rename(columns={"R_date": "date"}), p, "dlycumfacshr")
    adj = (s["f_R"] / s["f_t2"]).where(s["f_t2"].gt(0) & s["f_R"].gt(0)).fillna(1.0)
    s["price_t2"] = s["prc_t2"].abs() * adj               # negative CRSP price = bid-ask midpoint
    s = s[s["price_t2"].gt(0)].copy()
    funnel.step("CRSP price at t-2", len(s))
    s["surprise"] = s["actual"] - s["consensus_median"]
    s["sue_price"] = s["surprise"] / s["price_t2"]
    ok = s["n_analysts"].ge(3) & s["dispersion"].gt(0)
    s["sue_disp"] = (s["surprise"] / s["dispersion"]).where(ok)
    return s.drop(columns=["prc_t2", "f_t2", "f_R"]).reset_index(drop=True)


def check_no_lookahead(used: pd.DataFrame) -> None:
    assert (used["est_date"] < used["R_date"]).all(), "an estimate dated on or after R was used"
    assert (used["est_actdats"] < used["R_date"]).all(), "an estimate recorded on or after R was used"
    print(f"look-ahead check passed on {len(used):,} estimates")


def build_sue(cal: pd.DataFrame) -> pd.DataFrame:
    """Full part 3 SUE build from the WRDS cache. Returns one row per I/B/E/S event."""
    t0 = time.time()
    act = load_cache("ibes_actuals")
    link = load_cache("link_ibes_crsp")
    det = load_cache("ibes_detail", columns=["ticker", "estimator", "analys", "pdf", "fpi", "value",
                                             "curr", "report_curr", "fpedats", "anndats", "actdats"])
    stops = load_cache("ibes_stopped")
    px = load_cache("crsp_daily", columns=["permno", "date", "dlyprc", "dlycumfacshr"])
    f = Funnel()
    ev = build_events(act, link, cal[["date", "close_et"]], f)
    cons, used, _ = build_consensus(ev, det, px, stops)
    check_no_lookahead(used)
    s = add_sue(ev, cons, px, f)
    print(f.frame().to_string(index=False))
    print(f"SUE rebuilt: {len(s):,} I/B/E/S events in {time.time() - t0:.0f}s")
    keep = ["event_id", "ticker", "permno", "pends", "actual", "R_et", "R_date", "reaction_date",
            "t2_date", "price_t2", "consensus_median", "dispersion", "n_analysts",
            "est_latest_date", "surprise", "sue_price", "sue_disp"]
    s = s[keep].rename(columns={"event_id": "sue_event_id", "ticker": "ibes_ticker"})
    s["R_et"] = s["R_et"].astype(str)
    return s
