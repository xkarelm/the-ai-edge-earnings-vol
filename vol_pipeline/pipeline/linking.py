"""Links: 8-K CIK -> CRSP PERMNO, PERMNO -> ticker, 8-K -> I/B/E/S SUE event."""
import pandas as pd

import config as C

_SOURCE_PRIORITY = {"CRSP/Compustat Merged": 0, "Compustat Security": 1, "Compustat Company": 2, "Capital IQ": 3}


def link_permno(ev: pd.DataFrame) -> pd.DataFrame:
    """cik -> gvkey -> permno, valid on the release date. Inner join: unlinked events drop."""
    lk = pd.read_parquet(C.CACHE / "link_cik_gvkey.parquet")
    lk = lk[lk["gvkey"].notna() & lk["link_desc"].ne("No Link")].copy()
    lk["cik"] = lk["cik"].astype(int)
    lk["prio"] = (lk["source"].map(_SOURCE_PRIORITY).fillna(9)
                  + lk["link_desc"].ne("Valid CIK-GVKEY Link") * 10)
    lk = lk.sort_values("prio").drop_duplicates(["cik", "gvkey"])[["cik", "gvkey", "prio"]]

    ccm = pd.read_parquet(C.CACHE / "link_ccm.parquet")
    ccm = ccm[ccm["linktype"].isin(["LC", "LU"]) & ccm["linkprim"].isin(["P", "C"]) & ccm["permno"].notna()]
    ccm = ccm.assign(linkenddt=ccm["linkenddt"].fillna(pd.Timestamp("2099-12-31")))

    k = ev[["event_id", "cik", "release_date"]].merge(lk, on="cik").merge(ccm, on="gvkey")
    k = k[(k["release_date"] >= k["linkdt"]) & (k["release_date"] <= k["linkenddt"])]
    k = k.assign(p=k["linkprim"].ne("P")).sort_values(["event_id", "prio", "p"]).drop_duplicates("event_id")
    return ev.merge(k[["event_id", "permno", "gvkey"]].astype({"permno": "int64"}), on="event_id", how="inner")


def add_ticker(ev: pd.DataFrame) -> pd.DataFrame:
    nm = pd.read_parquet(C.CACHE / "crsp_names.parquet", columns=["permno", "namedt", "nameenddt", "ticker"])
    nm = nm.astype({"permno": "int64"}).sort_values("namedt")
    ev = pd.merge_asof(ev.sort_values("release_date"), nm, left_on="release_date", right_on="namedt",
                       by="permno", direction="backward")
    ev.loc[ev["release_date"] > ev["nameenddt"], "ticker"] = pd.NA
    return ev


def link_sue(ev: pd.DataFrame, sue: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Attach SUE: same PERMNO, 8-K filed within SUE_LINK_DAYS of the I/B/E/S release.

    Prefer the same reaction session, then the nearest date; one filing per
    I/B/E/S event and vice versa. Leakage guard: every consensus estimate and
    the t-2 price must predate the filing's release.
    """
    s = sue.copy()
    for c in ["R_date", "reaction_date", "t2_date", "est_latest_date"]:
        s[c] = pd.to_datetime(s[c])
    k = ev[["event_id", "permno", "release_date", "session_date", "pre_end"]].merge(s, on="permno")
    k["gap"] = (k["release_date"] - k["R_date"]).dt.days
    k = k[k["gap"].between(*C.SUE_LINK_DAYS)].copy()
    k["same_session"] = k["reaction_date"].eq(k["session_date"])
    k = (k.assign(a=k["gap"].abs())
          .sort_values(["event_id", "same_session", "a"], ascending=[True, False, True]).drop_duplicates("event_id")
          .sort_values(["sue_event_id", "same_session", "a"], ascending=[True, False, True])
          .drop_duplicates("sue_event_id"))
    n_link = len(k)
    k = k[(k["est_latest_date"] < k["release_date"]) & (k["t2_date"] <= k["pre_end"])]
    stats = {"linked": n_link, "removed_by_guard": n_link - len(k), "kept": len(k),
             "same_session_share": float(k["same_session"].mean())}
    ev = ev.merge(k[["event_id", "sue_event_id", "sue_price", "sue_disp", "n_analysts", "same_session"]]
                  .rename(columns={"same_session": "sue_same_session"}), on="event_id", how="left")
    ev["has_sue"] = ev["sue_price"].notna()
    return ev, stats
