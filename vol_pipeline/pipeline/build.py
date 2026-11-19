"""Orchestrates the build: explore -> windows -> links -> targets -> features -> SUE -> outputs."""
from __future__ import annotations

import gc

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config as C
from pipeline import linking, sue, volatility
from pipeline.trading_calendar import load_calendar
from pipeline.windows import add_windows

FUNNEL: list[tuple[str, int]] = []


def step(name: str, n: int) -> None:
    FUNNEL.append((name, n))
    print(f"  [{n:>9,}] {name}")


def profile(name: str, df: pd.DataFrame, key: str | None = None) -> None:
    print(f"\n--- {name}: {len(df):,} rows x {df.shape[1]} cols")
    print(df.dtypes.to_string())
    if key:
        print("rows per", key, "->", df.groupby(key).size().value_counts().sort_index().to_dict())


def banner(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def run(rebuild_sue: bool = False) -> pd.DataFrame:
    FUNNEL.clear()
    C.OUT.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ explore ----
    banner("1. EXPLORE")
    ev = pd.read_csv(C.EVENT_FILE, dtype={"accession": str}, low_memory=False)
    ev["event_id"] = ev["cik"].astype(str) + "_" + ev["accession"]
    profile("event_table.csv", ev, "event_id")
    assert ev["event_id"].is_unique, "duplicate event ids"
    print("when:", ev["when"].value_counts().to_dict())

    cal = load_calendar()
    cal.to_csv(C.OUT_CALENDAR, index=False)

    crsp_files = sorted((C.CACHE / "crsp_daily").glob("*.parquet"))
    print(f"\n--- crsp_daily: {len(crsp_files)} yearly files, "
          f"{sum(pq.ParquetFile(f).metadata.num_rows for f in crsp_files):,} rows")
    print("  ", ", ".join(f"{c.name}:{c.type}" for c in pq.ParquetFile(crsp_files[0]).schema_arrow))

    if rebuild_sue or not C.OUT_SUE.exists():
        print("\nRebuilding SUE with the part 3 method ...")
        sue.build_sue(cal).to_parquet(C.OUT_SUE, index=False)
        gc.collect()
    s = pd.read_parquet(C.OUT_SUE)
    profile("SUE (I/B/E/S events)", s, "sue_event_id")

    print("\nColumn mapping chosen:")
    for k, v in C.COLUMN_MAP.items():
        print(f"  {k:<18} <- {v}")

    # -------------------------------------------------------------- build ----
    banner("2. BUILD (rows remaining after each step)")
    step("8-K earnings events in event_table", len(ev))
    ev["release_et"] = pd.to_datetime(ev["release_et"], utc=True).dt.tz_convert(C.TZ)
    ev = ev[ev["release_et"].notna()].copy()
    step("valid release_et", len(ev))
    ev["release_date"] = ev["release_et"].dt.tz_localize(None).dt.normalize()

    ev = add_windows(ev, cal)
    step(f"{C.PRE_DAYS}-day pre and {max(C.POST_HORIZONS.values())}-day post window inside CRSP calendar", len(ev))
    print(f"     note: stored session_date differs from recomputed for "
          f"{ev['session_date_orig'].ne(ev['session_date']).sum():,} events")

    ev = linking.link_permno(ev)
    step("CIK linked to a CRSP PERMNO as of release date", len(ev))

    crsp = pd.concat([pd.read_parquet(f, columns=["permno", "date", "dlyret", "dlycap", "siccd"])
                      for f in crsp_files], ignore_index=True)
    crsp = crsp[crsp["date"] >= cal["date"].iloc[0]]
    ev = ev[ev["permno"].isin(set(crsp["permno"].unique()))].copy()
    step("PERMNO present in CRSP daily (US common stock)", len(ev))

    # targets and v_pre
    permnos = np.sort(ev["permno"].unique())
    crsp = crsp[crsp["permno"].isin(permnos)]
    days = cal["date"].to_numpy("datetime64[ns]")
    R = volatility.return_grid(crsp, permnos, days, int(cal["in_crsp"].sum()))
    rows = np.searchsorted(permnos, ev["permno"].to_numpy())
    ev["v_pre"], ev["n_pre_obs"] = volatility.logvol(R, rows, ev["i_pre_end"].to_numpy() - C.PRE_DAYS + 1,
                                                     C.PRE_DAYS, C.MIN_PRE_OBS)
    for name, k in C.POST_HORIZONS.items():
        ev[name], _ = volatility.logvol(R, rows, ev["i_post"].to_numpy(), k, k)

    # market cap and SIC at pre_end (last close before release)
    snap = crsp[["permno", "date", "dlycap", "siccd"]].rename(columns={"date": "pre_end"})
    ev = ev.merge(snap, on=["permno", "pre_end"], how="left")
    del crsp, snap, R
    gc.collect()

    ev = ev[ev["v_pre"].notna()]
    step(f"v_pre: >= {C.MIN_PRE_OBS} of {C.PRE_DAYS} pre returns, non-zero vol", len(ev))
    for name, k in C.POST_HORIZONS.items():
        ev = ev[ev[name].notna()]
        step(f"{name}: all {k} post returns present, non-zero vol", len(ev))

    cap = ev["dlycap"].astype(float)
    ev["log_mktcap"] = np.log(cap.where(cap > 0))
    sic = ev["siccd"].where(ev["siccd"] > 0)
    ev["sic2"] = (sic // 100).astype("Int64")
    bins, labels = C.SIC_DIVISIONS
    ev["sector"] = pd.cut(sic, bins=bins, labels=labels, include_lowest=True).astype("string")

    ev = linking.add_ticker(ev)
    ev, st = linking.link_sue(ev, s)
    print(f"     SUE: {st['linked']:,} filings linked to an I/B/E/S event; {st['removed_by_guard']:,} "
          f"unlinked by the leakage guard (estimate or t-2 price not before release); kept {st['kept']:,}")
    print(f"     SUE: I/B/E/S reaction session == 8-K session for {st['same_session_share']:.1%}")

    ev = ev.sort_values(["release_et", "event_id"]).reset_index(drop=True)   # chronological, never shuffled
    ev["split"] = sue.split_of(ev["session_date"])
    cols = ["event_id", "cik", "accession", "permno", "ticker", "release_et", "when", "session_date", "split",
            "v_post_3", "v_post", "v_post_15", "v_pre",
            "sue_price", "sue_disp", "n_analysts", "has_sue", "sue_same_session", "sue_event_id",
            "log_mktcap", "sector", "sic2"]
    ev[cols].to_parquet(C.OUT_MODEL_ALL, index=False)

    # Primary sample: 8-K and I/B/E/S agree on the session -> release_et is when the news became public.
    ev = ev[ev["sue_same_session"].eq(True)].reset_index(drop=True)
    step("primary: has SUE and 8-K / I/B/E/S agree on the session", len(ev))
    model = ev[cols]
    model.to_parquet(C.OUT_MODEL, index=False)
    ev[["event_id", "release_et", "when", "session_date", "session_date_orig", "pre_start", "pre_end"]
       + [f"post_end_{k}" for k in C.POST_HORIZONS.values()] + ["n_pre_obs"]].to_parquet(C.OUT_WINDOWS, index=False)

    report(model)
    return model


def report(model: pd.DataFrame) -> None:
    banner("3. RESULT")
    print("Rows per step:")
    prev = None
    for name, n in FUNNEL:
        print(f"  {name:<68} {n:>9,}" + ("" if prev is None else f"  (-{prev - n:,})"))
        prev = n
    print(f"\nFinal: {len(model):,} rows, {model['release_et'].min():%Y-%m-%d} .. "
          f"{model['release_et'].max():%Y-%m-%d}; with SUE: {model['has_sue'].sum():,}; "
          f"with market cap: {model['log_mktcap'].notna().sum():,}; with sector: {model['sector'].notna().sum():,}")
    tv = ["v_post_3", "v_post", "v_post_15", "v_pre"]
    print("\nTargets and v_pre (log daily vol):")
    print(model[tv].describe(percentiles=[.01, .25, .5, .75, .99]).round(3).to_string())
    print("\ncorr:", model[tv].corr().round(3).to_string(), sep="\n")
    print("\nB1 features:")
    print(model[["sue_price", "sue_disp", "log_mktcap"]].describe(percentiles=[.01, .5, .99]).round(4).to_string())
    print("when:", model["when"].value_counts().to_dict())
    print("sector:", model["sector"].value_counts(dropna=False).to_dict())
    print("events per year:", model["release_et"].dt.year.value_counts().sort_index().to_dict())
    print("split:", model["split"].value_counts().reindex(list(C.SPLITS)).to_dict())
    print("v_post mean by split:", model.groupby("split")["v_post"].mean().round(3).to_dict())
    print(f"\nwrote to {C.OUT}: {C.OUT_MODEL.name}, {C.OUT_MODEL_ALL.name}, {C.OUT_WINDOWS.name}, "
          f"{C.OUT_SUE.name}, {C.OUT_CALENDAR.name}")
    print("\nColumn mapping used:")
    for k, v in C.COLUMN_MAP.items():
        print(f"  {k:<18} <- {v}")
