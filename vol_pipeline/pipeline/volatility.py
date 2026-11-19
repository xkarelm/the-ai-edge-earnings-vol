"""Realized volatility: v = ln(sqrt(mean(r_t^2))) over a window of daily returns."""
import warnings

import numpy as np
import pandas as pd


def return_grid(crsp: pd.DataFrame, permnos: np.ndarray, days: np.ndarray, n_days: int) -> np.ndarray:
    """[permno x session] matrix of dlyret, NaN where missing. permnos must be sorted."""
    pi = np.searchsorted(permnos, crsp["permno"].to_numpy())
    di = np.searchsorted(days, crsp["date"].to_numpy("datetime64[ns]"))
    R = np.full((len(permnos), n_days), np.nan, dtype=np.float64)
    R[pi, di] = crsp["dlyret"].to_numpy(dtype=float, na_value=np.nan)
    return R


def logvol(R: np.ndarray, rows: np.ndarray, start: np.ndarray, k: int, min_obs: int
           ) -> tuple[np.ndarray, np.ndarray]:
    """Log realized vol over sessions start..start+k-1 for each (row, start).

    NaN when fewer than min_obs returns are present or when every return is zero.
    Returns (v, number of returns present).
    """
    w = R[rows[:, None], start[:, None] + np.arange(k)]
    nobs = np.isfinite(w).sum(1)
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        v = np.log(np.sqrt(np.nanmean(w ** 2, axis=1)))
    v[(nobs < min_obs) | ~np.isfinite(v)] = np.nan
    return v, nobs
