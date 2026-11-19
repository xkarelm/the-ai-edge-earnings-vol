"""Unit tests on synthetic data (no inputs needed).

    python -m unittest discover -s tests      # from vol_pipeline/
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as C  # noqa: E402
from pipeline.volatility import logvol  # noqa: E402
from pipeline.windows import window_index  # noqa: E402

# Wed 2024-07-03 is a 13:00 early close; Thu 07-04 is a holiday; 07-06/07 is a weekend.
CAL = pd.DataFrame({"date": pd.to_datetime(["2024-07-01", "2024-07-02", "2024-07-03", "2024-07-05",
                                            "2024-07-08", "2024-07-09"]),
                    "close_et": ["16:00", "16:00", "13:00", "16:00", "16:00", "16:00"],
                    "in_crsp": True})


def windows(*stamps):
    t = pd.Series(pd.to_datetime(list(stamps))).dt.tz_localize(C.TZ)
    i_post, i_pre = window_index(t, CAL)
    d = CAL["date"].dt.strftime("%m-%d").to_numpy()
    return [(d[a], d[b]) for a, b in zip(i_pre, i_post)]   # (pre_end, post_start)


class TestWindows(unittest.TestCase):
    def test_pre_open(self):
        self.assertEqual(windows("2024-07-02 07:00"), [("07-01", "07-02")])

    def test_intraday_day_in_neither_window(self):
        self.assertEqual(windows("2024-07-02 11:00"), [("07-01", "07-03")])

    def test_after_close(self):
        self.assertEqual(windows("2024-07-02 16:05"), [("07-02", "07-03")])

    def test_exactly_at_open_and_close(self):
        # 09:30:00 is not strictly before the open -> intraday; 16:00:00 is not strictly after the close
        self.assertEqual(windows("2024-07-02 09:30", "2024-07-02 16:00"),
                         [("07-01", "07-03"), ("07-01", "07-03")])

    def test_early_close(self):
        # 14:00 on a 13:00 close day is after the close: that day belongs to the pre window
        self.assertEqual(windows("2024-07-03 14:00"), [("07-03", "07-05")])

    def test_holiday_and_weekend(self):
        self.assertEqual(windows("2024-07-04 10:00", "2024-07-06 12:00"),
                         [("07-03", "07-05"), ("07-05", "07-08")])

    def test_windows_never_overlap(self):
        t = pd.Series(pd.date_range("2024-07-01", "2024-07-08 23:00", freq="17min")).dt.tz_localize(C.TZ)
        i_post, i_pre = window_index(t, CAL)
        self.assertTrue((i_pre < i_post).all())


class TestLogVol(unittest.TestCase):
    R = np.array([[0.01, -0.02, 0.03, np.nan, 0.0, 0.0]])

    def test_formula(self):
        v, n = logvol(self.R, np.array([0]), np.array([0]), 3, 3)
        self.assertAlmostEqual(v[0], np.log(np.sqrt((0.01**2 + 0.02**2 + 0.03**2) / 3)))
        self.assertEqual(n[0], 3)

    def test_missing_return(self):
        v, _ = logvol(self.R, np.array([0]), np.array([1]), 3, 3)     # window holds the NaN
        self.assertTrue(np.isnan(v[0]))
        v, _ = logvol(self.R, np.array([0]), np.array([1]), 3, 2)     # allowed with min_obs=2
        self.assertAlmostEqual(v[0], np.log(np.sqrt((0.02**2 + 0.03**2) / 2)))

    def test_zero_vol_is_nan(self):
        v, _ = logvol(self.R, np.array([0]), np.array([4]), 2, 2)
        self.assertTrue(np.isnan(v[0]))


if __name__ == "__main__":
    unittest.main()
