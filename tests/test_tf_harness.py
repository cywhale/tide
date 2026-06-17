"""T-F observation harness math (score_station): de-meaned RMSE, datum
bias, and coverage — locked in the suite so the gate scoring is correct
before real observations are supplied. (The binding T-F gate itself is
'script ready, gate not executed' until real tide-gauge data is given.)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dev_tpxo10" / "scripts"))
import tf_observation as TF  # noqa: E402


def test_score_perfect_match_after_datum_offset():
    # model == obs shifted by a constant datum offset -> RMSE 0, bias = offset
    obs = np.array([10.0, -5.0, 3.0, 8.0, -2.0])
    model = obs + 40.0
    s = TF.score_station(model, obs)
    assert s["rmse_cm"] == 0.0
    assert s["bias_cm"] == 40.0
    assert s["coverage"] == 1.0 and s["n"] == 5


def test_score_rmse_is_demeaned():
    obs = np.array([0.0, 2.0, 4.0, 6.0])
    model = np.array([1.0, 3.0, 5.0, 7.0])     # obs + 1 (pure offset)
    s = TF.score_station(model, obs)
    assert s["rmse_cm"] == 0.0                 # offset removed by de-mean
    assert np.isclose(s["bias_cm"], 1.0)


def test_score_real_residual():
    obs = np.array([0.0, 10.0, 0.0, -10.0])
    model = np.array([0.0, 8.0, 0.0, -8.0])    # 20% amplitude shrink, no offset
    s = TF.score_station(model, obs)
    assert s["rmse_cm"] > 0 and s["bias_cm"] == 0.0


def test_score_coverage_and_nan_handling():
    obs = np.array([1.0, np.nan, 3.0, 4.0, np.nan])
    model = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    s = TF.score_station(model, obs)
    assert s["n"] == 2                         # only indices 0 and 3 overlap
    assert s["coverage"] == round(2 / 5, 3)


def test_score_too_few_points():
    s = TF.score_station(np.array([1.0, 2.0]), np.array([1.0, np.nan]))
    assert s["rmse_cm"] is None and s["n"] == 1


def test_days_since_epoch_naive_is_utc():
    d = TF._days_since_epoch(["1992-01-02T00:00:00"])
    assert np.isclose(d[0], 1.0)
    d2 = TF._days_since_epoch(["1992-01-01T12:00:00+00:00"])
    assert np.isclose(d2[0], 0.5)
