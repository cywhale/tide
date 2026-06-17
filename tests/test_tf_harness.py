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


# ---------- report-only diagnostics (round 27) ----------

def _m2(t_min, amp=100.0, phase_min=0.0):
    # an M2-like sinusoid sampled at t_min minutes, phase shift in minutes
    period = 12.42 * 60.0
    return amp * np.sin(2*np.pi*(t_min - phase_min)/period)


def test_diag_detects_zero_lag_and_unit_amplitude():
    t = np.arange(0, 24*60, 60.0)              # 24 h hourly (dt=60min)
    obs = _m2(t); model = obs.copy()
    d = TF.phase_amplitude_diagnostics(model, obs, 60.0)
    assert abs(d["corr_at_zero_lag"] - 1.0) < 1e-6
    assert d["best_lag_min"] == 0.0
    assert abs(d["std_ratio"] - 1.0) < 1e-6


def test_diag_detects_phase_lag():
    t = np.arange(0, 48*60, 30.0)              # 48 h, 30-min sampling
    obs = _m2(t)
    model = _m2(t, phase_min=60.0)             # model leads obs by 60 min
    d = TF.phase_amplitude_diagnostics(model, obs, 30.0)
    assert abs(d["best_lag_min"] - 60.0) <= 30.0     # detects ~+60 min lead
    assert d["corr_at_best_lag"] > d["corr_at_zero_lag"]
    assert TF.classify_failure(d) == "phase-lag"


def test_diag_detects_amplitude_shrink():
    t = np.arange(0, 48*60, 30.0)
    obs = _m2(t, amp=100.0)
    model = _m2(t, amp=70.0)                    # 30% amplitude shrink, no lag
    d = TF.phase_amplitude_diagnostics(model, obs, 30.0)
    assert abs(d["std_ratio"] - 0.70) < 0.02
    assert d["best_lag_min"] == 0.0
    assert TF.classify_failure(d) == "amplitude"


def test_diag_none_on_too_few_points():
    assert TF.phase_amplitude_diagnostics([1.0, 2.0], [1.0, 2.0], 60.0) is None


# ---------- CLI behavior (round 26) ----------

def test_no_args_does_not_open_store(monkeypatch, capsys):
    """no-args must print 'SCRIPT READY' and return 0 WITHOUT opening any
    store (round 26 F1)."""
    import src.store_adapter as SA
    calls = []
    monkeypatch.setattr(SA, "open_store", lambda *a, **k: calls.append(a) or 1/0)
    monkeypatch.setattr(sys, "argv", ["tf_observation.py"])
    rc = TF.main()
    assert rc == 0 and calls == []
    assert "SCRIPT READY, GATE NOT EXECUTED" in capsys.readouterr().out


def test_observations_gate_requires_both_stores(monkeypatch, tmp_path, capsys):
    """--observations with a missing TPXO9 baseline must FAIL (no silent
    old->new fallback that would compare TPXO10 to itself; round 26 F2)."""
    import src.store_adapter as SA
    opened = []
    monkeypatch.setattr(SA, "open_store", lambda p, *a, **k: opened.append(p))
    obs = tmp_path / "obs.json"
    obs.write_text('{"s1": {"lon": 120.0, "lat": 24.0, '
                   '"times_utc": ["2023-07-25T00:00:00"], "heights_cm": [50.0]}}')
    missing_old = tmp_path / "nope_tpxo9.zarr"
    present_new = tmp_path / "tpxo10.zarr"; present_new.mkdir()
    monkeypatch.setattr(sys, "argv", [
        "tf_observation.py", "--observations", str(obs),
        "--old", str(missing_old), "--new", str(present_new)])
    rc = TF.main()
    assert rc == 2                       # nonzero: aborts
    assert opened == []                  # never opened either store
    assert "requires both stores" in capsys.readouterr().out
