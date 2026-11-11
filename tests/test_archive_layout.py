"""Archiwum w układzie "model x doba": dołożenie lub zmiana modelu liczy tylko ten model."""

import numpy as np
import pandas as pd
import pytest

from prognozowanie.combine.runner import combine
from prognozowanie.config import CombinationConfig
from tests.conftest import needs_data

ORIGINS = pd.date_range("2002-08-01", periods=2, freq="D")


def _cfg(experiment, names):
    models = [m for m in experiment.models if m.name in names]
    comb = experiment.combination.model_copy(
        update={"base_models": ["SNAIVE", "HONG"], "extra_pools": {}}
    )
    return experiment.model_copy(update={"models": models, "combination": comb})


def _mtimes(run_dir, model):
    return {p.name: p.stat().st_mtime_ns for p in (run_dir / "forecasts" / model).glob("*")}


@needs_data
@pytest.mark.slow
def test_new_model_is_computed_alone(dataset, experiment):
    from prognozowanie.backtest import load_archive, run_backtest
    from prognozowanie.backtest.engine import run_dir_for

    base = _cfg(experiment, ["SNAIVE", "HONG"])
    assert all(r.status == "ok" for r in run_backtest(base, dataset, origins=ORIGINS))
    run_dir = run_dir_for(base)
    before = {m: _mtimes(run_dir, m) for m in ("SNAIVE", "HONG")}

    extended = _cfg(experiment, ["SNAIVE", "HONG", "MR_ARIMA"])
    res = run_backtest(extended, dataset, origins=ORIGINS)
    assert len(res) == 2 and all(r.status == "ok" for r in res)
    assert {m: _mtimes(run_dir, m) for m in ("SNAIVE", "HONG")} == before
    assert len(list((run_dir / "forecasts" / "MR_ARIMA").glob("*.parquet"))) == 2
    assert run_backtest(extended, dataset, origins=ORIGINS) == []

    arch = load_archive(run_dir)
    fitted = arch[arch["kind"] == "fitted"]
    # wartości dopasowane ze 168 h dla każdego modelu, także spoza puli i pierwszego stopnia
    hours = fitted.groupby(["model", "origin"]).size()
    assert set(hours) == {extended.forecast.fitted_hours_saved}
    assert {"SNAIVE", "HONG", "MR_ARIMA", "MR_ARIMA__stage1"} <= set(fitted["model"])
    assert (fitted["target"] < fitted["origin"]).all()
    # wczytanie z listą modeli pomija modele spoza listy
    only = load_archive(run_dir, models=["SNAIVE"])
    assert set(only["model"]) == {"SNAIVE"}


@needs_data
@pytest.mark.slow
def test_changed_model_spec_needs_explicit_recompute(dataset, experiment):
    from prognozowanie.backtest import run_backtest
    from prognozowanie.backtest.engine import run_dir_for

    base = _cfg(experiment, ["SNAIVE", "HONG"])
    run_backtest(base, dataset, origins=ORIGINS)
    run_dir = run_dir_for(base)
    hong_before = _mtimes(run_dir, "HONG")

    models = [
        m.model_copy(update={"params": {"lag": 24}}) if m.name == "SNAIVE" else m
        for m in base.models
    ]
    changed = base.model_copy(update={"models": models})
    with pytest.raises(RuntimeError, match="inną specyfikacją"):
        run_backtest(changed, dataset, origins=ORIGINS)
    res = run_backtest(changed, dataset, origins=ORIGINS, recompute=["SNAIVE"])
    assert len(res) == 2
    assert _mtimes(run_dir, "HONG") == hong_before
    assert run_backtest(changed, dataset, origins=ORIGINS) == []


@needs_data
def test_combination_and_model_list_are_outside_global_snapshot(dataset, experiment):
    from prognozowanie.backtest.engine import prepare_run_dir, run_dir_for

    run_dir = run_dir_for(experiment)
    prepare_run_dir(experiment, dataset, run_dir)
    comb = experiment.combination.model_copy(update={"k_days": [7]})
    prepare_run_dir(experiment.model_copy(update={"combination": comb}), dataset, run_dir)
    fewer = experiment.model_copy(update={"models": experiment.models[:-1]})
    prepare_run_dir(fewer, dataset, run_dir)
    window = experiment.forecast.model_copy(update={"training_window_days": 56})
    with pytest.raises(RuntimeError, match="innymi wejściami"):
        prepare_run_dir(experiment.model_copy(update={"forecast": window}), dataset, run_dir)


@needs_data
def test_failed_model_is_recorded_and_not_retried(dataset, experiment, monkeypatch):
    from prognozowanie.backtest import run_backtest
    from prognozowanie.backtest.engine import forecast_file, run_dir_for
    from prognozowanie.evaluate.report import failures
    from prognozowanie.models.statistical import HongRegression

    def boom(self, hist):
        raise ValueError("awaria testowa")

    monkeypatch.setattr(HongRegression, "fit", boom)
    base = _cfg(experiment, ["SNAIVE", "HONG"])
    o = ORIGINS[:1]
    res = run_backtest(base, dataset, origins=o)
    assert res[0].status.startswith("błędy modeli")
    run_dir = run_dir_for(base)
    assert not forecast_file(run_dir, "HONG", o[0]).exists()
    assert forecast_file(run_dir, "SNAIVE", o[0]).exists()
    f = failures(run_dir)
    assert list(f["model"]) == ["HONG"] and "awaria testowa" in f["error"].iloc[0]
    assert run_backtest(base, dataset, origins=o) == []  # awaria jest wynikiem, nie brakiem


@needs_data
def test_old_archive_layout_is_rejected(dataset, experiment):
    from prognozowanie.backtest.engine import prepare_run_dir, run_dir_for

    run_dir = run_dir_for(experiment)
    (run_dir / "forecasts").mkdir(parents=True)
    pd.DataFrame({"x": [1]}).to_parquet(run_dir / "forecasts" / "2002-08-01.parquet")
    with pytest.raises(RuntimeError, match="starym układzie"):
        prepare_run_dir(experiment, dataset, run_dir)


def test_c0_variants_use_requested_fitted_hours():
    """C0 z 48 h widzi tylko końcówkę okna (model A tam bezbłędny), C0_H168 cały tydzień
    (model A zaszumiony w starszych godzinach, B dokładny)."""
    rng = np.random.default_rng(3)
    idx = pd.date_range("2020-01-01", periods=24 * 30, freq="h")
    actual = pd.Series(500 + 50 * np.sin(np.arange(len(idx)) / 24 * 2 * np.pi), index=idx)
    rows = []
    for o in pd.date_range("2020-01-10", periods=5, freq="D"):
        tgt = pd.date_range(o, periods=48, freq="h")
        fit_idx = pd.date_range(o - pd.Timedelta(hours=168), periods=168, freq="h")
        y_fit = actual[fit_idx].to_numpy()
        old = np.arange(168) < 120
        fa = y_fit + np.where(old, rng.normal(0, 50, 168), 0.0)
        fb = y_fit + rng.normal(0, 5, 168)
        for m, fv in (("A", fa), ("B", fb)):
            base = {"origin": o, "model": m, "draw": 0}
            rows.append(
                pd.DataFrame(
                    {
                        **base,
                        "scenario": "oracle",
                        "kind": "forecast",
                        "target": tgt,
                        "horizon": np.arange(1, 49),
                        "value": actual[tgt].to_numpy(),
                    }
                )
            )
            rows.append(
                pd.DataFrame(
                    {
                        **base,
                        "scenario": "fitted",
                        "kind": "fitted",
                        "target": fit_idx,
                        "horizon": np.arange(-167, 1),
                        "value": fv,
                    }
                )
            )
    arch = pd.concat(rows, ignore_index=True)

    class _C:
        combination = CombinationConfig(
            base_models=["A", "B"], methods=["C0"], c0_fitted_hours=[48, 168]
        )

    comb, w = combine(_C, arch, actual)
    assert {"C0", "C0_H168"} == set(comb["model"])
    w48 = w[w["fitted_hours"] == 48]
    w168 = w[w["fitted_hours"] == 168]
    assert np.allclose(w48["w_A"], 1, atol=1e-6) and np.allclose(w48["w_B"], 0, atol=1e-6)
    assert (w168["w_B"] > 0.8).all()

    class _Too:
        combination = CombinationConfig(
            base_models=["A", "B"], methods=["C0"], c0_fitted_hours=[200]
        )

    with pytest.raises(RuntimeError, match="wymaga 200"):
        combine(_Too, arch, actual)
