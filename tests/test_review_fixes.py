"""Testy usterek znalezionych w niezależnym przeglądzie kodu (11.11.2026)."""

import numpy as np
import pandas as pd
import pytest

from prognozowanie.combine.runner import combine, min_history
from prognozowanie.evaluate.metrics import diebold_mariano
from prognozowanie.models import arima_css, build_model
from tests.conftest import EXPERIMENT_YAML, needs_data, synthetic_frame
from tests.test_combine import _archive, _Cfg


def test_single_model_failure_does_not_wipe_k_days():
    """Awaria jednego modelu w jednej dobie: wagi z K = 7 liczą się dalej na 6 dobach."""
    arch, actual = _archive(days=40)
    fail_day = pd.Timestamp("2020-02-20")
    arch = arch[~((arch.origin == fail_day) & (arch.model == "B"))]
    comb, weights = combine(_Cfg, arch, actual)
    c2 = weights[
        (weights.pool == "main")
        & (weights.method == "C2")
        & (weights.k == 7)
        & (weights.block == 2)
    ]
    after = c2[(c2.origin > fail_day) & (c2.origin <= fail_day + pd.Timedelta(days=8))]
    assert len(after) == 8  # żadna doba po awarii nie wypadła
    assert set(after["n_history_days"]) == {6, 7}
    # sama doba awarii: brak prognozy łączonej (nie da się połączyć brakującej prognozy)
    assert fail_day not in set(comb[comb.model == "C2_K7"]["origin"])


def test_min_history_rule():
    assert [min_history(k) for k in (2, 3, 7, 14, 28)] == [2, 3, 6, 12, 24]


def test_hln_factor_for_h1():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=333) ** 2, rng.normal(size=333) ** 2
    r = diebold_mariano(a, b, max_lag=0)
    d = a - b
    dm_raw = d.mean() / np.sqrt(np.mean((d - d.mean()) ** 2) / len(d))
    assert r["dm"] == pytest.approx(dm_raw * np.sqrt((333 - 1) / 333))


def test_css_residuals_for_seasonal_ma_without_differencing():
    """Specyfikacja reszt ES_ARIMA: (1,0,1)(0,0,1); wcześniej rekursja się wysypywała."""
    from statsforecast.models import ARIMA

    y = synthetic_frame(days=40)["load"].to_numpy()
    y = y - y.mean()
    m = ARIMA(order=(1, 0, 1), seasonal_order=(0, 0, 1), season_length=24, method="CSS").fit(y)
    mine = arima_css.residuals(y, m.model_["coef"], m.model_["arma"])
    n_cond = int(m.model_.get("n_cond", 0))
    assert np.allclose(mine[n_cond:], np.asarray(m.model_["residuals"])[n_cond:], atol=1e-6)


def test_es_reports_convergence():
    f = synthetic_frame(days=50)
    m = build_model({"type": "exp_smoothing", "season": 24, "trend": None}).fit(f)
    assert m.convergence_code in (0, 1) and "converged" in m.coef_


def test_svr_has_iteration_limit_and_status():
    f = synthetic_frame(days=30)
    m = build_model({"type": "svr", "kernel": "rbf", "C": 1.0}).fit(f)
    svr = m._est.regressor_.steps[-1][1]
    assert svr.max_iter == 2_000_000
    assert m.convergence_code in (0, 1)


def test_identification_variants_exclude_estimated_init():
    from prognozowanie.identify.search import ES_VARIANTS, TECHNICAL

    assert all(v["initialization"] == "heuristic" for v in ES_VARIANTS)
    assert TECHNICAL["svr"]["max_iter"] == 2_000_000


@needs_data
def test_resume_with_changed_dataset_corrections_is_rejected(dataset, experiment):
    from prognozowanie.backtest.engine import prepare_run_dir, run_dir_for
    from prognozowanie.data.dataset import Dataset

    run_dir = run_dir_for(experiment)
    prepare_run_dir(experiment, dataset, run_dir)
    corr = dataset.config.corrections.model_copy(update={"fix": []})
    changed = Dataset(
        config=dataset.config.model_copy(update={"corrections": corr}),
        frame=dataset.frame,
        log=dataset.log,
        source_sha256=dataset.source_sha256,
    )
    with pytest.raises(RuntimeError, match="innymi wejściami"):
        prepare_run_dir(experiment, changed, run_dir)


@needs_data
@pytest.mark.slow
def test_no_leakage_all_models_and_noisy_scenario(dataset):
    """Test zatrutej przyszłości dla wszystkich modeli i scenariusza losowego."""
    from prognozowanie.backtest.engine import run_origin
    from prognozowanie.config import load_experiment_config

    cfg = load_experiment_config(EXPERIMENT_YAML, use_identified=False)
    scen = [
        s.model_copy(update={"draws": 2}) if s.kind == "noisy" else s
        for s in cfg.temperature_scenarios
    ]
    cfg = cfg.model_copy(update={"temperature_scenarios": scen})
    origin = pd.Timestamp("2003-01-15")
    models = [m.model_dump() for m in cfg.models]
    clean, _ = run_origin(cfg, dataset.frame, origin, models)
    bad = dataset.frame.copy()
    bad.loc[bad.index >= origin, ["load", "load_raw"]] = 1e6
    dirty, _ = run_origin(cfg, bad, origin, models)
    assert set(clean["model"]) >= {m["name"] for m in models}
    pd.testing.assert_frame_equal(clean, dirty)


def test_noise_levels_share_random_numbers():
    """Poziomy błędu temperatury różnią się tylko skalą: losowanie i jest tym samym błędem."""
    from prognozowanie.backtest.scenarios import scenario_temperatures
    from prognozowanie.config import TemperatureScenario

    idx = pd.date_range("2002-08-01", periods=48, freq="h")
    fut = pd.DataFrame({"temp": np.zeros(48)}, index=idx)
    hist = pd.DataFrame({"temp": np.zeros(24)}, index=idx[:24] - pd.Timedelta(days=1))
    lo = TemperatureScenario(name="lo", kind="noisy", sd_day1=1, sd_day2=1.5, rho=0.5, draws=50)
    hi = TemperatureScenario(name="hi", kind="noisy", sd_day1=2, sd_day2=3, rho=0.5, draws=50)
    a = np.array([t for _, t in scenario_temperatures(lo, hist, fut, idx[0], seed=7)])
    b = np.array([t for _, t in scenario_temperatures(hi, hist, fut, idx[0], seed=7)])
    assert np.allclose(b, 2 * a)
    # błąd stały w obrębie doby, różny między dobami, inny w innej dobie wydania
    assert np.allclose(a[:, :24], a[:, [0]]) and not np.allclose(a[:, 0], a[:, 24])
    other = np.array(
        [t for _, t in scenario_temperatures(lo, hist, fut, idx[0] + pd.Timedelta(days=1), 7)]
    )
    assert not np.allclose(other, a)


@needs_data
@pytest.mark.slow
def test_lightgbm_benchmark_is_tuned_in_identification(dataset, experiment, tmp_path):
    """Benchmark LightGBM strojony tą samą procedurą co modele bazowe; wynik daje się nałożyć
    na specyfikację i zbudować."""
    import yaml

    from prognozowanie.config import apply_identified
    from prognozowanie.identify.search import run_identification

    models = [m for m in experiment.models if m.name in ("MR_ARIMA", "ARIMA", "LGBM")]
    ident = experiment.identification.model_copy(update={"n_trials": 3, "tuning_origins": 2})
    cfg = experiment.model_copy(update={"models": models, "identification": ident})
    path = run_identification(cfg, dataset, tmp_path / "ident", steps=("tune",))
    result = yaml.safe_load(path.read_text(encoding="utf-8"))
    out = result["models"]["LGBM"]
    assert out["type"] == "lightgbm" and out["lags"] == [48, 72, 168]
    assert {"n_estimators", "learning_rate", "num_leaves", "verbose"} <= set(out)
    # samo strojenie: wynik niekompletny, model pozostaje wstępny
    assert result["steps_done"] == ["tune"] and result["complete"] is False
    assert result["dataset_sha256"] == dataset.source_sha256
    assert (tmp_path / "ident" / "sieve_lightgbm.csv").exists()  # sito pod persystencją
    specs = [m.model_dump() for m in models]
    partial = next(s for s in apply_identified(specs, result) if s["name"] == "LGBM")
    assert partial["provisional"] is True
    full = {**result, "steps_done": ["arima", "tune"]}
    spec = next(s for s in apply_identified(specs, full) if s["name"] == "LGBM")
    assert spec["provisional"] is False
    build_model(spec, seed=1).fit(synthetic_frame(days=30))
