import numpy as np
import pandas as pd
import pytest

from prognozowanie.evaluate.metrics import diebold_mariano
from prognozowanie.models import REGISTRY, arima_css, build_model
from tests.conftest import synthetic_frame


def test_registry_contains_all_models():
    assert {
        "regression",
        "hong_regression",
        "sarima",
        "exp_smoothing",
        "svr",
        "mlp",
        "lightgbm",
        "seasonal_naive",
        "two_stage",
    } <= set(REGISTRY)


def test_unknown_model_type():
    with pytest.raises(KeyError, match="nieznany typ"):
        build_model({"type": "nie_ma_takiego"})


def test_css_recursion_matches_statsforecast():
    from statsforecast.models import ARIMA

    f = synthetic_frame(days=40)
    y = f["load"].to_numpy()
    m = ARIMA(order=(1, 0, 1), seasonal_order=(0, 1, 1), season_length=24, method="CSS").fit(y)
    mine = arima_css.residuals(y, m.model_["coef"], m.model_["arma"])
    assert np.allclose(mine, m.model_["residuals"], atol=1e-8)
    fc = arima_css.forecast(y, m.model_["residuals"], m.model_["coef"], m.model_["arma"], 48)
    assert fc.shape == (48,) and np.all(np.isfinite(fc))
    assert abs(fc.mean() - y[-168:].mean()) < 50


@pytest.mark.parametrize(
    "spec",
    [
        {"type": "seasonal_naive"},
        {"type": "regression"},
        {"type": "hong_regression"},
        {"type": "sarima", "order": [1, 0, 0], "seasonal_order": [0, 1, 1], "season": 24},
        {
            "type": "two_stage",
            "first": {"type": "regression"},
            "residual": {
                "type": "sarima",
                "order": [1, 0, 0],
                "seasonal_order": [0, 0, 1],
                "season": 24,
            },
        },
        {"type": "two_stage", "first": {"type": "regression"}, "residual": None},
        {"type": "mlp", "hidden_layer_sizes": 4, "n_seeds": 2},
        {"type": "svr", "kernel": "rbf", "C": 10.0},
        {"type": "lightgbm", "n_estimators": 50},
        {"type": "exp_smoothing", "season": 24},
    ],
)
def test_every_model_fits_and_predicts(spec):
    f = synthetic_frame(days=60)
    origin = f.index[0] + pd.Timedelta(days=50)
    hist = f.loc[f.index < origin]
    fut = f.loc[origin : origin + pd.Timedelta(hours=47)].drop(columns=["load", "load_raw"])
    m = build_model(spec, seed=1).fit(hist)
    p = m.predict(fut)
    assert p.shape == (48,) and np.all(np.isfinite(p))
    actual = f.loc[fut.index, "load"].to_numpy()
    assert np.mean(np.abs(p - actual)) < 80  # rozsądny rząd wielkości
    assert len(m.fitted()) == len(hist)


def test_mlp_is_reproducible():
    f = synthetic_frame(days=30)
    o = f.index[0] + pd.Timedelta(days=25)
    hist, fut = f.loc[f.index < o], f.loc[o : o + pd.Timedelta(hours=47)].drop(columns=["load"])
    spec = {"type": "mlp", "hidden_layer_sizes": 4, "n_seeds": 2}
    p1 = build_model(spec, seed=7).fit(hist).predict(fut.drop(columns=["load_raw"]))
    p2 = build_model(spec, seed=7).fit(hist).predict(fut.drop(columns=["load_raw"]))
    assert np.array_equal(p1, p2)


def test_diebold_mariano_direction():
    rng = np.random.default_rng(0)
    good = np.abs(rng.normal(0, 1, 300))
    bad = good + 0.5
    r = diebold_mariano(good, bad)
    assert r["dm"] < 0 and r["p_value"] < 0.01
    r0 = diebold_mariano(good, good + rng.normal(0, 0.01, 300))
    assert r0["p_value"] > 0.01


def test_mcs_and_bootstrap():
    from prognozowanie.evaluate.inference import bootstrap_ci, model_confidence_set

    # ziarno, przy którym "same" nie jest przypadkiem istotnie lepszy (różnica średnich 0,04)
    rng = np.random.default_rng(3)
    base = np.abs(rng.normal(10, 2, 200))
    losses = pd.DataFrame({"good": base, "same": base + rng.normal(0, 2.0, 200), "bad": base + 3})
    mcs = model_confidence_set(losses, n_boot=500)
    assert not mcs.loc["bad", "in_mcs"]
    assert mcs.loc["good", "in_mcs"] and mcs.loc["same", "in_mcs"]
    ci = bootstrap_ci(losses, reference="good", n_boot=500)
    assert ci.loc["bad", "diff_ci_low"] > 0
    assert ci.loc["same", "diff_ci_low"] < 0 < ci.loc["same", "diff_ci_high"]
