import numpy as np
import pandas as pd
import pytest

from prognozowanie.combine import methods as M
from prognozowanie.combine import training_origins
from prognozowanie.combine.runner import combine


def test_training_origins_respect_information_set():
    d = pd.Timestamp("2002-11-20")
    # doba D+1 (blok 2): prognozy wydane od D-K-1 do D-2
    tr = training_origins(d, block=2, k=3)
    assert list(tr) == list(pd.date_range("2002-11-16", "2002-11-18"))
    # doba D (blok 1): od D-K do D-1
    tr = training_origins(d, block=1, k=3)
    assert list(tr) == list(pd.date_range("2002-11-17", "2002-11-19"))
    # doba docelowa ostatniej użytej prognozy zakończyła się przed chwilą D
    for block in (1, 2):
        last = training_origins(d, block, 7).max()
        assert last + pd.Timedelta(days=block - 1) < d


def test_simplex_weights():
    rng = np.random.default_rng(0)
    y = rng.normal(500, 20, 200)
    F = np.column_stack([y + rng.normal(0, 1, 200), y + rng.normal(0, 30, 200), y + 50])
    _, w, b0 = M.simplex(F, y, F)
    assert np.all(w >= 0) and np.isclose(w.sum(), 1) and b0 == 0
    assert w[0] > 0.9


def test_inverse_mse_prefers_accurate_model():
    y = np.linspace(0, 10, 50)
    F = np.column_stack([y + 0.1, y + 2.0])
    _, w, _ = M.inverse_mse(F, y, F)
    assert w[0] > w[1] and np.isclose(w.sum(), 1)


def _archive(days=40, seed=1):
    """Archiwum syntetyczne: model A dokładny, model B obciążony."""
    rng = np.random.default_rng(seed)
    origins = pd.date_range("2020-02-01", periods=days, freq="D")
    rows, actual = [], {}
    for o in origins:
        target = pd.date_range(o, periods=48, freq="h")
        y = 500 + 50 * np.sin(np.arange(48) / 24 * 2 * np.pi)
        for t, v in zip(target, y, strict=True):
            actual[t] = v
        for model, val in (("A", y + rng.normal(0, 1, 48)), ("B", y + 30)):
            rows.append(
                pd.DataFrame(
                    {
                        "origin": o,
                        "model": model,
                        "scenario": "oracle",
                        "draw": 0,
                        "kind": "forecast",
                        "target": target,
                        "horizon": np.arange(1, 49),
                        "value": val,
                    }
                )
            )
            fit_idx = pd.date_range(o - pd.Timedelta(hours=48), periods=48, freq="h")
            rows.append(
                pd.DataFrame(
                    {
                        "origin": o,
                        "model": model,
                        "scenario": "fitted",
                        "draw": 0,
                        "kind": "fitted",
                        "target": fit_idx,
                        "horizon": np.arange(-47, 1),
                        "value": val,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True), pd.Series(actual).sort_index()


class _Cfg:
    from prognozowanie.config import CombinationConfig

    combination = CombinationConfig(
        base_models=["A", "B"],
        methods=["C0", "C1", "C2", "C3", "C4", "C5"],
        k_days=[3, 7],
        extra_pools={"only_a": ["A"]},
    )


def test_combine_end_to_end():
    arch, actual = _archive()
    # wartości rzeczywiste dla dopasowań (okno przed każdą dobą)
    fit = arch[arch.kind == "fitted"]
    extra = pd.Series(500 + 50 * np.sin(np.arange(len(fit)) * 0), index=fit.target)
    actual = actual.combine_first(extra[~extra.index.duplicated()])
    comb, weights = combine(_Cfg, arch, actual)
    assert {"C2_K7", "C3", "C4", "C5_K3"} <= set(comb["model"])
    # rozgrzewka: pierwsze doby nie mają wariantów wymagających historii
    first = comb["origin"].min()
    assert "C2_K7" not in set(comb[comb.origin == first]["model"])
    late = weights[(weights.method == "C2") & (weights.k == 7) & (weights.pool == "main")]
    assert (late["w_A"] > 0.95).all()
    # pula dodatkowa: warianty z przedrostkiem, jeden model = jego własna prognoza
    assert "only_a/C3" in set(comb["model"])
    a = arch[(arch.kind == "forecast") & (arch.model == "A")].set_index(["origin", "target"])
    c = comb[comb.model == "only_a/C3"].set_index(["origin", "target"])
    assert np.allclose(c["value"], a.loc[c.index, "value"])


def test_ols_recovers_weights():
    rng = np.random.default_rng(3)
    F = rng.normal(size=(300, 2))
    y = 2 + 0.3 * F[:, 0] + 0.7 * F[:, 1]
    _, w, b0 = M.ols(F, y, F)
    assert w == pytest.approx([0.3, 0.7]) and b0 == pytest.approx(2)


def test_c2_known_optimum_independent_of_units():
    """Uwaga audytora: wynik C2 nie może zależeć od jednostki zapotrzebowania."""
    from prognozowanie.combine.methods import simplex

    D = np.diag([1.0, 2, 3, 4, 5])
    E = np.vstack([D] * 134)
    y = np.full(len(E), 1_500_000.0)
    expected = (1 / np.diag(D) ** 2) / (1 / np.diag(D) ** 2).sum()  # optimum analityczne
    for scale in (1e-3, 1.0, 1e4):
        F = y[:, None] + scale * E
        _, w, b0 = simplex(F, y, F[:1])
        assert np.allclose(w, expected, atol=1e-9) and b0 == 0.0
    rng = np.random.default_rng(5)
    y = rng.normal(1.5e6, 1e4, 200)
    F = (
        y[:, None]
        + rng.normal(0, 1, (200, 5)) * [3e3, 5e3, 4e3, 8e3, 2e4]
        + rng.normal(0, 2e3, (200, 1))
    )
    _, w_kw, _ = simplex(F, y, F[:1])
    _, w_mw, _ = simplex(F / 1000, y / 1000, F[:1] / 1000)
    assert np.allclose(w_kw, w_mw, atol=1e-10)


def test_c2_matches_dense_search_and_handles_duplicates():
    from prognozowanie.combine.methods import simplex

    rng = np.random.default_rng(9)
    y = rng.normal(600, 50, 72)
    F = y[:, None] + rng.normal(0, 1, (72, 3)) * [10, 15, 30] + rng.normal(0, 5, (72, 1))
    _, w, _ = simplex(F, y, F[:1])
    sse = ((F @ w - y) ** 2).sum()
    g = np.linspace(0, 1, 201)
    grid = np.array([(a, b, 1 - a - b) for a in g for b in g if a + b <= 1 + 1e-12])
    best_grid = (((F @ grid.T) - y[:, None]) ** 2).sum(axis=0).min()
    assert sse <= best_grid + 1e-9 and np.all(w >= 0) and np.isclose(w.sum(), 1)
    # zdublowana kolumna: ta sama funkcja celu i ta sama prognoza łączona
    F2 = np.column_stack([F, F[:, 0]])
    pred1, _, _ = simplex(F, y, F)
    pred2, w2, _ = simplex(F2, y, F2)
    assert np.isclose(((F2 @ w2 - y) ** 2).sum(), sse) and np.allclose(pred1, pred2)
