"""Analizy dodatkowe raportu i metoda C6."""

import numpy as np
import pandas as pd

from prognozowanie.combine import methods as M
from prognozowanie.evaluate import analyses as an

DAY2 = an.DAY2


def test_select_best_puts_all_weight_on_lowest_mse():
    y = np.linspace(0, 10, 40)
    F = np.column_stack([y + 3, y + 0.5, y - 1])
    pred, w, b0 = M.select_best(F, y, F[:5])
    assert list(w) == [0, 1, 0] and b0 == 0 and np.allclose(pred, F[:5, 1])


def test_holm_matches_textbook_example():
    p = pd.Series([0.01, 0.04, 0.03, np.nan], index=list("abcd"))
    adj = an.holm(p)
    assert np.allclose(adj[["a", "c", "b"]], [0.03, 0.06, 0.06]) and np.isnan(adj["d"])


def _scen(models_err: dict[str, np.ndarray], origins: int = 6) -> pd.DataFrame:
    rows = []
    for i in range(origins):
        o = pd.Timestamp("2020-01-01") + pd.Timedelta(days=i)
        for m, err in models_err.items():
            rows.append(
                pd.DataFrame(
                    {
                        "origin": o,
                        "model": m,
                        "draw": 0,
                        "target": pd.date_range(o, periods=48, freq="h"),
                        "horizon": np.arange(1, 49),
                        "error": err[i],
                        "block": np.where(np.arange(1, 49) <= 24, "D (1-24 h)", DAY2),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def test_diversity_decomposition_is_exact():
    rng = np.random.default_rng(0)
    errs = {m: rng.normal(0, s, (6, 48)) for m, s in (("A", 1.0), ("B", 2.0), ("C", 3.0))}
    d = an.diversity(_scen(errs), {"": ["A", "B", "C"]}).iloc[0]
    assert np.isclose(d["mse_of_mean"], d["mean_model_mse"] - d["diversity"])
    assert d["pool"] == "main" and -1 < d["mean_error_corr"] < 1


def test_single_window_counts_wins_against_ex_post_best():
    errs = {
        "A": np.tile([[1.0], [3.0], [1.0], [3.0], [1.0], [3.0]], 48),
        "B": np.tile([[3.0], [1.0], [3.0], [1.0], [3.0], [1.0]], 48),
        "C3": np.full((6, 48), 2.0),  # średnia nigdy nie wygra z najlepszym po fakcie
        "C9": np.full((6, 48), 0.5),  # zawsze wygrywa
    }
    days, summ = an.single_window(_scen(errs), ["A", "B"], ["C3", "C9"])
    s = summ.set_index("combination")["share_beats_best_base"]
    assert s["C3"] == 0 and s["C9"] == 1 and len(days) == 6


def test_break_even_interpolates_crossing():
    sens = pd.DataFrame(
        {
            "model": ["T"] * 3 + ["N"] * 3,
            "sd_day2": [0.0, 1.0, 2.0] * 2,
            "MAE": [8.0, 10.0, 12.0, 11.0, 11.0, 11.0],
        }
    )
    be = an.break_even(sens).set_index("model")
    assert be.loc["T", "reference"] == "N" and np.isclose(be.loc["T", "break_even_sd_day2"], 1.5)


def test_k_profile_parses_pools_and_reference():
    summ = pd.DataFrame(
        {
            "block": [DAY2] * 4,
            "model": ["C2_K7", "C3", "plus_lgbm/C6_K14", "plus_lgbm/C3"],
            "MAE": [9.0, 10.0, 8.0, 9.5],
            "RMSE": [1.0] * 4,
        }
    )
    kp = an.k_profile(summ).set_index(["pool", "method", "k"])
    assert np.isclose(kp.loc[("main", "C2", 7), "vs_C3"], -1.0)
    assert np.isclose(kp.loc[("plus_lgbm", "C6", 14), "vs_C3"], -1.5)


def test_primary_tests_apply_holm_within_family():
    rng = np.random.default_rng(1)
    n = 60
    base = rng.normal(10, 1, n)
    losses = pd.DataFrame(
        {
            "C3": base,
            "C1_K7": base - 0.5 + rng.normal(0, 0.2, n),
            "C2_K7": base + rng.normal(0, 0.2, n),
            "A": base + 1.0 + rng.normal(0, 0.2, n),
        }
    )
    fams = an.primary_families(["A"], [], [7], ["C1", "C2", "C3"])
    res = an.primary_tests(losses, fams)
    h1 = res[res["family"].str.startswith("H1")].set_index("model")
    assert set(h1.index) == {"C1_K7", "C2_K7"}
    assert (h1["p_holm"] >= h1["p_value"]).all()
    assert h1.loc["C1_K7", "mean_diff"] < 0
    h3 = res[res["family"].str.startswith("H3")]
    assert list(h3["model"]) == ["A"] and np.isclose(h3["p_holm"], h3["p_value"]).all()
