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
    be = an.break_even(sens, ["N"]).set_index("model")
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


def test_primary_tests_apply_holm_within_family_and_keep_missing():
    from prognozowanie.config import HypothesisFamily

    rng = np.random.default_rng(1)
    n = 60
    dates = pd.date_range("2002-08-01", periods=n, freq="D")
    base = rng.normal(10, 1, n)
    losses = pd.DataFrame(
        {
            "C3": base,
            "C1_K2": base - 0.5 + rng.normal(0, 0.2, n),
            "C1_K3": base + rng.normal(0, 0.2, n),
            "A": base + 1.0 + rng.normal(0, 0.2, n),
        },
        index=dates,
    )
    fams = {
        "H1": HypothesisFamily(reference="C3", compare=["C1_K2", "C1_K3", "C1_K7"]),
        "H3": HypothesisFamily(reference="C1_K2", compare=["A"]),
    }
    res = an.primary_tests(losses, fams)
    h1 = res[res["family"] == "H1"].set_index("model")
    # brakujący wariant zostaje w tabeli i liczy się do liczebności rodziny
    assert h1.loc["C1_K7", "status"] == "nieoceniony" and np.isnan(h1.loc["C1_K7", "p_value"])
    assert (
        h1.loc["C1_K2", "mean_diff"] < 0 and h1.loc["C1_K2", "p_holm"] >= h1.loc["C1_K2", "p_value"]
    )
    assert np.isclose(h1.loc["C1_K2", "p_holm"], min(1, 3 * h1.loc["C1_K2", "p_value"]))
    h3 = res[res["family"] == "H3"].set_index("model")
    assert h3.loc["A", "mean_diff"] > 0 and {"p_h2", "p_bw14", "p_no_hln"} <= set(res.columns)


def test_confirmation_requires_both_scenarios_and_same_direction():
    from prognozowanie.config import HypothesisFamily

    fams = {"H1": HypothesisFamily(reference="C3", compare=["X", "Y"])}
    k = {"family": ["H1", "H1"], "reference": ["C3", "C3"], "model": ["X", "Y"]}
    t_o = pd.DataFrame({**k, "mean_diff": [-1.0, -1.0], "p_value": [0.001, 0.001]})
    t_p = pd.DataFrame({**k, "mean_diff": [-0.5, 0.3], "p_value": [0.01, 0.001]})
    c = an.confirm_across_scenarios({"oracle": t_o, "persistence": t_p}, fams, 0.05).set_index(
        "model"
    )
    assert np.isclose(c.loc["X", "p_max"], 0.01) and bool(c.loc["X", "confirmed"])
    assert not c.loc["Y", "same_direction"] and not c.loc["Y", "confirmed"]


def test_mae_cancellation_identity_and_asymmetric_loss():
    rng = np.random.default_rng(4)
    errs = {m: rng.normal(0, 1, (6, 48)) for m in ("A", "B", "C")}
    frame = _scen(errs)
    summ, by_hour = an.mae_cancellation(frame, ["A", "B", "C"])
    d2 = frame[frame["block"] == DAY2]
    e = d2.pivot_table(index=["origin", "target"], columns="model", values="error")
    assert np.isclose(summ["MAE_C3"].iloc[0], e.mean(axis=1).abs().mean())
    assert np.isclose(
        summ["MAE_mean_of_models"].iloc[0] - summ["cancellation"].iloc[0], summ["MAE_C3"].iloc[0]
    )
    assert len(by_hour) == 24
    asym = an.asymmetric_loss(frame, "A")
    assert np.allclose(asym["U"] + asym["O"], asym["MAE"])
