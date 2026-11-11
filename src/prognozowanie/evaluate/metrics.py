"""Miary błędu i test Diebolda-Mariano (sekcja 10 metodyki).

MAPE nie jest liczony domyślnie: dodanie stałej do danych (anonimizacja) zmienia jego
mianownik, a MAE, RMSE i MASE od tej stałej nie zależą.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

DAY_TYPES = ("roboczy", "sobota", "niedziela", "święto", "pomostowy")


def attach_actuals(fc: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    a = actuals.rename(columns={"load": "actual"})
    out = fc.merge(
        a[["actual", "holiday", "bridge", "dow", "suspicious"]],
        left_on="target",
        right_index=True,
        how="left",
    )
    out["error"] = out["value"] - out["actual"]
    out["block"] = np.where(out["horizon"] <= 24, "D (1-24 h)", "D+1 (25-48 h)")
    out["day_type"] = _day_type(out)
    return out


def _day_type(df: pd.DataFrame) -> pd.Series:
    t = pd.Series("roboczy", index=df.index)
    t[df["dow"] == 5] = "sobota"
    t[df["dow"] == 6] = "niedziela"
    t[df["bridge"] == 1] = "pomostowy"
    t[df["holiday"] == 1] = "święto"
    return t


def mase_scale(
    actuals: pd.Series, origins: pd.DatetimeIndex, window_days: int, lag: int = 168
) -> pd.Series:
    """Skala MASE dla każdej doby: średni błąd modelu naiwnego sezonowego w oknie uczącym."""
    out = {}
    for o in origins:
        y = actuals.loc[(actuals.index >= o - pd.Timedelta(days=window_days)) & (actuals.index < o)]
        out[o] = float(np.mean(np.abs(y.to_numpy()[lag:] - y.to_numpy()[:-lag])))
    return pd.Series(out, name="scale")


def summary(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = df.groupby(by, observed=True)
    out = pd.DataFrame(
        {
            "MAE": g["error"].apply(lambda e: np.abs(e).mean()),
            "RMSE": g["error"].apply(lambda e: np.sqrt(np.mean(e**2))),
            "n": g["error"].size(),
        }
    )
    if "scale" in df.columns:
        out["MASE"] = g.apply(
            lambda x: np.mean(np.abs(x["error"]) / x["scale"]), include_groups=False
        )
    return out


def daily_loss(df: pd.DataFrame, loss: str = "abs") -> pd.DataFrame:
    """Strata uśredniona po godzinach bloku: wiersze = doby wydania, kolumny = modele."""
    f = np.abs if loss == "abs" else np.square
    d = df.assign(loss=f(df["error"]))
    return d.pivot_table(index="origin", columns="model", values="loss", aggfunc="mean")


def diebold_mariano(l1: np.ndarray, l2: np.ndarray, max_lag: int | None = None) -> dict:
    """Test DM z wariancją Neweya-Westa i poprawką Harveya, Leybourne'a i Newbolda (h = 1).

    l1, l2: dzienne straty dwóch modeli (te same doby). Ujemna statystyka: model 1 lepszy.
    """
    d = np.asarray(l1, float) - np.asarray(l2, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return {"n": n, "mean_diff": np.nan, "dm": np.nan, "p_value": np.nan}
    L = max_lag if max_lag is not None else int(np.floor(n ** (1 / 3)))
    dc = d - d.mean()
    gamma0 = dc @ dc / n
    lr = gamma0
    for lag in range(1, L + 1):
        g = dc[lag:] @ dc[:-lag] / n
        lr += 2 * (1 - lag / (L + 1)) * g
    dm = d.mean() / np.sqrt(lr / n)
    hln = dm * np.sqrt((n - 1) / n)  # HLN dla h = 1: (n + 1 - 2h + h(h - 1)/n) / n
    p = 2 * stats.t.sf(abs(hln), df=n - 1)
    return {
        "n": n,
        "mean_diff": float(d.mean()),
        "dm": float(hln),
        "p_value": float(p),
        "nw_lags": L,
    }


def dm_table(losses: pd.DataFrame, reference: str) -> pd.DataFrame:
    rows = []
    for m in losses.columns:
        if m == reference:
            continue
        ok = losses[[m, reference]].dropna()
        rows.append({"model": m, **diebold_mariano(ok[m].to_numpy(), ok[reference].to_numpy())})
    return pd.DataFrame(rows).set_index("model").sort_values("mean_diff")
