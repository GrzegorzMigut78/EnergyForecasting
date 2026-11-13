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


def diebold_mariano(
    l1: np.ndarray,
    l2: np.ndarray,
    max_lag: int | None = None,
    dates: pd.DatetimeIndex | None = None,
    h: int = 1,
    hln: bool = True,
) -> dict:
    """Test DM z wariancją Neweya-Westa i poprawką Harveya, Leybourne'a i Newbolda.

    l1, l2: dzienne straty dwóch modeli. Ujemna statystyka: model 1 lepszy.
    dates: daty dób; luki w datach (doby wykluczone) są uwzględniane w autokowariancjach
    według rzeczywistych odstępów, a nie kolejności wierszy. h: horyzont w poprawce HLN
    (konwencja główna h = 1; kontrola h = 2: blok D+1 to druga doba od końca historii).
    """
    d = np.asarray(l1, float) - np.asarray(l2, float)
    if dates is not None:
        ser = pd.Series(d, index=pd.DatetimeIndex(dates)).dropna().sort_index()
        cal = pd.date_range(ser.index.min(), ser.index.max(), freq="D") if len(ser) else ser.index
        full = ser.reindex(cal).to_numpy()
    else:
        full = d[np.isfinite(d)]
    ok = np.isfinite(full)
    n = int(ok.sum())
    if n < 10:
        return {"n": n, "mean_diff": np.nan, "dm": np.nan, "p_value": np.nan}
    mean = float(np.nanmean(full))
    if np.allclose(full[ok], full[ok][0], rtol=0, atol=1e-12):
        # stała różnica (np. identyczne straty): brak zmienności, rozstrzyga sam znak
        dm0 = 0.0 if mean == 0 else float(np.sign(mean) * np.inf)
        return {"n": n, "mean_diff": mean, "dm": dm0, "p_value": 1.0 if mean == 0 else 0.0}
    L = max_lag if max_lag is not None else int(np.floor(n ** (1 / 3)))
    dc = np.where(ok, full - mean, 0.0)
    lr = dc @ dc / n
    for lag in range(1, L + 1):
        g = dc[lag:] @ dc[:-lag] / n  # pary z luką wnoszą zero (iloczyn z zerem)
        lr += 2 * (1 - lag / (L + 1)) * g
    if lr <= 0:
        return {"n": n, "mean_diff": mean, "dm": np.nan, "p_value": np.nan}
    dm = mean / np.sqrt(lr / n)
    if hln:
        dm = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(abs(dm), df=n - 1)
    return {
        "n": n,
        "mean_diff": mean,
        "dm": float(dm),
        "p_value": float(p),
        "nw_lags": L,
        "gaps": int((~ok).sum()),
    }


def dm_table(losses: pd.DataFrame, reference: str) -> pd.DataFrame:
    rows = []
    for m in losses.columns:
        if m == reference:
            continue
        ok = losses[[m, reference]].dropna()
        rows.append(
            {
                "model": m,
                **diebold_mariano(
                    ok[m].to_numpy(), ok[reference].to_numpy(), dates=pd.DatetimeIndex(ok.index)
                ),
            }
        )
    return pd.DataFrame(rows).set_index("model").sort_values("mean_diff")
