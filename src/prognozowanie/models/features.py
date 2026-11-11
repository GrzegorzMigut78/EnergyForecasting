"""Macierze zmiennych objaśniających."""

from __future__ import annotations

import numpy as np
import pandas as pd

DOW_NAMES = ["pon", "wt", "sr", "czw", "pia", "sob"]  # baza: niedziela
BASE_HOUR = 3  # baza: godzina 3, jak w tabeli 1 artykułu


def trend(index: pd.DatetimeIndex, origin: pd.Timestamp) -> np.ndarray:
    """Numer godziny liczony od początku okna (1, 2, ...); dla przyszłości kontynuacja."""
    return ((index - origin) / pd.Timedelta(hours=1)).to_numpy() + 1.0


def article_features(
    df: pd.DataFrame, window_start: pd.Timestamp, use_trend: bool = True
) -> pd.DataFrame:
    """Zmienne modelu regresji z artykułu: temperatura, trend, święto, 6 dni tygodnia,
    23 godziny."""
    out = {"temp": df["temp"].to_numpy(dtype=float)}
    if use_trend:
        out["t"] = trend(df.index, window_start)
    out["holiday"] = df["holiday"].to_numpy(dtype=float)
    dow = df["dow"].to_numpy()
    for d, name in enumerate(DOW_NAMES):
        out[name] = (dow == d).astype(float)
    hour = df["hour"].to_numpy()
    for h in range(1, 25):
        if h != BASE_HOUR:
            out[f"g{h:02d}"] = (hour == h).astype(float)
    return pd.DataFrame(out, index=df.index)


def hong_features(df: pd.DataFrame, window_start: pd.Timestamp) -> pd.DataFrame:
    """Regresja benchmarkowa Honga (GEFCom2012) dostosowana do okna kilku miesięcy
    i temperatury dobowej: trend, dzień tygodnia x godzina, święto,
    wielomian temperatury 3. stopnia oraz jego interakcje z godziną."""
    idx = df.index
    t = trend(idx, window_start)
    temp = df["temp"].to_numpy(dtype=float)
    dow = df["dow"].to_numpy()
    hour = df["hour"].to_numpy()
    cols: dict[str, np.ndarray] = {"t": t, "holiday": df["holiday"].to_numpy(dtype=float)}
    # dzień x godzina: 7*24 komórek, baza = niedziela, godzina 3
    for d in range(7):
        for h in range(1, 25):
            if d == 6 and h == BASE_HOUR:
                continue
            cols[f"d{d}h{h:02d}"] = ((dow == d) & (hour == h)).astype(float)
    for p in (1, 2, 3):
        tp = temp**p
        cols[f"T{p}"] = tp
        for h in range(1, 25):
            if h != BASE_HOUR:
                cols[f"T{p}h{h:02d}"] = tp * (hour == h)
    return pd.DataFrame(cols, index=idx)
