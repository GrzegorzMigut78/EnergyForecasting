"""Scenariusze temperatury dla okresu prognozy (doba D i D+1).

oracle       temperatura zmierzona (idealna prognoza pogody)
persistence  przebieg temperatury z doby D-1 jako prognoza na D i D+1
noisy        temperatura zmierzona + losowy błąd dobowy, większy dla D+1, skorelowany między
             dobami

Wspólne liczby losowe: błąd standaryzowany (z1, z2) zależy tylko od (seed, data wydania,
numer losowania), nie od nazwy scenariusza ani modelu. Błąd scenariusza to
    e_D   = sd_day1 * z1
    e_D+1 = sd_day2 * (rho * z1 + sqrt(1 - rho^2) * z2),
więc scenariusze o różnym poziomie błędu różnią się wyłącznie skalą, a nie losowaniem
(gładka krzywa wrażliwości, mniejsza wariancja porównań między poziomami).
"""

from __future__ import annotations

import zlib
from collections.abc import Iterator

import numpy as np
import pandas as pd

from prognozowanie.config import TemperatureScenario

# wersja generatora błędów; wchodzi do migawki przebiegu, żeby archiwum policzone starszym
# generatorem nie zostało po cichu wznowione nowym
NOISE_GENERATOR = 2


def standard_errors(seed: int, origin: pd.Timestamp, draw: int) -> np.ndarray:
    """Błąd standaryzowany (z1, z2) dla doby wydania i numeru losowania."""
    key = f"{seed}|noisy|{origin:%Y-%m-%d}|{draw}".encode()
    return np.random.default_rng(zlib.crc32(key)).standard_normal(2)


def scenario_temperatures(
    scen: TemperatureScenario,
    hist: pd.DataFrame,
    fut: pd.DataFrame,
    origin: pd.Timestamp,
    seed: int,
) -> Iterator[tuple[int, np.ndarray]]:
    """Zwraca pary (numer losowania, temperatura dla wierszy fut)."""
    day = ((fut.index - origin) // pd.Timedelta(days=1)).to_numpy()  # 0 = doba D, 1 = D+1
    if scen.kind == "oracle":
        yield 0, fut["temp"].to_numpy(dtype=float)
    elif scen.kind == "persistence":
        # przebieg temperatury z doby D-1 powtórzony na D i D+1 (dla danych dobowych
        # to po prostu temperatura z D-1; dla godzinowych: ta sama godzina dnia poprzedniego)
        last24 = hist["temp"].to_numpy(dtype=float)[-24:]
        yield 0, np.resize(last24, len(fut))
    elif scen.kind == "noisy":
        base = fut["temp"].to_numpy(dtype=float)
        for draw in range(scen.draws):
            z1, z2 = standard_errors(seed, origin, draw)
            err = np.array(
                [
                    scen.sd_day1 * z1,
                    scen.sd_day2 * (scen.rho * z1 + np.sqrt(1 - scen.rho**2) * z2),
                ]
            )
            yield draw, base + err[np.clip(day, 0, 1)]
    else:  # pragma: no cover - chronione przez walidację konfiguracji
        raise ValueError(scen.kind)
