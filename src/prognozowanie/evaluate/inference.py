"""Wnioskowanie o różnicach trafności: blokowy bootstrap po dobach i zbiór modeli najlepszych
(Model Confidence Set, Hansen, Lunde i Nason 2011, statystyka T_max).

Wejście: macierz dziennych strat (wiersze = doby wydania, kolumny = modele). Kolejne doby
są zależne, dlatego losujemy bloki kolejnych dób (moving block bootstrap). Bloki są
kolejnymi dobami KALENDARZA: doby wykluczone z porównania (luki) zostają w kalendarzu jako
puste i nie wnoszą nic do średniej bloku, więc blok nie skleja dób odległych w czasie.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def block_indices(n: int, block: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Indeksy n_boot pseudoprób długości n złożonych z bloków kolejnych dób."""
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)
    return idx[:, :n]


def _calendar(losses: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Straty na pełnym kalendarzu dób (luki jako NaN) i liczba luk."""
    L = losses.dropna()
    if not isinstance(L.index, pd.DatetimeIndex) or L.empty:
        return L, 0
    cal = pd.date_range(L.index.min(), L.index.max(), freq="D")
    return L.reindex(cal), len(cal) - len(L)


def bootstrap_ci(
    losses: pd.DataFrame,
    reference: str | None = None,
    block: int = 7,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """Przedziały ufności średniej straty każdego modelu oraz (jeśli podano model odniesienia)
    różnicy względem niego. Ujemna różnica: model lepszy od odniesienia."""
    L, gaps = _calendar(losses)
    rng = np.random.default_rng(seed)
    idx = block_indices(len(L), block, n_boot, rng)
    X = L.to_numpy()
    with np.errstate(all="ignore"):
        boot = np.nanmean(X[idx], axis=1)  # n_boot x modele
    a = (1 - level) / 2
    out = pd.DataFrame(
        {
            "mean_loss": np.nanmean(X, axis=0),
            "ci_low": np.quantile(boot, a, axis=0),
            "ci_high": np.quantile(boot, 1 - a, axis=0),
        },
        index=L.columns,
    )
    if reference is not None:
        r = list(L.columns).index(reference)
        diff = boot - boot[:, [r]]
        out["diff_vs_ref"] = np.nanmean(X, axis=0) - np.nanmean(X[:, r])
        out["diff_ci_low"] = np.quantile(diff, a, axis=0)
        out["diff_ci_high"] = np.quantile(diff, 1 - a, axis=0)
    out.attrs.update(
        {"n_days": len(L) - gaps, "gaps": gaps, "block": block, "n_boot": n_boot, "level": level}
    )
    return out


def model_confidence_set(
    losses: pd.DataFrame,
    alpha: float = 0.10,
    block: int = 7,
    n_boot: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """MCS z eliminacją według T_max. Zwraca kolejność eliminacji, p-wartości MCS
    i przynależność do zbioru najlepszych na poziomie alpha."""
    L, gaps = _calendar(losses)
    names = list(L.columns)
    X = L.to_numpy()
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = block_indices(n, block, n_boot, rng)
    alive = list(range(len(names)))
    records, p_running = [], 0.0
    while len(alive) > 1:
        Xa = X[:, alive]
        d = Xa - Xa.mean(axis=1, keepdims=True)  # strata względem średniej w zbiorze
        dbar = np.nanmean(d, axis=0)
        with np.errstate(all="ignore"):
            boot = np.nanmean(d[idx], axis=1)  # n_boot x |alive|
        var = ((boot - dbar) ** 2).mean(axis=0)
        t = dbar / np.sqrt(np.maximum(var, 1e-300))
        t_max = t.max()
        t_boot = ((boot - dbar) / np.sqrt(np.maximum(var, 1e-300))).max(axis=1)
        p = float((t_boot >= t_max).mean())
        p_running = max(p_running, p)
        worst = alive[int(np.argmax(t))]
        records.append(
            {"model": names[worst], "eliminated_order": len(records) + 1, "mcs_p_value": p_running}
        )
        alive.remove(worst)
    records.append(
        {"model": names[alive[0]], "eliminated_order": len(records) + 1, "mcs_p_value": 1.0}
    )
    out = pd.DataFrame(records).set_index("model")
    out["in_mcs"] = out["mcs_p_value"] >= alpha
    out.attrs.update(
        {"alpha": alpha, "n_days": n - gaps, "gaps": gaps, "block": block, "n_boot": n_boot}
    )
    return out.sort_values("mcs_p_value", ascending=False)
