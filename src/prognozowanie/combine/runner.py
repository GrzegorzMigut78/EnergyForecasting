"""Łączenie prognoz na archiwum backtestu, bez zaglądania w przyszłość.

Dla doby wydania D i bloku b (b = 1: godziny 1-24, doba D; b = 2: godziny 25-48, doba D+1)
wagi wyznaczamy na prognozach tego samego bloku wydanych w dniach
    o = D - b - K + 1, ..., D - b,
czyli tylko takich, których doba docelowa (o + b - 1) zakończyła się przed chwilą D.
Wariant C0 (pierwotny) używa wartości dopasowanych z ostatnich 48 godzin okna; warianty
C0_H<n> z ostatnich n godzin (archiwum przechowuje forecast.fitted_hours_saved godzin).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from prognozowanie.combine import methods as M
from prognozowanie.config import ExperimentConfig

log = logging.getLogger(__name__)
BLOCKS = {1: (1, 24), 2: (25, 48)}
HISTORY_METHODS = ("C1", "C2", "C5", "C6")  # metody uczone na historii prognoz (parametr K)


def min_history(k: int) -> int:
    """Minimalna liczba dób historii, przy której wyznaczamy wagi: dopuszczamy jedną brakującą
    dobę na tydzień (awaria modelu), żeby pojedyncza awaria nie wycinała K kolejnych dób."""
    return max(2, k - k // 7)


def training_origins(origin: pd.Timestamp, block: int, k: int) -> pd.DatetimeIndex:
    """Doby wydania, których prognozy bloku ``block`` wolno użyć do wag w dobie ``origin``."""
    last = origin - pd.Timedelta(days=block)
    return pd.date_range(last - pd.Timedelta(days=k - 1), last, freq="D")


def _wide(fc: pd.DataFrame, models: list[str]) -> dict:
    """Słownik (origin, block) -> (macierz godzin x modele, wektor celów)."""
    out = {}
    for (origin, block), g in fc.groupby(["origin", "block"], sort=False):
        w = g.pivot(index="target", columns="model", values="value")
        if set(models) - set(w.columns):
            continue
        out[(origin, block)] = w[models]
    return out


def combine(
    cfg: ExperimentConfig,
    archive: pd.DataFrame,
    actual: pd.Series,
    origins: pd.DatetimeIndex | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Zwraca (prognozy łączone w formacie archiwum, wagi) dla wszystkich pul łączenia.

    Warianty puli głównej nazywają się jak metoda ("C2_K7"), pul dodatkowych
    "<pula>/<metoda>" ("plus_hong/C2_K7")."""
    out_rows: list[tuple] = []
    w_rows: list[dict] = []
    for pool_name, models in cfg.combination.pools().items():
        prefix = f"{pool_name}/" if pool_name else ""
        _combine_pool(cfg, archive, actual, origins, models, prefix, out_rows, w_rows)
    return _assemble(out_rows), pd.DataFrame(w_rows)


def _combine_pool(cfg, archive, actual, origins, models, prefix, out_rows, w_rows) -> None:
    fc = archive[(archive["kind"] == "forecast") & archive["model"].isin(models)].copy()
    if fc.empty:
        return
    fc["block"] = np.where(fc["horizon"] <= 24, 1, 2)
    fitted = archive[(archive["kind"] == "fitted") & archive["model"].isin(models)]
    fitted_w = {
        o: g.pivot(index="target", columns="model", values="value").reindex(columns=models)
        for o, g in fitted.groupby("origin")
    }
    origins = origins if origins is not None else pd.DatetimeIndex(sorted(fc["origin"].unique()))
    variants = method_variants(cfg)
    first_archived = fc["origin"].min()
    for (scen, draw), part in fc.groupby(["scenario", "draw"], sort=False):
        wide = _wide(part, models)
        for origin in origins:
            for block in (1, 2):
                key = (origin, block)
                if key not in wide:
                    continue  # brak prognozy któregoś modelu w tej dobie
                F = wide[key].to_numpy()
                target = wide[key].index
                for method, k, hours, label in variants:
                    res = _one(
                        method, k, hours, origin, block, F, wide, fitted_w, actual, first_archived
                    )
                    if res is None:
                        continue
                    pred, w, b0, n_used = res
                    name = prefix + label
                    out_rows.append((origin, name, scen, draw, target, pred))
                    w_rows.append(
                        {
                            "pool": prefix.rstrip("/") or "main",
                            "origin": origin,
                            "block": block,
                            "scenario": scen,
                            "draw": draw,
                            "method": method,
                            "k": k,
                            "fitted_hours": hours,
                            "intercept": b0,
                            "n_history_days": n_used,
                            **{f"w_{m}": float(x) for m, x in zip(models, w, strict=True)},
                        }
                    )


def method_variants(cfg: ExperimentConfig) -> list[tuple[str, int | None, int | None, str]]:
    """(metoda, K, godziny dopasowania C0, nazwa wariantu) dla wszystkich wariantów łączenia."""
    out = []
    for method in cfg.combination.methods:
        if method == "C0":
            out += [("C0", None, h, label) for label, h in cfg.combination.c0_variants()]
        elif method in HISTORY_METHODS:
            out += [(method, k, None, f"{method}_K{k}") for k in cfg.combination.k_days]
        else:
            out.append((method, None, None, method))
    return out


def _assemble(rows: list[tuple]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    n = [len(r[5]) for r in rows]
    origin = np.repeat(np.array([r[0] for r in rows], dtype="datetime64[ns]"), n)
    target = np.concatenate([r[4].to_numpy(dtype="datetime64[ns]") for r in rows])
    df = pd.DataFrame(
        {
            "origin": origin,
            "model": np.repeat([r[1] for r in rows], n),
            "scenario": np.repeat([r[2] for r in rows], n),
            "draw": np.repeat([r[3] for r in rows], n).astype(np.int16),
            "kind": "forecast",
            "target": target,
            "value": np.concatenate([r[5] for r in rows]),
        }
    )
    df.insert(
        6, "horizon", (((df["target"] - df["origin"]) / pd.Timedelta(hours=1)) + 1).astype(np.int16)
    )
    return df


def _one(method, k, hours, origin, block, F, wide, fitted_w, actual, first_archived):
    """Zwraca (prognoza, wagi, wyraz wolny, liczba dób historii) albo None."""
    if method == "C3":
        return (*M.mean(F), 0)
    if method == "C4":
        return (*M.median(F), 0)
    if method == "C0":
        fw = fitted_w.get(origin)
        if fw is None:
            return None
        if len(fw) < hours:
            raise RuntimeError(
                f"C0: archiwum ma {len(fw)} h wartości dopasowanych, wariant wymaga {hours} h"
            )
        fw = fw.iloc[-hours:]
        if fw.isna().any().any():
            return None
        if fw.index.max() >= origin:
            raise RuntimeError("C0: wartości dopasowane z przyszłości")
        return (*M.ols(fw.to_numpy(), actual.reindex(fw.index).to_numpy(), F), 0)
    # C1, C2, C5, C6: historia prognoz tego samego bloku
    tr = training_origins(origin, block, k)
    if tr[0] < first_archived:
        return None  # rozgrzewka: okno historii sięga przed pierwszą zapisaną dobę
    mats = [m for m in (wide.get((o, block)) for o in tr) if m is not None]
    if len(mats) < min_history(k):
        return None  # zbyt wiele dób z awarią któregoś modelu
    n_used = len(mats)
    Ft = pd.concat(mats)
    if Ft.index.max() >= origin:
        raise RuntimeError("łączenie: realizacja spoza zakresu informacji")
    yt = actual.reindex(Ft.index).to_numpy()
    Ft = Ft.to_numpy()
    if method == "C1":
        return (*M.ols(Ft, yt, F), n_used)
    if method == "C2":
        return (*M.simplex(Ft, yt, F), n_used)
    if method == "C5":
        return (*M.inverse_mse(Ft, yt, F), n_used)
    if method == "C6":
        return (*M.select_best(Ft, yt, F), n_used)
    raise ValueError(method)


def run_combination(cfg: ExperimentConfig, run_dir: Path) -> tuple[Path, Path]:
    from prognozowanie.backtest import load_archive

    archive = load_archive(run_dir, models=sorted(cfg.combination.all_pooled_models()))
    actual = pd.read_parquet(run_dir / "actuals.parquet")["load"]
    combined, weights = combine(cfg, archive, actual)
    p1, p2 = run_dir / "combined.parquet", run_dir / "weights.parquet"
    combined.to_parquet(p1, index=False)
    weights.to_parquet(p2, index=False)
    log.info("łączenie: %d wierszy prognoz, %d zestawów wag", len(combined), len(weights))
    return p1, p2
