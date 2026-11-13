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
# metody uczone na historii prognoz (parametr K)
HISTORY_METHODS = ("C1", "C2", "C5", "C6", "C7", "C8")


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
    # C1, C2, C5-C8: historia prognoz tego samego bloku
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
    if method == "C7":
        return (*M.calibrate_level(Ft, yt, F), n_used)
    if method == "C8":
        return (*M.calibrate_affine(Ft, yt, F), n_used)
    raise ValueError(method)


def run_combination(cfg: ExperimentConfig, run_dir: Path) -> tuple[Path, Path]:
    """Łączenie partiami: osobno dla każdej pary (scenariusz, losowanie), z zapisem do
    combined/<scenariusz>__<losowanie>.parquet i weights/<...>.parquet. W pamięci jest
    naraz tylko jedna partia wyników (pełna konfiguracja to dziesiątki milionów wierszy)."""
    from prognozowanie.backtest import load_archive

    archive = load_archive(run_dir, models=sorted(cfg.combination.all_pooled_models()))
    for col in ("model", "scenario", "kind"):
        archive[col] = archive[col].astype("category")
    actual = pd.read_parquet(run_dir / "actuals.parquet")["load"]
    cdir, wdir = run_dir / "combined", run_dir / "weights"
    for d in (cdir, wdir):
        d.mkdir(exist_ok=True)
        for old in d.glob("*.parquet"):  # wyniki łączenia liczone zawsze od nowa
            old.unlink()
    for name in ("combined.parquet", "weights.parquet"):  # stary układ (jeden plik)
        (run_dir / name).unlink(missing_ok=True)
    fitted = archive[archive["kind"] == "fitted"]
    fc = archive[archive["kind"] == "forecast"]
    n_rows = n_w = 0
    for (scen, draw), part in fc.groupby(["scenario", "draw"], sort=True, observed=True):
        sub = pd.concat([part, fitted], ignore_index=True)
        combined, weights = combine(cfg, sub, actual)
        stem = f"{scen}__{int(draw)}"
        if not combined.empty:
            combined.to_parquet(cdir / f"{stem}.parquet", index=False)
            weights.to_parquet(wdir / f"{stem}.parquet", index=False)
        n_rows += len(combined)
        n_w += len(weights)
        log.info("łączenie %s: %d wierszy prognoz", stem, len(combined))
    log.info("łączenie: razem %d wierszy prognoz, %d zestawów wag", n_rows, n_w)
    return cdir, wdir


def leave_one_day_out(
    archive: pd.DataFrame,
    actual: pd.Series,
    models: list[str],
    origins: pd.DatetimeIndex,
    k: int,
) -> pd.DataFrame:
    """Diagnostyka stabilności C1 (uzupełnienie audytu): dla każdej doby wydania usuwamy
    kolejno jedną całą dobę z historii wag (blok D+1), ponownie szacujemy regresję łączącą
    i stosujemy ją do tych samych bieżących prognoz. Zapisujemy sumę modułów zmian wag
    i średnią godzinową zmianę prognozy (w jednostkach zapotrzebowania). To wpływ jednej
    doby uczącej, nie przedział predykcji. Tylko wydania z pełną historią K dób."""
    fc = archive[(archive["kind"] == "forecast") & archive["model"].isin(models)].copy()
    fc = fc[fc["horizon"] > 24]
    wide = {
        o: g.pivot(index="target", columns="model", values="value")[models]
        for o, g in fc.groupby("origin")
        if set(models) <= set(g["model"].unique())
    }
    rows = []
    for origin in origins:
        if origin not in wide:
            continue
        tr = training_origins(origin, 2, k)
        mats = [wide[o] for o in tr if o in wide]
        if len(mats) < k:
            continue
        F = wide[origin].to_numpy()
        Ft = pd.concat(mats)
        pred0, w0, _ = M.ols(Ft.to_numpy(), actual.reindex(Ft.index).to_numpy(), F)
        dw, dp = [], []
        for i in range(k):
            Fi = pd.concat(mats[:i] + mats[i + 1 :])
            pred, w, _ = M.ols(Fi.to_numpy(), actual.reindex(Fi.index).to_numpy(), F)
            dw.append(float(np.abs(w - w0).sum()))
            dp.append(float(np.abs(pred - pred0).mean()))
        rows.append(
            {
                "origin": origin,
                "k": k,
                "median_weight_change": float(np.median(dw)),
                "max_weight_change": float(np.max(dw)),
                "median_forecast_change": float(np.median(dp)),
                "max_forecast_change": float(np.max(dp)),
            }
        )
    return pd.DataFrame(rows)


def load_combined(run_dir: Path, scenarios: list[str] | None = None, what: str = "combined"):
    """Wyniki łączenia (what = "combined" albo "weights"), opcjonalnie tylko dla scenariuszy."""
    d = run_dir / what
    if not d.exists():
        return pd.DataFrame()
    files = sorted(d.glob("*.parquet"))
    if scenarios is not None:
        files = [f for f in files if f.stem.rsplit("__", 1)[0] in set(scenarios)]
    if not files:
        return pd.DataFrame()
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
