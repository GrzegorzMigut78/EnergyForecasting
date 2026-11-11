"""Analizy dodatkowe raportu (plan analizy, docs/Plan_analizy.md, sekcja 4).

Wszystkie liczone z archiwum i prognoz łączonych, bez nowych prognoz:
  k_profile       MAE doby D+1 w funkcji K dla metod uczonych na historii, z C3 jako odniesieniem
                  ("zagadka łączenia prognoz": szum estymacji wag a nieaktualność wag)
  single_window   jak często pojedyncze okno 48 h (układ pierwotnej wersji artykułu) dałoby
                  wniosek "łączenie lepsze od najlepszego modelu"
  diversity       rozkład błędu średniej: średni błąd modeli minus różnorodność (Krogh, Vedelsby)
  arima_ablation  model dwustopniowy a jego pierwszy stopień (czy korekta ARIMA pomaga)
  compute_cost    czas obliczeń modeli (z metadanych)
  break_even      poziom błędu prognozy temperatury, przy którym model z temperaturą przestaje
                  wygrywać z najlepszym modelem bez temperatury
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from prognozowanie.evaluate import metrics as mt

DAY2 = "D+1 (25-48 h)"
STAGE1 = "__stage1"
_K = re.compile(r"^(?:(?P<pool>[^/]+)/)?(?P<method>C[0-9])_K(?P<k>\d+)$")


def holm(p: pd.Series) -> pd.Series:
    """Poprawka Holma-Bonferroniego dla rodziny porównań (wartości p w dowolnej kolejności)."""
    p = p.astype(float)
    ok = p.dropna().sort_values()
    m = len(ok)
    adj = np.maximum.accumulate(np.minimum(1.0, (m - np.arange(m)) * ok.to_numpy()))
    return pd.Series(adj, index=ok.index).reindex(p.index)


def k_profile(summary: pd.DataFrame) -> pd.DataFrame:
    """Wiersze: (pula, metoda, K) z MAE doby D+1; C3 puli jako odniesienie (kolumna vs_C3)."""
    d2 = summary[summary["block"] == DAY2]
    rows = []
    for _, r in d2.iterrows():
        m = _K.match(r["model"])
        if m:
            rows.append(
                {
                    "pool": m["pool"] or "main",
                    "method": m["method"],
                    "k": int(m["k"]),
                    "MAE": r["MAE"],
                    "RMSE": r["RMSE"],
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    ref = {
        pool: d2.loc[d2["model"] == ("C3" if pool == "main" else f"{pool}/C3"), "MAE"]
        for pool in out["pool"].unique()
    }
    out["MAE_C3"] = out["pool"].map({p: (v.iloc[0] if len(v) else np.nan) for p, v in ref.items()})
    out["vs_C3"] = out["MAE"] - out["MAE_C3"]
    return out.sort_values(["pool", "method", "k"]).reset_index(drop=True)


def single_window(
    scen_df: pd.DataFrame, base_models: list[str], combos: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dla każdej doby wydania: MAE z 48 h (oba bloki) łączenia i najlepszego modelu bazowego
    wybranego po fakcie, jak w pierwotnej wersji artykułu (jedno okno, najlepszy model ex post).

    Zwraca (tabela dób, podsumowanie: odsetek dób, w których łączenie wygrywa)."""
    models = [m for m in [*base_models, *combos] if m in set(scen_df["model"])]
    per = (
        scen_df[scen_df["model"].isin(models)]
        .assign(ae=lambda d: d["error"].abs())
        .groupby(["origin", "model"])
        .agg(mae=("ae", "mean"), hours=("ae", "size"), draws=("draw", "nunique"))
        .reset_index()
    )
    per = per[per["hours"] == 48 * per["draws"]]  # tylko doby z kompletnym oknem 48 h
    wide = per.pivot(index="origin", columns="model", values="mae").dropna()
    base = [m for m in base_models if m in wide.columns]
    if wide.empty or not base:
        return pd.DataFrame(), pd.DataFrame()
    wide["best_base"] = wide[base].min(axis=1)
    wide["best_base_model"] = wide[base].idxmin(axis=1)
    rows = []
    for c in combos:
        if c not in wide.columns:
            continue
        diff = wide[c] - wide["best_base"]
        wide[f"diff_{c}"] = diff
        rows.append(
            {
                "combination": c,
                "days": len(diff),
                "share_beats_best_base": float((diff < 0).mean()),
                "median_diff": float(diff.median()),
                "q10_diff": float(diff.quantile(0.1)),
                "q90_diff": float(diff.quantile(0.9)),
            }
        )
    return wide, pd.DataFrame(rows)


def diversity(scen_df: pd.DataFrame, pools: dict[str, list[str]]) -> pd.DataFrame:
    """Rozkład błędu kwadratowego średniej puli dla doby D+1:
    MSE(średnia) = średni MSE modeli - różnorodność, gdzie różnorodność to średni kwadrat
    odchylenia prognoz modeli od ich średniej. Do tego średnia korelacja błędów par modeli."""
    d2 = scen_df[scen_df["block"] == DAY2]
    rows = []
    for pool, models in pools.items():
        part = d2[d2["model"].isin(models)]
        E = part.pivot_table(
            index=["origin", "target", "draw"], columns="model", values="error"
        ).dropna()
        if E.shape[1] != len(models) or E.empty:
            continue
        e = E.to_numpy()
        ebar = e.mean(axis=1, keepdims=True)
        avg_mse = float((e**2).mean())
        div = float(((e - ebar) ** 2).mean())
        corr = np.corrcoef(e.T)
        iu = np.triu_indices(len(models), 1)
        rows.append(
            {
                "pool": pool or "main",
                "models": len(models),
                "mean_model_mse": avg_mse,
                "diversity": div,
                "mse_of_mean": float((ebar**2).mean()),
                "diversity_share": div / avg_mse,
                "mean_error_corr": float(corr[iu].mean()),
                "min_error_corr": float(corr[iu].min()),
            }
        )
    return pd.DataFrame(rows)


def arima_ablation(losses: pd.DataFrame) -> pd.DataFrame:
    """Model dwustopniowy a jego pierwszy stopień: dzienne straty doby D+1, test DM."""
    rows = []
    for m in losses.columns:
        if m.endswith(STAGE1):
            full = m[: -len(STAGE1)]
            if full in losses.columns:
                ok = losses[[full, m]].dropna()
                res = mt.diebold_mariano(ok[full].to_numpy(), ok[m].to_numpy())
                rows.append(
                    {
                        "model": full,
                        "MAE_two_stage": float(ok[full].mean()),
                        "MAE_first_stage": float(ok[m].mean()),
                        **res,
                    }
                )
    return pd.DataFrame(rows)


def compute_cost(run_dir: Path) -> pd.DataFrame:
    """Czas dopasowania i prognozy modeli (sekundy na dobę) z metadanych przebiegu."""
    rows = []
    for f in (run_dir / "meta").glob("*/*.json"):
        info = json.loads(f.read_text(encoding="utf-8"))
        rows.append({"model": info["model"], "seconds": info.get("seconds", np.nan)})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return (
        df.groupby("model")["seconds"]
        .agg(days="size", mean="mean", median="median", max="max", total="sum")
        .sort_values("mean")
    )


def break_even(sens: pd.DataFrame) -> pd.DataFrame:
    """Dla każdego modelu korzystającego z temperatury: poziom błędu prognozy temperatury
    (sd dla D+1), przy którym jego MAE doby D+1 zrównuje się z najlepszym modelem bez
    temperatury (interpolacja liniowa między poziomami siatki).

    Model "bez temperatury" rozpoznajemy po wyniku: MAE stałe na całej siatce."""
    curve = sens.dropna(subset=["sd_day2"]).pivot(index="sd_day2", columns="model", values="MAE")
    curve = curve.sort_index()
    flat = [m for m in curve.columns if np.ptp(curve[m].to_numpy()) < 1e-9]
    if not flat:
        return pd.DataFrame()
    ref_model = curve[flat].iloc[0].idxmin()
    ref = float(curve[ref_model].iloc[0])
    rows = []
    x = curve.index.to_numpy(dtype=float)
    for m in curve.columns:
        if m in flat:
            continue
        y = curve[m].to_numpy() - ref
        if y[0] >= 0:
            be, note = 0.0, "gorszy już przy temperaturze zmierzonej"
        elif np.all(y < 0):
            be, note = np.nan, f"lepszy na całej siatce (do {x[-1]:g})"
        else:
            i = int(np.argmax(y >= 0))
            be = float(x[i - 1] + (x[i] - x[i - 1]) * (-y[i - 1]) / (y[i] - y[i - 1]))
            note = ""
        rows.append(
            {
                "model": m,
                "MAE_oracle": float(curve[m].iloc[0]),
                "reference": ref_model,
                "MAE_reference": ref,
                "break_even_sd_day2": be,
                "note": note,
            }
        )
    return pd.DataFrame(rows).sort_values("MAE_oracle")


def primary_families(
    base_models: list[str], benchmarks: list[str], k_days: list[int], methods: list[str]
) -> dict[str, list[str]]:
    """Rodziny hipotez głównych (plan analizy, sekcja 3); każde porównanie: model vs C3."""
    fam = {
        "H1 wagi estymowane vs C3": [
            f"{m}_K{k}" for m in ("C1", "C2", "C5") if m in methods for k in k_days
        ],
        "H2 procedura pierwotna C0 vs C3": ["C0"] if "C0" in methods else [],
        "H3 modele pojedyncze vs C3": [*base_models, *benchmarks],
        "H4 wybór C6 vs C3": [f"C6_K{k}" for k in k_days] if "C6" in methods else [],
    }
    return {k: v for k, v in fam.items() if v}


def primary_tests(losses: pd.DataFrame, families: dict[str, list[str]]) -> pd.DataFrame:
    """Testy DM wobec C3 w rodzinach hipotez głównych, poprawka Holma w obrębie rodziny.
    Ujemna średnia różnica: model lepszy od C3."""
    rows = []
    for fam, models in families.items():
        part = []
        for m in models:
            if m not in losses.columns or "C3" not in losses.columns:
                continue
            ok = losses[[m, "C3"]].dropna()
            part.append(
                {
                    "family": fam,
                    "model": m,
                    **mt.diebold_mariano(ok[m].to_numpy(), ok["C3"].to_numpy()),
                }
            )
        if part:
            df = pd.DataFrame(part)
            df["p_holm"] = holm(df["p_value"]).to_numpy()
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def primary_set(base_models: list[str], benchmarks: list[str], models: list[str]) -> list[str]:
    """Zbiór do MCS w analizie głównej: modele pojedyncze i warianty łączenia puli głównej
    (bez pierwszych stopni i pul rozszerzonych)."""
    combos = [m for m in models if m.startswith("C") and "/" not in m and STAGE1 not in m]
    return [m for m in [*base_models, *benchmarks, *combos] if m in models]
