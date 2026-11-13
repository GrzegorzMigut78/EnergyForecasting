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


def clean_of_excluded(
    frame: pd.DataFrame, excluded: list[tuple[pd.Timestamp, pd.Timestamp]], window_days: int
) -> pd.Series:
    """Maska wierszy, których prognoza nie korzysta z wyłączonych okresów danych.

    Prognoza z doby wydania D używa danych od D - W do końca D - 1 (W: okno uczące). Wariant
    łączenia z historią K bloku b szacuje wagi na prognozach wydanych w dobach D-b-K+1 ... D-b,
    więc sięga do D-b-K+1-W. Metody bez historii (C0, C3, C4) i modele: tylko okno doby D."""
    ok = pd.Series(True, index=frame.index)
    if not excluded or frame.empty:
        return ok
    origin = pd.to_datetime(frame["origin"]).dt.normalize()
    k = frame["model"].str.extract(_K)["k"].astype(float).fillna(0)
    b = np.where(frame["block"] == DAY2, 2, 1)
    back = window_days + np.where(k > 0, k + b - 1, 0)
    start = origin - pd.to_timedelta(back, unit="D")
    last = origin - pd.Timedelta(days=1)
    for ex_start, ex_end in excluded:
        ok &= (start > ex_end) | (last < ex_start)
    return ok


def holm(p: pd.Series, m: int | None = None) -> pd.Series:
    """Poprawka Holma-Bonferroniego dla rodziny porównań.

    m: planowana liczebność rodziny. Brakujące wartości p (kontrast nieoceniony) liczą się
    jak p = 1, czyli rodzina nie kurczy się po cichu, gdy któregoś wariantu brakuje."""
    p = p.astype(float)
    ok = p.dropna().sort_values()
    m = max(m or 0, len(ok))
    adj = np.maximum.accumulate(np.minimum(1.0, (m - np.arange(len(ok))) * ok.to_numpy()))
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
    """Czasy modeli (sekundy na dobę) z metadanych przebiegu, tylko udane doby:
    dopasowanie, jedna prognoza 48 h (poranne uruchomienie = dopasowanie + jedna prognoza),
    pętla po wszystkich scenariuszach i losowaniach (koszt badania, nie operacji), całość."""
    rows = []
    for f in (run_dir / "meta").glob("*/*.json"):
        info = json.loads(f.read_text(encoding="utf-8"))
        if info.get("status") != "ok":
            continue
        rows.append(
            {
                "model": info["model"],
                "fit": info.get("seconds_fit", np.nan),
                "predict_one": info.get("seconds_predict_one", np.nan),
                "scenarios": info.get("seconds_scenarios", np.nan),
                "total": info.get("seconds", np.nan),
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["operational"] = df["fit"] + df["predict_one"]
    out = df.groupby("model").agg(
        days=("total", "size"),
        fit_median=("fit", "median"),
        predict_one_median=("predict_one", "median"),
        operational_median=("operational", "median"),
        operational_p90=("operational", lambda x: x.quantile(0.9)),
        research_total_hours=("total", lambda x: x.sum() / 3600),
    )
    return out.sort_values("operational_median")


def break_even(sens: pd.DataFrame, no_temperature: list[str]) -> pd.DataFrame:
    """Dla każdego wariantu korzystającego z temperatury: poziom błędu prognozy temperatury
    (sd dla D+1), przy którym jego MAE doby D+1 zrównuje się z najlepszym modelem bez
    temperatury (interpolacja liniowa między poziomami siatki).

    no_temperature: modele, które z definicji nie korzystają z temperatury (z wejść modeli).
    Odniesienie (najlepszy z nich) jest wybrane opisowo na teście, co raport zaznacza.
    Raportujemy liczbę przecięć; przy kilku przecięciach albo różnicy bliskiej zeru wynik
    jest oznaczony jako niepewny."""
    curve = sens.dropna(subset=["sd_day2"]).pivot(index="sd_day2", columns="model", values="MAE")
    curve = curve.sort_index()
    flat = [m for m in no_temperature if m in curve.columns]
    if not flat:
        return pd.DataFrame()
    ref_model = curve[flat].iloc[0].idxmin()
    ref = float(curve[ref_model].iloc[0])
    rows = []
    x = curve.index.to_numpy(dtype=float)
    for m in curve.columns:
        if m in flat or m in no_temperature:
            continue
        y = curve[m].to_numpy() - ref
        sign = np.sign(y)
        crossings = np.where(np.diff(sign) != 0)[0]
        if np.allclose(y, 0.0, atol=1e-9):  # np. wybór (C6) zawsze trafiający w odniesienie
            be, note = np.nan, "taki sam jak odniesienie na całej siatce"
        elif y[0] >= 0:
            be, note = 0.0, "gorszy już przy temperaturze zmierzonej"
        elif not len(crossings):
            be, note = np.nan, f"lepszy na całej siatce (do {x[-1]:g})"
        else:
            i = int(crossings[0]) + 1
            be = float(x[i - 1] + (x[i] - x[i - 1]) * (-y[i - 1]) / (y[i] - y[i - 1]))
            note = "kilka przecięć: niepewny" if len(crossings) > 1 else ""
        if np.nanmin(np.abs(y)) < 0.01 * ref and not note:
            note = "różnica bliska zeru: niepewny"
        rows.append(
            {
                "model": m,
                "MAE_oracle": float(curve[m].iloc[0]),
                "reference": ref_model,
                "reference_choice": "opisowo na teście",
                "MAE_reference": ref,
                "break_even_sd_day2": be,
                "n_crossings": len(crossings),
                "note": note,
            }
        )
    return pd.DataFrame(rows).sort_values("MAE_oracle")


def family_models(families) -> list[str]:
    """Wszystkie warianty występujące w rodzinach hipotez (porównywane i odniesienia)."""
    out: list[str] = []
    for f in families.values():
        out += [f.reference, *f.compare]
    return list(dict.fromkeys(out))


def primary_tests(losses: pd.DataFrame, families) -> pd.DataFrame:
    """Testy DM w rodzinach hipotez głównych (konfiguracja evaluation.families), poprawka
    Holma w obrębie rodziny z planowaną liczebnością. Kontrast bez danych zostaje w tabeli
    jako nieoceniony. Kolumny kontrolne: h = 2 w poprawce HLN, pasmo 14 dób, HAC bez HLN.
    Ujemna średnia różnica: wariant porównywany lepszy od odniesienia."""
    rows = []
    for fam, spec in families.items():
        part = []
        for m in spec.compare:
            base = {"family": fam, "reference": spec.reference, "model": m}
            if m not in losses.columns or spec.reference not in losses.columns:
                part.append({**base, "status": "nieoceniony", "p_value": np.nan})
                continue
            ok = losses[[m, spec.reference]].dropna()
            dates = pd.DatetimeIndex(ok.index)
            a, b = ok[m].to_numpy(), ok[spec.reference].to_numpy()
            main = mt.diebold_mariano(a, b, dates=dates)
            part.append(
                {
                    **base,
                    "status": "oceniony",
                    **main,
                    "p_h2": mt.diebold_mariano(a, b, dates=dates, h=2)["p_value"],
                    "p_bw14": mt.diebold_mariano(a, b, dates=dates, max_lag=14)["p_value"],
                    "p_no_hln": mt.diebold_mariano(a, b, dates=dates, hln=False)["p_value"],
                }
            )
        df = pd.DataFrame(part)
        df["p_holm"] = holm(df["p_value"], m=len(spec.compare)).to_numpy()
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def confirm_across_scenarios(
    tables: dict[str, pd.DataFrame], families, alpha: float
) -> pd.DataFrame:
    """Reguła potwierdzenia (plan analizy): dla każdego kontrastu większa z wartości p
    w scenariuszach, poprawka Holma w rodzinie (planowana liczebność) i zgodny kierunek."""
    scen = list(tables)
    keys = ["family", "reference", "model"]
    merged = None
    for s_name, t in tables.items():
        part = t[[*keys, "mean_diff", "p_value"]].rename(
            columns={"mean_diff": f"mean_diff_{s_name}", "p_value": f"p_{s_name}"}
        )
        merged = part if merged is None else merged.merge(part, on=keys, how="outer")
    if merged is None:
        return pd.DataFrame()
    pcols = [f"p_{s}" for s in scen]
    merged["p_max"] = merged[pcols].max(axis=1, skipna=False)
    signs = np.sign(merged[[f"mean_diff_{s}" for s in scen]])
    merged["same_direction"] = signs.nunique(axis=1).eq(1) & signs.notna().all(axis=1)
    merged["p_max_holm"] = np.nan
    for fam, spec in families.items():
        sel = merged["family"] == fam
        merged.loc[sel, "p_max_holm"] = holm(merged.loc[sel, "p_max"], m=len(spec.compare))
    merged["confirmed"] = merged["same_direction"] & (merged["p_max_holm"] < alpha)
    return merged


def coverage(
    frame: pd.DataFrame,
    planned: list[str],
    n_test_days: int,
    main_models: set[str],
    main_days: int,
    all_days: dict[str, int],
) -> pd.DataFrame:
    """Planowane warianty (uwaga audytora): doby testu, doby dostępne (z prognozą D+1)
    i doby użyte w analizie głównej oraz w zestawieniu wszystkich wariantów (all_days: doby
    D+1 danego wariantu w tym zestawieniu)."""
    d2 = frame[frame["block"] == DAY2]
    avail = d2.groupby("model")["origin"].nunique()
    rows = []
    for m in planned:
        n_av = int(avail.get(m, 0))
        rows.append(
            {
                "variant": m,
                "planned_days": n_test_days,
                "available_days": n_av,
                "main_analysis": m in main_models,
                "used_main_days": main_days if m in main_models and n_av else 0,
                "used_all_days": int(all_days.get(m, 0)),
            }
        )
    return pd.DataFrame(rows)


def mae_cancellation(frame: pd.DataFrame, models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rozkład MAE średniej w mierze głównej (uzupełnienie audytu): dla każdej godziny
    średnia po modelach |e| - |błąd średniej| = 2 min(P, N), gdzie P i N to średnie po
    modelach z dodatnich i ujemnych części błędów. Uśrednione po godzinach doby D+1.

    Zwraca (podsumowanie z porównaniem z każdym modelem, przebieg według godziny doby)."""
    d2 = frame[(frame["block"] == DAY2) & frame["model"].isin(models)]
    E = d2.pivot_table(index=["origin", "target", "draw"], columns="model", values="error").dropna()
    if E.shape[1] != len(models) or E.empty:
        return pd.DataFrame(), pd.DataFrame()
    e = E[models].to_numpy()
    P = np.clip(e, 0, None).mean(axis=1)
    N = np.clip(-e, 0, None).mean(axis=1)
    canc = 2 * np.minimum(P, N)
    avg_mae = float((P + N).mean())
    mae_mean = float(np.abs(P - N).mean())
    rows = []
    for j, m in enumerate(models):
        mae_j = float(np.abs(e[:, j]).mean())
        rows.append(
            {
                "model": m,
                "MAE_model": mae_j,
                "MAE_mean_of_models": avg_mae,
                "MAE_C3": mae_mean,
                "cancellation": float(canc.mean()),
                "needed_to_beat": avg_mae - mae_j,
                "C3_beats_model": mae_mean < mae_j,
            }
        )
    hours = E.index.get_level_values("target").hour + 1
    by_hour = (
        pd.DataFrame({"hour": hours, "cancellation": canc, "mixed_signs": (P > 0) & (N > 0)})
        .groupby("hour")
        .mean()
    )
    for j, m in enumerate(models):
        by_hour[f"needed_vs_{m}"] = (
            pd.Series(P + N - np.abs(e[:, j]), index=hours).groupby(level=0).mean()
        )
    summary = pd.DataFrame(rows)
    summary.attrs["share_mixed_signs"] = float(((P > 0) & (N > 0)).mean())
    return summary, by_hour


def asymmetric_loss(frame: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Niedoszacowanie U i przeszacowanie O (dodatnie części błędów, doba D+1); MAE = U + O.
    Strata przy stosunku kosztów r (niedoszacowanie do przeszacowania): r U + O. Dwa warianty
    zamieniają kolejność przy r* = -dO/dU (wobec odniesienia). Analiza wrażliwości przy
    liniowej funkcji strat, bez przeliczania na pieniądze."""
    d2 = frame[frame["block"] == DAY2]
    g = d2.assign(U=np.clip(-d2["error"], 0, None), O=np.clip(d2["error"], 0, None))
    t = g.groupby("model")[["U", "O"]].mean()
    t["MAE"] = t["U"] + t["O"]
    if reference in t.index:
        dU = t["U"] - t.loc[reference, "U"]
        dO = t["O"] - t.loc[reference, "O"]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = -dO / dU
        t["r_star_vs_ref"] = r.where((dU != 0) & (r > 0))
        for ratio in (0.5, 1.0, 2.0, 4.0):
            t[f"loss_r{ratio:g}"] = ratio * t["U"] + t["O"]
    return t.sort_values("MAE")


def fitted_vs_forecast_pairs(
    losses: pd.DataFrame, weights: pd.DataFrame | None, seed: int
) -> pd.DataFrame:
    """C0 z n godzinami dopasowań kontra C1 z K = n/24 dobami prognoz: ta sama liczba
    realizacji w regresji (uzupełnienie audytu). Tylko doby, w których C1 miał pełną
    historię K dób. Para 48 h / K = 2 jest główna (H2), pozostałe kontrolne."""
    from prognozowanie.evaluate.inference import bootstrap_ci

    pairs = [("C0", "C1_K2", 2), ("C0_H72", "C1_K3", 3), ("C0_H168", "C1_K7", 7)]
    rows = []
    for c0, c1, k in pairs:
        if c0 not in losses.columns or c1 not in losses.columns:
            continue
        ok = losses[[c0, c1]].dropna()
        if weights is not None and not weights.empty:
            full = weights[
                (weights["method"] == "C1")
                & (weights["k"] == k)
                & (weights["block"] == 2)
                & (weights["n_history_days"] == k)
            ]["origin"].unique()
            ok = ok[ok.index.isin(full)]
        if len(ok) < 10:
            continue
        dm = mt.diebold_mariano(
            ok[c0].to_numpy(), ok[c1].to_numpy(), dates=pd.DatetimeIndex(ok.index)
        )
        ci = bootstrap_ci(ok, reference=c1, n_boot=2000, seed=seed)
        rows.append(
            {
                "fitted_variant": c0,
                "forecast_variant": c1,
                "days": len(ok),
                "mean_diff": float((ok[c0] - ok[c1]).mean()),
                "ci_low": float(ci.loc[c0, "diff_ci_low"]),
                "ci_high": float(ci.loc[c0, "diff_ci_high"]),
                "p_value": dm["p_value"],
                "role": "H2 (główna)" if k == 2 else "kontrola",
            }
        )
    return pd.DataFrame(rows)


def primary_set(families, models: list[str]) -> list[str]:
    """Zbiór analizy głównej (MCS, bootstrap): warianty z rodzin hipotez."""
    return [m for m in family_models(families) if m in models]
