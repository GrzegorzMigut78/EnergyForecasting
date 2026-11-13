"""Kontrola techniczna przebiegu BEZ miar dokładności (plan analizy: między kwartałami testu
oglądamy wyłącznie kompletność, awarie, zbieżność, czasy i rozmiar archiwum)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from prognozowanie.config import ExperimentConfig

CONV_KEYS = ("convergence_code", "first_convergence_code", "residual_convergence_code")


def _codes(x: pd.Series) -> str:
    """Rozkład niezerowych kodów zbieżności, np. 'res2:18' (człon reszt, kod 2, 18 dób).
    Kod 2 z CSS (statsforecast) oznacza zwykle zatrzymanie optymalizatora przy optimum
    (sprawdzone w przeglądzie kodu: różnica SSE rzędu 1e-14); oceniamy go osobno od awarii."""
    vals = pd.Series([c for v in x for c in v.split(",") if c])
    if vals.empty:
        return ""
    return " ".join(f"{k}:{n}" for k, n in vals.value_counts().sort_index().items())


def run_status(cfg: ExperimentConfig, run_dir: Path, origins: pd.DatetimeIndex) -> str:
    rows = []
    for f in (run_dir / "meta").glob("*/*.json"):
        info = json.loads(f.read_text(encoding="utf-8"))
        rows.append(
            {
                "model": info["model"],
                "origin": pd.Timestamp(info["origin"]),
                "ok": info.get("status") == "ok",
                "codes": ",".join(
                    f"{k.split('_')[0][:3]}{info[k]}"
                    for k in CONV_KEYS
                    if info.get(k) not in (None, 0)
                ),
                "fitted_error": "fitted_error" in info,
                "fit": info.get("seconds_fit"),
                "predict_one": info.get("seconds_predict_one"),
                "total": info.get("seconds"),
            }
        )
    meta = pd.DataFrame(rows)
    lines = [f"# Status techniczny: {run_dir.name}", ""]
    lines.append(
        f"Doby w zakresie: {len(origins)} ({origins.min():%Y-%m-%d} - {origins.max():%Y-%m-%d})."
    )
    lines.append("Brak miar dokładności: zgodnie z planem analizy ocena dopiero po całym teście.")
    lines.append("")
    if meta.empty:
        lines.append("Archiwum jest puste.")
        return "\n".join(lines)
    meta = meta[meta["origin"].isin(origins)]
    names = [m.name for m in cfg.models]
    tab = (
        meta.groupby("model")
        .agg(
            done=("origin", "nunique"),
            failed=("ok", lambda x: int((~x).sum())),
            nonzero_codes=("codes", lambda x: _codes(x)),
            fitted_errors=("fitted_error", "sum"),
            fit_median_s=("fit", "median"),
            predict_one_median_s=("predict_one", "median"),
            total_hours=("total", lambda x: x.sum() / 3600),
        )
        .reindex(names)
    )
    tab["done"] = tab["done"].fillna(0).astype(int)
    tab["missing"] = len(origins) - tab["done"]
    size = sum(p.stat().st_size for p in (run_dir / "forecasts").rglob("*.parquet")) / 2**20
    lines += [
        tab.round(3).to_markdown(),
        "",
        "Kody zbieżności (con: model, fir: pierwszy stopień, res: człon reszt): SARIMA metodą CSS "
        "minimalizowana BFGS (scipy), kod 1 = limit iteracji, kod 2 = utrata precyzji w "
        "wyszukiwaniu liniowym przy końcu optymalizacji (sprawdzone 13.11.2026: ponowny start "
        "BFGS i L-BFGS-B z tego punktu kończą się w nim z kodem 0, inny punkt startowy nie daje "
        "lepszej wartości funkcji celu), kod 3 = wartości nieskończone. Awarie: kolumna failed.",
        "",
        f"Rozmiar archiwum prognoz: {size:.1f} MB.",
    ]
    return "\n".join(lines)
