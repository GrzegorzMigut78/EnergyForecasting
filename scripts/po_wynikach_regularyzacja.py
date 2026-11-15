# Uruchomienie:
#   uv run python scripts/po_wynikach_regularyzacja.py <katalog results> <katalog wyjściowy>
"""Plan analizy, sekcja 10.1: analiza dodana po wynikach, zatwierdzona 15.11.2026 przed obliczeniem.

Regresja łącząca ze ściąganiem wag do równych: C9 = psi * C1_K + (1 - psi) * C3, z archiwalnych
prognoz połączonych (nic nie jest szacowane od nowa). Wartości psi ustalone z góry: reguła Stocka
i Watsona z T w godzinach (a) i w dobach (b), kappa w {0,25; 0,5; 1}, oraz stała siatka
{0,25; 0,5; 0,75}. Doba D+1, scenariusze oracle i persystencja, doby testu z regułą wyłączonych
okresów (dla K jak w C1_K). Porównania opisowe: C9 minus C1_K i C9 minus C3, bootstrap blokowy
7 dób, DM bez korekty Holma. Archiwum tylko do odczytu; wyniki do osobnego katalogu.
"""

import sys
from pathlib import Path

import pandas as pd

from prognozowanie.config import load_experiment_config
from prognozowanie.evaluate import analyses as an
from prognozowanie.evaluate import metrics as mt
from prognozowanie.evaluate.inference import bootstrap_ci

RES, OUT = Path(sys.argv[1]), Path(sys.argv[2])
ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "ElecEner": ("main", "configs/experiments/main.yaml"),
    "GEFCom2012": ("gefcom2012", "configs/experiments/gefcom2012.yaml"),
}
KS = [2, 3, 7, 14, 28]
N_MODELS = 5
KAPPAS = [0.25, 0.5, 1.0]
FIXED = [0.25, 0.5, 0.75]
DAY2 = "D+1 (25-48 h)"


def psi_values(k: int) -> dict[str, float]:
    out = {}
    for label, t in (("SW_godziny", 24 * k), ("SW_doby", k)):
        for kap in KAPPAS:
            den = t - 1 - N_MODELS
            psi = 0.0 if den <= 0 else max(0.0, 1 - kap * N_MODELS / den)
            out[f"{label}_kappa{kap}"] = psi
    for p in FIXED:
        out[f"stale_{p}"] = p
    return out


rows = []
for ds, (run, cfg_path) in RUNS.items():
    cfg = load_experiment_config(ROOT / cfg_path, use_identified=False)
    t0, t1 = pd.Timestamp(cfg.periods.test.start), pd.Timestamp(cfg.periods.test.end)
    excl = [(pd.Timestamp(p.start), pd.Timestamp(p.end)) for p in cfg.evaluation.excluded_data]
    win = cfg.forecast.training_window_days
    actuals = pd.read_parquet(RES / run / "actuals.parquet")
    for scen in ("oracle", "persistence"):
        c = pd.read_parquet(RES / run / "combined" / f"{scen}__0.parquet")
        c = c[(c.kind == "forecast") & (c.origin >= t0) & (c.origin <= t1)]
        key = ["origin", "target", "horizon"]
        c3 = c[c.model == "C3"].set_index(key)["value"]
        for k in KS:
            c1 = c[c.model == f"C1_K{k}"].copy()
            c1 = mt.attach_actuals(c1, actuals)
            c1 = c1[(c1.block == DAY2) & an.clean_of_excluded(c1, excl, win)]
            base = c1.set_index(key)
            mean = c3.reindex(base.index)
            ok = mean.notna()
            base, mean = base[ok], mean[ok]
            frames = {"C1": base["value"], "C3": mean}
            for lab, psi in psi_values(k).items():
                frames[f"C9_{lab}"] = psi * base["value"] + (1 - psi) * mean
            act = base["actual"]
            loss = pd.DataFrame({m: (v - act).abs() for m, v in frames.items()})
            daily = loss.groupby(level="origin").mean()
            ci1 = bootstrap_ci(daily, reference="C1")
            ci3 = bootstrap_ci(daily, reference="C3")
            for lab, psi in psi_values(k).items():
                m = f"C9_{lab}"
                d1 = mt.diebold_mariano(
                    daily[m].to_numpy(), daily["C1"].to_numpy(), dates=daily.index
                )
                d3 = mt.diebold_mariano(
                    daily[m].to_numpy(), daily["C3"].to_numpy(), dates=daily.index
                )
                rows.append(
                    {
                        "dataset": ds,
                        "scenario": scen,
                        "K": k,
                        "psi_rule": lab,
                        "psi": round(psi, 4),
                        "days": len(daily),
                        "MAE_C9": daily[m].mean(),
                        "MAE_C1": daily["C1"].mean(),
                        "MAE_C3": daily["C3"].mean(),
                        "diff_vs_C1": ci1.loc[m, "diff_vs_ref"],
                        "ci_C1_low": ci1.loc[m, "diff_ci_low"],
                        "ci_C1_high": ci1.loc[m, "diff_ci_high"],
                        "p_dm_vs_C1": d1["p_value"],
                        "diff_vs_C3": ci3.loc[m, "diff_vs_ref"],
                        "ci_C3_low": ci3.loc[m, "diff_ci_low"],
                        "ci_C3_high": ci3.loc[m, "diff_ci_high"],
                        "p_dm_vs_C3": d3["p_value"],
                    }
                )

out = pd.DataFrame(rows)
OUT.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT / "regularyzacja_C9.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)
cols = [
    "dataset",
    "scenario",
    "K",
    "psi_rule",
    "psi",
    "days",
    "MAE_C9",
    "MAE_C1",
    "MAE_C3",
    "diff_vs_C3",
    "ci_C3_low",
    "ci_C3_high",
]
print(out[cols].round(3).to_string())
