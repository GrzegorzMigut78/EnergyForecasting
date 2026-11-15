# Uruchomienie:
#   uv run python scripts/po_wynikach_pula_lgbm.py <katalog results> <katalog wyjściowy>
"""Plan analizy, sekcja 10.2: pula z LightGBM w roli puli głównej (analiza po wynikach).

Rodziny H1-H3 z wariantami puli plus_lgbm, te same doby, DM, Holm z planowaną liczebnością
i reguła dwóch scenariuszy, wynik wyłącznie opisowy. Dodatkowo ściąganie wag do równych
(jak 10.1, n = 6) i różnica plus_lgbm/C3 minus C3. Archiwum tylko do odczytu.
"""

import glob
import sys
from pathlib import Path

import pandas as pd

from prognozowanie.config import HypothesisFamily, load_experiment_config
from prognozowanie.evaluate import analyses as an
from prognozowanie.evaluate import metrics as mt
from prognozowanie.evaluate.inference import bootstrap_ci

RES, OUT = Path(sys.argv[1]), Path(sys.argv[2])
ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "ElecEner": ("main", "configs/experiments/main.yaml"),
    "GEFCom2012": ("gefcom2012", "configs/experiments/gefcom2012.yaml"),
}
SINGLE = ["MR_ARIMA", "ARIMA", "ES_ARIMA", "SVR_ARIMA", "NN_ARIMA", "SNAIVE", "HONG", "LGBM"]
P = "plus_lgbm/"
FAMILIES = {
    "H1": HypothesisFamily(reference=P + "C3", compare=[P + f"C1_K{k}" for k in (2, 3, 7)]),
    "H2": HypothesisFamily(reference=P + "C1_K2", compare=[P + "C0"]),
    "H3": HypothesisFamily(reference=P + "C1_K2", compare=SINGLE),
}
KS = [2, 3, 7, 14, 28]
N_MODELS = 6
DAY2 = "D+1 (25-48 h)"


def psi_values(k: int) -> dict[str, float]:
    out = {}
    for label, t in (("SW_godziny", 24 * k), ("SW_doby", k)):
        for kap in (0.25, 0.5, 1.0):
            den = t - 1 - N_MODELS
            out[f"{label}_kappa{kap}"] = 0.0 if den <= 0 else max(0.0, 1 - kap * N_MODELS / den)
    for p in (0.25, 0.5, 0.75):
        out[f"stale_{p}"] = p
    return out


def frame(run, scen, t0, t1):
    parts = []
    for m in SINGLE:
        for f in sorted(glob.glob(str(RES / run / "forecasts" / m / "*.parquet"))):
            d = pd.Timestamp(Path(f).stem[:10])
            if t0 <= d <= t1:
                x = pd.read_parquet(f)
                parts.append(x[(x.scenario == scen) & (x.kind == "forecast")])
    c = pd.read_parquet(RES / run / "combined" / f"{scen}__0.parquet")
    c = c[(c.kind == "forecast") & (c.origin >= t0) & (c.origin <= t1)]
    c = c[c.model.str.startswith(P) | (c.model == "C3")]
    act = pd.read_parquet(RES / run / "actuals.parquet")
    fr = mt.attach_actuals(pd.concat(parts + [c], ignore_index=True), act)
    return fr[fr.block == DAY2]


tests, c9rows, poolrows = [], [], []
for ds, (run, cfg_path) in RUNS.items():
    cfg = load_experiment_config(ROOT / cfg_path, use_identified=False)
    t0, t1 = pd.Timestamp(cfg.periods.test.start), pd.Timestamp(cfg.periods.test.end)
    excl = [(pd.Timestamp(p.start), pd.Timestamp(p.end)) for p in cfg.evaluation.excluded_data]
    win = cfg.forecast.training_window_days
    tables = {}
    for scen in ("oracle", "persistence"):
        fr = frame(run, scen, t0, t1)
        fr = fr[an.clean_of_excluded(fr, excl, win)]
        daily = mt.daily_loss(fr)
        fam_models = sorted({m for s in FAMILIES.values() for m in [s.reference, *s.compare]})
        main = daily[fam_models].dropna()
        t = an.primary_tests(main, FAMILIES)
        t["dataset"], t["scenario"], t["days"] = ds, scen, len(main)
        tables[scen] = t
        tests.append(t)
        pair = daily[[P + "C3", "C3"]].dropna()
        ci = bootstrap_ci(pair, reference="C3").loc[P + "C3"]
        dm = mt.diebold_mariano(pair[P + "C3"].to_numpy(), pair["C3"].to_numpy(), dates=pair.index)
        poolrows.append(
            {
                "dataset": ds,
                "scenario": scen,
                "days": len(pair),
                "MAE_plus_lgbm_C3": pair[P + "C3"].mean(),
                "MAE_C3": pair["C3"].mean(),
                "diff": ci["diff_vs_ref"],
                "ci_low": ci["diff_ci_low"],
                "ci_high": ci["diff_ci_high"],
                "p_dm": dm["p_value"],
            }
        )
        key = ["origin", "target", "horizon"]
        c3 = fr[fr.model == P + "C3"].set_index(key)["value"]
        for k in KS:
            c1 = fr[fr.model == P + f"C1_K{k}"].set_index(key)
            mean = c3.reindex(c1.index)
            ok = mean.notna()
            c1, mean = c1[ok], mean[ok]
            loss = {"C1": (c1["value"] - c1["actual"]).abs(), "C3": (mean - c1["actual"]).abs()}
            for lab, psi in psi_values(k).items():
                f9 = psi * c1["value"] + (1 - psi) * mean
                loss[lab] = (f9 - c1["actual"]).abs()
            dl = pd.DataFrame(loss).groupby(level="origin").mean()
            ci1 = bootstrap_ci(dl, reference="C1")
            ci3 = bootstrap_ci(dl, reference="C3")
            for lab, psi in psi_values(k).items():
                x = dl[lab].to_numpy()
                d1 = mt.diebold_mariano(x, dl["C1"].to_numpy(), dates=dl.index)
                d3 = mt.diebold_mariano(x, dl["C3"].to_numpy(), dates=dl.index)
                c9rows.append(
                    {
                        "dataset": ds,
                        "scenario": scen,
                        "K": k,
                        "psi_rule": lab,
                        "psi": round(psi, 4),
                        "days": len(dl),
                        "MAE_C9": dl[lab].mean(),
                        "MAE_C1": dl["C1"].mean(),
                        "MAE_C3": dl["C3"].mean(),
                        "diff_vs_C1": ci1.loc[lab, "diff_vs_ref"],
                        "ci_C1_low": ci1.loc[lab, "diff_ci_low"],
                        "ci_C1_high": ci1.loc[lab, "diff_ci_high"],
                        "p_dm_vs_C1": d1["p_value"],
                        "diff_vs_C3": ci3.loc[lab, "diff_vs_ref"],
                        "ci_C3_low": ci3.loc[lab, "diff_ci_low"],
                        "ci_C3_high": ci3.loc[lab, "diff_ci_high"],
                        "p_dm_vs_C3": d3["p_value"],
                    }
                )
    conf = an.confirm_across_scenarios(tables, FAMILIES, alpha=cfg.evaluation.alpha)
    conf["dataset"] = ds
    conf = conf.rename(columns={"confirmed": "spelnia_regule_opisowo"})
    OUT.mkdir(parents=True, exist_ok=True)
    conf.to_csv(OUT / f"regula_dwoch_scenariuszy_{ds}.csv", index=False)
    pd.set_option("display.width", 250)
    print(ds)
    print(conf.round(4).to_string())

OUT.mkdir(parents=True, exist_ok=True)
pd.concat(tests).to_csv(OUT / "testy_rodzin.csv", index=False)
pd.DataFrame(c9rows).to_csv(OUT / "regularyzacja_C9_pula_lgbm.csv", index=False)
pd.DataFrame(poolrows).to_csv(OUT / "pula_lgbm_C3_kontra_C3.csv", index=False)
print(pd.DataFrame(poolrows).round(3).to_string())
c9 = pd.DataFrame(c9rows)
print(c9[c9.psi_rule.isin(["stale_0.25", "stale_0.5", "stale_0.75"])].round(3).to_string())
