# Uruchomienie: uv run python scripts/porownanie_kwartalne.py <katalog results>
"""Plan, punkt 18: parametry stałe (main) kontra odświeżane co kwartał (main_kwartalny),
C1_K2 i modele pojedyncze, wydania 1.11.2002-29.06.2003, doba D+1, oracle i persystencja.
Różnica MAE (odświeżane minus stałe; ujemna: odświeżanie pomaga), bootstrap blokowy 7 dób,
DM (HAC + HLN, h = 1) bez korekty Holma, opisowo. Dodatkowo podział na kwartały."""

import sys, glob
import pandas as pd
from prognozowanie.evaluate import metrics as mt
from prognozowanie.evaluate.inference import bootstrap_ci

K = sys.argv[1]
MODELS = ["MR_ARIMA", "ARIMA", "ES_ARIMA", "SVR_ARIMA", "NN_ARIMA", "SNAIVE", "HONG", "LGBM"]
START, END = pd.Timestamp("2002-11-01"), pd.Timestamp("2003-06-29")


def load(run, scen):
    parts = []
    for m in MODELS:
        for f in sorted(glob.glob(f"{K}/{run}/forecasts/{m}/*.parquet")):
            d = pd.Timestamp(f.split("/")[-1][:10])
            if START <= d <= END:
                x = pd.read_parquet(f)
                parts.append(
                    x[
                        (x.scenario == scen) & (x.kind == "forecast")
                        if "kind" in x
                        else (x.scenario == scen)
                    ]
                )
    c = pd.read_parquet(f"{K}/{run}/combined/{scen}__0.parquet")
    c = c[
        (c.model.isin(["C1_K2", "C3"]))
        & (c.kind == "forecast")
        & (c.origin >= START)
        & (c.origin <= END)
    ]
    fc = pd.concat(parts + [c], ignore_index=True)
    act = pd.read_parquet(f"{K}/{run}/actuals.parquet")
    fr = mt.attach_actuals(fc, act)
    fr = fr[fr.block == "D+1 (25-48 h)"]
    return mt.daily_loss(fr)


rows = []
for scen in ["oracle", "persistence"]:
    a, b = load("main", scen), load("main_kwartalny", scen)
    for m in ["C1_K2", "C3"] + MODELS:
        pair = pd.concat({"stale": a[m], "kwartalne": b[m]}, axis=1).dropna()
        r = mt.diebold_mariano(
            pair["kwartalne"].to_numpy(), pair["stale"].to_numpy(), dates=pair.index
        )
        bc = bootstrap_ci(pair, reference="stale")
        ci = (bc.set_index("model") if "model" in bc else bc).loc["kwartalne"]
        q = {}
        for lab, lo, hi in [
            ("Q XI-I", "2002-11-01", "2003-01-31"),
            ("Q II-IV", "2003-02-01", "2003-04-30"),
            ("Q V-VI", "2003-05-01", "2003-06-29"),
        ]:
            s = pair.loc[lo:hi]
            q[lab] = s["kwartalne"].mean() - s["stale"].mean()
        rows.append(
            {
                "scenario": scen,
                "model": m,
                "days": len(pair),
                "MAE_stale": pair["stale"].mean(),
                "MAE_kwartalne": pair["kwartalne"].mean(),
                "diff": ci["diff_vs_ref"],
                "ci_low": ci["diff_ci_low"],
                "ci_high": ci["diff_ci_high"],
                "p_dm": r["p_value"],
                **q,
            }
        )
out = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(out.round(3).to_string())
out.to_csv(f"{K}/kwartal_porownanie.csv", index=False)
