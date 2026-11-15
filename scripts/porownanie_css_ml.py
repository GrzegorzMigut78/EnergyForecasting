# Uruchomienie: uv run python scripts/porownanie_css_ml.py <katalog z main i main_ml_kontrola>
"""Kontrola estymacji (plan, sekcja 4): przebieg główny (SARIMA metodą CSS) kontra przebieg
kontrolny (dokładna wiarygodność), te same doby wydania 1-31.10.2002, doba D+1, oracle
i persystencja. MAE wariantów analizy głównej, ranking (Spearman) i kierunki kontrastów H1-H3."""

import glob
import sys

import pandas as pd

from prognozowanie.evaluate import metrics as mt

K = sys.argv[1]
SINGLE = ["MR_ARIMA", "ARIMA", "ES_ARIMA", "SVR_ARIMA", "NN_ARIMA", "SNAIVE", "HONG", "LGBM"]
COMB = ["C0", "C1_K2", "C1_K3", "C1_K7", "C3"]
START, END = pd.Timestamp("2002-10-01"), pd.Timestamp("2002-10-31")


def losses(run, scen):
    parts = []
    for m in SINGLE:
        for f in sorted(glob.glob(f"{K}/{run}/forecasts/{m}/*.parquet")):
            d = pd.Timestamp(f.split("/")[-1][:10])
            if START <= d <= END:
                x = pd.read_parquet(f)
                parts.append(x[(x.scenario == scen) & (x.kind == "forecast")])
    c = pd.read_parquet(f"{K}/{run}/combined/{scen}__0.parquet")
    c = c[c.model.isin(COMB) & (c.kind == "forecast") & (c.origin >= START) & (c.origin <= END)]
    fr = mt.attach_actuals(
        pd.concat(parts + [c], ignore_index=True), pd.read_parquet(f"{K}/{run}/actuals.parquet")
    )
    return mt.daily_loss(fr[fr.block == "D+1 (25-48 h)"])


CONTRASTS = [("H1", m, "C3") for m in ["C1_K2", "C1_K3", "C1_K7"]] + [("H2", "C0", "C1_K2")]
CONTRASTS += [("H3", m, "C1_K2") for m in SINGLE]
rows, ranks = [], []
for scen in ["oracle", "persistence"]:
    a, b = losses("main", scen), losses("main_ml_kontrola", scen)
    mae = pd.DataFrame({"CSS": a.mean(), "ML": b.mean()})
    mae["scenario"] = scen
    ranks.append(mae)
    rho = mae["CSS"].rank().corr(mae["ML"].rank())
    print(f"{scen}: doby {len(a)}/{len(b)}, Spearman rang {rho:.3f}")
    for fam, m, ref in CONTRASTS:
        d_css, d_ml = a[m].mean() - a[ref].mean(), b[m].mean() - b[ref].mean()
        rows.append(
            {
                "scenario": scen,
                "family": fam,
                "model": m,
                "ref": ref,
                "diff_CSS": d_css,
                "diff_ML": d_ml,
                "same_sign": (d_css > 0) == (d_ml > 0),
            }
        )
pd.set_option("display.width", 200)
print(pd.concat(ranks).round(2).to_string())
out = pd.DataFrame(rows)
print(out.round(2).to_string())
