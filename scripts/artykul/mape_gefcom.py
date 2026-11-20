# Uruchomienie:
#   uv run python scripts/artykul/mape_gefcom.py <katalog przebiegu gefcom2012>
"""MAPE doby D+1 na GEFCom2012 (sekcja 3.1 artykułu): modele pojedyncze i wybrane sposoby
łączenia, oba scenariusze główne, doby testu 1.06.2007-28.06.2008. Tylko odczyt archiwum;
MAE liczone przy okazji jako kontrola zgodności z tabelą 6."""

import sys
from pathlib import Path

import pandas as pd

RUN = Path(sys.argv[1])
MODELS = ["MR_ARIMA", "ARIMA", "ES_ARIMA", "SVR_ARIMA", "NN_ARIMA", "SNAIVE", "HONG", "LGBM"]
COMB = ["C3", "C1_K2", "C1_K28", "C2_K28"]
y = pd.read_parquet(RUN / "actuals.parquet")["load"]
for sc in ("oracle", "persistence"):
    c = pd.read_parquet(RUN / "combined" / f"{sc}__0.parquet")
    parts = [c[c.model.isin(COMB)]]
    for m in MODELS:
        d = pd.concat(pd.read_parquet(p) for p in sorted((RUN / "forecasts" / m).glob("*.parquet")))
        parts.append(d[(d.scenario == sc) & (d.kind == "forecast") & (d.draw == 0)])
    f = pd.concat(parts)
    f = f[(f.horizon > 24) & (f.origin >= "2007-06-01") & (f.origin <= "2008-06-28")].copy()
    f["y"] = y.reindex(pd.to_datetime(f.target)).to_numpy()
    f["ae"] = (f.value - f.y).abs()
    f["ape"] = 100 * f.ae / f.y
    g = f.groupby("model").agg(MAE_tys=("ae", "mean"), MAPE=("ape", "mean"), n=("ae", "size"))
    g["MAE_tys"] /= 1000
    print(sc)
    print(g.round(2).to_string())
