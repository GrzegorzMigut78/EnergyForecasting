# Uruchomienie:
#   uv run python scripts/artykul/testy_pelne_s1.py <archiwum> <raport> <seed> \
#       <start testu> <koniec testu> <out.csv>
"""Pełna tabela testów hipotez H1-H3 do suplementu S1: różnica MAE doby D+1, przedział
bootstrapowy (bloki 7 dób, 2000 powtórzeń, ten sam generator co w raporcie), p DM-HLN, p po
Holmie oraz kontrole testu (h = 2, szerokość 14 dób, bez poprawki HLN) i wariant bez 26 dób
oglądanych przed zamrożeniem planu, każdy z poprawką Holma w rodzinie. Tylko odczyt archiwum."""

import sys
from pathlib import Path

import pandas as pd

from prognozowanie.evaluate import inference as inf
from prognozowanie.evaluate.analyses import holm

RUN, REP, SEED = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
t0, t1, OUT = pd.Timestamp(sys.argv[4]), pd.Timestamp(sys.argv[5]), Path(sys.argv[6])
MODELS = [
    "C3",
    "C1_K2",
    "C1_K3",
    "C1_K7",
    "C0",
    "MR_ARIMA",
    "ARIMA",
    "ES_ARIMA",
    "SVR_ARIMA",
    "NN_ARIMA",
    "SNAIVE",
    "HONG",
    "LGBM",
]
y = pd.read_parquet(RUN / "actuals.parquet")["load"]
rows = []
for sc in ("oracle", "persistence"):
    c = pd.read_parquet(RUN / "combined" / f"{sc}__0.parquet")
    parts = [c[c.model.isin(MODELS)]]
    for m in MODELS:
        if (RUN / "forecasts" / m).exists():
            d = pd.concat(
                pd.read_parquet(p) for p in sorted((RUN / "forecasts" / m).glob("*.parquet"))
            )
            parts.append(d[(d.scenario == sc) & (d.kind == "forecast") & (d.draw == 0)])
    f = pd.concat(parts)
    f = f[(f.horizon > 24) & (f.origin >= t0) & (f.origin <= t1)].copy()
    f["ae"] = (f.value - y.reindex(pd.to_datetime(f.target)).to_numpy()).abs()
    L = f.pivot_table(index="origin", columns="model", values="ae", aggfunc="mean")[MODELS]
    ci3 = inf.bootstrap_ci(L, reference="C3", seed=SEED)
    ci1 = inf.bootstrap_ci(L, reference="C1_K2", seed=SEED)
    pt = pd.read_csv(REP / f"primary_tests_{sc}.csv")
    ws = (
        pd.read_csv(REP / f"primary_tests_without_seen_days_{sc}.csv")
        if (REP / f"primary_tests_without_seen_days_{sc}.csv").exists()
        else None
    )
    for col in ("p_h2", "p_bw14", "p_no_hln"):
        pt[col + "_holm"] = pt.groupby("family")[col].transform(holm)
    if ws is not None:
        pt["p_without_seen_holm"] = ws["p_holm"].to_numpy()
    for _, r in pt.iterrows():
        ci = ci3 if r.reference == "C3" else ci1
        rows.append(
            {
                "scenario": sc,
                "family": r.family,
                "model": r.model,
                "reference": r.reference,
                "MAE_check": L[r.model].mean(),
                "diff": r.mean_diff,
                "ci_low": ci.loc[r.model, "diff_ci_low"],
                "ci_high": ci.loc[r.model, "diff_ci_high"],
                "p": r.p_value,
                "p_holm": r.p_holm,
                "p_h2_holm": r.p_h2_holm,
                "p_bw14_holm": r.p_bw14_holm,
                "p_no_hln_holm": r.p_no_hln_holm,
                "p_without_seen_holm": r.get("p_without_seen_holm"),
            }
        )
out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
print(out.round(4).to_string())
