# Uruchomienie:
#   uv run python scripts/artykul/tabele_wyniki.py <raport main> <raport gefcom2012> \
#       <kwartal_porownanie.csv> <plik wyjściowy .json>
"""Tabele 6-9 sekcji wyników artykułu, sformatowane wprost z plików raportów (tylko odczyt).
ElecEner: MAE w jednostkach danych, jedno miejsce po przecinku; GEFCom2012: tysiące kW."""

import json
import sys
from pathlib import Path

import pandas as pd

RM, RG, KW, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
D2 = "D+1 (25-48 h)"


def f1(x):
    t = f"{x:.1f}"
    return ("0.0" if t == "-0.0" else t).replace("-", "−")


def fk(x):
    return f1(x / 1000)


def fp(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def summ(rep, scen):
    s = pd.read_csv(rep / f"summary_main_{scen}.csv")
    return s[s.block == D2].set_index("model")


def scen_table(rep):
    s = pd.read_csv(rep / "by_scenario_day2.csv")
    return s.pivot_table(index="model", columns="scenario", values="MAE")


tables = {}

# Tabela 6: modele pojedyncze
names = {
    "MR_ARIMA": "MR+ARIMA",
    "ARIMA": "ARIMA",
    "ES_ARIMA": "ES+ARIMA",
    "SVR_ARIMA": "SVR+ARIMA",
    "NN_ARIMA": "NN+ARIMA",
    "SNAIVE": "SNAIVE",
    "HONG": "HONG",
    "LGBM": "LGBM",
    "C3": "Simple mean (C3)",
}
em, ep, gm, gp = (
    summ(RM, "oracle"),
    summ(RM, "persistence"),
    summ(RG, "oracle"),
    summ(RG, "persistence"),
)
rows = [
    [
        "Model",
        "ElecEner, meas.",
        "ElecEner, pers.",
        "ElecEner, MASE",
        "GEFCom2012, meas.",
        "GEFCom2012, pers.",
        "GEFCom2012, MASE",
    ]
]
for m, lab in names.items():
    rows.append(
        [
            lab,
            f1(em.loc[m, "MAE"]),
            f1(ep.loc[m, "MAE"]),
            f"{em.loc[m, 'MASE']:.2f}",
            fk(gm.loc[m, "MAE"]),
            fk(gp.loc[m, "MAE"]),
            f"{gm.loc[m, 'MASE']:.2f}",
        ]
    )
tables["T6"] = rows

# Tabela 7: hipotezy
lab7 = {
    ("H1", "C1_K2"): "H1: C1, K = 2 vs mean",
    ("H1", "C1_K3"): "H1: C1, K = 3 vs mean",
    ("H1", "C1_K7"): "H1: C1, K = 7 vs mean",
    ("H2", "C0"): "H2: C0 vs C1, K = 2",
}
rows = [["Contrast", "ElecEner Δ", "p", "Conf.", "GEFCom2012 Δ", "p", "Conf."]]
ce, cg = pd.read_csv(RM / "primary_confirmed.csv"), pd.read_csv(RG / "primary_confirmed.csv")
for (fam, mod), rowe in ce.set_index(["family", "model"]).iterrows():
    rowg = cg.set_index(["family", "model"]).loc[(fam, mod)]
    lab = lab7.get((fam, mod), f"H3: {names.get(mod, mod)} vs C1, K = 2")

    def conf(r):
        if r["confirmed"]:
            return "yes"
        return "no" if r["same_direction"] else "no (dir.)"

    rows.append(
        [
            lab,
            f"{f1(rowe['mean_diff_oracle'])} / {f1(rowe['mean_diff_persistence'])}",
            fp(rowe["p_max_holm"]),
            conf(rowe),
            f"{fk(rowg['mean_diff_oracle'])} / {fk(rowg['mean_diff_persistence'])}",
            fp(rowg["p_max_holm"]),
            conf(rowg),
        ]
    )
tables["T7"] = rows

# Tabela 8: sposoby łączenia
se, sg = scen_table(RM), scen_table(RG)
lab8 = [
    ("C0", "C0 regression on fitted values, 48 h (first version)"),
    ("C0_H168", "C0 regression on fitted values, 168 h"),
    ("C1_K2", "C1 regression on forecasts, K = 2"),
    ("C1_K28", "C1 regression on forecasts, K = 28"),
    ("C2_K2", "C2 constrained weights, K = 2"),
    ("C2_K28", "C2 constrained weights, K = 28"),
    ("C3", "C3 simple mean"),
    ("C4", "C4 median"),
    ("C5_K2", "C5 inverse-MSE weights, K = 2"),
    ("C5_K28", "C5 inverse-MSE weights, K = 28"),
    ("C6_K2", "C6 selection of one model, K = 2"),
    ("C6_K28", "C6 selection of one model, K = 28"),
    ("C7_K2", "C7 mean plus level correction, K = 2"),
    ("C8_K2", "C8 a + b · mean, K = 2"),
    ("plus_hong/C3", "Simple mean, pool + HONG+ARIMA"),
    ("plus_lgbm/C3", "Simple mean, pool + LGBM"),
]
rows = [["Method", "ElecEner, meas.", "ElecEner, pers.", "GEFCom2012, meas.", "GEFCom2012, pers."]]
for m, lab in lab8:
    rows.append(
        [
            lab,
            f1(se.loc[m, "oracle"]),
            f1(se.loc[m, "persistence"]),
            fk(sg.loc[m, "oracle"]),
            fk(sg.loc[m, "persistence"]),
        ]
    )
tables["T8"] = rows

# Tabela 9: reidentyfikacja kwartalna
q = pd.read_csv(KW)
rows = [["Model", "Fixed settings", "Quarterly re-identification", "Difference (95% CI)", "p"]]
order = ["MR_ARIMA", "ARIMA", "ES_ARIMA", "SVR_ARIMA", "NN_ARIMA", "LGBM", "C3", "C1_K2"]
qo = q[q.scenario == "oracle"].set_index("model")
for m in order:
    r = qo.loc[m]
    lab = {"C3": "Simple mean (C3)", "C1_K2": "C1, K = 2"}.get(m, names.get(m, m))
    rows.append(
        [
            lab,
            f1(r["MAE_stale"]),
            f1(r["MAE_kwartalne"]),
            f"{f1(r['diff'])} ({f1(r['ci_low'])} to {f1(r['ci_high'])})",
            fp(r["p_dm"]),
        ]
    )
tables["T9"] = rows

OUT.write_text(json.dumps(tables, ensure_ascii=False, indent=1), encoding="utf-8")
for k, v in tables.items():
    print(k)
    for r in v:
        print("  ", " | ".join(r))
