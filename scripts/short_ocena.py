# Uruchomienie:
#   uv run python scripts/short_ocena.py results/short <katalog przebiegu main> results/short/ocena
"""Plan analizy 10.3, części C i D: ocena opisowa przebiegów na pliku pierwszej wersji
oraz te same doby wydania w archiwum przebiegu głównego ElecEner Long (tylko odczyt)."""

import glob
import sys
from pathlib import Path

import pandas as pd

from prognozowanie.evaluate import metrics as mt

RES, LONG, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
RUNS = ["short_oryginal_84", "short_long_84", "short_oryginal_56", "short_long_56"]
SINGLE = ["MR_ARIMA", "ARIMA", "ES_ARIMA", "SVR_ARIMA", "NN_ARIMA", "SNAIVE", "HONG", "LGBM"]
COMB = [
    "C0",
    "C0_H72",
    "C0_H168",
    "C1_K2",
    "C1_K3",
    "C1_K7",
    "C1_K14",
    "C1_K28",
    "C2_K2",
    "C2_K7",
    "C3",
    "C4",
    "C5_K2",
    "C5_K7",
    "C6_K2",
    "C6_K7",
    "C7_K2",
    "C8_K2",
]
BLOCK = {"D+1": "D+1 (25-48 h)", "D": "D (1-24 h)"}


def frame(run_dir, scen, t0, t1):
    parts = []
    for m in SINGLE:
        for f in sorted(glob.glob(str(run_dir / "forecasts" / m / "*.parquet"))):
            d = pd.Timestamp(Path(f).stem[:10])
            if t0 <= d <= t1:
                x = pd.read_parquet(f)
                parts.append(x[(x.scenario == scen) & (x.kind == "forecast")])
    c = pd.read_parquet(run_dir / "combined" / f"{scen}__0.parquet")
    c = c[(c.kind == "forecast") & (c.origin >= t0) & (c.origin <= t1) & c.model.isin(COMB)]
    fr = mt.attach_actuals(
        pd.concat(parts + [c], ignore_index=True), pd.read_parquet(run_dir / "actuals.parquet")
    )
    fr["ape"] = (fr["error"].abs() / fr["actual"]) * 100
    fr["ae"] = fr["error"].abs()
    return fr


def summary(fr, label):
    rows = []
    for b, bl in BLOCK.items():
        part = fr[fr.block == bl]
        g = part.groupby("model")
        days = g["origin"].nunique()
        rows.append(
            pd.DataFrame(
                {
                    "przebieg": label,
                    "blok": b,
                    "doby": days,
                    "MAE": g["ae"].mean(),
                    "MAPE": g["ape"].mean(),
                }
            )
        )
    return pd.concat(rows).reset_index()


tabs, dms = [], []
for run in RUNS:
    rd = RES / run
    if not (rd / "combined").exists():
        continue
    win = 84 if run.endswith("84") else 56
    t0 = pd.Timestamp("2003-01-24" if win == 84 else "2002-12-27")
    t1 = pd.Timestamp("2003-01-30")
    for scen in ("oracle", "persistence"):
        fr = frame(rd, scen, t0, t1)
        s = summary(fr, run)
        s["scenariusz"] = scen
        tabs.append(s)
        # wspólne doby dla porównań kluczowych (D+1): C1_K2 dostępne od trzeciej doby
        day2 = fr[fr.block == BLOCK["D+1"]]
        daily = day2.pivot_table(index="origin", columns="model", values="ae", aggfunc="mean")
        for a, b in [
            ("C1_K2", "C3"),
            ("C0", "C3"),
            ("C0", "C1_K2"),
            ("C3", "NN_ARIMA"),
            ("C3", "HONG"),
            ("C1_K2", "NN_ARIMA"),
        ]:
            ok = daily[[a, b]].dropna()
            if len(ok) < 5:
                continue
            d = mt.diebold_mariano(ok[a].to_numpy(), ok[b].to_numpy(), dates=ok.index)
            dms.append(
                {
                    "przebieg": run,
                    "scenariusz": scen,
                    "porownanie": f"{a} - {b}",
                    "doby": len(ok),
                    "roznica_MAE": float((ok[a] - ok[b]).mean()),
                    "p_DM": d["p_value"],
                    "doby_wygranych_a": int((ok[a] < ok[b]).sum()),
                }
            )

tab = pd.concat(tabs)
tab.to_csv(OUT / "C_tabela.csv", index=False)
pd.DataFrame(dms).to_csv(OUT / "C_dm.csv", index=False)

# część D: te same doby w archiwum ElecEner Long (przebieg główny), doba D+1
drows = []
for label, t0 in (("7 dób (24-30.01.2003)", "2003-01-24"), ("35 dób (27.12-30.01)", "2002-12-27")):
    for scen in ("oracle", "persistence"):
        fr = frame(LONG, scen, pd.Timestamp(t0), pd.Timestamp("2003-01-30"))
        s = summary(fr, f"Long {label}")
        s["scenariusz"] = scen
        drows.append(s)
dtab = pd.concat(drows)
dtab.to_csv(OUT / "D_long_te_same_doby.csv", index=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)
keep = ["C0", "C0_H168", "C1_K2", "C1_K3", "C1_K7", "C3", "C4", "C5_K2", "C6_K2", "C8_K2", *SINGLE]
for t, name in ((tab, "C"), (dtab, "D")):
    t = t[t.model.isin(keep)]
    p = t.pivot_table(index="model", columns=["przebieg", "scenariusz", "blok"], values="MAE")
    print(name, "MAE")
    print(p.reindex(keep).round(2).to_string())
print(
    tab[tab.model.isin(keep) & (tab.blok == "D+1")]
    .pivot_table(index="model", columns=["przebieg", "scenariusz"], values="MAPE")
    .reindex(keep)
    .round(2)
    .to_string()
)
print(
    tab[tab.blok == "D+1"]
    .pivot_table(index="model", columns=["przebieg"], values="doby", aggfunc="max")
    .reindex(keep)
    .to_string()
)
print(pd.DataFrame(dms).round(3).to_string())
