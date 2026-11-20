# Uruchomienie:
#   uv run python scripts/artykul/bezwladnosc_temperatury.py <data/raw/ElecEner_long.xlsx>
"""Obserwacja eksploracyjna do Dyskusji (sekcja 4.4): jak silnie średnie dobowe zapotrzebowanie
ElecEner wiąże się z temperaturą dnia bieżącego i dni poprzednich. Regresja liniowa średniego
dobowego zapotrzebowania na temperaturę (średnia z dziennej i nocnej, jak w modelach) z
zerojedynkowymi dniami tygodnia i świętem; okres testu 1.08.2002-29.06.2003, R^2."""

import sys

import numpy as np
import pandas as pd

raw = pd.read_excel(sys.argv[1])
raw["d"] = pd.to_datetime(raw["Data"])
day = raw.groupby("d").agg(
    load=("Zapotrzebowanie", "mean"),
    td=("sr_temp_dz", "mean"),
    tn=("sr_temp_noc", "mean"),
    hol=("święto", "max"),
)
t = (day["td"] + day["tn"]) / 2
dow = pd.get_dummies(day.index.day_of_week, prefix="w", drop_first=True).astype(float)
dow.index = day.index


def r2(cols):
    df = pd.concat([day["load"], *cols, dow, day["hol"]], axis=1).dropna()
    df = df.loc["2002-08-01":"2003-06-29"]
    y = df.iloc[:, 0].to_numpy()
    a = np.column_stack([np.ones(len(df)), df.iloc[:, 1:].to_numpy()])
    b, *_ = np.linalg.lstsq(a, y, rcond=None)
    return 1 - ((y - a @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum(), len(df)


for lab, cols in (
    ("dzień D", [t]),
    ("dzień D-1", [t.shift(1)]),
    ("dzień D-2", [t.shift(2)]),
    ("D, D-1, D-2 razem", [t, t.shift(1), t.shift(2)]),
):
    v, n = r2(cols)
    print(f"{lab:20s} R2 = {v:.3f}  (n = {n})")
