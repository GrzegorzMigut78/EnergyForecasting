# Uruchomienie:
#   uv run python scripts/short_okno_artykulu.py \
#       data/raw/ElecEner_Short_d.xlsx results/short/okno_artykulu
"""Plan analizy 10.3, części A i B: okno z artykułu i stabilność wag, z kolumn pliku pierwszej
wersji (bez szacowania modeli).

Układ pliku: modele oszacowane na wierszach 0-2015 (84 doby), regresja łącząca na wierszach
1968-2015 (ostatnie 48 h, wartości dopasowane), prognoza na wierszach 2016-2063 (doby 85-86).
Kolumny ostatecznych modeli: MR+ARIMA, ARIMA, ES+ARIMA, SVR+ARIMA, NN+ARIMA. Kolumna ARIMA ma
wartości tylko w wierszach 1968-2063, więc część B (przesuwane okna) używa czterech modeli
z pełnym zakresem wartości dopasowanych.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from prognozowanie.combine import methods as cm

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
x = pd.read_excel(SRC)
y = x["Demand"].to_numpy(float)
COLS = {
    "MR_ARIMA": "Model M01(1)t+",
    "ARIMA": "ARIMA",
    "ES_ARIMA": "WW(1)+",
    "SVR_ARIMA": "SVR(1)+",
    "NN_ARIMA": "NN(1)+",
}
F = pd.DataFrame({k: x[v].to_numpy(float) for k, v in COLS.items()})
TR, TE = slice(1968, 2016), slice(2016, 2064)  # okno łączące, prognoza 48 h
BLOCKS = {
    "48 h": slice(0, 48),
    "doba D (1-24 h)": slice(0, 24),
    "doba D+1 (25-48 h)": slice(24, 48),
}


def err(pred, actual):
    e = pred - actual
    return {
        "MAPE": float(np.mean(np.abs(e) / actual) * 100),
        "MAE": float(np.mean(np.abs(e))),
        "bias": float(np.mean(e)),
    }


# ---------------------------------------------------------------- część A
Ftr, Fte = F.iloc[TR].to_numpy(), F.iloc[TE].to_numpy()
ytr, yte = y[TR], y[TE]
combos = {
    "C0 regresja łącząca (artykuł)": cm.ols(Ftr, ytr, Fte)[0],
    "C2 średnia ważona (wagi >= 0, suma 1)": cm.simplex(Ftr, ytr, Fte)[0],
    "C3 średnia": cm.mean(Fte)[0],
    "C4 mediana": cm.median(Fte)[0],
    "C5 wagi odwrotne do MSE": cm.inverse_mse(Ftr, ytr, Fte)[0],
    "C6 wybór najlepszego": cm.select_best(Ftr, ytr, Fte)[0],
    "C7 średnia + poziom": cm.calibrate_level(Ftr, ytr, Fte)[0],
    "C8 a + b * średnia": cm.calibrate_affine(Ftr, ytr, Fte)[0],
}
for k in COLS:
    combos[k] = F[k].to_numpy()[TE]
rows = []
for name, pred in combos.items():
    for b, sl in BLOCKS.items():
        rows.append({"wariant": name, "blok": b, **err(np.asarray(pred)[sl], yte[sl])})
tab_a = pd.DataFrame(rows)
tab_a.to_csv(OUT / "A_okno_artykulu_bloki.csv", index=False)
check = err(x["Regresja końcowa"].to_numpy(float)[TE], yte)["MAPE"]

# w próbie: dopasowanie regresji i modeli na oknie łączącym
ins = [{"wariant": "C0 regresja łącząca (artykuł)", **err(cm.ols(Ftr, ytr, Ftr)[0], ytr)}]
ins += [{"wariant": k, **err(Ftr[:, i], ytr)} for i, k in enumerate(COLS)]
ins += [{"wariant": "C3 średnia", **err(Ftr.mean(1), ytr)}]
pd.DataFrame(ins).to_csv(OUT / "A_w_probie.csv", index=False)

# współczynniki z błędami standardowymi (MNK, jak w artykule)
ols = sm.OLS(ytr, sm.add_constant(pd.DataFrame(Ftr, columns=list(COLS)))).fit()
coef = pd.DataFrame({"wsp": ols.params, "bl_std": ols.bse, "t": ols.tvalues, "p": ols.pvalues})
coef.to_csv(OUT / "A_wspolczynniki.csv")
vif = {
    k: 1 / (1 - sm.OLS(Ftr[:, i], sm.add_constant(np.delete(Ftr, i, 1))).fit().rsquared)
    for i, k in enumerate(COLS)
}
pd.Series(vif, name="VIF").to_csv(OUT / "A_vif.csv")
corr_fit = pd.DataFrame(Ftr, columns=list(COLS)).corr()
corr_err = pd.DataFrame(Fte - yte[:, None], columns=list(COLS)).corr()
corr_fit.to_csv(OUT / "A_korelacje_dopasowanych.csv")
corr_err.to_csv(OUT / "A_korelacje_bledow_prognozy.csv")

# godziny: błąd bezwzględny regresji, średniej i najlepszego modelu
hour = pd.DataFrame(
    {
        "godzina_horyzontu": np.arange(1, 49),
        "rzeczywiste": yte,
        "C0": combos["C0 regresja łącząca (artykuł)"],
        "C3": combos["C3 średnia"],
        **{k: F[k].to_numpy()[TE] for k in COLS},
    }
)
hour.to_csv(OUT / "A_godziny.csv", index=False)

# ---------------------------------------------------------------- część B
M4 = ["MR_ARIMA", "ES_ARIMA", "SVR_ARIMA", "NN_ARIMA"]
F4 = F[M4].to_numpy()
first = int(np.max([np.argmax(~np.isnan(F4[:, i])) for i in range(4)]))
rowsb = []
# okna 48 h przesuwane co 24 h, kończące się najpóźniej w wierszu 2015
for end in range(2016, first + 48 - 1, -24):
    sl = slice(end - 48, end)
    for pool, cols in (("4 modele", M4), ("5 modeli", list(COLS))):
        Ft = F[cols].to_numpy()
        if np.isnan(Ft[sl]).any():
            continue
        pred, w, a = cm.ols(Ft[sl], y[sl], Ft[TE])
        ins_mape = err(cm.ols(Ft[sl], y[sl], Ft[sl])[0], y[sl])["MAPE"]
        r = {
            "pula": pool,
            "okno_konczy_sie_wiersz": end,
            "doby_przed_prognoza": (2016 - end) // 24,
            "MAPE_w_probie": ins_mape,
            "wyraz_wolny": a,
            **{f"w_{c}": v for c, v in zip(cols, w, strict=True)},
        }
        for b, bsl in BLOCKS.items():
            r[f"MAPE {b}"] = err(pred[bsl], yte[bsl])["MAPE"]
            r[f"MAE {b}"] = err(pred[bsl], yte[bsl])["MAE"]
        rowsb.append(r)
tab_b = pd.DataFrame(rowsb)
tab_b.to_csv(OUT / "B_okna_48h.csv", index=False)

# długość okna łączącego (4 modele, okna kończące się w wierszu 2015)
rowsl = []
for hours in [48, 72, 96, 168, 336, 504, 672, 1008, 1344, 2016 - first]:
    sl = slice(2016 - hours, 2016)
    pred, w, a = cm.ols(F4[sl], y[sl], F4[TE])
    r = {
        "godziny_okna": hours,
        "MAPE_w_probie": err(cm.ols(F4[sl], y[sl], F4[sl])[0], y[sl])["MAPE"],
        "wyraz_wolny": a,
        **{f"w_{c}": v for c, v in zip(M4, w, strict=True)},
    }
    for b, bsl in BLOCKS.items():
        r[f"MAPE {b}"] = err(pred[bsl], yte[bsl])["MAPE"]
    rowsl.append(r)
ref4 = {f"MAPE {b}": err(F4[TE].mean(1)[bsl], yte[bsl])["MAPE"] for b, bsl in BLOCKS.items()}
pd.DataFrame(rowsl).to_csv(OUT / "B_dlugosc_okna.csv", index=False)

pd.set_option("display.width", 250)
print("kontrola: MAPE kolumny 'Regresja końcowa' =", round(check, 3))
print(tab_a.pivot(index="wariant", columns="blok", values="MAPE").round(3).to_string())
print(pd.DataFrame(ins).round(3).to_string())
print(coef.round(4).to_string())
print("VIF", {k: round(v, 1) for k, v in vif.items()})
print("korelacje błędów prognozy\n", corr_err.round(2).to_string())
print("B: okna 48 h, liczba:", tab_b.groupby("pula").size().to_dict())
print(
    tab_b.groupby("pula")[[c for c in tab_b if c.startswith("MAPE")]]
    .describe()
    .T.round(3)
    .to_string()
)
print("B: średnia 4 modeli", {k: round(v, 3) for k, v in ref4.items()})
print(pd.DataFrame(rowsl).round(3).to_string())
