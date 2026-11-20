# Uruchomienie:
#   uv run python scripts/artykul/figury_wyniki.py <raport main> <raport gefcom2012> \
#       <data/raw/ElecEner_long.xlsx> <katalog wyjściowy>
"""Figury 2-4 sekcji wyników artykułu, wprost z plików raportów przebiegów (tylko odczyt).

Figura 2: dane ElecEner (średnie dobowe zapotrzebowania i temperatury, profil tygodniowy).
Figura 3: błąd sposobów łączenia z wagami szacowanymi w zależności od długości historii K,
względem zwykłej średniej, oba zbiory, temperatura zmierzona.
Figura 4: błąd w zależności od błędu prognozy temperatury, oba zbiory.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

RM, RG, RAW, OUT = (Path(a) for a in sys.argv[1:5])
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
COL = {
    "C1": "#c0392b",
    "C2": "#e67e22",
    "C5": "#2e86c1",
    "C6": "#7f8c8d",
    "C3": "#000000",
    "C0": "#8e44ad",
    "C1_K2": "#c0392b",
    "C1_K28": "#e67e22",
    "HONG": "#27ae60",
    "NN_ARIMA": "#16a085",
    "LGBM": "#2e86c1",
    "MR_ARIMA": "#7f8c8d",
}
NAME = {
    "C1": "C1 regression",
    "C2": "C2 constrained weights",
    "C5": "C5 inverse-MSE weights",
    "C6": "C6 selection",
    "C3": "C3 simple mean",
    "C0": "C0 first version",
    "C1_K2": "C1, K = 2",
    "C1_K28": "C1, K = 28",
    "HONG": "HONG",
    "NN_ARIMA": "NN+ARIMA",
    "LGBM": "LGBM",
    "MR_ARIMA": "MR+ARIMA",
}

# ---------------------------------------------------------------- figura 2
raw = pd.read_excel(RAW)
ts = pd.to_datetime(raw["Data"])
day = (
    raw.assign(d=ts)
    .groupby("d")
    .agg(load=("Zapotrzebowanie", "mean"), temp=("sr_temp_dz", "mean"), tn=("sr_temp_noc", "mean"))
)
day["temp"] = (day["temp"] + day["tn"]) / 2
day = day[day.index >= "2001-11-01"]
fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 2.8), gridspec_kw={"width_ratios": [1.6, 1]})
a.plot(day.index, day["load"], color="#2e86c1", lw=0.7)
a.set_ylabel("Mean daily demand (shifted units)")
a2 = a.twinx()
a2.plot(day.index, day["temp"], color="#c0392b", lw=0.5, alpha=0.7)
a2.set_ylabel("Daily temperature (°C)", color="#c0392b")
a.axvspan(pd.Timestamp("2002-08-01"), pd.Timestamp("2003-06-29"), color="#9dc3e6", alpha=0.18)
a.axvspan(pd.Timestamp("2001-11-01"), pd.Timestamp("2002-03-31"), color="#bbbbbb", alpha=0.18)
a.axvspan(pd.Timestamp("2002-04-01"), pd.Timestamp("2002-04-30"), color="#f3d9a4", alpha=0.5)
a.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
a.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
a.set_title("(a) Demand and temperature", loc="left", fontsize=8.5)
test = raw.assign(d=ts)[(ts >= "2002-08-01") & (ts <= "2003-06-29")].copy()
test["how"] = test["d"].dt.day_of_week * 24 + test["Godzina"] - 1
prof = test.groupby("how")["Zapotrzebowanie"].mean()
b.plot(prof.index, prof.values, color="#2e86c1", lw=0.9)
b.set_xticks(range(12, 168, 24), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
for x in range(24, 168, 24):
    b.axvline(x, color="#dddddd", lw=0.5)
b.set_title("(b) Average weekly profile, test period", loc="left", fontsize=8.5)
fig.tight_layout()
fig.savefig(OUT / "Figure2.png", dpi=300)

# ---------------------------------------------------------------- figura 3
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=False)
for ax, rep, title in ((axes[0], RM, "(a) ElecEner"), (axes[1], RG, "(b) GEFCom2012")):
    k = pd.read_csv(rep / "k_profile_oracle.csv")
    k = k[k.pool == "main"]
    for m in ("C1", "C2", "C5", "C6"):
        x = k[k.method == m]
        ax.plot(
            x["k"], 100 * x["vs_C3"] / x["MAE_C3"], marker="o", ms=3, color=COL[m], label=NAME[m]
        )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([2, 3, 7, 14, 28], ["2", "3", "7", "14", "28"])
    ax.minorticks_off()
    ax.set_xlabel("Length of history K (days)")
    ax.set_title(title, loc="left", fontsize=8.5)
    ax.grid(color="#eeeeee", lw=0.5)
axes[0].set_ylabel("MAE relative to the simple mean (%)")
axes[1].legend(fontsize=7.5, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "Figure3.png", dpi=300)

# ---------------------------------------------------------------- figura 4
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
for ax, rep, title, unit, models, scale in (
    (axes[0], RM, "(a) ElecEner", "°C", ["C3", "C0", "C1_K2", "C1_K28", "HONG", "MR_ARIMA"], 1),
    (
        axes[1],
        RG,
        "(b) GEFCom2012",
        "°F",
        ["C3", "C0", "C1_K2", "C1_K28", "NN_ARIMA", "LGBM"],
        1000,
    ),
):
    n = pd.read_csv(rep / "noise_sensitivity_day2.csv")
    n["sd"] = n["scenario"].str.replace("noisy_", "", regex=False)
    n.loc[n.scenario == "oracle", "sd"] = "0"
    grid = n[n.scenario != "persistence"].copy()
    grid["sd"] = grid["sd"].astype(float)
    xmax = grid["sd"].max()
    for m in models:
        g = grid[grid.model == m].sort_values("sd")
        p = n[(n.model == m) & (n.scenario == "persistence")]["MAE"].iloc[0]
        ax.plot(g["sd"], g["MAE"] / scale, marker="o", ms=2.5, color=COL[m], label=NAME[m])
        ax.plot([xmax * 1.18], [p / scale], marker="s", ms=4, color=COL[m])
    ax.axvline(xmax * 1.09, color="#bbbbbb", lw=0.6, ls=":")
    ticks = sorted(grid["sd"].unique())
    ax.set_xticks([*ticks, xmax * 1.18], [f"{t:g}" for t in ticks] + ["pers."])
    ax.set_xlabel(f"SD of the temperature error, day D+1 ({unit})")
    ax.set_title(title, loc="left", fontsize=8.5)
    ax.grid(color="#eeeeee", lw=0.5)
axes[0].set_ylabel("MAE for day D+1")
axes[1].set_ylabel("MAE for day D+1 (thousand kW)")
axes[0].set_ylim(16.8, 30.5)
axes[1].set_ylim(80, 300)
axes[0].legend(fontsize=7, frameon=False, ncol=2, loc="upper left")
axes[1].legend(fontsize=7, frameon=False, ncol=2, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "Figure4.png", dpi=300)
print("figury w", OUT)
