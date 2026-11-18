# Uruchomienie:
#   uv run python scripts/przygotuj_short.py \
#       data/raw/ElecEner_Short_d.xlsx data/raw/ElecEner_Short.csv
"""Plan analizy 10.3: plik pierwszej wersji (ElecEner Short) w układzie loadera csv_hourly.

Plik nie ma roku: 1.11 to piątek, więc listopad i grudzień 2002, styczeń 2003. Daty odtworzone
z miesiąca, dnia miesiąca i godziny; skrypt sprawdza zgodność z kolumną dnia tygodnia i ciągłość
godzin. Surowy plik nie jest zmieniany, wynik nie trafia do repozytorium.
"""

import sys

import pandas as pd

src, dst = sys.argv[1], sys.argv[2]
raw = pd.read_excel(src)
month = raw["Month"].map({"November": 11, "December": 12, "January": 1})
year = month.map({11: 2002, 12: 2002, 1: 2003})
date = pd.to_datetime(pd.DataFrame({"year": year, "month": month, "day": raw["Day of Month"]}))
if not (date.dt.day_name() == raw["Day of Week"]).all():
    raise SystemExit("niezgodny dzień tygodnia: rok odtworzony błędnie")
out = pd.DataFrame(
    {
        "Data": date.dt.date,
        "Godzina": raw["Hour"].astype(int),
        "Zapotrzebowanie": raw["Demand"].astype(float),
        "Temperatura": raw["Aver Daily Temp"].astype(float),
    }
)
ts = date + pd.to_timedelta(out["Godzina"] - 1, unit="h")
if not (ts.diff().dropna() == pd.Timedelta(hours=1)).all():
    raise SystemExit("luka lub powtórzenie w godzinach")
out.to_csv(dst, index=False)
print(f"{len(out)} godzin, {out['Data'].nunique()} dób, {out['Data'].min()} - {out['Data'].max()}")
