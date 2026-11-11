"""Loadery surowych plików. Każdy zwraca ramkę ze wspólnymi kolumnami:
indeks godzinowy 'timestamp' (godzina 1 = 00:00-01:00), 'load', 'temp'.

Nowy format pliku = nowa funkcja zarejestrowana w LOADERS.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from prognozowanie.config import SourceConfig


def _from_frame(raw: pd.DataFrame, src: SourceConfig) -> pd.DataFrame:
    missing = {src.date_column, src.hour_column, src.load_column, *src.temperature.columns} - set(
        raw.columns
    )
    if missing:
        raise KeyError(f"w pliku {src.path.name} brak kolumn: {sorted(missing)}")
    date = pd.to_datetime(raw[src.date_column]).dt.normalize()
    hour0 = raw[src.hour_column].astype(int) - src.hour_base
    ts = date + pd.to_timedelta(hour0, unit="h")
    temp_cols = raw[src.temperature.columns].astype(float)
    temp = temp_cols.mean(axis=1) if src.temperature.aggregate == "mean" else temp_cols.iloc[:, 0]
    out = pd.DataFrame(
        {"load": raw[src.load_column].astype(float).to_numpy(), "temp": temp.to_numpy()},
        index=pd.DatetimeIndex(ts, name="timestamp"),
    )
    return out.sort_index()


def load_excel_hourly(src: SourceConfig) -> pd.DataFrame:
    return _from_frame(pd.read_excel(src.path, sheet_name=src.sheet), src)


def load_csv_hourly(src: SourceConfig) -> pd.DataFrame:
    return _from_frame(pd.read_csv(src.path), src)


def _gefcom_long(path, id_col: str, ids: list[int] | None) -> pd.DataFrame:
    """Plik GEFCom2012 (układ szeroki: id, year, month, day, h1..h24) -> kolumny = id."""
    raw = pd.read_csv(path, thousands=",")
    if ids is not None:
        raw = raw[raw[id_col].isin(ids)]
        if raw[id_col].nunique() != len(ids):
            raise KeyError(f"{path}: nie znaleziono wszystkich identyfikatorów {ids}")
    hours = [f"h{h}" for h in range(1, 25)]
    long = raw.melt(
        id_vars=[id_col, "year", "month", "day"], value_vars=hours, var_name="h", value_name="v"
    )
    ts = pd.to_datetime(long[["year", "month", "day"]]) + pd.to_timedelta(
        long["h"].str[1:].astype(int) - 1, unit="h"
    )
    return long.assign(timestamp=ts).pivot(index="timestamp", columns=id_col, values="v")


def load_gefcom2012(src: SourceConfig) -> pd.DataFrame:
    """GEFCom2012, ścieżka obciążenia (Hong, Pinson, Fan 2014): Load_history.csv
    i temperature_history.csv. Zapotrzebowanie = suma wybranych stref, temperatura =
    średnia wybranych stacji (godzinowa). Luki w danych konkursowych (tygodnie do prognozy
    wstecznej) trzeba wyciąć przez source.period."""
    load = _gefcom_long(src.path, "zone_id", src.zones)
    temp = _gefcom_long(src.temperature.path, "station_id", src.temperature.stations)
    idx = load.index.intersection(temp.index)
    # suma stref: brak w dowolnej strefie daje brak (min_count), a nie zaniżoną sumę
    out = pd.DataFrame(
        {
            "load": load.loc[idx].sum(axis=1, min_count=load.shape[1]),
            "temp": temp.loc[idx].mean(axis=1),
        },
        index=pd.DatetimeIndex(idx, name="timestamp"),
    )
    return out.sort_index()


LOADERS: dict[str, Callable[[SourceConfig], pd.DataFrame]] = {
    "excel_hourly": load_excel_hourly,
    "csv_hourly": load_csv_hourly,
    "gefcom2012": load_gefcom2012,
}


def crop(df: pd.DataFrame, src: SourceConfig) -> pd.DataFrame:
    if src.period is None:
        return df
    day = df.index.normalize()
    return df[(day >= pd.Timestamp(src.period.start)) & (day <= pd.Timestamp(src.period.end))]
