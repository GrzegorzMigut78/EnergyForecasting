"""Pobiera dane GEFCom2012 (ścieżka obciążenia) i zapisuje je w układzie plików konkursowych
(Load_history.csv, temperature_history.csv), który czyta loader 'gefcom2012'.

Źródło: pakiet R 'tscompdata' (R. J. Hyndman), dane zgodne z konkursem
(Hong, Pinson, Fan, International Journal of Forecasting 30(2), 2014):
https://github.com/robjhyndman/tscompdata

Szereg godzinowy zaczyna się 2004-01-01 godz. 1 (00:00-01:00). Wyrównanie jest sprawdzane:
luki w obciążeniu muszą wypaść w ośmiu tygodniach wyciętych w konkursie do prognozy wstecznej.

    uv run --extra gefcom python scripts/fetch_gefcom2012.py
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://raw.githubusercontent.com/robjhyndman/tscompdata/master/data/"
START = pd.Timestamp("2004-01-01 00:00")
# tygodnie bez obciążenia w danych konkursowych (pierwsza i ostatnia doba luki)
EXPECTED_GAPS = [
    ("2005-03-06", "2005-03-12"),
    ("2005-06-20", "2005-06-26"),
    ("2005-09-10", "2005-09-16"),
    ("2005-12-25", "2005-12-31"),
    ("2006-02-13", "2006-02-19"),
    ("2006-05-25", "2006-05-31"),
    ("2006-08-02", "2006-08-08"),
    ("2006-11-22", "2006-11-28"),
]


def _read_rda(name: str) -> dict[str, np.ndarray]:
    import rdata

    with urllib.request.urlopen(BASE + f"{name}.rda", timeout=120) as r:
        raw = r.read()
    parsed = rdata.parser.parse_data(io.BytesIO(raw).read())
    obj = rdata.conversion.convert(parsed)[name]
    return {str(k): np.asarray(v, dtype=float) for k, v in obj.items()}


def _wide(series: dict[str, np.ndarray], id_col: str) -> pd.DataFrame:
    rows = []
    for key, values in series.items():
        idx = pd.date_range(START, periods=len(values), freq="h")
        df = pd.DataFrame({"v": values, "date": idx.normalize(), "h": idx.hour + 1})
        w = df.pivot(index="date", columns="h", values="v")
        w.columns = [f"h{h}" for h in w.columns]
        w.insert(0, "day", w.index.day)
        w.insert(0, "month", w.index.month)
        w.insert(0, "year", w.index.year)
        w.insert(0, id_col, int(key))
        rows.append(w.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True)


def _check_gaps(series: dict[str, np.ndarray]) -> None:
    v = next(iter(series.values()))
    idx = pd.date_range(START, periods=len(v), freq="h")
    missing_days = pd.DatetimeIndex(idx[np.isnan(v)].normalize().unique())
    expected = pd.DatetimeIndex(
        np.concatenate([pd.date_range(a, b, freq="D") for a, b in EXPECTED_GAPS])
    )
    if not missing_days.equals(expected):
        raise RuntimeError("luki w danych nie zgadzają się z konkursem: błędne wyrównanie dat")


def main(out_dir: Path = Path("data/raw/gefcom2012")) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    load = _read_rda("gefcom2012_load")
    temp = _read_rda("gefcom2012_temp")
    _check_gaps(load)
    _wide(load, "zone_id").to_csv(out_dir / "Load_history.csv", index=False)
    _wide(temp, "station_id").to_csv(out_dir / "temperature_history.csv", index=False)
    n = len(next(iter(load.values())))
    end = START + pd.Timedelta(hours=n - 1)
    print(f"Zapisano {out_dir}: {len(load)} stref, {len(temp)} stacji, {START} - {end}")


if __name__ == "__main__":
    main()
