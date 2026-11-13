"""Wczytanie zbioru danych: loader, korekty z konfiguracji, kalendarz, kontrola jakości.

Surowy plik nigdy nie jest zmieniany. Wszystkie korekty są zadeklarowane w YAML,
wykonywane w kodzie i zapisywane w dzienniku zmian (atrybut ``Dataset.log``).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from prognozowanie.config import DatasetConfig
from prognozowanie.data.calendar import build_calendar
from prognozowanie.data.loaders import LOADERS, crop


@dataclass
class Dataset:
    config: DatasetConfig
    frame: pd.DataFrame  # indeks godzinowy, kolumny opisane w load_dataset
    log: list[dict] = field(default_factory=list)
    source_sha256: str = ""

    @property
    def days(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.frame.index.normalize().unique())


def load_dataset(cfg: DatasetConfig) -> Dataset:
    """Zwraca ramkę z kolumnami:
    load, load_raw, temp, dow (0 = pn), hour (1-24), holiday, holiday_statutory,
    holiday_eve, holiday_after, bridge, dst_day, suspicious.
    """
    raw = crop(LOADERS[cfg.source.loader](cfg.source), cfg.source)
    log: list[dict] = []
    df = raw.copy()
    df["load_raw"] = df["load"]

    # 1. wykluczenia (tylko na brzegach szeregu, żeby nie rozerwać ciągłości)
    for ex in cfg.corrections.exclude:
        mask = (df.index.normalize() >= pd.Timestamp(ex.start)) & (
            df.index.normalize() <= pd.Timestamp(ex.end)
        )
        log.append(
            {
                "type": "exclude",
                "what": f"{ex.start}..{ex.end}",
                "rows": int(mask.sum()),
                "reason": ex.reason,
            }
        )
        df = df.loc[~mask]

    # 2. korekty pojedynczych wartości
    for fx in cfg.corrections.fix:
        ts = pd.Timestamp(fx.timestamp)
        if ts not in df.index:
            raise KeyError(f"korekta {ts}: brak takiej godziny w danych")
        i = df.index.get_loc(ts)
        old = float(df["load"].iloc[i])
        new = float((df["load"].iloc[i - 1] + df["load"].iloc[i + 1]) / 2)
        df.iloc[i, df.columns.get_loc("load")] = new
        log.append(
            {"type": "fix", "what": str(ts), "before": old, "after": new, "reason": fx.reason}
        )

    # 3. flagi podejrzanych godzin (bez zmiany wartości)
    df["suspicious"] = 0
    for fl in cfg.corrections.flag:
        ts = pd.Timestamp(fl.timestamp)
        if ts in df.index:
            df.loc[ts, "suspicious"] = 1

    _check_hourly(df)

    # 4. kalendarz
    cal = build_calendar(df.index, cfg.calendar)
    day = df.index.normalize()
    for col in cal.columns:
        df[col] = cal[col].reindex(day).to_numpy()
    df["dow"] = df.index.day_of_week
    df["hour"] = df.index.hour + 1

    return Dataset(config=cfg, frame=df, log=log, source_sha256=source_fingerprint(cfg))


def _check_hourly(df: pd.DataFrame) -> None:
    if df.index.has_duplicates:
        dup = df.index[df.index.duplicated()][:5]
        raise ValueError(f"powtórzone godziny: {list(dup)}")
    expected = pd.date_range(df.index.min(), df.index.max(), freq="h")
    if len(expected) != len(df):
        gaps = expected.difference(df.index)[:5]
        raise ValueError(
            f"szereg nie jest ciągły (brakuje {len(expected) - len(df)} godzin, np. {list(gaps)}). "
            "Wykluczenia wolno stosować tylko na początku lub końcu szeregu."
        )
    for col in ("load", "temp"):
        n = int(df[col].isna().sum())
        if n:
            raise ValueError(f"kolumna {col}: {n} braków")
    if not np.isfinite(df["load"]).all():
        raise ValueError("nieskończone wartości w load")


def source_fingerprint(cfg: DatasetConfig) -> str:
    """Odcisk wszystkich plików źródłowych zbioru. Przy jednym pliku: jego SHA-256 (jak w
    data/README.md); przy osobnym pliku temperatur (GEFCom): SHA-256 z obu skrótów, więc
    podmiana któregokolwiek pliku zmienia odcisk i blokuje wznowienie przebiegu."""
    main = _sha256(cfg.source.path)
    tpath = cfg.source.temperature.path
    if tpath is None:
        return main
    combined = f"{main}|{_sha256(tpath)}".encode()
    return hashlib.sha256(combined).hexdigest()


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
