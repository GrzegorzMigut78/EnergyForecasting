"""Loader GEFCom2012 na małych plikach w formacie konkursowym (bez prawdziwych danych)."""

import numpy as np
import pandas as pd
import yaml

from prognozowanie.config import load_dataset_config
from prognozowanie.data import load_dataset


def _write(path, id_col, ids, days, value):
    rows = []
    for i in ids:
        for d in days:
            rows.append(
                {
                    id_col: i,
                    "year": d.year,
                    "month": d.month,
                    "day": d.day,
                    **{f"h{h}": value(i, d, h) for h in range(1, 25)},
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_gefcom_loader_sums_zones_and_averages_stations(tmp_path):
    days = pd.date_range("2004-01-01", periods=10, freq="D")
    _write(
        tmp_path / "Load_history.csv", "zone_id", [1, 2], days, lambda i, d, h: f"{1000 * i + h:,}"
    )  # format z separatorem tysięcy jak w konkursie
    _write(
        tmp_path / "temperature_history.csv", "station_id", [1, 2, 3], days, lambda i, d, h: 30 + i
    )
    (tmp_path / "pyproject.toml").write_text("")
    cfg_path = tmp_path / "g.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "name": "g",
                "source": {
                    "loader": "gefcom2012",
                    "path": "Load_history.csv",
                    "temperature": {"path": "temperature_history.csv"},
                    "period": {"start": "2004-01-02", "end": "2004-01-09"},
                },
                "calendar": {"country": "US"},
            }
        )
    )
    ds = load_dataset(load_dataset_config(cfg_path))
    f = ds.frame
    assert len(f) == 8 * 24
    assert f.index[0] == pd.Timestamp("2004-01-02 00:00")
    # godzina h: 1000*1 + h + 1000*2 + h = 3000 + 2h
    assert np.allclose(f["load"].to_numpy()[:24], 3000 + 2 * np.arange(1, 25))
    assert np.allclose(f["temp"], 32.0)  # średnia z 31, 32, 33
    assert f.loc["2004-01-01":"2004-01-02", "holiday"].sum() == 0  # 1.01 wycięty okresem
