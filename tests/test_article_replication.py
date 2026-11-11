"""Czy regresja z kodu projektu to ten sam model co w artykule?

Plik Short (dane pierwotnej wersji artykułu, 01.11.2002-31.01.2003) zawiera wartości
dopasowane ze STATISTICA (kolumna 'Model 01 (1)t'). Regresja projektu na pierwszych
2016 obserwacjach musi odtworzyć tabelę 1 i te wartości.
"""

import numpy as np
import pandas as pd
import pytest

from prognozowanie.models import build_model
from tests.conftest import ROOT

SHORT = ROOT / "data" / "raw" / "ElecEner_Short_d.xlsx"
pytestmark = pytest.mark.skipif(not SHORT.exists(), reason="brak pliku Short z artykułu")

# tabela 1 artykułu (wybrane współczynniki)
TABLE_1 = {
    "const": 257.617,
    "temp": -2.170,
    "t": 0.016,
    "holiday": -62.595,
    "pon": 41.864,
    "pia": 54.704,
    "sob": 32.598,
    "g01": 18.611,
    "g02": 4.917,
    "g18": 100.029,
    "g24": 40.472,
}


@pytest.fixture(scope="module")
def short_frame():
    s = pd.read_excel(SHORT)
    idx = pd.date_range("2002-11-01", periods=len(s), freq="h", name="timestamp")
    f = pd.DataFrame(
        {
            "load": s["Demand"].to_numpy(float),
            "temp": s["Aver Daily Temp"].to_numpy(float),
            "holiday": s["Holiday"].to_numpy(int),
            "dow": idx.day_of_week,
            "hour": idx.hour + 1,
        },
        index=idx,
    )
    # kontrola: kalendarz z dat zgadza się z kolumnami pliku
    for col, d in zip(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"], range(6), strict=True):
        assert np.array_equal(s[col].to_numpy() == 1, f["dow"].to_numpy() == d), col
    assert np.array_equal(s["Hour"].to_numpy(), f["hour"].to_numpy())
    return f, s


def test_regression_reproduces_table_1(short_frame):
    f, _ = short_frame
    m = build_model({"type": "regression"}).fit(f.iloc[:2016])
    coef = dict(zip(["const", *m._cols], m.coef_, strict=True))
    for name, value in TABLE_1.items():
        assert coef[name] == pytest.approx(value, abs=0.0015), name


def test_regression_reproduces_statistica_fitted_values(short_frame):
    f, s = short_frame
    m = build_model({"type": "regression"}).fit(f.iloc[:2016])
    ref = s["Model 01 (1)t"].to_numpy()[:2016]
    assert np.max(np.abs(m.fitted().to_numpy() - ref)) < 0.05
