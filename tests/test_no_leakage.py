"""Najważniejszy test projektu: prognoza nie może zależeć od niczego, co wydarzyło się
po chwili jej wydania (poza temperaturą w scenariuszu wyroczni, który z definicji
ją zna)."""

import numpy as np
import pandas as pd
import pytest

from prognozowanie.backtest.engine import run_origin, split_origin
from tests.conftest import needs_data

ORIGIN = pd.Timestamp("2002-11-20")


def _poison(frame: pd.DataFrame, origin: pd.Timestamp, temps: bool) -> pd.DataFrame:
    bad = frame.copy()
    after = bad.index >= origin
    bad.loc[after, ["load", "load_raw"]] = 1e6
    if temps:
        bad.loc[after, "temp"] = -99.0
    return bad


@needs_data
def test_forecasts_ignore_future_load(dataset, experiment):
    models = [m.model_dump() for m in experiment.models]
    clean, _ = run_origin(experiment, dataset.frame, ORIGIN, models)
    dirty, _ = run_origin(experiment, _poison(dataset.frame, ORIGIN, temps=False), ORIGIN, models)
    assert len(clean) > 0
    pd.testing.assert_frame_equal(clean, dirty)


@needs_data
def test_persistence_scenario_ignores_future_temperature(dataset, experiment):
    models = [m.model_dump() for m in experiment.models]
    clean, _ = run_origin(experiment, dataset.frame, ORIGIN, models)
    dirty, _ = run_origin(experiment, _poison(dataset.frame, ORIGIN, temps=True), ORIGIN, models)
    keep = clean["scenario"].isin(["persistence", "fitted"])
    pd.testing.assert_frame_equal(
        clean[keep].reset_index(drop=True),
        dirty[dirty["scenario"].isin(["persistence", "fitted"])].reset_index(drop=True),
    )
    # wyrocznia z definicji zna przyszłą temperaturę, więc tu prognozy modeli z temperaturą
    # muszą się różnić (kontrola, że test w ogóle coś sprawdza)
    o_clean = clean[(clean.scenario == "oracle") & (clean.model == "HONG")]["value"].to_numpy()
    o_dirty = dirty[(dirty.scenario == "oracle") & (dirty.model == "HONG")]["value"].to_numpy()
    assert not np.allclose(o_clean, o_dirty)


@needs_data
def test_window_and_future_frame(dataset):
    hist, fut = split_origin(dataset.frame, ORIGIN, 84, 48)
    assert hist.index.max() == ORIGIN - pd.Timedelta(hours=1)
    assert len(hist) == 84 * 24
    assert fut.index[0] == ORIGIN and len(fut) == 48
    assert "load" not in fut.columns and "load_raw" not in fut.columns


def test_model_rejects_future_with_load():
    from prognozowanie.models.base import check_future

    with pytest.raises(RuntimeError):
        check_future(pd.DataFrame({"load": [1.0]}))
