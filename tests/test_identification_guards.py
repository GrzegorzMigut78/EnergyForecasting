"""Zabezpieczenia identyfikacji i przebiegu (uwagi audytora z 11.11.2026)."""

import datetime as dt
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from prognozowanie.config import (
    _check_as_of,
    apply_identified,
    check_identification_data,
    identification_complete,
)
from prognozowanie.data.dataset import source_fingerprint
from prognozowanie.identify.search import identification_period, validation_origins
from tests.conftest import needs_data


def test_validation_origins_spread_over_identification(experiment):
    period = identification_period(experiment)
    window = experiment.forecast.training_window_days
    for n in (10, 20):
        idx = validation_origins(experiment, n, period=period)
        assert len(idx) == n and idx.is_monotonic_increasing
        assert idx.min() >= period[0] + pd.Timedelta(days=window)
        assert idx.max() + pd.Timedelta(days=1) <= period[1]
        steps = {d.days for d in idx[1:] - idx[:-1]}
        assert len(steps) == 1 and steps.pop() % 7 != 0  # różne dni tygodnia
        assert len(set(idx.day_of_week)) > 1
        # rozłożone, nie skupione na końcu okresu
        assert (idx.max() - idx.min()).days > 0.6 * (period[1] - period[0]).days - window


def test_partial_identification_keeps_models_provisional():
    models = [{"name": "M", "type": "sarima", "provisional": True, "order": [1, 0, 1]}]
    ident = {"models": {"M": {"order": [2, 0, 1]}}, "steps_done": ["tune"]}
    assert not identification_complete(ident)
    assert apply_identified(models, ident)[0]["provisional"] is True
    ident["steps_done"] = ["arima", "tune"]
    out = apply_identified(models, ident)[0]
    assert out["provisional"] is False and out["order"] == [2, 0, 1]


def test_identification_must_end_before_first_forecast():
    _check_as_of({"as_of": "2002-04-01"}, dt.date(2002, 7, 1), "x")
    with pytest.raises(ValueError, match="as_of"):
        _check_as_of({"as_of": "2002-07-02"}, dt.date(2002, 7, 1), "x")
    with pytest.raises(ValueError, match="as_of"):
        _check_as_of({}, dt.date(2002, 7, 1), "x")


def test_fingerprint_covers_separate_temperature_file(tmp_path):
    load, temp = tmp_path / "load.csv", tmp_path / "temp.csv"
    load.write_text("a")
    temp.write_text("1")
    cfg = SimpleNamespace(source=SimpleNamespace(path=load, temperature=SimpleNamespace(path=temp)))
    before = source_fingerprint(cfg)
    temp.write_text("2")
    assert source_fingerprint(cfg) != before
    single = SimpleNamespace(
        source=SimpleNamespace(path=load, temperature=SimpleNamespace(path=None))
    )
    import hashlib

    assert source_fingerprint(single) == hashlib.sha256(b"a").hexdigest()


@needs_data
def test_run_refuses_provisional_and_foreign_identification(dataset, experiment, tmp_path):
    from prognozowanie.backtest import run_backtest

    origins = pd.date_range("2002-08-01", periods=1, freq="D")
    strict = experiment.model_copy(
        update={"execution": experiment.execution.model_copy(update={"allow_provisional": False})}
    )
    with pytest.raises(RuntimeError, match="wstępnymi"):
        run_backtest(strict, dataset, origins=origins)
    # plik identyfikacji z innych danych blokuje przebieg
    f = tmp_path / "identified.yaml"
    f.write_text(yaml.safe_dump({"dataset_sha256": "inne", "as_of": "2002-04-01"}))
    foreign = experiment.model_copy(update={"identified_params": f})
    with pytest.raises(RuntimeError, match="innych danych"):
        check_identification_data(foreign, dataset.source_sha256)
    f.write_text(yaml.safe_dump({"dataset_sha256": dataset.source_sha256}))
    check_identification_data(foreign, dataset.source_sha256)
