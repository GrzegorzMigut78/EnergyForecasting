import pandas as pd
import pytest
from pydantic import ValidationError

from prognozowanie.config import ExperimentConfig, Periods, load_experiment_config
from tests.conftest import EXPERIMENT_YAML, needs_data


def test_main_config_is_valid():
    cfg = load_experiment_config(EXPERIMENT_YAML)
    assert cfg.periods.test.start.isoformat() == "2002-08-01"
    assert set(cfg.combination.base_models) <= {m.name for m in cfg.models}


def test_overlapping_periods_rejected():
    with pytest.raises(ValidationError, match="identyfikacji"):
        Periods.model_validate(
            {
                "identification": {"start": "2001-11-01", "end": "2002-09-01"},
                "forecast_start": "2002-07-01",
                "test": {"start": "2002-08-01", "end": "2003-06-29"},
            }
        )


def test_unknown_key_rejected():
    raw = load_experiment_config(EXPERIMENT_YAML).model_dump()
    raw["literowka"] = 1
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(raw)


def test_combination_needs_defined_models():
    raw = load_experiment_config(EXPERIMENT_YAML).model_dump()
    raw["combination"]["base_models"].append("NIEZNANY")
    with pytest.raises(ValidationError, match="bez definicji"):
        ExperimentConfig.model_validate(raw)


@needs_data
@pytest.mark.slow
def test_run_resumes_without_recomputing(dataset, experiment):
    from prognozowanie.backtest import run_backtest

    origins = pd.date_range("2002-08-01", periods=2, freq="D")
    first = run_backtest(experiment, dataset, origins=origins)
    assert len(first) == 2 and all(r.status == "ok" for r in first)
    second = run_backtest(experiment, dataset, origins=origins)
    assert second == []  # nic do policzenia


@needs_data
def test_changed_config_in_same_run_dir_is_rejected(dataset, experiment):
    from prognozowanie.backtest.engine import prepare_run_dir, run_dir_for

    run_dir = run_dir_for(experiment)
    prepare_run_dir(experiment, dataset, run_dir)
    changed = experiment.model_copy(update={"seed": experiment.seed + 1})
    with pytest.raises(RuntimeError, match="innymi wejściami"):
        prepare_run_dir(changed, dataset, run_dir)


def test_identified_params_replace_first_stage_entirely():
    from prognozowanie.config import apply_identified

    models = [
        {
            "name": "SVR_ARIMA",
            "type": "two_stage",
            "first": {"type": "svr", "kernel": "poly", "degree": 2, "coef0": 1.5},
            "residual": {"type": "sarima", "order": [1, 0, 1], "method": "css"},
        }
    ]
    ident = {
        "models": {
            "SVR_ARIMA": {
                "first": {"type": "svr", "kernel": "rbf", "C": 10.0},
                "residual": {"order": [2, 0, 1]},
            }
        },
        "steps_done": ["arima", "tune"],
    }
    out = apply_identified(models, ident)[0]
    assert out["first"] == {"type": "svr", "kernel": "rbf", "C": 10.0}  # bez starego degree
    assert out["residual"] == {"type": "sarima", "order": [2, 0, 1], "method": "css"}
    assert out["provisional"] is False
    assert models[0]["first"]["kernel"] == "poly"  # oryginał nienaruszony


def test_identification_schedule(tmp_path):
    import datetime as dt

    import yaml

    from prognozowanie.config import ScheduleEntry, model_specs_for

    cfg = load_experiment_config(EXPERIMENT_YAML)
    p1, p2 = tmp_path / "a.yaml", tmp_path / "b.yaml"
    p1.write_text(
        yaml.safe_dump({"as_of": "2002-04-01", "models": {"ARIMA": {"order": [1, 0, 0]}}})
    )
    p2.write_text(
        yaml.safe_dump({"as_of": "2002-11-01", "models": {"ARIMA": {"order": [2, 0, 1]}}})
    )
    sched = [
        ScheduleEntry(valid_from=dt.date(2002, 4, 1), params=p1),
        ScheduleEntry(valid_from=dt.date(2002, 11, 1), params=p2),
    ]
    c = cfg.model_copy(update={"identified_schedule": sched})

    def arima(day):
        return next(s for s in model_specs_for(c, day) if s["name"] == "ARIMA")["order"]

    assert arima(dt.date(2002, 8, 1)) == [1, 0, 0]
    assert arima(dt.date(2002, 11, 1)) == [2, 0, 1]
    # identyfikacja na danych z przyszłości względem doby obowiązywania: odrzucona
    p2.write_text(yaml.safe_dump({"as_of": "2002-12-01", "models": {}}))
    with pytest.raises(ValueError, match="as_of"):
        model_specs_for(c, dt.date(2002, 11, 5))
