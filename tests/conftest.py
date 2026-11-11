from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prognozowanie.config import load_dataset_config, load_experiment_config

ROOT = Path(__file__).resolve().parents[1]
DATASET_YAML = ROOT / "configs" / "datasets" / "elecener.yaml"
EXPERIMENT_YAML = ROOT / "configs" / "experiments" / "main.yaml"
RAW = ROOT / "data" / "raw" / "ElecEner_long.xlsx"

needs_data = pytest.mark.skipif(not RAW.exists(), reason="brak surowego pliku danych")

FAST_MODELS = ["MR_ARIMA", "ARIMA", "SNAIVE", "HONG", "LGBM"]


@pytest.fixture(scope="session")
def dataset():
    from prognozowanie.data import load_dataset

    return load_dataset(load_dataset_config(DATASET_YAML))


@pytest.fixture
def experiment(tmp_path):
    """Eksperyment główny z szybkimi modelami i wynikami w katalogu tymczasowym."""
    cfg = load_experiment_config(EXPERIMENT_YAML)
    models = [m for m in cfg.models if m.name in FAST_MODELS]
    comb = cfg.combination.model_copy(update={"base_models": ["MR_ARIMA", "ARIMA"]})
    scen = [s for s in cfg.temperature_scenarios if s.kind != "noisy"]
    ex = cfg.execution.model_copy(update={"output_dir": tmp_path, "n_jobs": 1})
    return cfg.model_copy(
        update={
            "models": models,
            "combination": comb,
            "temperature_scenarios": scen,
            "execution": ex,
        }
    )


def synthetic_frame(days: int = 120, seed: int = 0) -> pd.DataFrame:
    """Sztuczny szereg z cyklem dobowym i tygodniowym, w formacie warstwy danych."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-06", periods=days * 24, freq="h")
    hour = idx.hour + 1
    dow = idx.day_of_week
    temp = np.repeat(rng.normal(5, 5, days), 24)
    load = (
        500
        + 60 * np.sin(2 * np.pi * (hour - 6) / 24)
        - 40 * (dow >= 5)
        - 2 * temp
        + rng.normal(0, 5, len(idx))
    )
    return pd.DataFrame(
        {
            "load": load,
            "load_raw": load,
            "temp": temp,
            "dow": dow,
            "hour": hour,
            "holiday": 0,
            "holiday_statutory": 0,
            "holiday_eve": 0,
            "holiday_after": 0,
            "bridge": 0,
            "dst_day": 0,
            "suspicious": 0,
        },
        index=pd.DatetimeIndex(idx, name="timestamp"),
    )
