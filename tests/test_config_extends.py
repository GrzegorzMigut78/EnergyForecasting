"""Konfiguracja przebiegu próbnego dziedziczy z eksperymentu głównego ('extends')."""

from pathlib import Path

import pytest

from prognozowanie.config import ExperimentConfig, _read_experiment_yaml

ROOT = Path(__file__).resolve().parents[1]


def test_pilot_differs_from_main_only_in_name_and_periods():
    main = _read_experiment_yaml(ROOT / "configs/experiments/main.yaml")
    pilot = _read_experiment_yaml(ROOT / "configs/experiments/pilot_lipiec.yaml")
    assert pilot["name"] == "pilot_lipiec"
    assert str(pilot["periods"]["test"]["start"]) == "2002-07-10"
    assert str(pilot["periods"]["test"]["end"]) == "2002-07-31"
    # okres identyfikacji odziedziczony mimo nadpisania innych kluczy 'periods'
    assert pilot["periods"]["identification"] == main["periods"]["identification"]
    same = set(main) - {"name", "description", "periods", "evaluation"}
    assert {k: pilot[k] for k in same} == {k: main[k] for k in same}
    # w ocenie różni się tylko wyłączenie kwietnia (okna lipca go obejmują)
    assert pilot["evaluation"]["excluded_data"] == []
    assert main["evaluation"]["excluded_data"]
    rest = set(main["evaluation"]) - {"excluded_data"}
    assert {k: pilot["evaluation"][k] for k in rest} == {k: main["evaluation"][k] for k in rest}
    raw = {k: v for k, v in pilot.items() if not str(k).startswith("x-")}
    raw.pop("identified_params")
    ExperimentConfig.model_validate(raw)


def test_extends_cycle_rejected(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\nname: a\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\nname: b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cykliczne"):
        _read_experiment_yaml(tmp_path / "a.yaml")


def test_extends_lists_replaced_dicts_merged(tmp_path):
    (tmp_path / "base.yaml").write_text(
        "name: base\nlst: [1, 2]\nd: {a: 1, b: {c: 2, e: 3}}\n", encoding="utf-8"
    )
    (tmp_path / "child.yaml").write_text(
        "extends: base.yaml\nlst: [9]\nd: {b: {c: 5}}\n", encoding="utf-8"
    )
    got = _read_experiment_yaml(tmp_path / "child.yaml")
    assert got == {"name": "base", "lst": [9], "d": {"a": 1, "b": {"c": 5, "e": 3}}}


def test_ml_control_differs_from_main_only_in_sarima_method():
    """Kontrola estymacji: te same modele co w main.yaml, tylko SARIMA metodą ML."""
    import json

    main = _read_experiment_yaml(ROOT / "configs/experiments/main.yaml")
    ml = _read_experiment_yaml(ROOT / "configs/experiments/main_ml_kontrola.yaml")
    a = json.dumps(main["models"], sort_keys=True).replace('"css"', '"ml"')
    assert a == json.dumps(ml["models"], sort_keys=True)
    assert '"css"' not in json.dumps(ml["models"])
    same = set(main) - {"name", "description", "periods", "models", "execution"}
    assert {k: ml[k] for k in same} == {k: main[k] for k in same}
    # wykonanie: tylko liczba równoległych dób (pamięć SARIMA ML), poza migawką wejść
    assert ml["execution"] == {**main.get("execution", {}), "n_jobs": 1}
