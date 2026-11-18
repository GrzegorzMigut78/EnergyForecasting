# Uruchomienie:
#   uv run python scripts/konfiguracje_short.py
"""Plan analizy 10.3: konfiguracje ćwiczenia na pliku pierwszej wersji (ElecEner Short).

Cztery przebiegi: okno uczące 84 albo 56 dni, parametry modeli z pierwszej wersji albo
z identyfikacji na ElecEner Long. Parametry z Long są wpisane wprost do listy modeli
(identyfikacja pochodzi z innych danych, więc nie podajemy jej jako identified_params,
którego strażnik słusznie odrzuca dla innego zbioru). Modele pomocnicze i pule rozszerzone
pominięte; scenariusze temperatury: zmierzona i persystencja.
"""

import copy
import json
from pathlib import Path

import yaml

from prognozowanie.config import _read_experiment_yaml, _read_yaml, apply_identified

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "experiments" / "short"
main = _read_experiment_yaml(ROOT / "configs/experiments/main.yaml")
ident = _read_yaml(ROOT / "results/identify/main/base/identified.yaml")

POOL = ["MR_ARIMA", "ARIMA", "ES_ARIMA", "SVR_ARIMA", "NN_ARIMA", "SNAIVE", "HONG", "LGBM"]
# kopia przez JSON: kotwice YAML dzielą słowniki między modelami, a zmiana rzędów jednego
# modelu nie może przechodzić na inne
base_models = [json.loads(json.dumps(m)) for m in main["models"] if m["name"] in POOL]

# rzędy SARIMA z opisu pliku pierwszej wersji (Opis_ElecEner_Short_d.docx), okres 168 h
ORIGINAL_ORDERS = {
    "MR_ARIMA": ("residual", [1, 0, 1], [0, 1, 1]),
    "ARIMA": (None, [1, 1, 3], [1, 1, 1]),
    "ES_ARIMA": ("residual", [1, 0, 3], [1, 1, 1]),
    # w opisie (1,0,0)(0,1,3): sezonowe MA rzędu 3 przy okresie 168 daje stan wymiaru ponad 500,
    # którego inicjalizacja w statsforecast przekracza pamięć (std::bad_alloc, 19.11.2026);
    # najbliższy wykonalny rząd: sezonowe MA rzędu 1 (odstępstwo, plan sekcja 9)
    "SVR_ARIMA": ("residual", [1, 0, 0], [0, 1, 1]),
    "NN_ARIMA": ("residual", [1, 0, 2], [1, 1, 1]),
}


def original_models():
    out = []
    for m in copy.deepcopy(base_models):
        if m["name"] in ORIGINAL_ORDERS:
            part, order, sorder = ORIGINAL_ORDERS[m["name"]]
            tgt = m[part] if part else m
            tgt["order"], tgt["seasonal_order"] = order, sorder
        m["provisional"] = False  # parametry pierwszej wersji z założenia ćwiczenia
        out.append(m)
    return out


def long_models():
    return [m for m in apply_identified(copy.deepcopy(base_models), ident)]


VARIANTS = {
    "oryginal": (original_models, "parametry pierwszej wersji (SVR, sieć, rzędy ARIMA z opisu)"),
    "long": (long_models, "parametry z identyfikacji ElecEner Long (do 31.03.2002)"),
}
WINDOWS = {84: "2003-01-24", 56: "2002-12-27"}

OUT.mkdir(parents=True, exist_ok=True)
for var, (fn, desc) in VARIANTS.items():
    for win, start in WINDOWS.items():
        name = f"short_{var}_{win}"
        cfg = {
            "name": name,
            "description": f"Plan 10.3: ElecEner Short, okno {win} dni, {desc}.",
            "dataset": "configs/datasets/elecener_short.yaml",
            "seed": main["seed"],
            "periods": {
                # okres identyfikacji ElecEner Long (informacyjnie: z niego pochodzą parametry)
                "identification": main["periods"]["identification"],
                "forecast_start": start,
                "test": {"start": start, "end": "2003-01-30"},
            },
            "forecast": {**main["forecast"], "training_window_days": win},
            "temperature_scenarios": main["temperature_scenarios"][:2],
            "models": fn(),
            "combination": {
                "base_models": main["combination"]["base_models"],
                "methods": main["combination"]["methods"],
                "k_days": main["combination"]["k_days"],
                "c0_fitted_hours": main["combination"]["c0_fitted_hours"],
            },
            "evaluation": {
                "excluded_data": [],
                "families": main["evaluation"]["families"],
                "confirm_scenarios": main["evaluation"]["confirm_scenarios"],
                "alpha": main["evaluation"]["alpha"],
            },
            "execution": {"n_jobs": 2, "output_dir": "results/short"},
        }
        head = "# Wygenerowane przez scripts/konfiguracje_short.py (plan 10.3); nie edytować.\n"
        (OUT / f"{name}.yaml").write_text(
            head + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        print(OUT / f"{name}.yaml")
