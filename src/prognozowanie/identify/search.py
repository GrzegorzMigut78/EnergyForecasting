"""Identyfikacja struktur na okresie identyfikacji (sekcje 4-5 metodyki).

1. Strojenie hiperparametrów SVR, sieci i benchmarku LightGBM (Optuna, walidacja
   chronologiczna): model uczy się na oknie przed dobą walidacyjną i prognozuje 48 godzin;
   kryterium = MAE dla doby D+1. Benchmark strojony tą samą procedurą co modele bazowe,
   żeby porównanie z nim było uczciwe.
2. Wybór rzędów ARIMA (na szeregu i na resztach modeli pierwszego stopnia) tym samym
   kryterium, spośród kandydatów z konfiguracji.

Wszystkie doby walidacyjne i ich horyzonty leżą wewnątrz okresu identyfikacji.
Wynik trafia do identified.yaml, który eksperyment wczytuje przez identified_params.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import yaml

from prognozowanie.backtest.engine import split_origin
from prognozowanie.config import ExperimentConfig, apply_identified
from prognozowanie.data import Dataset
from prognozowanie.models import build_model
from prognozowanie.models.statistical import Sarima

log = logging.getLogger(__name__)


def identification_period(
    cfg: ExperimentConfig, as_of: dt.date | None = None
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Okres identyfikacji: z konfiguracji albo, przy ponownej identyfikacji, okres tej samej
    długości kończący się dobę przed 'as_of'."""
    start = pd.Timestamp(cfg.periods.identification.start)
    end = pd.Timestamp(cfg.periods.identification.end)
    if as_of is not None:
        length = end - start
        end = pd.Timestamp(as_of) - pd.Timedelta(days=1)
        start = end - length
    return start, end


def validation_origins(
    cfg: ExperimentConfig, n: int, step: int = 1, period: tuple | None = None
) -> pd.DatetimeIndex:
    """Ostatnie n dób wydania (co 'step' dób), których horyzont mieści się w identyfikacji."""
    start, end = period or identification_period(cfg)
    h_days = cfg.forecast.horizon_hours // 24
    last = end - pd.Timedelta(days=h_days - 1)
    idx = pd.date_range(end=last, periods=n * step, freq="D")[::-1][::step][::-1]
    first_needed = idx.min() - pd.Timedelta(days=cfg.forecast.training_window_days)
    if first_needed < start:
        raise ValueError("okres identyfikacji za krótki na okno uczące i doby walidacyjne")
    return idx


def _mae_day2(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(pred[24:48] - actual[24:48])))


# ---------------------------------------------------------------- przestrzenie strojenia


def _suggest_svr(trial: optuna.Trial) -> dict[str, Any]:
    kernel = trial.suggest_categorical("kernel", ["rbf", "poly"])
    p = {
        "type": "svr",
        "kernel": kernel,
        # zmienna celu standaryzowana: C i epsilon w jednostkach odchylenia standardowego
        "C": trial.suggest_float("C", 0.1, 100.0, log=True),
        "epsilon": trial.suggest_float("epsilon", 0.005, 0.5, log=True),
        "gamma": trial.suggest_float("gamma", 1e-3, 1.0, log=True),
    }
    if kernel == "poly":
        p["degree"] = trial.suggest_int("degree", 2, 3)
        p["coef0"] = trial.suggest_float("coef0", 0.0, 3.0)
    return p


def _suggest_mlp(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "type": "mlp",
        "hidden_layer_sizes": trial.suggest_int("hidden_layer_sizes", 2, 16),
        "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
        "alpha": trial.suggest_float("alpha", 1e-7, 1e-2, log=True),
        "tol": trial.suggest_float("tol", 1e-6, 1e-3, log=True),
    }


def _suggest_lightgbm(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "type": "lightgbm",
        "n_estimators": trial.suggest_int("n_estimators", 100, 1500, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 4, 64, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


SPACES = {"svr": _suggest_svr, "mlp": _suggest_mlp, "lightgbm": _suggest_lightgbm}
# modele samodzielne (bez członu ARIMA) strojone w identyfikacji niezależnie od roli
STANDALONE_TUNED = ("lightgbm",)
SPACE_VERSION = "2026-11-11"  # zmiana przestrzeni strojenia = nowe badanie Optuny

# ustawienia techniczne niestrojone, identyczne w strojeniu i w przebiegu (zapisywane w wyniku)
TECHNICAL = {
    "svr": {"max_iter": 2_000_000, "cache_size": 1000},
    "mlp": {"solver": "lbfgs"},
    "lightgbm": {"verbose": -1},
}


# parametry techniczne przenoszone bez strojenia z konfiguracji (np. liczba startów sieci),
# żeby strojenie i przebieg używały identycznego modelu
FIXED_KEYS = ("n_seeds", "max_iter", "cache_size", "trend", "lags")


def tune_first_stage(
    cfg: ExperimentConfig,
    ds: Dataset,
    base_first: dict[str, Any],
    out_dir: Path,
    period: tuple,
) -> dict[str, Any]:
    model_type = base_first["type"]
    fixed = {
        **TECHNICAL.get(model_type, {}),
        **{k: v for k, v in base_first.items() if k in FIXED_KEYS},
    }
    origins = validation_origins(cfg, cfg.identification.tuning_origins, step=2, period=period)
    splits = [_split(cfg, ds, o) for o in origins]

    def objective(trial: optuna.Trial) -> float:
        spec = {**SPACES[model_type](trial), **fixed}
        scores = []
        for i, (hist, fut, actual) in enumerate(splits):
            m = build_model(spec, seed=cfg.seed).fit(hist)
            scores.append(_mae_day2(m.predict(fut), actual))
            trial.report(float(np.mean(scores)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=cfg.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=3)
    # postęp zapisywany w SQLite: przerwane strojenie wznawia się od miejsca przerwania
    storage = f"sqlite:///{(out_dir / 'optuna.db').as_posix()}"
    # nazwa badania zależy od wszystkich wejść strojenia: wznowienie po zmianie danych, dób
    # walidacyjnych, przestrzeni albo ustawień stałych zaczyna nowe badanie zamiast mieszać próby
    fingerprint = json.dumps(
        {
            "space": SPACE_VERSION,
            "type": model_type,
            "fixed": fixed,
            "origins": [str(o) for o in origins],
            "window": cfg.forecast.training_window_days,
            "data": ds.source_sha256,
            "corrections": ds.config.corrections.model_dump(mode="json"),
            "seed": cfg.seed,
        },
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(fingerprint.encode()).hexdigest()[:12]
    study = optuna.create_study(
        study_name=f"{model_type}-{digest}",
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )
    finished = sum(t.state.is_finished() for t in study.trials)
    remaining = max(cfg.identification.n_trials - finished, 0)
    timeout = (cfg.identification.timeout_minutes or 0) * 60 or None
    if remaining:
        log.info("%s: strojenie, %d prób do wykonania (%d już są)", model_type, remaining, finished)
        study.optimize(objective, n_trials=remaining, timeout=timeout)
    study.trials_dataframe().to_csv(out_dir / f"trials_{model_type}.csv", index=False)
    best = {**dict(study.best_params), **fixed}
    best["validation_mae_day2"] = study.best_value
    log.info("%s: najlepsze MAE(D+1) %.3f, parametry %s", model_type, study.best_value, best)
    return best


# warianty wygładzania wykładniczego rozważane w identyfikacji: wariant z artykułu (trend
# addytywny) i jego alternatywy; wybór na dobach walidacyjnych, nie na teście
# Inicjalizacja 'estimated' (170+ dodatkowych parametrów) w większości dób nie osiąga zbieżności
# optymalizatora (przegląd kodu: 26-29 na 34 doby), dlatego rozważamy tylko 'heuristic'.
ES_VARIANTS: list[dict[str, Any]] = [
    {"trend": "add", "damped": False, "initialization": "heuristic"},
    {"trend": "add", "damped": True, "initialization": "heuristic"},
    {"trend": None, "damped": False, "initialization": "heuristic"},
]
MAX_NONCONVERGED_SHARE = 0.2  # wariant z częstszym brakiem zbieżności jest odrzucany


def select_es_variant(
    cfg: ExperimentConfig, ds: Dataset, base_first: dict[str, Any], out_dir: Path, period: tuple
) -> dict[str, Any]:
    """Wybiera wariant wygładzania wykładniczego (trend, tłumienie, inicjalizacja) o najmniejszym
    MAE doby D+1 pierwszego stopnia na tych samych dobach co strojenie SVR i sieci."""
    origins = validation_origins(cfg, cfg.identification.tuning_origins, step=2, period=period)
    splits = [_split(cfg, ds, o) for o in origins]
    rows = []
    for variant in ES_VARIANTS:
        spec = {**base_first, **variant}
        scores, nonconv = [], 0
        for hist, fut, actual in splits:
            try:
                m = build_model(spec).fit(hist)
                nonconv += int(getattr(m, "convergence_code", 0) != 0)
                scores.append(_mae_day2(m.predict(fut), actual))
            except Exception as exc:
                log.warning("wygładzanie %s: %s", variant, exc)
                scores.append(np.inf)
        share = nonconv / len(splits)
        mae = float(np.mean(scores))
        rows.append(
            {
                **variant,
                "mae_day2": mae,
                "nonconverged_share": share,
                "eligible": bool(np.isfinite(mae) and share <= MAX_NONCONVERGED_SHARE),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "es_variants.csv", index=False)
    eligible = [r for r in rows if r["eligible"]]
    if not eligible:
        raise RuntimeError(
            "żaden wariant wygładzania wykładniczego nie spełnia kryteriów "
            f"(zbieżność, skończony błąd); zob. {out_dir / 'es_variants.csv'}"
        )
    best = min(eligible, key=lambda r: r["mae_day2"])
    log.info("wygładzanie wykładnicze: wariant %s (MAE D+1 %.3f)", best, best["mae_day2"])
    keep = ("trend", "damped", "initialization")
    return {**base_first, **{k: best[k] for k in keep}}


def _split(cfg, ds, origin):
    hist, fut = split_origin(
        ds.frame, origin, cfg.forecast.training_window_days, cfg.forecast.horizon_hours
    )
    actual = ds.frame["load"].reindex(fut.index).to_numpy()
    return hist, fut, actual


def select_arima_orders(
    cfg: ExperimentConfig,
    ds: Dataset,
    model_specs: dict[str, dict],
    out_dir: Path,
    period: tuple,
) -> dict[str, dict]:
    """Dla każdego modelu z członem ARIMA wybiera rzędy o najmniejszym MAE(D+1)."""
    origins = validation_origins(cfg, cfg.identification.validation_origins, period=period)
    splits = [_split(cfg, ds, o) for o in origins]
    cands = [(tuple(o), tuple(s)) for o, s in cfg.identification.arima_orders]
    rows, chosen = [], {}
    for name, spec in model_specs.items():
        t0 = time.perf_counter()
        if spec["type"] == "sarima":
            base = None
            ar_template = {
                k: v for k, v in spec.items() if k not in ("name", "role", "provisional")
            }
        elif spec["type"] == "two_stage" and spec.get("residual"):
            base = spec["first"]
            ar_template = dict(spec["residual"])
        else:
            continue
        # pierwszy stopień liczony raz na dobę, potem kandydaci ARIMA na jego resztach
        prepared = []
        for hist, fut, actual in splits:
            if base is None:
                prepared.append((hist["load"], np.zeros(len(fut)), actual))
            else:
                m1 = build_model(copy.deepcopy(base), seed=cfg.seed).fit(hist)
                resid = (hist["load"] - m1.fitted()).dropna()
                prepared.append((resid, m1.predict(fut), actual))
        for order, sorder in cands:
            scores = []
            for series, first_fc, actual in prepared:
                ar = Sarima({**ar_template, "order": list(order), "seasonal_order": list(sorder)})
                try:
                    ar.fit_series(series)
                    pred = first_fc + ar.forecast(len(first_fc))
                    scores.append(_mae_day2(pred, actual))
                except Exception as exc:  # kandydat niezbieżny: odrzucony
                    log.warning("%s %s%s: %s", name, order, sorder, exc)
                    scores.append(np.inf)
            rows.append(
                {
                    "model": name,
                    "order": list(order),
                    "seasonal_order": list(sorder),
                    "mae_day2": float(np.mean(scores)),
                }
            )
        best = min((r for r in rows if r["model"] == name), key=lambda r: r["mae_day2"])
        if not np.isfinite(best["mae_day2"]):
            raise RuntimeError(f"{name}: żaden kandydat ARIMA nie dał prognozy")
        chosen[name] = {"order": best["order"], "seasonal_order": best["seasonal_order"]}
        log.info(
            "%s: rzędy %s (MAE D+1 %.3f, %.0f s)",
            name,
            chosen[name],
            best["mae_day2"],
            time.perf_counter() - t0,
        )
    table = pd.DataFrame(rows)
    table[["order", "seasonal_order"]] = table[["order", "seasonal_order"]].astype(str)
    table.to_csv(out_dir / "arima_candidates.csv", index=False)
    return chosen


def run_identification(
    cfg: ExperimentConfig,
    ds: Dataset,
    out_dir: Path,
    steps: tuple[str, ...] = ("tune", "arima"),
    as_of: dt.date | None = None,
) -> Path:
    """Wykonuje identyfikację i zapisuje identified.yaml w out_dir.

    Kroki można uruchamiać osobno: wynik wcześniejszego kroku (plik identified.yaml
    w tym samym katalogu) jest wczytywany i uzupełniany.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    period = identification_period(cfg, as_of)
    path = out_dir / "identified.yaml"
    previous = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None
    if previous and previous.get("identification_period") != [
        str(period[0].date()),
        str(period[1].date()),
    ]:
        raise RuntimeError(f"{path} pochodzi z innego okresu identyfikacji; użyj innego katalogu")
    base_specs = [m.model_dump() for m in cfg.models if m.role == "base"]
    if previous:
        base_specs = apply_identified(base_specs, previous)
    specs = {s["name"]: s for s in base_specs}
    result: dict[str, Any] = {
        "experiment": cfg.name,
        "identification_period": [str(period[0].date()), str(period[1].date())],
        "as_of": str((period[1] + pd.Timedelta(days=1)).date()),
        "models": copy.deepcopy(previous["models"]) if previous else {},
    }
    if "tune" in steps:
        tuned: dict[str, dict] = {}
        for name, spec in specs.items():
            first = spec.get("first", {})
            if spec["type"] == "two_stage" and first.get("type") in SPACES:
                key = first["type"]
                if key not in tuned:  # jeden typ modelu strojony raz
                    best = tune_first_stage(cfg, ds, first, out_dir, period)
                    params = {k: v for k, v in best.items() if k != "validation_mae_day2"}
                    tuned[key] = {"type": key, **params}
                    if key == "mlp":
                        tuned[key]["solver"] = "lbfgs"
                spec["first"] = copy.deepcopy(tuned[key])
                result["models"].setdefault(name, {})["first"] = copy.deepcopy(tuned[key])
            elif spec["type"] == "two_stage" and first.get("type") == "exp_smoothing":
                chosen = select_es_variant(cfg, ds, first, out_dir, period)
                spec["first"] = copy.deepcopy(chosen)
                result["models"].setdefault(name, {})["first"] = copy.deepcopy(chosen)
        # benchmarki i inne modele samodzielne strojonych typów (LightGBM)
        for m in cfg.models:
            if m.type in STANDALONE_TUNED:
                own = {
                    k: v
                    for k, v in m.model_dump().items()
                    if k not in ("name", "role", "provisional")
                }
                best = tune_first_stage(cfg, ds, own, out_dir, period)
                params = {k: v for k, v in best.items() if k != "validation_mae_day2"}
                result["models"][m.name] = {"type": m.type, **params}
    if "tune" in steps and "arima" not in steps:
        # nowy pierwszy stopień unieważnia rzędy ARIMA dobrane do jego reszt
        for name in result["models"]:
            if result["models"][name].pop("residual", None) is not None:
                log.warning("%s: rzędy ARIMA reszt usunięte; uruchom krok 'arima' ponownie", name)
    if "arima" in steps:
        orders = select_arima_orders(cfg, ds, specs, out_dir, period)
        for name, o in orders.items():
            key = "residual" if specs[name]["type"] == "two_stage" else None
            if key:
                result["models"].setdefault(name, {})[key] = o
            else:
                result["models"].setdefault(name, {}).update(o)
    path.write_text(yaml.safe_dump(result, allow_unicode=True, sort_keys=False), encoding="utf-8")
    log.info("wynik identyfikacji: %s", path)
    return path
