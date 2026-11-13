"""Symulacja operacyjna: dla każdej doby D model uczy się na oknie [D - W, D) i prognozuje
48 godzin od północy.

Archiwum ma układ "model x doba": forecasts/<model>/<RRRR-MM-DD>.parquet oraz
meta/<model>/<RRRR-MM-DD>.json. Plik metadanych zapisywany jest jako ostatni i oznacza, że
para (model, doba) jest policzona (także gdy model zawiódł). Dzięki temu:
  - przerwany przebieg wznawia się od brakujących par,
  - dołożenie modelu liczy tylko ten model, a pozostałe zostają,
  - zmiana specyfikacji jednego modelu wymaga przeliczenia tylko tego modelu
    (wznowienie z inną specyfikacją jest błędem, chyba że model wskazano w ``recompute``).
"""

from __future__ import annotations

import json
import logging
import os
import platform
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from prognozowanie.backtest.scenarios import NOISE_GENERATOR, scenario_temperatures
from prognozowanie.config import ExperimentConfig, check_identification_data, model_specs_for
from prognozowanie.data import Dataset
from prognozowanie.models import build_model

log = logging.getLogger(__name__)

TARGET_COLUMNS = ("load", "load_raw")
STAGE1_SUFFIX = "__stage1"  # prognoza modelu dwustopniowego bez korekty ARIMA
HIDDEN_FROM_MODELS = (*TARGET_COLUMNS, "suspicious")


@dataclass(frozen=True)
class OriginResult:
    origin: pd.Timestamp
    status: str
    seconds: float
    path: Path | None = None
    error: str | None = None


def origins_for(cfg: ExperimentConfig) -> pd.DatetimeIndex:
    start = pd.Timestamp(cfg.periods.forecast_start)
    end = pd.Timestamp(cfg.periods.test.end)
    return pd.date_range(start, end, freq="D") + pd.Timedelta(hours=cfg.forecast.origin_hour)


def split_origin(
    frame: pd.DataFrame, origin: pd.Timestamp, window_days: int, horizon: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Okno uczące (wyłącznie przed chwilą prognozy) i ramka przyszłości bez zmiennej celu."""
    start = origin - pd.Timedelta(days=window_days)
    hist = frame.loc[(frame.index >= start) & (frame.index < origin)]
    if len(hist) != window_days * 24:
        raise ValueError(f"{origin}: okno ma {len(hist)} godzin zamiast {window_days * 24}")
    if hist.index.max() >= origin:  # strażnik informacji
        raise RuntimeError("okno uczące zawiera dane z chwili prognozy lub późniejsze")
    fut_index = pd.date_range(origin, periods=horizon, freq="h")
    fut = frame.reindex(fut_index).drop(columns=list(HIDDEN_FROM_MODELS), errors="ignore")
    if fut.isna().any().any():
        raise ValueError(f"{origin}: brak kalendarza lub temperatury w horyzoncie prognozy")
    return hist, fut


def run_origin(
    cfg: ExperimentConfig, frame: pd.DataFrame, origin: pd.Timestamp, models: list[dict[str, Any]]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Prognozy jednej doby dla podanych modeli.

    Zwraca (archiwum w formacie długim z kolumną ``model``, metadane). Wiersze
    ``<model>__stage1`` (pierwszy stopień modelu dwustopniowego) należą do modelu nadrzędnego,
    co zapisuje kolumna techniczna ``_owner`` (usuwana przy zapisie)."""
    h = cfg.forecast.horizon_hours
    hist, fut = split_origin(frame, origin, cfg.forecast.training_window_days, h)
    pooled = cfg.combination.all_pooled_models()
    n_fit = cfg.forecast.fitted_hours_saved
    rows: list[tuple[str, pd.DataFrame]] = []
    meta: dict[str, Any] = {"origin": str(origin), "models": {}}

    # temperatury scenariuszy liczone raz dla doby
    scen_temps = [
        (s.name, draw, temp)
        for s in cfg.temperature_scenarios
        for draw, temp in scenario_temperatures(s, hist, fut, origin, cfg.seed)
    ]

    for spec in models:
        name = spec["name"]
        t0 = time.perf_counter()
        info: dict[str, Any] = {"spec": spec_key(spec)}
        model_rows: list[pd.DataFrame] = []  # dopisywane do archiwum tylko przy sukcesie
        try:
            model = build_model(spec, seed=cfg.seed)
            model.fit(hist)
            # czasy rozdzielone (uwaga audytora, R3): dopasowanie, jedna prognoza 48 h i cała
            # pętla scenariuszy; poranne uruchomienie operacyjne = dopasowanie + jedna prognoza
            info["seconds_fit"] = time.perf_counter() - t0
            t_loop = time.perf_counter()
            two_stage = hasattr(model, "first_stage_forecast")
            cache: tuple[np.ndarray, np.ndarray | None] | None = None
            for scen_name, draw, temp in scen_temps:
                if model.uses_temperature or cache is None:
                    t_one = time.perf_counter()
                    f = fut.copy()
                    f["temp"] = temp
                    if two_stage:
                        # prognoza pierwszego stopnia archiwizowana osobno: ablacja
                        # "bez korekty ARIMA" bez ponownego liczenia
                        stage1 = model.first_stage_forecast(f)
                        pred = stage1 + model.residual_forecast(len(f))
                    else:
                        stage1, pred = None, model.predict(f)
                    if not np.all(np.isfinite(pred)):
                        raise FloatingPointError("prognoza zawiera wartości nieskończone lub NaN")
                    info.setdefault("seconds_predict_one", time.perf_counter() - t_one)
                    if not model.uses_temperature:
                        cache = (pred, stage1)
                else:
                    pred, stage1 = cache
                model_rows.append(_long(origin, name, scen_name, draw, "forecast", fut.index, pred))
                if two_stage and model.residual is not None:
                    model_rows.append(
                        _long(
                            origin,
                            f"{name}{STAGE1_SUFFIX}",
                            scen_name,
                            draw,
                            "forecast",
                            fut.index,
                            stage1,
                        )
                    )
            info["seconds_scenarios"] = time.perf_counter() - t_loop
            # wartości dopasowane z końca okna dla wszystkich modeli (wariant C0 z różną
            # liczbą godzin i C0 na pierwszym stopniu bez ponownego liczenia)
            fitted_series = [(name, model)]
            if two_stage and model.residual is not None:
                fitted_series.append((f"{name}{STAGE1_SUFFIX}", model.first))
            for fname, fmodel in fitted_series:
                try:
                    fv = fmodel.fitted().iloc[-n_fit:]
                    if fv.isna().any():
                        raise ValueError(f"brak wartości dopasowanych w ostatnich {n_fit} h")
                except Exception as exc:
                    if fname in pooled:
                        raise
                    info.setdefault("fitted_error", {})[fname] = f"{type(exc).__name__}: {exc}"
                    continue
                model_rows.append(
                    _long(origin, fname, "fitted", 0, "fitted", fv.index, fv.to_numpy())
                )
            rows.extend((name, r) for r in model_rows)
            info["status"] = "ok"
            for attr in ("coef_", "convergence_code"):
                if hasattr(model, attr):
                    info[attr] = _jsonable(getattr(model, attr))
            if two_stage:
                first = model.first
                for attr in ("coef_", "convergence_code"):
                    if hasattr(first, attr):
                        info[f"first_{attr}"] = _jsonable(getattr(first, attr))
                if model.residual is not None:
                    info["residual_coef"] = _jsonable(model.residual.coef_)
                    if hasattr(model.residual, "convergence_code"):
                        info["residual_convergence_code"] = model.residual.convergence_code
            # święto w horyzoncie, którego okno uczące nie zawiera (zmienna nieznana modelowi)
            if fut["holiday"].max() > 0 and hist["holiday"].max() == 0:
                info["holiday_unseen_in_window"] = True
        except Exception as exc:  # błąd jednego modelu nie zatrzymuje doby
            info["status"] = "error"
            info["error"] = f"{type(exc).__name__}: {exc}"
            info["traceback"] = traceback.format_exc(limit=3)
        info["seconds"] = round(time.perf_counter() - t0, 3)
        meta["models"][name] = info

    if rows:
        out = pd.concat([r.assign(_owner=o) for o, r in rows], ignore_index=True)
    else:
        out = pd.DataFrame(columns=[*ARCHIVE_COLUMNS, "_owner"])
    return out, meta


ARCHIVE_COLUMNS = ("origin", "model", "scenario", "draw", "kind", "target", "horizon", "value")


def spec_key(spec: dict[str, Any]) -> dict[str, Any]:
    """Specyfikacja modelu w postaci kanonicznej (porównywana przy wznowieniu)."""
    return json.loads(json.dumps(spec, sort_keys=True, default=str))


def _long(origin, model, scenario, draw, kind, index, values) -> pd.DataFrame:
    hours = ((index - origin) / pd.Timedelta(hours=1)).to_numpy().astype(int) + 1
    return pd.DataFrame(
        {
            "origin": origin,
            "model": model,
            "scenario": scenario,
            "draw": np.int16(draw),
            "kind": kind,
            "target": index,
            "horizon": hours.astype(np.int16),  # 1..48 prognoza; <= 0 wartości dopasowane
            "value": np.asarray(values, dtype=float),
        }
    )


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in x]
    return x


# ---------------------------------------------------------------- przebieg


def run_dir_for(cfg: ExperimentConfig, run_name: str | None = None) -> Path:
    return cfg.execution.output_dir / (run_name or cfg.name)


ARCHIVE_FORMAT = 2


def _strip_paths(d: Any) -> Any:
    """Ścieżki plików zastąpione nazwami: migawka nie zależy od położenia projektu na dysku
    (treść danych sprawdza skrót SHA-256)."""
    if isinstance(d, dict):
        return {k: (Path(v).name if k == "path" and v else _strip_paths(v)) for k, v in d.items()}
    if isinstance(d, list):
        return [_strip_paths(v) for v in d]
    return d


def global_inputs(cfg: ExperimentConfig, ds: Dataset) -> dict[str, Any]:
    """Wejścia wspólne dla wszystkich modeli: od nich zależy każda prognoza w archiwum.
    Lista modeli, łączenie i ocena nie należą tu: modele mają własne migawki (spec w
    metadanych pary model x doba), a łączenie i ocena liczą się z archiwum od nowa."""
    return {
        "archive_format": ARCHIVE_FORMAT,
        "dataset": _strip_paths(json.loads(ds.config.model_dump_json())),
        "dataset_sha256": ds.source_sha256,
        "forecast": json.loads(cfg.forecast.model_dump_json()),
        "temperature_scenarios": [
            json.loads(t.model_dump_json()) for t in cfg.temperature_scenarios
        ],
        "seed": cfg.seed,
        "noise_generator": NOISE_GENERATOR,
    }


def prepare_run_dir(cfg: ExperimentConfig, ds: Dataset, run_dir: Path) -> None:
    fdir = run_dir / "forecasts"
    if fdir.exists() and any(fdir.glob("*.parquet")):
        raise RuntimeError(
            f"{run_dir} ma archiwum w starym układzie (jeden plik na dobę). "
            "Policz przebieg pod nową nazwą (--run-name)."
        )
    fdir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta").mkdir(exist_ok=True)
    (run_dir / "runs").mkdir(exist_ok=True)
    snap = run_dir / "config.yaml"
    current = yaml.safe_dump(global_inputs(cfg, ds), allow_unicode=True, sort_keys=False)
    if snap.exists() and snap.read_text(encoding="utf-8") != current:
        raise RuntimeError(
            f"{run_dir} zawiera wyniki policzone z innymi wejściami wspólnymi (dane, korekty, "
            "horyzont, okno, scenariusze temperatury lub ziarno). Użyj innej nazwy przebiegu "
            "(--run-name) albo usuń katalog."
        )
    snap.write_text(current, encoding="utf-8")
    actual = ds.frame[
        [
            "load",
            "load_raw",
            "temp",
            "holiday",
            "holiday_statutory",
            "bridge",
            "dow",
            "hour",
            "suspicious",
        ]
    ]
    actual.to_parquet(run_dir / "actuals.parquet")


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def _log_invocation(cfg: ExperimentConfig, ds: Dataset, run_dir: Path, info: dict) -> None:
    """Każde wywołanie przebiegu zostawia ślad: pełna konfiguracja, środowisko, zakres prac."""
    stamp = time.strftime("%Y%m%dT%H%M%S")
    rec = {
        "experiment": json.loads(cfg.model_dump_json()),
        "identified_schedule": {
            str(e.valid_from): yaml.safe_load(Path(e.params).read_text(encoding="utf-8"))
            for e in cfg.identified_schedule
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu": _cpu_name(),
            "cpu_count": os.cpu_count(),
            "packages": _versions(),
            "dataset_sha256": ds.source_sha256,
            "dataset_log": ds.log,
        },
        **info,
    }
    path = run_dir / "runs" / f"{stamp}_{os.getpid()}.json"
    path.write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    # environment.json: środowisko ostatniego wywołania (pełna historia w runs/)
    (run_dir / "environment.json").write_text(
        json.dumps(rec["environment"], indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for pkg in (
        "numpy",
        "pandas",
        "scipy",
        "statsmodels",
        "statsforecast",
        "scikit-learn",
        "lightgbm",
        "optuna",
        "prognozowanie",
    ):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "brak"
    return out


def forecast_file(run_dir: Path, model: str, origin: pd.Timestamp) -> Path:
    return run_dir / "forecasts" / model / f"{origin:%Y-%m-%d}.parquet"


def meta_file(run_dir: Path, model: str, origin: pd.Timestamp) -> Path:
    return run_dir / "meta" / model / f"{origin:%Y-%m-%d}.json"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _work(cfg, frame, origin, models, run_dir) -> OriginResult:
    t0 = time.perf_counter()
    try:
        df, meta = run_origin(cfg, frame, origin, models)
        for name, info in meta["models"].items():
            path = forecast_file(run_dir, name, origin)
            if info["status"] == "ok":
                # najpierw prognozy, potem metadane (oba zapisy atomowe): plik metadanych
                # oznacza, że para (model, doba) jest kompletna
                part = df[df["_owner"] == name].drop(columns="_owner")
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                part.to_parquet(tmp, index=False)
                os.replace(tmp, path)
            else:
                path.unlink(missing_ok=True)  # nieaktualna prognoza z wcześniejszej specyfikacji
            rec = {"origin": meta["origin"], "model": name, **info}
            _atomic_write_text(
                meta_file(run_dir, name, origin), json.dumps(rec, ensure_ascii=False, indent=1)
            )
        errors = [m for m, i in meta["models"].items() if i["status"] != "ok"]
        status = "ok" if not errors else f"błędy modeli: {errors}"
        return OriginResult(origin, status, time.perf_counter() - t0, None)
    except Exception as exc:
        return OriginResult(origin, "error", time.perf_counter() - t0, error=repr(exc))


def plan_work(
    cfg: ExperimentConfig,
    run_dir: Path,
    origins: pd.DatetimeIndex,
    recompute: set[str] | frozenset[str] = frozenset(),
) -> dict[pd.Timestamp, list[dict[str, Any]]]:
    """Pary (doba, model) do policzenia. Para jest gotowa, gdy istnieją jej metadane
    z tą samą specyfikacją modelu (i plik prognoz, jeśli model nie zawiódł)."""
    unknown = set(recompute) - {m.name for m in cfg.models}
    if unknown:
        raise KeyError(f"recompute: modele spoza konfiguracji {sorted(unknown)}")
    todo: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    mismatch: dict[str, int] = {}
    for o in origins:
        for spec in model_specs_for(cfg, o.date()):
            name = spec["name"]
            mp = meta_file(run_dir, name, o)
            if name in recompute or not mp.exists():
                todo.setdefault(o, []).append(spec)
                continue
            done = json.loads(mp.read_text(encoding="utf-8"))
            if done.get("spec") != spec_key(spec):
                mismatch[name] = mismatch.get(name, 0) + 1
            elif done["status"] == "ok" and not forecast_file(run_dir, name, o).exists():
                todo.setdefault(o, []).append(spec)
    if mismatch:
        raise RuntimeError(
            "archiwum zawiera prognozy policzone z inną specyfikacją modeli "
            f"{dict(sorted(mismatch.items()))} (liczba dób). Jeśli zmiana jest zamierzona, "
            "przelicz te modele (--recompute NAZWA) albo nadaj modelowi nową nazwę."
        )
    return todo


def run_backtest(
    cfg: ExperimentConfig,
    ds: Dataset,
    run_name: str | None = None,
    origins: pd.DatetimeIndex | None = None,
    n_jobs: int | None = None,
    recompute: list[str] | None = None,
    allow_provisional: bool | None = None,
) -> list[OriginResult]:
    allow = cfg.execution.allow_provisional if allow_provisional is None else allow_provisional
    provisional = [m.name for m in cfg.models if m.provisional]
    if provisional and not allow:
        raise RuntimeError(
            f"modele z parametrami wstępnymi (bez kompletnej identyfikacji): {provisional}. "
            "Wskaż wynik identyfikacji (identified_params) albo, do próby technicznej, "
            "uruchom z --allow-provisional."
        )
    check_identification_data(cfg, ds.source_sha256)
    run_dir = run_dir_for(cfg, run_name)
    prepare_run_dir(cfg, ds, run_dir)
    origins = origins if origins is not None else origins_for(cfg)
    last_needed = origins.max() + pd.Timedelta(hours=cfg.forecast.horizon_hours - 1)
    if last_needed > ds.frame.index.max():
        raise ValueError(
            f"dane kończą się przed końcem horyzontu ostatniej prognozy ({last_needed})"
        )
    todo = plan_work(cfg, run_dir, origins, set(recompute or ()))
    n_pairs = sum(len(v) for v in todo.values())
    per_model: dict[str, int] = {}
    for specs in todo.values():
        for sp in specs:
            per_model[sp["name"]] = per_model.get(sp["name"], 0) + 1
    log.info(
        "przebieg %s: %d dób, do policzenia %d par (model, doba) w %d dobach: %s",
        run_dir.name,
        len(origins),
        n_pairs,
        len(todo),
        per_model,
    )
    _log_invocation(
        cfg,
        ds,
        run_dir,
        {
            "origins": [str(origins.min()), str(origins.max()), len(origins)],
            "todo_per_model": per_model,
            "n_jobs": n_jobs or cfg.execution.n_jobs,
            "recompute": sorted(recompute or ()),
        },
    )
    jobs = n_jobs or cfg.execution.n_jobs
    results = Parallel(n_jobs=jobs, return_as="generator_unordered")(
        delayed(_work)(cfg, ds.frame, o, specs, run_dir) for o, specs in todo.items()
    )
    out = []
    for i, r in enumerate(results, 1):
        out.append(r)
        level = logging.INFO if r.status == "ok" else logging.WARNING
        log.log(
            level,
            "[%d/%d] %s %s (%.1f s) %s",
            i,
            len(todo),
            f"{r.origin:%Y-%m-%d}",
            r.status,
            r.seconds,
            r.error or "",
        )
    return out


def archived_models(run_dir: Path) -> list[str]:
    return sorted(p.name for p in (run_dir / "forecasts").iterdir() if p.is_dir())


def load_archive(run_dir: Path, models: list[str] | None = None) -> pd.DataFrame:
    """Archiwum prognoz. ``models``: nazwy modeli z konfiguracji (z ich wierszami
    ``__stage1``); None = wszystkie modele obecne w archiwum."""
    fdir = run_dir / "forecasts"
    if fdir.exists() and any(fdir.glob("*.parquet")):
        raise RuntimeError(f"{run_dir}: archiwum w starym układzie (jeden plik na dobę)")
    names = models if models is not None else archived_models(run_dir)
    files = [f for m in names for f in sorted((fdir / m).glob("*.parquet"))]
    if not files:
        raise FileNotFoundError(f"brak prognoz w {fdir}")
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
