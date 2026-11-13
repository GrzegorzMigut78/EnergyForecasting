"""Schematy konfiguracji (pydantic) dla zbiorów danych i eksperymentów.

Każdy plik YAML jest sprawdzany przy wczytaniu: brakujące pola, literówki w nazwach
i niespójne okresy zatrzymują uruchomienie, zanim cokolwiek zostanie policzone.
"""

from __future__ import annotations

import copy
import datetime as dt
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------- zbiór danych


class Period(_Strict):
    start: dt.date
    end: dt.date

    @model_validator(mode="after")
    def _ordered(self) -> Period:
        if self.end < self.start:
            raise ValueError(f"koniec okresu {self.end} przed początkiem {self.start}")
        return self


class TemperatureSource(_Strict):
    columns: list[str] = Field(default_factory=list)  # dla excel_hourly / csv_hourly
    aggregate: Literal["mean", "first"] = "mean"
    path: Path | None = None  # dla gefcom2012: plik temperature_history.csv
    stations: list[int] | None = None  # dla gefcom2012: None = wszystkie stacje


class SourceConfig(_Strict):
    loader: Literal["excel_hourly", "csv_hourly", "gefcom2012"]
    path: Path
    sheet: int | str = 0
    date_column: str | None = None
    hour_column: str | None = None
    hour_base: Literal[0, 1] = 1  # 1: godzina 1 = 00:00-01:00
    load_column: str | None = None
    temperature: TemperatureSource
    zones: list[int] | None = None  # dla gefcom2012: None = suma wszystkich stref
    period: Period | None = None  # opcjonalne przycięcie do okresu bez luk

    @model_validator(mode="after")
    def _required_columns(self) -> SourceConfig:
        if self.loader in ("excel_hourly", "csv_hourly"):
            missing = [
                k for k in ("date_column", "hour_column", "load_column") if getattr(self, k) is None
            ]
            if missing or not self.temperature.columns:
                raise ValueError(
                    f"loader {self.loader} wymaga: {missing or ['temperature.columns']}"
                )
        if self.loader == "gefcom2012" and self.temperature.path is None:
            raise ValueError("loader gefcom2012 wymaga temperature.path")
        return self


class HolidayFeature(_Strict):
    """Definicja zmiennej 'święto' używanej przez modele."""

    exclude_weekdays: list[int] = Field(default_factory=list)  # 0 = poniedziałek, 6 = niedziela
    extra_month_days: list[str] = Field(default_factory=list)  # np. "12-24" (Wigilia)


class CalendarConfig(_Strict):
    country: str  # kod dla biblioteki holidays, np. "PL", "US"
    subdivision: str | None = None
    timezone: str | None = None  # np. "Europe/Warsaw"; służy tylko do oznaczenia dób zmiany czasu
    holiday_feature: HolidayFeature = HolidayFeature()


class Exclusion(_Strict):
    start: dt.date
    end: dt.date
    reason: str


class PointFix(_Strict):
    timestamp: dt.datetime
    method: Literal["interpolate_neighbors"] = "interpolate_neighbors"
    reason: str


class SuspiciousHour(_Strict):
    timestamp: dt.datetime
    reason: str = ""


class CorrectionsConfig(_Strict):
    exclude: list[Exclusion] = Field(default_factory=list)
    fix: list[PointFix] = Field(default_factory=list)
    flag: list[SuspiciousHour] = Field(default_factory=list)


class DatasetConfig(_Strict):
    name: str
    description: str = ""
    source: SourceConfig
    units: dict[str, str] = Field(default_factory=dict)
    dst_policy: Literal["as_is"] = "as_is"
    calendar: CalendarConfig
    corrections: CorrectionsConfig = CorrectionsConfig()


# ---------------------------------------------------------------- eksperyment


class Periods(_Strict):
    identification: Period
    forecast_start: dt.date  # pierwsza doba z prognozami (początek rozgrzewki)
    test: Period  # doby, dla których oceniamy prognozy

    @model_validator(mode="after")
    def _consistent(self) -> Periods:
        if self.forecast_start > self.test.start:
            raise ValueError("forecast_start musi być nie później niż początek testu")
        if self.identification.end >= self.test.start:
            raise ValueError("okres identyfikacji nie może zachodzić na okres testowy")
        return self


class ForecastConfig(_Strict):
    horizon_hours: int = 48
    training_window_days: int = 84
    origin_hour: int = 0  # prognoza wydawana o północy
    # ile ostatnich godzin wartości dopasowanych zapisywać w archiwum dla każdego modelu
    # (wariant C0 może potem użyć dowolnej liczby godzin nie większej niż ta)
    fitted_hours_saved: int = 168

    @field_validator("horizon_hours")
    @classmethod
    def _whole_days(cls, v: int) -> int:
        if v % 24:
            raise ValueError("horyzont musi być wielokrotnością 24 godzin")
        return v


class TemperatureScenario(_Strict):
    name: str
    kind: Literal["oracle", "persistence", "noisy"]
    sd_day1: float = 0.0  # odchylenie błędu prognozy temperatury dla doby D [°C]
    sd_day2: float = 0.0  # dla doby D+1 [°C]
    rho: float = 0.0  # korelacja błędu między dobami
    draws: int = 1

    @model_validator(mode="after")
    def _noise(self) -> TemperatureScenario:
        if self.sd_day1 < 0 or self.sd_day2 < 0 or not -1 <= self.rho <= 1 or self.draws < 1:
            raise ValueError(f"scenariusz {self.name}: sd >= 0, rho w [-1, 1], draws >= 1")
        return self


class ModelSpec(_Strict):
    """Specyfikacja modelu; 'type' wybiera klasę z rejestru modeli."""

    model_config = ConfigDict(extra="allow", frozen=True)
    name: str
    type: str
    # base: pula główna; benchmark: model odniesienia (H3); auxiliary: model pomocniczy do pul
    # rozszerzonych i ablacji, poza rodzinami hipotez
    role: Literal["base", "benchmark", "auxiliary"] = "base"
    provisional: bool = False  # True: parametry wymagają potwierdzenia w identyfikacji


class CombinationConfig(_Strict):
    base_models: list[str]
    methods: list[Literal["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]]
    k_days: list[int] = Field(default_factory=lambda: [2, 3, 7, 14, 28])
    # godziny wartości dopasowanych dla wariantu C0; 48 = procedura artykułu ("C0"),
    # pozostałe wartości dają warianty "C0_H<godziny>", np. "C0_H168"
    c0_fitted_hours: list[int] = Field(default_factory=lambda: [48])
    # dodatkowe pule łączone na tym samym archiwum, np. {"plus_hong": [..., "HONG"]};
    # nazwy wariantów w wynikach: "<pula>/<metoda>", np. "plus_hong/C2_K7"
    extra_pools: dict[str, list[str]] = Field(default_factory=dict)

    @field_validator("c0_fitted_hours", mode="before")
    @classmethod
    def _c0_list(cls, v):
        return [v] if isinstance(v, int) else v

    def c0_variants(self) -> list[tuple[str, int]]:
        return [("C0" if h == 48 else f"C0_H{h}", h) for h in self.c0_fitted_hours]

    def pools(self) -> dict[str, list[str]]:
        """Pula główna ma pustą nazwę (warianty bez przedrostka)."""
        return {"": list(self.base_models), **{k: list(v) for k, v in self.extra_pools.items()}}

    def all_pooled_models(self) -> set[str]:
        return {m for pool in self.pools().values() for m in pool}


class HypothesisFamily(_Strict):
    """Rodzina hipotez głównych: każde porównanie to wariant z 'compare' kontra 'reference'
    (różnica dziennych strat compare - reference; ujemna: wariant lepszy)."""

    question: str = ""
    reference: str
    compare: list[str]


class EvaluationConfig(_Strict):
    # rodziny hipotez głównych (plan analizy); poprawka Holma w obrębie rodziny
    families: dict[str, HypothesisFamily] = Field(default_factory=dict)
    # scenariusze temperatury, w których wniosek musi się potwierdzić: większa z wartości p,
    # Holm w rodzinie, zgodny kierunek różnicy
    confirm_scenarios: list[str] = Field(default_factory=lambda: ["oracle", "persistence"])
    alpha: float = 0.05
    # odniesienie dla opisowej tabeli DM i przedziałów bootstrapowych
    reference: str = "C3"
    # doby testowe oglądane przed zamrożeniem planu (pilotaże, przebiegi techniczne);
    # analiza wrażliwości: hipotezy główne bez tych dób
    seen_days: list[dt.date] = Field(default_factory=list)
    # okresy danych, które nie mogą wejść do oceny (plan, sekcja 1: kwiecień 2002 bez
    # potwierdzenia): para (wariant, doba wydania, blok) jest oceniana tylko wtedy, gdy żadne
    # okno uczące użyte do jej prognozy, łącznie z oknami prognoz z historii wag, nie obejmuje
    # tych okresów; stąd K = 14 od 8.08 i K = 28 od 22.08.2002
    excluded_data: list[Period] = Field(default_factory=list)


class ExecutionConfig(_Strict):
    n_jobs: int = 1
    output_dir: Path = Path("results")
    # przebieg na parametrach wstępnych (sprzed kompletnej identyfikacji) tylko świadomie:
    # do prób technicznych, nigdy do wyników
    allow_provisional: bool = False


class IdentificationConfig(_Strict):
    arima_orders: list[list[list[int]]] = Field(default_factory=list)
    validation_origins: int = 20  # doby walidacyjne dla wyboru rzędów ARIMA
    tuning_origins: int = 10  # doby walidacyjne dla strojenia SVR i sieci (co druga doba)
    n_trials: int = 100
    timeout_minutes: int | None = None  # limit czasu strojenia jednego modelu


class ScheduleEntry(_Strict):
    valid_from: dt.date  # pierwsza doba wydania, dla której obowiązują parametry
    params: Path  # identified.yaml z identyfikacji wykonanej na danych sprzed valid_from


class ExperimentConfig(_Strict):
    name: str
    description: str = ""
    dataset: Path
    seed: int = 12345
    periods: Periods
    forecast: ForecastConfig = ForecastConfig()
    temperature_scenarios: list[TemperatureScenario]
    models: list[ModelSpec]
    combination: CombinationConfig
    identification: IdentificationConfig = IdentificationConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    execution: ExecutionConfig = ExecutionConfig()
    # plik z wynikiem identyfikacji (prognozowanie identify); nadpisuje parametry modeli
    identified_params: Path | None = None
    # wariant z ponowną identyfikacją: od danej doby obowiązują parametry z danego pliku
    identified_schedule: list[ScheduleEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _names(self) -> ExperimentConfig:
        names = [m.name for m in self.models]
        if len(names) != len(set(names)):
            raise ValueError(f"powtórzone nazwy modeli: {names}")
        missing = self.combination.all_pooled_models() - set(names)
        if missing:
            raise ValueError(f"modele w pulach łączenia bez definicji: {sorted(missing)}")
        if self.identified_params and self.identified_schedule:
            raise ValueError("podaj identified_params albo identified_schedule, nie oba")
        dates = [e.valid_from for e in self.identified_schedule]
        if dates != sorted(dates):
            raise ValueError("identified_schedule musi być uporządkowany według valid_from")
        scen = [s.name for s in self.temperature_scenarios]
        if len(scen) != len(set(scen)):
            raise ValueError("powtórzone nazwy scenariuszy temperatury")
        saved = self.forecast.fitted_hours_saved
        if saved > self.forecast.training_window_days * 24:
            raise ValueError("fitted_hours_saved przekracza długość okna uczącego")
        c0 = self.combination.c0_fitted_hours
        if len(c0) != len(set(c0)) or not all(0 < h <= saved for h in c0):
            raise ValueError(
                f"c0_fitted_hours {c0}: wartości muszą być różne i nie większe niż "
                f"forecast.fitted_hours_saved ({saved})"
            )
        if any("/" in n or n.endswith("__stage1") for n in names):
            raise ValueError("nazwa modelu nie może zawierać '/' ani kończyć się '__stage1'")
        return self


# ---------------------------------------------------------------- wczytywanie


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_dataset_config(path: str | Path, root: Path | None = None) -> DatasetConfig:
    path = Path(path)
    cfg = DatasetConfig.model_validate(_read_yaml(path))
    base = root or _project_root(path)
    src = cfg.source
    upd: dict[str, Any] = {}
    if not src.path.is_absolute():
        upd["path"] = (base / src.path).resolve()
    tp = src.temperature.path
    if tp is not None and not tp.is_absolute():
        upd["temperature"] = src.temperature.model_copy(update={"path": (base / tp).resolve()})
    if upd:
        cfg = cfg.model_copy(update={"source": src.model_copy(update=upd)})
    return cfg


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Słowniki łączone rekurencyjnie; listy i wartości proste z 'over' zastępują bazowe."""
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read_experiment_yaml(path: Path, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Wczytuje eksperyment; 'extends: plik.yaml' (ścieżka względem bieżącego pliku) bierze
    konfigurację bazową i nadpisuje w niej tylko podane klucze (np. okresy przebiegu próbnego)."""
    path = path.resolve()
    if path in _seen:
        raise ValueError(f"cykliczne 'extends': {path}")
    raw = _read_yaml(path)
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    return _merge(_read_experiment_yaml(path.parent / parent, (*_seen, path)), raw)


def load_experiment_config(path: str | Path, use_identified: bool = True) -> ExperimentConfig:
    """Wczytuje eksperyment. Jeśli podano identified_params, parametry modeli z tego pliku
    nadpisują wartości wstępne, a flaga provisional jest zdejmowana. use_identified=False
    pomija wynik identyfikacji (testy: parametry wstępne niezależnie od stanu results/)."""
    path = Path(path)
    raw = _read_experiment_yaml(path)
    # klucze 'x-...' służą wyłącznie jako szablony YAML (kotwice) i nie są częścią schematu
    raw = {k: v for k, v in raw.items() if not str(k).startswith("x-")}
    if not use_identified:
        raw.pop("identified_params", None)
        raw.pop("identified_schedule", None)
    base = _project_root(path)
    ident = raw.get("identified_params")
    identified = None
    if ident:
        ident_path = Path(ident) if Path(ident).is_absolute() else base / ident
        identified = _read_yaml(ident_path)
        raw["models"] = apply_identified(raw["models"], identified)
    cfg = ExperimentConfig.model_validate(raw)
    if identified is not None:
        # identyfikacja musi kończyć się przed pierwszą prognozą (także rozgrzewkową)
        _check_as_of(identified, cfg.periods.forecast_start, ident)
    upd: dict[str, Any] = {}
    if not cfg.dataset.is_absolute():
        upd["dataset"] = (base / cfg.dataset).resolve()
    if not cfg.execution.output_dir.is_absolute():
        upd["execution"] = cfg.execution.model_copy(
            update={"output_dir": (base / cfg.execution.output_dir).resolve()}
        )
    if cfg.identified_schedule:
        upd["identified_schedule"] = [
            e.model_copy(
                update={
                    "params": e.params if e.params.is_absolute() else (base / e.params).resolve()
                }
            )
            for e in cfg.identified_schedule
        ]
    return cfg.model_copy(update=upd) if upd else cfg


IDENTIFICATION_STEPS = frozenset({"tune", "arima"})


def identification_complete(identified: dict[str, Any]) -> bool:
    """Wynik identyfikacji jest kompletny, gdy zawiera oba kroki (strojenie i rzędy ARIMA
    dobrane do reszt strojonego modelu)."""
    return set(identified.get("steps_done", [])) >= IDENTIFICATION_STEPS


def _check_as_of(identified: dict[str, Any], first_day: dt.date, source: Any) -> None:
    as_of = identified.get("as_of")
    if as_of is None or dt.date.fromisoformat(str(as_of)) > first_day:
        raise ValueError(f"{source}: identyfikacja musi kończyć się przed {first_day} (as_of)")


def apply_identified(models: list[dict[str, Any]], identified: dict[str, Any]) -> list[dict]:
    """Nakłada wynik identyfikacji na specyfikacje modeli.

    Człon 'first' jest zastępowany w całości (strojenie mogło zmienić np. jądro SVR, więc
    stare parametry nie mogą zostać); rzędy ARIMA są nadpisywane w obrębie członu.
    Flaga 'provisional' jest zdejmowana tylko przy kompletnym wyniku identyfikacji: samo
    strojenie bez kroku ARIMA zostawia startowe rzędy, więc model pozostaje wstępny.
    """
    complete = identification_complete(identified)
    overrides = identified.get("models", {})
    out = []
    for spec in models:
        spec = copy.deepcopy(spec)
        ov = overrides.get(spec["name"])
        if ov:
            for k, v in ov.items():
                if k == "first" or not isinstance(v, dict) or not isinstance(spec.get(k), dict):
                    spec[k] = copy.deepcopy(v)
                else:
                    spec[k].update(v)
            if complete:
                spec["provisional"] = False
        out.append(spec)
    return out


def model_specs_for(cfg: ExperimentConfig, day: dt.date) -> list[dict[str, Any]]:
    """Specyfikacje modeli obowiązujące dla doby wydania 'day' (harmonogram identyfikacji)."""
    specs = [m.model_dump() for m in cfg.models]
    entries = [e for e in cfg.identified_schedule if e.valid_from <= day]
    if not cfg.identified_schedule:
        return specs
    if not entries:
        raise ValueError(f"brak parametrów z identyfikacji obowiązujących dla doby {day}")
    entry = entries[-1]
    identified = _read_yaml(entry.params)
    _check_as_of(identified, entry.valid_from, entry.params)
    return apply_identified(specs, identified)


def identification_files(cfg: ExperimentConfig, project_root: Path | None = None) -> list[Path]:
    """Pliki identyfikacji użyte przez eksperyment (pojedynczy albo z harmonogramu)."""
    files = [e.params for e in cfg.identified_schedule]
    if cfg.identified_params:
        p = cfg.identified_params
        if not p.is_absolute() and project_root is not None:
            p = project_root / p
        files.append(p)
    return files


def check_identification_data(cfg: ExperimentConfig, dataset_sha256: str) -> None:
    """Każdy plik identyfikacji musi pochodzić z tych samych danych co przebieg."""
    root = _project_root(cfg.dataset) if cfg.identified_params else None
    for f in identification_files(cfg, root):
        got = _read_yaml(f).get("dataset_sha256")
        if got != dataset_sha256:
            raise RuntimeError(
                f"{f}: identyfikację wykonano na innych danych (odcisk {got}, przebieg "
                f"{dataset_sha256}); powtórz identyfikację"
            )


def _project_root(config_path: Path) -> Path:
    """Katalog projektu: pierwszy przodek zawierający pyproject.toml."""
    for parent in config_path.resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return config_path.resolve().parent
