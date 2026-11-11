"""Wspólny interfejs modeli i rejestr typów.

Każdy model:
  * ``fit(hist)``   uczy się na oknie ``hist`` (indeks godzinowy, kolumna ``load`` i zmienne
                    kalendarzowe oraz ``temp``); widzi wyłącznie dane sprzed chwili prognozy,
  * ``predict(fut)`` zwraca prognozę dla wierszy ``fut`` (te same zmienne, BEZ ``load``),
  * ``fitted()``    zwraca wartości dopasowane w oknie (potrzebne tylko wariantowi C0).

Nowy model = nowa klasa oznaczona dekoratorem ``@register("nazwa_typu")``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np
import pandas as pd

REGISTRY: dict[str, type[Forecaster]] = {}


def register(type_name: str):
    def deco(cls: type[Forecaster]) -> type[Forecaster]:
        if type_name in REGISTRY:
            raise ValueError(f"typ modelu '{type_name}' już zarejestrowany")
        REGISTRY[type_name] = cls
        cls.type_name = type_name
        return cls

    return deco


class Forecaster(ABC):
    type_name: ClassVar[str] = ""
    uses_temperature: ClassVar[bool] = False
    # dozwolone parametry; None = bez kontroli (np. LightGBM przyjmuje własne parametry)
    allowed_params: ClassVar[frozenset[str] | None] = frozenset()

    def __init__(self, params: dict[str, Any] | None = None, seed: int = 0) -> None:
        self.params = dict(params or {})
        if self.allowed_params is not None:
            unknown = set(self.params) - set(self.allowed_params)
            if unknown:
                raise KeyError(
                    f"{type(self).__name__}: nieznane parametry {sorted(unknown)}; "
                    f"dozwolone: {sorted(self.allowed_params)}"
                )
        self.seed = seed
        self._fitted: pd.Series | None = None

    @abstractmethod
    def fit(self, hist: pd.DataFrame) -> Forecaster: ...

    @abstractmethod
    def predict(self, fut: pd.DataFrame) -> np.ndarray: ...

    def fitted(self) -> pd.Series:
        if self._fitted is None:
            raise RuntimeError(f"{type(self).__name__}: brak wartości dopasowanych (wywołaj fit)")
        return self._fitted


def build_model(spec: dict[str, Any], seed: int = 0) -> Forecaster:
    """Tworzy model ze specyfikacji konfiguracyjnej (słownik z kluczem 'type')."""
    spec = dict(spec)
    type_name = spec.pop("type")
    for key in ("name", "role", "provisional"):
        spec.pop(key, None)
    if type_name not in REGISTRY:
        raise KeyError(f"nieznany typ modelu '{type_name}'; dostępne: {sorted(REGISTRY)}")
    return REGISTRY[type_name](spec, seed=seed)


def check_future(fut: pd.DataFrame) -> None:
    """Strażnik: ramka przyszłości nie może zawierać zmiennej celu."""
    if "load" in fut.columns or "load_raw" in fut.columns:
        raise RuntimeError(
            "ramka przyszłości zawiera zapotrzebowanie: naruszenie zasady informacji"
        )
