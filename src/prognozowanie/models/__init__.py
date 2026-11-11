"""Import modułów rejestruje typy modeli w REGISTRY."""

from prognozowanie.models import composite, ml, statistical  # noqa: F401
from prognozowanie.models.base import REGISTRY, Forecaster, build_model

__all__ = ["REGISTRY", "Forecaster", "build_model"]
