"""Model dwustopniowy: model pierwszego stopnia + SARIMA na jego resztach (rys. 1 artykułu)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from prognozowanie.models.base import Forecaster, build_model, check_future, register
from prognozowanie.models.statistical import Sarima


@register("two_stage")
class TwoStage(Forecaster):
    """params: first: {type: ...}, residual: {type: sarima, ...} lub residual: null
    (wariant ablacyjny bez korekty ARIMA)."""

    allowed_params = frozenset({"first", "residual"})

    def __init__(self, params: dict[str, Any] | None = None, seed: int = 0) -> None:
        super().__init__(params, seed)
        self.first = build_model(dict(self.params["first"]), seed=seed)
        res = self.params.get("residual")
        self.residual = Sarima(dict(res), seed=seed) if res else None
        self.uses_temperature = self.first.uses_temperature  # type: ignore[misc]
        self._res_fc: np.ndarray | None = None

    def fit(self, hist: pd.DataFrame) -> TwoStage:
        self.first.fit(hist)
        f1 = self.first.fitted()
        self._index = hist.index
        if self.residual is None:
            self._fitted = f1
            return self
        resid = (hist["load"] - f1).dropna()
        self.residual.fit_series(resid)
        self._fitted = f1 + self.residual.fitted().reindex(hist.index)
        self._res_fc = None
        return self

    def first_stage_forecast(self, fut: pd.DataFrame) -> np.ndarray:
        return self.first.predict(fut)

    def residual_forecast(self, h: int) -> np.ndarray:
        if self.residual is None:
            return np.zeros(h)
        # prognoza reszt nie zależy od temperatury: liczona raz, używana dla wszystkich
        # scenariuszy temperatury
        if self._res_fc is None or len(self._res_fc) < h:
            self._res_fc = self.residual.forecast(h)
        return self._res_fc[:h]

    def predict(self, fut: pd.DataFrame) -> np.ndarray:
        check_future(fut)
        if fut.index[0] != self._index[-1] + pd.Timedelta(hours=1):
            raise ValueError("prognoza musi zaczynać się tuż po końcu okna")
        return self.first_stage_forecast(fut) + self.residual_forecast(len(fut))
