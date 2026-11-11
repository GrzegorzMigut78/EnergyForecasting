"""Modele uczenia maszynowego: SVR, perceptron wielowarstwowy (średnia z kilku startów),
LightGBM z opóźnieniami. Standaryzacja liczona wyłącznie na oknie uczącym."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from prognozowanie.models.base import Forecaster, check_future, register
from prognozowanie.models.features import article_features


class _FeatureModel(Forecaster):
    uses_temperature = True

    def _make(self):
        raise NotImplementedError

    def fit(self, hist: pd.DataFrame) -> _FeatureModel:
        self._t0 = hist.index[0]
        X = article_features(hist, self._t0, use_trend=self.params.get("trend", True))
        self._cols = list(X.columns)
        y = hist["load"].to_numpy(dtype=float)
        self._est = self._make()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._est.fit(X.to_numpy(), y)
        self._fitted = pd.Series(self._est.predict(X.to_numpy()), index=hist.index)
        return self

    def predict(self, fut: pd.DataFrame) -> np.ndarray:
        check_future(fut)
        X = article_features(fut, self._t0, use_trend=self.params.get("trend", True))[self._cols]
        return np.asarray(self._est.predict(X.to_numpy()), dtype=float)


@register("svr")
class SVRModel(_FeatureModel):
    """SVR na zmiennych z regresji artykułu. Zmienna celu standaryzowana na oknie uczącym,
    więc C i epsilon są w jednostkach odchylenia standardowego zapotrzebowania. Dzięki temu
    ta sama przestrzeń strojenia działa dla zbiorów o różnej skali (ElecEner ok. 600,
    GEFCom ok. 1,5 mln). Parametry z artykułu (C = 805, epsilon = 5,08 w jednostkach danych)
    odpowiadają przy odchyleniu ok. 67 wartościom C ~ 12 i epsilon ~ 0,076."""

    _SVR_KEYS = ("kernel", "degree", "C", "epsilon", "gamma", "coef0", "cache_size", "max_iter")
    allowed_params = frozenset({*_SVR_KEYS, "trend"})

    def _make(self):
        kw = {k: self.params[k] for k in self._SVR_KEYS if k in self.params}
        kw.setdefault("cache_size", 1000)
        kw.setdefault("max_iter", 2_000_000)  # limit czasu pojedynczego dopasowania
        return TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), SVR(**kw)), transformer=StandardScaler()
        )

    def fit(self, hist: pd.DataFrame) -> SVRModel:
        super().fit(hist)
        svr = self._est.regressor_.steps[-1][1]
        # 0 = zbieżny; 1 = przerwany na limicie max_iter (zapisywane w metadanych przebiegu)
        self.convergence_code = int(getattr(svr, "fit_status_", 0))
        return self


class _MLPAverage:
    """Średnia z kilku sieci różniących się punktem startowym (wynik powtarzalny)."""

    def __init__(self, n_seeds: int, seed: int, **mlp_kw) -> None:
        self.nets = [
            TransformedTargetRegressor(
                regressor=make_pipeline(
                    StandardScaler(), MLPRegressor(random_state=seed + i, **mlp_kw)
                ),
                transformer=StandardScaler(),
            )
            for i in range(n_seeds)
        ]

    def fit(self, X, y):
        for net in self.nets:
            net.fit(X, y)
        return self

    def predict(self, X):
        return np.mean([net.predict(X) for net in self.nets], axis=0)


@register("mlp")
class MLPModel(_FeatureModel):
    _MLP_KEYS = ("hidden_layer_sizes", "activation", "solver", "alpha", "tol", "max_iter")
    allowed_params = frozenset({*_MLP_KEYS, "n_seeds", "trend"})

    def _make(self):
        kw = {k: self.params[k] for k in self._MLP_KEYS if k in self.params}
        if "hidden_layer_sizes" in kw:
            kw["hidden_layer_sizes"] = tuple(np.atleast_1d(kw["hidden_layer_sizes"]))
        kw.setdefault("solver", "lbfgs")
        kw.setdefault("activation", "relu")
        kw.setdefault("max_iter", 2000)
        return _MLPAverage(int(self.params.get("n_seeds", 5)), self.seed, **kw)


@register("lightgbm")
class LightGBMModel(Forecaster):
    """Wzmacniane drzewa z opóźnieniami dostępnymi dla całego horyzontu 48 h
    (domyślnie 48, 72 i 168 godzin), kalendarzem i temperaturą."""

    uses_temperature = True
    allowed_params = None  # 'lags' + dowolne parametry LGBMRegressor (sprawdza sama biblioteka)

    def _frame(self, df: pd.DataFrame, load: pd.Series) -> pd.DataFrame:
        X = pd.DataFrame(index=df.index)
        for lag in self._lags:
            X[f"lag{lag}"] = load.reindex(df.index - pd.Timedelta(hours=lag)).to_numpy()
        for c in ("temp", "holiday", "dow", "hour", "bridge"):
            if c in df.columns:
                X[c] = df[c].to_numpy(dtype=float)
        return X

    def fit(self, hist: pd.DataFrame) -> LightGBMModel:
        import lightgbm as lgb

        self._lags = [int(x) for x in self.params.get("lags", [48, 72, 168])]
        if min(self._lags) < 48:
            raise ValueError("opóźnienia krótsze niż 48 h niedostępne dla całego horyzontu")
        self._hist_load = hist["load"]
        X = self._frame(hist, self._hist_load)
        ok = X.notna().all(axis=1).to_numpy()
        kw = {k: v for k, v in self.params.items() if k != "lags"}
        kw.setdefault("n_estimators", 400)
        kw.setdefault("learning_rate", 0.05)
        kw.setdefault("num_leaves", 31)
        kw.setdefault("verbose", -1)
        self._est = lgb.LGBMRegressor(random_state=self.seed, n_jobs=1, **kw)
        self._est.fit(X[ok], hist["load"].to_numpy()[ok])
        fitted = pd.Series(np.nan, index=hist.index)
        fitted[ok] = self._est.predict(X[ok])
        self._fitted = fitted
        return self

    def predict(self, fut: pd.DataFrame) -> np.ndarray:
        check_future(fut)
        X = self._frame(fut, self._hist_load)
        if X.isna().any().any():
            raise ValueError("brak wartości opóźnionych dla horyzontu prognozy")
        return np.asarray(self._est.predict(X), dtype=float)
