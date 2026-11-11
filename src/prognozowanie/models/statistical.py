"""Modele statystyczne: naiwny sezonowy, regresja z artykułu, regresja Honga,
SARIMA (CSS lub dokładna wiarygodność), wygładzanie wykładnicze Holta-Wintersa."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from prognozowanie.models import arima_css
from prognozowanie.models.base import Forecaster, check_future, register
from prognozowanie.models.features import article_features, hong_features


@register("seasonal_naive")
class SeasonalNaive(Forecaster):
    """Wartość z tej samej godziny sprzed ``lag`` godzin (domyślnie tydzień)."""

    allowed_params = frozenset({"lag"})

    def fit(self, hist: pd.DataFrame) -> SeasonalNaive:
        self.lag = int(self.params.get("lag", 168))
        self._y = hist["load"]
        self._fitted = hist["load"].shift(self.lag)
        return self

    def predict(self, fut: pd.DataFrame) -> np.ndarray:
        check_future(fut)
        src = fut.index - pd.Timedelta(hours=self.lag)
        if src.max() >= self._y.index.max() + pd.Timedelta(hours=1):
            raise ValueError("horyzont dłuższy niż opóźnienie modelu naiwnego")
        return self._y.reindex(src).to_numpy(dtype=float)


class _LinearRegression(Forecaster):
    uses_temperature = True
    allowed_params = frozenset({"trend"})

    def _design(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def fit(self, hist: pd.DataFrame) -> _LinearRegression:
        self._t0 = hist.index[0]
        X = self._design(hist)
        self._cols = list(X.columns)
        A = np.column_stack([np.ones(len(X)), X.to_numpy()])
        self.coef_, *_ = np.linalg.lstsq(A, hist["load"].to_numpy(dtype=float), rcond=None)
        self._fitted = pd.Series(A @ self.coef_, index=hist.index)
        return self

    def predict(self, fut: pd.DataFrame) -> np.ndarray:
        check_future(fut)
        X = self._design(fut)[self._cols]
        return np.column_stack([np.ones(len(X)), X.to_numpy()]) @ self.coef_


@register("regression")
class ArticleRegression(_LinearRegression):
    """Regresja wieloraka z artykułu (tabela 1)."""

    def _design(self, df: pd.DataFrame) -> pd.DataFrame:
        return article_features(df, self._t0, use_trend=self.params.get("trend", True))


@register("hong_regression")
class HongRegression(_LinearRegression):
    """Regresja benchmarkowa Honga, wersja dla okna kilku miesięcy i temperatury dobowej."""

    allowed_params = frozenset()

    def _design(self, df: pd.DataFrame) -> pd.DataFrame:
        return hong_features(df, self._t0)


@register("sarima")
class Sarima(Forecaster):
    """Sezonowa ARIMA na samym szeregu (bez zmiennych objaśniających).

    params: order [p,d,q], seasonal_order [P,D,Q], season (np. 168),
            method: "css" (domyślnie), "css-ml" (statsforecast) lub "ml" (statsmodels,
            dokładna wiarygodność, wolna przy długim okresie sezonowym).
    """

    # "type" tolerowany, bo człon ARIMA w modelu dwustopniowym przychodzi z pełną specyfikacją
    allowed_params = frozenset(
        {"type", "order", "seasonal_order", "season", "method", "include_mean"}
    )

    def fit(self, hist: pd.DataFrame) -> Sarima:
        return self.fit_series(hist["load"])

    def fit_series(self, y: pd.Series) -> Sarima:
        order = tuple(self.params.get("order", (1, 0, 1)))
        sorder = tuple(self.params.get("seasonal_order", (0, 1, 1)))
        season = int(self.params.get("season", 168))
        method = self.params.get("method", "css").lower()
        values = y.to_numpy(dtype=float)
        self._index = y.index
        if method in ("css", "css-ml"):
            from statsforecast.models import ARIMA

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # kod zbieżności zapisujemy niżej
                m = ARIMA(
                    order=order,
                    seasonal_order=sorder,
                    season_length=season,
                    method=method.upper(),
                    include_mean=bool(self.params.get("include_mean", True)),
                ).fit(values)
            self.convergence_code = int(m.model_.get("code", 0))
            self._sf = m
            resid = np.asarray(m.model_["residuals"], dtype=float)
            self.coef_ = {k: float(v) for k, v in m.model_["coef"].items()}
            self._arma = tuple(m.model_["arma"])
            self._y, self._resid = values, resid
            self._fitted = pd.Series(values - resid, index=y.index)
            # pierwsze obserwacje nie mają reszt (warunkowanie CSS): brak dopasowania
            n_cond = int(m.model_.get("n_cond", 0))
            self._fitted.iloc[:n_cond] = np.nan
        elif method == "ml":
            from statsmodels.tsa.statespace.sarimax import SARIMAX

            # dokładna wiarygodność dla szeregu po różnicowaniu (simple_differencing):
            # bez tego wymiar stanu ~340 i brak pamięci przy okresie 168
            # stała jak w CSS (include_mean): tylko gdy szereg nie jest różnicowany
            no_diff = order[1] == 0 and sorder[1] == 0
            const = "c" if no_diff and bool(self.params.get("include_mean", True)) else None
            mod = SARIMAX(
                values,
                order=order,
                seasonal_order=(*sorder, season),
                trend=const,
                simple_differencing=True,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = mod.fit(disp=False)
            self._sm = res
            self._diff_poly = arima_css.polynomials({}, (0, 0, 0, 0, season, order[1], sorder[1]))[
                0
            ]
            self._y = values
            self.coef_ = dict(zip(res.param_names, map(float, res.params), strict=True))
            fitted = np.full(len(values), np.nan)
            n_lost = len(self._diff_poly) - 1
            fitted[n_lost:] = values[n_lost:] - res.resid
            self._fitted = pd.Series(fitted, index=y.index)
        else:
            raise ValueError(f"nieznana metoda estymacji SARIMA: {method}")
        self._method = method
        return self

    def forecast(self, h: int) -> np.ndarray:
        if self._method == "ml":
            w = np.asarray(self._sm.forecast(h), dtype=float)  # prognoza szeregu zróżnicowanego
            dp = self._diff_poly
            z = np.r_[self._y, np.zeros(h)]
            n = len(self._y)
            for t in range(n, n + h):  # odwrócenie różnicowania
                z[t] = w[t - n] - np.dot(dp[1:], z[t - 1 :: -1][: len(dp) - 1])
            return z[n:]
        if self._method == "css":
            # statsforecast po CSS nie ustawia stanu modelu: własna rekursja (arima_css)
            return arima_css.forecast(self._y, self._resid, self.coef_, self._arma, h)
        return np.asarray(self._sf.predict(h)["mean"], dtype=float)

    def predict(self, fut: pd.DataFrame) -> np.ndarray:
        check_future(fut)
        expected_start = self._index[-1] + pd.Timedelta(hours=1)
        if fut.index[0] != expected_start:
            raise ValueError("prognoza SARIMA musi zaczynać się tuż po końcu okna")
        return self.forecast(len(fut))


@register("exp_smoothing")
class ExpSmoothing(Forecaster):
    """Wygładzanie wykładnicze Holta-Wintersa: poziom, trend addytywny,
    sezonowość addytywna o okresie ``season``. Stałe wygładzania z optymalizatora
    minimalizującego błąd prognozy na 1 krok (zamiast siatki co 0,001, która dawałaby
    miliard kombinacji). Uwaga metodyczna: przy danych godzinowych optymalizator ustawia
    wysokie alfa, co osłabia prognozy na 25-48 h."""

    allowed_params = frozenset({"trend", "damped", "season", "initialization"})

    def fit(self, hist: pd.DataFrame) -> ExpSmoothing:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        y = hist["load"].to_numpy(dtype=float)
        self._index = hist.index
        trend = self.params.get("trend", "add")
        mod = ExponentialSmoothing(
            y,
            trend=trend,
            damped_trend=bool(self.params.get("damped", False)) and trend is not None,
            seasonal="add",
            seasonal_periods=int(self.params.get("season", 168)),
            initialization_method=self.params.get("initialization", "heuristic"),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._res = mod.fit(optimized=True)
        p = self._res.params

        def _get(key: str) -> float:
            v = p.get(key)
            return float("nan") if v is None else float(v)

        self.coef_ = {
            "alpha": _get("smoothing_level"),
            "gamma_trend": _get("smoothing_trend"),
            "delta_season": _get("smoothing_seasonal"),
            "phi_damping": _get("damping_trend"),
        }
        # zbieżność optymalizatora: 0 = zbieżny, 1 = brak zbieżności (zapisywane w metadanych)
        retvals = getattr(self._res, "mle_retvals", None)
        ok = bool(getattr(retvals, "success", True)) if retvals is not None else True
        self.convergence_code = 0 if ok else 1
        self.coef_["converged"] = ok
        self._fitted = pd.Series(self._res.fittedvalues, index=hist.index)
        return self

    def predict(self, fut: pd.DataFrame) -> np.ndarray:
        check_future(fut)
        return np.asarray(self._res.forecast(len(fut)), dtype=float)
