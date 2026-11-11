"""Prognoza i reszty sezonowej ARIMA z parametrów oszacowanych metodą CSS.

statsforecast w trybie CSS nie ustawia stanu modelu i zwraca zerowe prognozy, dlatego
prognozę liczymy tu standardową rekursją (Box i Jenkins):

    A(B) (y_t - mu) = M(B) e_t,
    A(B) = phi(B) Phi(B^s) (1 - B)^d (1 - B^s)^D,   M(B) = theta(B) Theta(B^s),

z konwencją znaków jak w R/statsforecast: phi(B) = 1 - phi_1 B - ..., theta(B) = 1 + theta_1 B + ...
Przyszłe zaburzenia e przyjmujemy równe zero.
"""

from __future__ import annotations

import numpy as np


def _seasonal(coefs: list[float], s: int, sign: float) -> np.ndarray:
    poly = np.zeros(len(coefs) * s + 1)
    poly[0] = 1.0
    for j, c in enumerate(coefs, start=1):
        poly[j * s] = sign * c
    return poly


def polynomials(coef: dict[str, float], arma: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Zwraca (A, M) w potęgach rosnących B. arma = (p, q, P, Q, s, d, D) jak w statsforecast."""
    p, q, P, Q, s, d, D = (int(x) for x in arma[:7])
    phi = [coef[f"ar{i}"] for i in range(1, p + 1)]
    theta = [coef[f"ma{i}"] for i in range(1, q + 1)]
    sphi = [coef[f"sar{i}"] for i in range(1, P + 1)]
    stheta = [coef[f"sma{i}"] for i in range(1, Q + 1)]
    A = np.convolve(np.r_[1.0, -np.asarray(phi)], _seasonal(sphi, s, -1.0))
    for _ in range(d):
        A = np.convolve(A, [1.0, -1.0])
    for _ in range(D):
        A = np.convolve(A, _seasonal([1.0], s, -1.0))
    M = np.convolve(np.r_[1.0, np.asarray(theta)], _seasonal(stheta, s, 1.0))
    return A, M


def forecast(
    y: np.ndarray, resid: np.ndarray, coef: dict[str, float], arma: tuple[int, ...], h: int
) -> np.ndarray:
    A, M = polynomials(coef, arma)
    mu = float(coef.get("intercept", 0.0))
    n = len(y)
    if n < len(A) or n < len(M):
        raise ValueError("szereg krótszy niż rząd modelu")
    z = np.r_[np.asarray(y, dtype=float) - mu, np.zeros(h)]
    e = np.r_[np.asarray(resid, dtype=float), np.zeros(h)]
    for t in range(n, n + h):
        ar = -np.dot(A[1:], z[t - 1 :: -1][: len(A) - 1])
        ma = np.dot(M[1:], e[t - 1 :: -1][: len(M) - 1])
        z[t] = ar + ma
    return z[n:] + mu


def residuals(y: np.ndarray, coef: dict[str, float], arma: tuple[int, ...]) -> np.ndarray:
    """Reszty CSS (warunkowe, zera dla pierwszych obserwacji); służą do kontroli zgodności."""
    A, M = polynomials(coef, arma)
    mu = float(coef.get("intercept", 0.0))
    z = np.asarray(y, dtype=float) - mu
    n0 = len(A) - 1
    e = np.zeros(len(z))
    q = len(M) - 1
    for t in range(n0, len(z)):
        w = np.dot(A, z[t::-1][: len(A)])
        k = min(q, t)  # na początku szeregu dostępnych jest mniej przeszłych reszt
        ma = np.dot(M[1 : k + 1], e[t - 1 :: -1][:k]) if k > 0 else 0.0
        e[t] = w - ma
    return e
