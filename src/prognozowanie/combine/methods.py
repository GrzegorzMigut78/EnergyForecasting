"""Sposoby łączenia prognoz (sekcja 6 metodyki).

Wszystkie funkcje dostają macierz prognoz F (wiersze = godziny, kolumny = modele) i,
jeśli wymagają estymacji, macierz uczącą F_train z wektorem realizacji y_train.
Zwracają (prognozę łączoną, wagi, wyraz wolny).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def ols(F_train, y_train, F):
    """C0 i C1: regresja z wyrazem wolnym, wagi bez ograniczeń."""
    A = np.column_stack([np.ones(len(F_train)), F_train])
    beta, *_ = np.linalg.lstsq(A, y_train, rcond=None)
    return beta[0] + F @ beta[1:], beta[1:], float(beta[0])


def simplex(F_train, y_train, F):
    """C2: wagi nieujemne sumujące się do 1, bez wyrazu wolnego (średnia ważona).

    Rozwiązanie dokładne i deterministyczne: dla każdego podzbioru modeli liczymy
    najmniejsze kwadraty z warunkiem sumy wag = 1 (układ KKT) i wybieramy najlepsze
    rozwiązanie z nieujemnymi wagami. Przy 5 modelach to 31 małych układów równań.
    (SLSQP potrafi po cichu zwrócić punkt startowy, ta metoda nie ma tego trybu awarii.)
    """
    k = F_train.shape[1]
    if k > 10:
        raise ValueError("dokładne wagi C2 przewidziane dla co najwyżej 10 modeli")
    # przy sumie wag = 1: F w - y = E w, gdzie E = macierz błędów prognoz. Praca na błędach
    # (rząd 10-30) zamiast na poziomach (rząd 500) daje dobrze uwarunkowany układ.
    E = F_train - y_train[:, None]
    G = E.T @ E
    best_w, best_sse = None, np.inf
    for size in range(1, k + 1):
        for subset in combinations(range(k), size):
            s = list(subset)
            kkt = np.zeros((size + 1, size + 1))
            kkt[:size, :size] = 2 * G[np.ix_(s, s)]
            kkt[:size, size] = 1.0
            kkt[size, :size] = 1.0
            rhs = np.r_[np.zeros(size), 1.0]
            sol = np.linalg.lstsq(kkt, rhs, rcond=None)[0][:size]
            if np.any(sol < -1e-12):
                continue
            w = np.zeros(k)
            w[s] = np.clip(sol, 0.0, None)
            w /= w.sum()
            sse = float(np.sum((F_train @ w - y_train) ** 2))
            if sse < best_sse:
                best_w, best_sse = w, sse
    return F @ best_w, best_w, 0.0


def mean(F):
    """C3: średnia arytmetyczna."""
    k = F.shape[1]
    return F.mean(axis=1), np.full(k, 1.0 / k), 0.0


def median(F):
    """C4: mediana (nie jest kombinacją liniową; wagi nieokreślone)."""
    return np.median(F, axis=1), np.full(F.shape[1], np.nan), 0.0


def inverse_mse(F_train, y_train, F):
    """C5: wagi odwrotnie proporcjonalne do średniego kwadratu błędu w historii."""
    mse = ((F_train - y_train[:, None]) ** 2).mean(axis=0)
    w = 1.0 / np.maximum(mse, 1e-12)
    w = w / w.sum()
    return F @ w, w, 0.0


def select_best(F_train, y_train, F):
    """C6: wybór zamiast łączenia. Cała waga dla modelu o najmniejszym średnim kwadracie
    błędu w historii (granica C5, gdy różnice błędów rosną); remis: pierwszy model z puli."""
    mse = ((F_train - y_train[:, None]) ** 2).mean(axis=0)
    w = np.zeros(F_train.shape[1])
    w[int(np.argmin(mse))] = 1.0
    return F @ w, w, 0.0
