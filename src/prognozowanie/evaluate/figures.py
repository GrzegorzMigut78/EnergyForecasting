"""Wykresy do artykułu (PNG 300 dpi + dane w CSV obok każdego wykresu).

Kolory: paleta kategoryczna sprawdzona pod kątem rozróżnialności przy zaburzeniach widzenia
barw; tożsamość serii niesiona dodatkowo stylem linii (druk czarno-biały) i legendą.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (8, 2))]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def _axes(title: str, xlabel: str, ylabel: str, figsize=(7.0, 3.6)):
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=8.5, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=8.5, color=MUTED)
    ax.tick_params(labelsize=8, colors=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    return fig, ax


def _save(fig, path: Path, data: pd.DataFrame) -> Path:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    data.to_csv(path.with_suffix(".csv"))
    return path


def lines(data: pd.DataFrame, path: Path, title: str, xlabel: str, ylabel: str) -> Path:
    """Wiersze = oś x, kolumny = serie (najwyżej 8; kolejność kolorów stała)."""
    data = data.iloc[:, :8]
    fig, ax = _axes(title, xlabel, ylabel)
    for i, col in enumerate(data.columns):
        ax.plot(
            data.index,
            data[col],
            color=PALETTE[i],
            linestyle=STYLES[i],
            linewidth=1.4,
            label=str(col),
        )
    ax.legend(
        fontsize=7.5,
        frameon=False,
        ncol=min(4, len(data.columns)),
        loc="upper left",
        bbox_to_anchor=(0, -0.18),
    )
    return _save(fig, path, data)


def mae_by_horizon(df: pd.DataFrame, models: list[str], path: Path) -> Path:
    piv = (
        df[df["model"].isin(models)]
        .assign(ae=lambda x: x["error"].abs())
        .pivot_table(index="horizon", columns="model", values="ae", aggfunc="mean")
    )
    piv = piv[[m for m in models if m in piv.columns]]
    return lines(piv, path, "Błąd prognozy rośnie z horyzontem", "Godzina horyzontu (1-48)", "MAE")


def weights_over_time(
    weights: pd.DataFrame,
    method: str,
    k: int,
    path: Path,
    pool: str = "main",
    block: int = 2,
    scenario: str = "oracle",
) -> Path | None:
    w = weights[
        (weights["method"] == method)
        & (weights["k"] == k)
        & (weights["pool"] == pool)
        & (weights["block"] == block)
        & (weights["scenario"] == scenario)
        & (weights["draw"] == 0)
    ]
    if w.empty:
        return None
    cols = [c for c in w.columns if c.startswith("w_") and w[c].notna().any()]
    data = w.set_index("origin")[cols].rename(columns=lambda c: c[2:]).sort_index()
    return lines(
        data, path, f"Wagi {method}, K = {k} dób, doba D+1", "Doba wydania prognozy", "Waga"
    )


def mae_by_month(df: pd.DataFrame, models: list[str], path: Path) -> Path:
    d = df[df["model"].isin(models)].assign(
        ae=lambda x: x["error"].abs(), month=lambda x: x["target"].dt.to_period("M")
    )
    piv = d.pivot_table(index="month", columns="model", values="ae", aggfunc="mean")
    piv = piv[[m for m in models if m in piv.columns]]
    piv.index = piv.index.astype(str)
    return lines(piv, path, "Błąd prognozy w kolejnych miesiącach (doba D+1)", "Miesiąc", "MAE")


def histograms(data: dict[str, pd.Series], path: Path, title: str, xlabel: str) -> Path:
    """Nałożone histogramy (najwyżej 4 serie) z pionową linią w zerze."""
    fig, ax = _axes(title, xlabel, "liczba dób")
    allv = pd.concat(list(data.values()))
    bins = 30 if len(allv) > 60 else 10
    edges = pd.cut(allv, bins=bins, retbins=True)[1]
    for i, (name, v) in enumerate(list(data.items())[:4]):
        ax.hist(v, bins=edges, histtype="step", color=PALETTE[i], linewidth=1.4, label=name)
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.legend(fontsize=7.5, frameon=False)
    return _save(fig, path, pd.DataFrame(data))
