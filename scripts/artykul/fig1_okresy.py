# Uruchomienie:
#   uv run python scripts/artykul/fig1_okresy.py <plik wyjściowy .png>
"""Figura 1 artykułu: okresy badania ElecEner w skali (u góry) i zbiór informacji jednej
prognozy (u dołu). Daty z configs/experiments/main.yaml i configs/datasets/elecener.yaml."""

import sys
from datetime import date, timedelta

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = sys.argv[1]
first_test = date(2002, 8, 1)
rows = [
    ("Excluded: October 2001", date(2001, 10, 1), date(2001, 10, 31), "#f3d9a4"),
    ("Identification", date(2001, 11, 1), date(2002, 3, 31), "#d9d9d9"),
    ("April 2002: excluded from all windows", date(2002, 4, 1), date(2002, 4, 30), "#f3d9a4"),
    (
        "Training window of the first test origin",
        first_test - timedelta(days=84),
        first_test - timedelta(days=1),
        "#f4b9a0",
    ),
    ("Warm-up: history of combination weights", date(2002, 7, 1), date(2002, 7, 31), "#d9d9d9"),
    ("Test: 333 issue days", date(2002, 8, 1), date(2003, 6, 29), "#9dc3e6"),
]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig = plt.figure(figsize=(7.2, 4.8))
ax = fig.add_axes([0.40, 0.42, 0.54, 0.52])
ax2 = fig.add_axes([0.02, 0.02, 0.96, 0.30])

for i, (lab, a, b, col) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.barh(
        y,
        (b - a).days + 1,
        left=mdates.date2num(a),
        height=0.55,
        color=col,
        edgecolor="#555555",
        linewidth=0.6,
    )
    ax.text(
        mdates.date2num(date(2001, 9, 20)),
        y,
        lab,
        ha="right",
        va="center",
        fontweight="bold" if lab.startswith("Test") else "normal",
    )
ax.set_yticks([])
ax.set_xlim(mdates.date2num(date(2001, 9, 25)), mdates.date2num(date(2003, 7, 10)))
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.grid(axis="x", color="#e5e5e5", linewidth=0.6)
for s in ("left", "right", "top"):
    ax.spines[s].set_visible(False)
ax.set_title("(a) Study periods of ElecEner", loc="left", fontsize=9)

# (b) schemat jednej prognozy
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 2.2)
ax2.axis("off")


def box(x, y, w, txt, fc, bold=False):
    ax2.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            0.5,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            fc=fc,
            ec="#555555",
            lw=0.6,
        )
    )
    ax2.text(
        x + w / 2, y + 0.25, txt, ha="center", va="center", fontweight="bold" if bold else "normal"
    )


ax2.text(0, 1.45, "Models", va="center")
ax2.text(0, 0.55, "Weights", va="center")
box(2.0, 1.2, 3.7, "84-day training window", "#ffffff")
box(6.1, 1.2, 1.65, "Day D (h 1–24)", "#deebf7")
box(7.85, 1.2, 2.1, "Day D+1 (h 25–48)", "#9dc3e6", bold=True)
box(2.95, 0.3, 2.75, "Past forecasts, last K days", "#f4b9a0")
ax2.axvline(5.9, ymin=0.02, ymax=0.98, color="#2f6fb3", ls="--", lw=0.9)
ax2.text(
    5.9,
    0.02,
    "origin: 00:00 of day D",
    ha="center",
    va="bottom",
    fontsize=8,
    color="#2f6fb3",
    bbox={"fc": "white", "ec": "none", "pad": 1},
)
ax2.set_title("(b) Information set of one forecast origin (schematic)", loc="left", fontsize=9)
fig.savefig(OUT, dpi=300)
print(OUT)
