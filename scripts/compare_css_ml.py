"""Czy wybór metody estymacji SARIMA (CSS czy dokładna wiarygodność) zmienia prognozy?

Dla losowych dób okresu testowego dopasowuje ARIMA(1,0,1)(0,1,1)_168 obiema metodami na tym
samym oknie i porównuje błędy prognoz 48 h. Wynik: results/css_vs_ml.csv.

    uv run python scripts/compare_css_ml.py configs/experiments/main.yaml --days 20 --jobs 2
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from prognozowanie.backtest.engine import split_origin
from prognozowanie.config import load_dataset_config, load_experiment_config
from prognozowanie.data import load_dataset
from prognozowanie.models.statistical import Sarima


def one(frame, origin, window, spec):
    hist, fut = split_origin(frame, origin, window, 48)
    actual = frame["load"].reindex(fut.index).to_numpy()
    out = {"origin": origin}
    for method in ("css", "ml"):
        t0 = time.perf_counter()
        m = Sarima({**spec, "method": method}).fit(hist)
        p = m.predict(fut)
        out[f"{method}_seconds"] = time.perf_counter() - t0
        out[f"{method}_mae_d1"] = float(np.mean(np.abs(p[:24] - actual[:24])))
        out[f"{method}_mae_d2"] = float(np.mean(np.abs(p[24:] - actual[24:])))
        out[f"{method}_coef"] = {k: round(v, 4) for k, v in m.coef_.items()}
        out[f"{method}_pred"] = p
    out["max_abs_pred_diff"] = float(np.max(np.abs(out["css_pred"] - out["ml_pred"])))
    out["mean_abs_pred_diff"] = float(np.mean(np.abs(out["css_pred"] - out["ml_pred"])))
    del out["css_pred"], out["ml_pred"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--days", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    cfg = load_experiment_config(a.config)
    ds = load_dataset(load_dataset_config(cfg.dataset))
    test = pd.date_range(cfg.periods.test.start, cfg.periods.test.end, freq="D")
    rng = np.random.default_rng(cfg.seed)
    origins = pd.DatetimeIndex(sorted(rng.choice(test, size=a.days, replace=False)))
    spec = {"order": [1, 0, 1], "seasonal_order": [0, 1, 1], "season": 168}
    path = cfg.execution.output_dir / "css_vs_ml.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    # dokładna wiarygodność zajmuje ok. 5 GB RAM na proces: przy małej pamięci --jobs 1;
    # wynik zapisywany po każdej porcji, więc przerwanie nie traci policzonych dób
    done = pd.read_csv(path, parse_dates=["origin"]) if path.exists() else pd.DataFrame()
    todo = [o for o in origins if done.empty or o not in set(done["origin"])]
    for i in range(0, len(todo), a.jobs):
        batch = todo[i : i + a.jobs]
        rows = Parallel(n_jobs=a.jobs)(
            delayed(one)(ds.frame, o, cfg.forecast.training_window_days, spec) for o in batch
        )
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        done.to_csv(path, index=False)
    df = done[done["origin"].isin(origins)].sort_values("origin")
    print(
        df[["origin", "css_mae_d2", "ml_mae_d2", "mean_abs_pred_diff", "css_seconds", "ml_seconds"]]
        .round(2)
        .to_string(index=False)
    )
    print(
        f"\nŚrednio: MAE D+1 CSS {df.css_mae_d2.mean():.2f}, ML {df.ml_mae_d2.mean():.2f}; "
        f"średnia |różnica prognoz| {df.mean_abs_pred_diff.mean():.2f}; "
        f"czas CSS {df.css_seconds.mean():.1f} s, ML {df.ml_seconds.mean():.1f} s"
    )


if __name__ == "__main__":
    main()
