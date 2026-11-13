"""Interfejs wiersza poleceń.

prognozowanie check-data   configs/datasets/elecener.yaml [--export plik.xlsx]
prognozowanie identify     configs/experiments/main.yaml [--steps tune arima]
prognozowanie run          configs/experiments/main.yaml [--run-name X] [--jobs N]
                           [--start 2002-08-01 --end 2002-08-10]
prognozowanie combine      configs/experiments/main.yaml [--run-name X]
prognozowanie evaluate     configs/experiments/main.yaml [--run-name X] [--scenario oracle]
prognozowanie all          configs/experiments/main.yaml   (run + combine + evaluate)
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("numba", "matplotlib", "optuna"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def cmd_check_data(a: argparse.Namespace) -> int:
    from prognozowanie.config import load_dataset_config
    from prognozowanie.data import load_dataset

    ds = load_dataset(load_dataset_config(a.config))
    f = ds.frame
    print(f"Zbiór: {ds.config.name}")
    print(f"Zakres: {f.index.min()} - {f.index.max()}, {len(f)} godzin, {len(ds.days)} dób")
    print(f"SHA-256 pliku źródłowego: {ds.source_sha256}")
    print("Korekty:")
    for entry in ds.log:
        print("  -", {k: v for k, v in entry.items() if k != "reason"})
    days = f.groupby(f.index.normalize()).first()
    print(
        f"Święta (zmienna modelu): {int(days['holiday'].sum())}, "
        f"dni pomostowe: {int(days['bridge'].sum())}, doby zmiany czasu: "
        f"{int(days['dst_day'].sum())}, godziny oznaczone: {int(f['suspicious'].sum())}"
    )
    if a.export:
        f.to_excel(a.export) if a.export.endswith(".xlsx") else f.to_csv(a.export)
        print(f"Zapisano: {a.export}")
    return 0


def _load(a):
    from prognozowanie.config import load_dataset_config, load_experiment_config
    from prognozowanie.data import load_dataset

    cfg = load_experiment_config(a.config)
    provisional = [m.name for m in cfg.models if m.provisional]
    if provisional:
        logging.getLogger("prognozowanie").warning(
            "parametry wstępne (przed identyfikacją): %s", ", ".join(provisional)
        )
    return cfg, load_dataset(load_dataset_config(cfg.dataset))


def cmd_identify(a):
    import datetime as dt

    from prognozowanie.identify import run_identification

    cfg, ds = _load(a)
    as_of = dt.date.fromisoformat(a.as_of) if a.as_of else None
    sub = f"as_of_{as_of}" if as_of else "base"
    out = cfg.execution.output_dir / "identify" / cfg.name / sub
    path = run_identification(cfg, ds, out, steps=tuple(a.steps), as_of=as_of)
    if as_of:
        print(
            f"Wynik: {path}\nDopisz do identified_schedule: {{valid_from: {as_of}, params: {path}}}"
        )
    else:
        print(f"Wynik: {path}\nDopisz do konfiguracji eksperymentu: identified_params: {path}")
    return 0


def cmd_run(a):
    from prognozowanie.backtest import origins_for, run_backtest

    cfg, ds = _load(a)
    origins = origins_for(cfg)
    if a.start:
        origins = origins[origins >= pd.Timestamp(a.start)]
    if a.end:
        origins = origins[origins <= pd.Timestamp(a.end)]
    res = run_backtest(
        cfg,
        ds,
        run_name=a.run_name,
        origins=origins,
        n_jobs=a.jobs,
        recompute=a.recompute,
        allow_provisional=a.allow_provisional or None,
    )
    bad = [r for r in res if r.status != "ok"]
    print(f"Policzone doby: {len(res)}, z problemami: {len(bad)}")
    return 1 if any(r.status == "error" for r in res) else 0


def cmd_combine(a):
    from prognozowanie.backtest.engine import run_dir_for
    from prognozowanie.combine import run_combination
    from prognozowanie.config import load_experiment_config

    cfg = load_experiment_config(a.config)
    run_combination(cfg, run_dir_for(cfg, a.run_name))
    return 0


def cmd_evaluate(a):
    from prognozowanie.backtest.engine import run_dir_for
    from prognozowanie.config import load_experiment_config
    from prognozowanie.evaluate import evaluate_run

    cfg = load_experiment_config(a.config)
    out = evaluate_run(cfg, run_dir_for(cfg, a.run_name), scenario=a.scenario)
    print(f"Raport: {out}")
    return 0


def cmd_all(a):
    rc = cmd_run(a)
    if rc:
        return rc
    cmd_combine(a)
    return cmd_evaluate(a)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="prognozowanie",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("check-data", help="wczytaj zbiór, pokaż korekty i kalendarz")
    s.add_argument("config")
    s.add_argument("--export", help="zapisz przygotowane dane (.xlsx lub .csv)")
    s.set_defaults(func=cmd_check_data)

    s = sub.add_parser("identify", help="strojenie SVR/sieci i wybór rzędów ARIMA")
    s.add_argument("config")
    s.add_argument("--steps", nargs="+", default=["tune", "arima"], choices=["tune", "arima"])
    s.add_argument("--as-of", help="ponowna identyfikacja na danych sprzed tej doby (RRRR-MM-DD)")
    s.set_defaults(func=cmd_identify)

    for name, func, hlp in (("run", cmd_run, "backtest"), ("all", cmd_all, "run+combine+evaluate")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("config")
        s.add_argument("--run-name")
        s.add_argument("--jobs", type=int)
        s.add_argument("--start")
        s.add_argument("--end")
        s.add_argument(
            "--recompute",
            nargs="+",
            metavar="MODEL",
            help="przelicz wskazane modele (np. po zmianie ich specyfikacji); pozostałe zostają",
        )
        s.add_argument(
            "--allow-provisional",
            action="store_true",
            help="dopuść parametry wstępne (tylko próba techniczna, nie wyniki)",
        )
        s.add_argument("--scenario", default="oracle")
        s.set_defaults(func=func)

    s = sub.add_parser("combine", help="łączenie prognoz na gotowym archiwum")
    s.add_argument("config")
    s.add_argument("--run-name")
    s.set_defaults(func=cmd_combine)

    s = sub.add_parser("evaluate", help="miary i testy")
    s.add_argument("config")
    s.add_argument("--run-name")
    s.add_argument("--scenario", default="oracle")
    s.set_defaults(func=cmd_evaluate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return args.func(args)
    except (ValueError, KeyError, FileNotFoundError, RuntimeError) as exc:
        logging.getLogger("prognozowanie").error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
