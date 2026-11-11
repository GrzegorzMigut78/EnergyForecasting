"""Raport z przebiegu (sekcja 10 metodyki): tabele w przekrojach, testy, wykresy.

Wszystkie tabele zapisują się jako CSV, raport zbiorczy jako Markdown, wykresy jako PNG
z danymi w CSV obok. Porównania są liczone wyłącznie na dobach wspólnych dla wszystkich
modeli (bez dób rozgrzewki i bez dób, w których któryś model zawiódł).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from prognozowanie.config import ExperimentConfig
from prognozowanie.evaluate import analyses as an
from prognozowanie.evaluate import figures as fg
from prognozowanie.evaluate import inference as inf
from prognozowanie.evaluate import metrics as mt

log = logging.getLogger(__name__)
DAY2 = "D+1 (25-48 h)"
BLOCK1 = "D (1-24 h)"


def failures(run_dir: Path) -> pd.DataFrame:
    """Doby, w których model zakończył się błędem (z metadanych par model x doba)."""
    rows = []
    for f in sorted((run_dir / "meta").glob("*/*.json")):
        info = json.loads(f.read_text(encoding="utf-8"))
        if info.get("status") != "ok":
            rows.append(
                {"origin": info["origin"], "model": info["model"], "error": info.get("error")}
            )
    return pd.DataFrame(rows, columns=["origin", "model", "error"])


def weight_stability(weights: pd.DataFrame) -> pd.DataFrame:
    w = weights[(weights["scenario"] == "oracle") & (weights["draw"] == 0)]
    w = w[w["method"].isin(["C0", "C1", "C2", "C5", "C6"])]
    if "fitted_hours" not in w.columns:
        w = w.assign(fitted_hours=pd.NA)
    cols = [c for c in w.columns if c.startswith("w_")]
    rows = []
    for key, g in w.groupby(["pool", "method", "k", "fitted_hours", "block"], dropna=False):
        g = g.sort_values("origin")
        vals = g[cols].dropna(axis=1, how="all")
        rows.append(
            {
                "pool": key[0],
                "method": key[1],
                "k": key[2],
                "fitted_hours": key[3],
                "block": key[4],
                "days": len(g),
                "mean_abs_daily_change": float(vals.diff().abs().sum(axis=1).iloc[1:].mean()),
                "share_negative": float((vals < 0).to_numpy().mean()),
                **{f"mean_{c[2:]}": float(vals[c].mean()) for c in vals.columns},
                **{f"sd_{c[2:]}": float(vals[c].std()) for c in vals.columns},
            }
        )
    return pd.DataFrame(rows)


def evaluate_run(cfg: ExperimentConfig, run_dir: Path, scenario: str = "oracle") -> Path:
    from prognozowanie.backtest import load_archive

    actuals = pd.read_parquet(run_dir / "actuals.parquet")
    # tylko modele z bieżącej konfiguracji (archiwum może zawierać modele usunięte z listy)
    fc = load_archive(run_dir, models=[m.name for m in cfg.models])
    fc = fc[fc["kind"] == "forecast"]
    comb_path = run_dir / "combined.parquet"
    if comb_path.exists():
        fc = pd.concat([fc, pd.read_parquet(comb_path)], ignore_index=True)

    test = pd.date_range(cfg.periods.test.start, cfg.periods.test.end, freq="D")
    fc = fc[fc["origin"].isin(test)]
    scale = mt.mase_scale(actuals["load"], test, cfg.forecast.training_window_days)
    df = mt.attach_actuals(fc, actuals).merge(scale, left_on="origin", right_index=True)

    out_dir = run_dir / "report"
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    # w scenariuszu losowym miary są uśrednione po wszystkich losowaniach
    scen_df = df[df["scenario"] == scenario]
    # wspólne pary (doba wydania, blok): każdy blok porównywany na tym samym zbiorze dób
    keys = pd.MultiIndex.from_frame(scen_df[["origin", "block"]])
    scen_df = scen_df.assign(_key=list(keys))
    per_model = scen_df.groupby("model")["_key"].unique().map(set)
    common = set.intersection(*per_model.to_list()) if len(per_model) else set()
    longest = max(map(len, per_model)) if len(per_model) else 0
    short = {m: len(o) // 2 for m, o in per_model.items() if len(o) < longest}
    scen_df = scen_df[scen_df["_key"].isin(common)].drop(columns="_key")
    common_days = {b: sum(1 for _, bb in common if bb == b) for b in (BLOCK1, DAY2)}
    d2 = scen_df[scen_df["block"] == DAY2]

    fails = failures(run_dir)
    fails.to_csv(out_dir / "failures.csv", index=False)

    main = mt.summary(scen_df, ["block", "model"]).reset_index()
    main.to_csv(out_dir / f"summary_{scenario}.csv", index=False)
    mt.summary(scen_df, ["block", "day_type", "model"]).reset_index().to_csv(
        out_dir / f"by_day_type_{scenario}.csv", index=False
    )
    mt.summary(
        d2.assign(month=d2["target"].dt.to_period("M").astype(str)), ["month", "model"]
    ).reset_index().to_csv(out_dir / f"by_month_{scenario}.csv", index=False)
    mt.summary(scen_df, ["horizon", "model"]).reset_index().to_csv(
        out_dir / f"by_horizon_{scenario}.csv", index=False
    )
    common_d2 = {o for o, b in common if b == DAY2}
    mt.summary(
        df[(df["block"] == DAY2) & df["origin"].isin(common_d2)], ["scenario", "model"]
    ).reset_index().to_csv(out_dir / "by_scenario_day2.csv", index=False)
    # wrażliwość na błąd prognozy temperatury: MAE doby D+1 wg poziomu błędu (0 = oracle);
    # persystencja jako punkt kontrolny w tej samej tabeli (bez wartości na osi)
    noise_levels = {s.name: s.sd_day2 for s in cfg.temperature_scenarios if s.kind == "noisy"}
    sens = None
    if noise_levels:
        d2all = df[(df["block"] == DAY2) & df["origin"].isin(common_d2)]
        sens = mt.summary(d2all, ["scenario", "model"]).reset_index()
        level = {"oracle": 0.0, **noise_levels}
        sens["sd_day2"] = sens["scenario"].map(level)
        sens.to_csv(out_dir / "noise_sensitivity_day2.csv", index=False)
    # wrażliwość: ocena bez godzin oznaczonych jako podejrzane (flaga w konfiguracji zbioru)
    clean = scen_df[scen_df["suspicious"] == 0]
    mt.summary(clean, ["block", "model"]).reset_index().to_csv(
        out_dir / f"summary_without_flagged_{scenario}.csv", index=False
    )

    # testy na dziennych stratach doby D+1 (średnia po losowaniach w scenariuszu losowym)
    losses = mt.daily_loss(d2)
    ref = "C3" if "C3" in losses.columns else losses.columns[0]
    dm = mt.dm_table(losses, ref)
    # rodzina porównań z odniesieniem: poprawka Holma (plan analizy, sekcja 5)
    dm["p_holm"] = an.holm(dm["p_value"])
    dm.to_csv(out_dir / f"dm_vs_{ref}_{scenario}.csv")
    benchmarks = [m.name for m in cfg.models if m.role == "benchmark"]
    fams = an.primary_families(
        cfg.combination.base_models, benchmarks, cfg.combination.k_days, cfg.combination.methods
    )
    prim = an.primary_tests(losses, fams)
    if not prim.empty:
        prim.to_csv(out_dir / f"primary_tests_{scenario}.csv", index=False)
    # MCS i bootstrap w analizie głównej: modele pojedyncze i łączenia puli głównej
    pset = an.primary_set(cfg.combination.base_models, benchmarks, list(losses.columns))
    plosses = losses[pset]
    enough = len(plosses.dropna()) >= 30
    ci = inf.bootstrap_ci(plosses, reference=ref, seed=cfg.seed) if enough else None
    mcs = inf.model_confidence_set(plosses, seed=cfg.seed) if enough else None
    if ci is not None:
        ci.to_csv(out_dir / f"bootstrap_ci_{scenario}.csv")
        mcs.to_csv(out_dir / f"mcs_{scenario}.csv")

    weights_path = run_dir / "weights.parquet"
    stab = None
    if weights_path.exists():
        weights = pd.read_parquet(weights_path)
        weights = weights[weights["origin"].isin(test)]  # bez dób rozgrzewki
        stab = weight_stability(weights)
        stab.to_csv(out_dir / "weight_stability.csv", index=False)
        for method in ("C1", "C2"):
            k = 7 if 7 in cfg.combination.k_days else cfg.combination.k_days[0]
            fg.weights_over_time(
                weights, method, k, out_dir / "figures" / f"weights_{method}_K{k}.png"
            )

    # analizy dodatkowe (plan analizy, sekcja 4)
    extra: dict[str, pd.DataFrame] = {}
    kp = an.k_profile(main)
    if not kp.empty:
        kp.to_csv(out_dir / f"k_profile_{scenario}.csv", index=False)
        extra["k"] = kp
        km = kp[kp["pool"] == "main"].pivot(index="k", columns="method", values="MAE")
        km["C3"] = kp.loc[kp["pool"] == "main", "MAE_C3"].iloc[0]
        fg.lines(
            km,
            out_dir / "figures" / f"k_profile_{scenario}.png",
            "MAE doby D+1 a długość historii K (pula główna)",
            "K (doby historii)",
            "MAE",
        )
    combos_main = sorted(m for m in scen_df["model"].unique() if m.startswith("C") and "/" not in m)
    sw_days, sw = an.single_window(scen_df, cfg.combination.base_models, combos_main)
    if not sw.empty:
        sw_days.to_csv(out_dir / f"single_window_days_{scenario}.csv")
        sw.to_csv(out_dir / f"single_window_{scenario}.csv", index=False)
        extra["sw"] = sw
        show_sw = {c: sw_days[f"diff_{c}"] for c in ("C0", "C3") if f"diff_{c}" in sw_days}
        if show_sw:
            fg.histograms(
                show_sw,
                out_dir / "figures" / f"single_window_{scenario}.png",
                "Pojedyncze okno 48 h: łączenie minus najlepszy model bazowy (po fakcie)",
                "różnica MAE (ujemna: łączenie lepsze)",
            )
    div = an.diversity(scen_df, cfg.combination.pools())
    if not div.empty:
        div.to_csv(out_dir / f"diversity_{scenario}.csv", index=False)
        extra["div"] = div
    abl = an.arima_ablation(losses)
    if not abl.empty:
        abl.to_csv(out_dir / f"arima_ablation_{scenario}.csv", index=False)
        extra["abl"] = abl
    cost = an.compute_cost(run_dir)
    if not cost.empty:
        cost.to_csv(out_dir / "compute_cost.csv")
    if sens is not None:
        be = an.break_even(sens)
        if not be.empty:
            be.to_csv(out_dir / "break_even_temperature.csv", index=False)
            extra["be"] = be

    # wykresy: modele bazowe, model naiwny, średnia i najlepsze warianty łączenia
    piv = main.pivot(index="model", columns="block", values="MAE")
    best_comb = [m for m in piv.sort_values(DAY2).index if m.startswith(("C", "plus"))][:2]
    show = [m for m in [*cfg.combination.base_models, "SNAIVE", "C3", *best_comb] if m in piv.index]
    show = list(dict.fromkeys(show))[:8]
    fg.mae_by_horizon(scen_df, show, out_dir / "figures" / f"mae_by_horizon_{scenario}.png")
    fg.mae_by_month(d2, show, out_dir / "figures" / f"mae_by_month_{scenario}.png")
    if sens is not None:
        curve = sens.dropna(subset=["sd_day2"]).pivot(
            index="sd_day2", columns="model", values="MAE"
        )
        curve = curve[[m for m in show if m in curve.columns]]
        fg.lines(
            curve,
            out_dir / "figures" / "noise_sensitivity_day2.png",
            "MAE doby D+1 a błąd prognozy temperatury",
            "odchylenie standardowe błędu temperatury dla D+1 (0 = temperatura zmierzona)",
            "MAE",
        )

    # raport zbiorczy
    md = [f"# Raport: {cfg.name} ({run_dir.name}), scenariusz temperatury: {scenario}", ""]
    md += [
        f"Doby testowe: {test.min():%Y-%m-%d} - {test.max():%Y-%m-%d}; doby wspólne dla "
        f"wszystkich modeli (podstawa porównania): blok D {common_days[BLOCK1]}, "
        f"blok D+1 {common_days[DAY2]}.",
        "",
    ]
    if short:
        md += [f"Modele z niepełną historią (liczba dób): {short}", ""]
    if not fails.empty:
        md += [
            "Błędy modeli (doby):",
            "",
            fails.groupby("model").size().rename("doby").to_frame().to_markdown(),
            "",
        ]
    tab = main.pivot(index="model", columns="block", values=["MAE", "RMSE", "MASE"])
    tab.columns = [f"{a} {b}" for a, b in tab.columns]
    md += [
        "## Miary błędu wg bloku horyzontu",
        "",
        tab.sort_values(f"MAE {DAY2}").round(3).to_markdown(),
        "",
    ]
    if not prim.empty:
        md += [
            "## Hipotezy główne (plan analizy): DM wobec C3, doba D+1, Holm w rodzinie",
            "",
            prim.round(4).to_markdown(index=False),
            "",
        ]
    md += [
        f"## Test Diebolda-Mariano, doba D+1, wobec {ref} (wszystkie warianty, opisowo)",
        "",
        "Ujemna średnia różnica: model lepszy od odniesienia.",
        "",
        dm.round(4).to_markdown(),
        "",
    ]
    if ci is not None:
        md += [
            f"## Przedziały bootstrapowe 95% (bloki 7 dób), doba D+1, różnica wobec {ref}",
            "",
            ci.sort_values("mean_loss").round(3).to_markdown(),
            "",
        ]
        md += [
            f"## Zbiór modeli najlepszych (MCS, alfa = {mcs.attrs['alpha']})",
            "",
            mcs.round(4).to_markdown(),
            "",
        ]
    else:
        md += ["Za mało dób (< 30) na bootstrap i MCS.", ""]
    if stab is not None and not stab.empty:
        md += [
            "## Stabilność wag (doba D+1)",
            "",
            stab[stab["block"] == 2].round(3).to_markdown(index=False),
            "",
        ]
    if "sw" in extra:
        md += [
            "## Pojedyncze okno 48 h (układ pierwotnej wersji artykułu)",
            "",
            "Odsetek dób, w których łączenie ma mniejsze MAE z 48 h niż najlepszy model bazowy "
            "wybrany po fakcie.",
            "",
            extra["sw"].round(3).to_markdown(index=False),
            "",
        ]
    if "div" in extra:
        md += [
            "## Różnorodność błędów (doba D+1)",
            "",
            "MSE średniej = średni MSE modeli - różnorodność.",
            "",
            extra["div"].round(3).to_markdown(index=False),
            "",
        ]
    if "abl" in extra:
        md += [
            "## Korekta ARIMA: model dwustopniowy a pierwszy stopień (doba D+1)",
            "",
            extra["abl"].round(4).to_markdown(index=False),
            "",
        ]
    if "be" in extra:
        md += [
            "## Próg opłacalności prognozy temperatury (doba D+1)",
            "",
            extra["be"].round(3).to_markdown(index=False),
            "",
        ]
    md += ["Wykresy: katalog `figures/` (PNG + dane w CSV).", ""]
    (out_dir / f"report_{scenario}.md").write_text("\n".join(md), encoding="utf-8")
    log.info("raport zapisany w %s", out_dir)
    return out_dir


__all__ = ["evaluate_run", "failures", "weight_stability"]
