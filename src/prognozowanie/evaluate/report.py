"""Raport z przebiegu (sekcja 10 metodyki): tabele w przekrojach, testy, wykresy.

Wszystkie tabele zapisują się jako CSV, raport zbiorczy jako Markdown, wykresy jako PNG
z danymi w CSV obok. Porównania są liczone wyłącznie na dobach wspólnych dla wszystkich
modeli (bez dób rozgrzewki i bez dób, w których któryś model zawiódł).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
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


def _no_temperature_models(cfg: ExperimentConfig) -> list[str]:
    """Modele, które z definicji nie korzystają z temperatury (wejścia modelu)."""
    from prognozowanie.models import build_model

    out = []
    for m in cfg.models:
        try:
            if not build_model(m.model_dump(), seed=0).uses_temperature:
                out.append(m.name)
        except Exception:  # model niebudowalny bez danych: pomijamy w liście
            pass
    return out


def _common(frame: pd.DataFrame, models: list[str] | None = None):
    """Wspólne pary (doba wydania, blok) dla podanych modeli (None = wszystkie obecne)."""
    if models is not None:
        frame = frame[frame["model"].isin(models)]
    keys = pd.MultiIndex.from_frame(frame[["origin", "block"]])
    frame = frame.assign(_key=list(keys))
    per_model = frame.groupby("model")["_key"].unique().map(set)
    common = set.intersection(*per_model.to_list()) if len(per_model) else set()
    out = frame[frame["_key"].isin(common)].drop(columns="_key")
    return out, common, per_model


def evaluate_run(cfg: ExperimentConfig, run_dir: Path, scenario: str = "oracle") -> Path:
    from prognozowanie.backtest import load_archive
    from prognozowanie.combine.runner import (
        leave_one_day_out,
        load_combined,
        method_variants,
    )

    actuals = pd.read_parquet(run_dir / "actuals.parquet")
    # tylko modele z bieżącej konfiguracji (archiwum może zawierać modele usunięte z listy)
    archive = load_archive(run_dir, models=[m.name for m in cfg.models])
    test = pd.date_range(cfg.periods.test.start, cfg.periods.test.end, freq="D")
    arch_fc = archive[(archive["kind"] == "forecast") & archive["origin"].isin(test)]
    scale = mt.mase_scale(actuals["load"], test, cfg.forecast.training_window_days)
    excluded = [(pd.Timestamp(p.start), pd.Timestamp(p.end)) for p in cfg.evaluation.excluded_data]

    def scenario_frame(scen: str) -> pd.DataFrame:
        parts = [arch_fc[arch_fc["scenario"] == scen]]
        comb = load_combined(run_dir, [scen])
        legacy = run_dir / "combined.parquet"  # stary układ: jeden plik łączenia
        if comb.empty and legacy.exists():
            comb = pd.read_parquet(legacy)
            comb = comb[comb["scenario"] == scen]
        if not comb.empty:
            parts.append(comb[comb["origin"].isin(test)])
        fc = pd.concat(parts, ignore_index=True)
        # plan, sekcja 1: bez prognoz, których okna uczące (także w historii wag) obejmują
        # wyłączone okresy danych (kwiecień 2002 bez potwierdzenia)
        fc = mt.attach_actuals(fc, actuals)  # tu powstaje kolumna block
        fc = fc[an.clean_of_excluded(fc, excluded, cfg.forecast.training_window_days)]
        return fc.merge(scale, left_on="origin", right_index=True)

    out_dir = run_dir / "report"
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    fams = cfg.evaluation.families
    ref = cfg.evaluation.reference

    # w scenariuszu losowym miary są uśrednione po wszystkich losowaniach
    scen_raw = scenario_frame(scenario)
    present = sorted(scen_raw["model"].unique())
    # analiza główna: warianty z rodzin hipotez, na ich własnych wspólnych dobach, żeby brak
    # wariantu dodatkowego (np. pula rozszerzona) nie wycinał dób z analizy głównej
    main_models = an.primary_set(fams, present) if fams else present
    scen_main, common_main, _ = _common(scen_raw, main_models)
    # zestawienie wszystkich wariantów (opisowo) na dobach analizy głównej: warianty z kompletem
    # tych dób na wspólnym zbiorze, warianty z niepełną historią (albo wyłączone regułą
    # kwietnia) na swoich dobach, z liczbą dób w kolumnie n; nie zawężają zbioru pozostałym
    keys = pd.MultiIndex.from_frame(scen_raw[["origin", "block"]])
    on_main = scen_raw[keys.isin(list(common_main))]
    per_model = on_main.assign(_key=list(pd.MultiIndex.from_frame(on_main[["origin", "block"]])))
    per_model = per_model.groupby("model")["_key"].unique().map(set)
    complete = [m for m, o in per_model.items() if o >= common_main]
    short = {m: len(o) // 2 for m, o in per_model.items() if not o >= common_main}
    scen_df, common, _ = _common(on_main, complete)
    common_days = {b: sum(1 for _, bb in common_main if bb == b) for b in (BLOCK1, DAY2)}
    used_all = {m: sum(1 for _, bb in o if bb == DAY2) for m, o in per_model.items()}
    d2 = scen_main[scen_main["block"] == DAY2]

    fails = failures(run_dir)
    fails.to_csv(out_dir / "failures.csv", index=False)

    planned = [m.name for m in cfg.models]
    for pool, _models in cfg.combination.pools().items():
        prefix = f"{pool}/" if pool else ""
        planned += [prefix + label for *_, label in method_variants(cfg)]
    cov = an.coverage(scen_raw, planned, len(test), set(main_models), common_days[DAY2], used_all)
    cov.to_csv(out_dir / f"coverage_{scenario}.csv", index=False)

    main = mt.summary(scen_main, ["block", "model"]).reset_index()
    main.to_csv(out_dir / f"summary_main_{scenario}.csv", index=False)
    summary_all = mt.summary(on_main, ["block", "model"]).reset_index()
    summary_all["full_main_days"] = summary_all["model"].isin(complete)
    summary_all.to_csv(out_dir / f"summary_{scenario}.csv", index=False)
    mt.summary(scen_main, ["block", "day_type", "model"]).reset_index().to_csv(
        out_dir / f"by_day_type_{scenario}.csv", index=False
    )
    day_counts = scen_main[scen_main["block"] == DAY2].groupby("day_type")["origin"].nunique()
    day_counts.rename("issue_days").to_csv(out_dir / f"day_type_counts_{scenario}.csv")
    mt.summary(
        d2.assign(month=d2["target"].dt.to_period("M").astype(str)), ["month", "model"]
    ).reset_index().to_csv(out_dir / f"by_month_{scenario}.csv", index=False)
    mt.summary(scen_main, ["horizon", "model"]).reset_index().to_csv(
        out_dir / f"by_horizon_{scenario}.csv", index=False
    )
    common_d2 = {o for o, b in common_main if b == DAY2}
    # wszystkie scenariusze temperatury na dobach analizy głównej, scenariusz po scenariuszu
    # (pełne łączenie wszystkich scenariuszy nie mieści się w pamięci naraz)
    per_scen = []
    for s_name in [t.name for t in cfg.temperature_scenarios]:
        fr = scen_raw if s_name == scenario else scenario_frame(s_name)
        fr = fr[(fr["block"] == DAY2) & fr["origin"].isin(common_d2) & fr["model"].isin(present)]
        if not fr.empty:
            per_scen.append(mt.summary(fr, ["scenario", "model"]).reset_index())
    by_scen = pd.concat(per_scen, ignore_index=True) if per_scen else pd.DataFrame()
    by_scen.to_csv(out_dir / "by_scenario_day2.csv", index=False)
    # wrażliwość na błąd prognozy temperatury: MAE doby D+1 wg poziomu błędu (0 = oracle)
    noise_levels = {s.name: s.sd_day2 for s in cfg.temperature_scenarios if s.kind == "noisy"}
    sens = None
    if noise_levels and not by_scen.empty:
        sens = by_scen.copy()
        level = {"oracle": 0.0, **noise_levels}
        sens["sd_day2"] = sens["scenario"].map(level)
        sens.to_csv(out_dir / "noise_sensitivity_day2.csv", index=False)
    # wrażliwość: ocena bez godzin oznaczonych jako podejrzane (flaga w konfiguracji zbioru)
    clean = scen_main[scen_main["suspicious"] == 0]
    mt.summary(clean, ["block", "model"]).reset_index().to_csv(
        out_dir / f"summary_without_flagged_{scenario}.csv", index=False
    )

    # testy na dziennych stratach doby D+1 (średnia po losowaniach w scenariuszu losowym)
    losses = mt.daily_loss(d2)
    losses_all = mt.daily_loss(on_main[on_main["block"] == DAY2])  # DM parami, na wspólnych dobach
    dm_ref = ref if ref in losses_all.columns else losses_all.columns[0]
    dm = mt.dm_table(losses_all, dm_ref)
    dm["p_holm_all"] = an.holm(dm["p_value"])  # opisowo: cała tabela jako jedna rodzina
    dm.to_csv(out_dir / f"dm_vs_{dm_ref}_{scenario}.csv")
    prim = an.primary_tests(losses, fams) if fams else pd.DataFrame()
    confirmed = pd.DataFrame()
    if not prim.empty:
        prim.to_csv(out_dir / f"primary_tests_{scenario}.csv", index=False)
        if cfg.evaluation.seen_days:  # plan, sekcja 7: bez dób oglądanych przed zamrożeniem
            seen = pd.DatetimeIndex([pd.Timestamp(d) for d in cfg.evaluation.seen_days])
            an.primary_tests(losses[~losses.index.isin(seen)], fams).to_csv(
                out_dir / f"primary_tests_without_seen_days_{scenario}.csv", index=False
            )
        files = {s_: out_dir / f"primary_tests_{s_}.csv" for s_ in cfg.evaluation.confirm_scenarios}
        if all(f.exists() for f in files.values()):
            tables = {s_: pd.read_csv(f) for s_, f in files.items()}
            confirmed = an.confirm_across_scenarios(tables, fams, cfg.evaluation.alpha)
            confirmed.to_csv(out_dir / "primary_confirmed.csv", index=False)
    # MCS i bootstrap w analizie głównej
    pset = [m for m in main_models if m in losses.columns]
    plosses = losses[pset]
    enough = len(plosses.dropna()) >= 30
    boot_ref = ref if ref in pset else pset[0]
    ci = inf.bootstrap_ci(plosses, reference=boot_ref, seed=cfg.seed) if enough else None
    mcs = inf.model_confidence_set(plosses, seed=cfg.seed) if enough else None
    if ci is not None:
        ci.to_csv(out_dir / f"bootstrap_ci_{scenario}.csv")
        mcs.to_csv(out_dir / f"mcs_{scenario}.csv")

    weights = load_combined(run_dir, [scenario], what="weights")
    if weights.empty and (run_dir / "weights.parquet").exists():
        weights = pd.read_parquet(run_dir / "weights.parquet")
        weights = weights[weights["scenario"] == scenario]
    stab = None
    if not weights.empty:
        weights = weights[weights["origin"].isin(test)]  # bez dób rozgrzewki
        if excluded:  # ta sama reguła wyłączonych okresów co dla prognoz
            k_lab = weights["k"].fillna(0).astype(int)
            probe = pd.DataFrame(
                {
                    "origin": weights["origin"],
                    "model": weights["method"] + np.where(k_lab > 0, "_K" + k_lab.astype(str), ""),
                    "block": np.where(weights["block"] == 2, DAY2, BLOCK1),
                },
                index=weights.index,
            )
            win = cfg.forecast.training_window_days
            weights = weights[an.clean_of_excluded(probe, excluded, win)]
        stab = weight_stability(weights)
        stab.to_csv(out_dir / f"weight_stability_{scenario}.csv", index=False)
        for method in ("C1", "C2"):
            k = 2 if 2 in cfg.combination.k_days else cfg.combination.k_days[0]
            fg.weights_over_time(
                weights, method, k, out_dir / "figures" / f"weights_{method}_K{k}_{scenario}.png"
            )

    # analizy dodatkowe (plan analizy, sekcja 4)
    extra: dict[str, pd.DataFrame] = {}
    # profil K na dobach wspólnych dla wszystkich K (porównywalność między K), z C3 puli
    k_models = [m for m in present if an._K.match(m) or m.split("/")[-1] == "C3"]
    k_frame, k_common, _ = _common(on_main, k_models)
    k_days = sum(1 for _, bb in k_common if bb == DAY2)
    kp = an.k_profile(mt.summary(k_frame, ["block", "model"]).reset_index())
    if not kp.empty:
        kp["days"] = k_days
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
    abl = an.arima_ablation(losses_all)
    if not abl.empty:
        abl.to_csv(out_dir / f"arima_ablation_{scenario}.csv", index=False)
        extra["abl"] = abl
    cost = an.compute_cost(run_dir)
    if not cost.empty:
        cost.to_csv(out_dir / "compute_cost.csv")
    if sens is not None:
        # tylko warianty z kompletem dób analizy głównej: porównanie na tych samych dobach
        be = an.break_even(sens[sens["model"].isin(complete)], _no_temperature_models(cfg))
        if not be.empty:
            be.to_csv(out_dir / "break_even_temperature.csv", index=False)
            extra["be"] = be

    canc, canc_hour = an.mae_cancellation(scen_df, cfg.combination.base_models)
    if not canc.empty:
        canc.to_csv(out_dir / f"mae_cancellation_{scenario}.csv", index=False)
        canc_hour.to_csv(out_dir / f"mae_cancellation_by_hour_{scenario}.csv")
        extra["canc"] = canc
    h3_ref = fams["H3"].reference if "H3" in fams else ref
    asym = an.asymmetric_loss(scen_main, h3_ref)
    asym.to_csv(out_dir / f"asymmetric_loss_{scenario}.csv")
    pairs = an.fitted_vs_forecast_pairs(losses_all, weights, cfg.seed)
    if not pairs.empty:
        pairs.to_csv(out_dir / f"fitted_vs_forecast_{scenario}.csv", index=False)
        extra["pairs"] = pairs
    if scenario in cfg.evaluation.confirm_scenarios:
        part = archive[archive["scenario"] == scenario]
        loo = pd.concat(
            [
                leave_one_day_out(part, actuals["load"], cfg.combination.base_models, test, k)
                for k in (2, 7)
                if k in cfg.combination.k_days
            ],
            ignore_index=True,
        )
        if not loo.empty:
            loo.to_csv(out_dir / f"leave_one_day_out_{scenario}.csv", index=False)
            extra["loo"] = loo.groupby("k")[
                ["median_forecast_change", "max_forecast_change", "max_weight_change"]
            ].quantile([0.5, 0.9])

    # wykresy: modele bazowe, model naiwny, średnia i najlepsze warianty łączenia
    piv = summary_all.pivot(index="model", columns="block", values="MAE")
    best_comb = [m for m in piv.sort_values(DAY2).index if m.startswith(("C", "plus"))][:2]
    show = [m for m in [*cfg.combination.base_models, "SNAIVE", "C3", *best_comb] if m in piv.index]
    show = list(dict.fromkeys(show))[:8]
    d2_all = scen_df[scen_df["block"] == DAY2]
    fg.mae_by_horizon(scen_df, show, out_dir / "figures" / f"mae_by_horizon_{scenario}.png")
    fg.mae_by_month(d2_all, show, out_dir / "figures" / f"mae_by_month_{scenario}.png")
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
        f"Doby testowe: {test.min():%Y-%m-%d} - {test.max():%Y-%m-%d} ({len(test)}). "
        f"Analiza główna (warianty z rodzin hipotez) na dobach wspólnych dla tych wariantów: "
        f"blok D {common_days[BLOCK1]}, blok D+1 {common_days[DAY2]}. Zestawienie wszystkich "
        f"wariantów (opisowo) na tych samych dobach; warianty bez kompletu tych dób na swoich "
        f"dobach (kolumna n w summary_{scenario}.csv). "
        f"Pokrycie każdego planowanego wariantu: coverage_{scenario}.csv.",
        "",
    ]
    if excluded:
        okresy = ", ".join(f"{a:%Y-%m-%d} - {b:%Y-%m-%d}" for a, b in excluded)
        md += [
            f"Wyłączone okresy danych: {okresy}. Para (wariant, doba wydania) jest oceniana, "
            "gdy żadne okno uczące jej prognozy, także w historii wag, ich nie obejmuje.",
            "",
        ]
    if short:
        md += [f"Warianty bez kompletu dób analizy głównej (liczba dób D+1): {short}", ""]
    if not fails.empty:
        md += [
            "Błędy modeli (doby):",
            "",
            fails.groupby("model").size().rename("doby").to_frame().to_markdown(),
            "",
        ]
    tab = summary_all.pivot(index="model", columns="block", values=["MAE", "RMSE", "MASE"])
    tab.columns = [f"{a} {b}" for a, b in tab.columns]
    tab["doby D+1"] = pd.Series(used_all).reindex(tab.index)
    md += [
        "## Miary błędu wg bloku horyzontu (wszystkie warianty, opisowo)",
        "",
        "RMSE w scenariuszu losowym: pierwiastek ze średniej MSE po godzinach i losowaniach "
        "(nie średnia z RMSE poszczególnych losowań).",
        "",
        # najpierw warianty na pełnym zbiorze dób, potem te z niepełną historią (mniej dób)
        tab.assign(_full=tab.index.isin(complete))
        .sort_values(["_full", f"MAE {DAY2}"], ascending=[False, True])
        .drop(columns="_full")
        .round(3)
        .to_markdown(),
        "",
    ]
    if not prim.empty:
        md += [
            "## Hipotezy główne (plan analizy): DM, doba D+1, Holm w rodzinie",
            "",
            "Różnica = wariant - odniesienie rodziny (ujemna: wariant lepszy). Kolumny "
            "p_h2, p_bw14, p_no_hln: kontrole konwencji testu.",
            "",
            prim.round(4).to_markdown(index=False),
            "",
        ]
    if not confirmed.empty:
        md += [
            "## Potwierdzenie w scenariuszach " + ", ".join(cfg.evaluation.confirm_scenarios),
            "",
            "Większa z wartości p, Holm w rodzinie, zgodny kierunek różnicy.",
            "",
            confirmed.round(4).to_markdown(index=False),
            "",
        ]
    md += [
        f"## Test Diebolda-Mariano, doba D+1, wobec {dm_ref} (wszystkie warianty, opisowo)",
        "",
        "Ujemna średnia różnica: model lepszy od odniesienia.",
        "",
        dm.round(4).to_markdown(),
        "",
    ]
    if ci is not None:
        md += [
            f"## Przedziały bootstrapowe 95% (bloki 7 dób), doba D+1, różnica wobec {boot_ref}",
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
    if "canc" in extra:
        md += [
            "## Rozkład MAE średniej: średnie MAE modeli - 2 min(P, N) (doba D+1)",
            "",
            f"Godziny z błędami obu znaków: {extra['canc'].attrs['share_mixed_signs']:.1%}.",
            "",
            extra["canc"].round(3).to_markdown(index=False),
            "",
        ]
    if "pairs" in extra:
        md += [
            "## Wartości dopasowane kontra prognozy przy tej samej liczbie obserwacji",
            "",
            extra["pairs"].round(4).to_markdown(index=False),
            "",
        ]
    if "loo" in extra:
        md += [
            "## Usunięcie jednej doby z historii wag C1 (mediana i 90. percentyl po wydaniach)",
            "",
            extra["loo"].round(3).to_markdown(),
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
