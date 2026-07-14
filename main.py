from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from alignment import align_quotes, quality_summary
from collector import collect_live
from config import load_config, load_local_env
from demo_data import generate_demo_quotes
from report import run_research, write_report
from schema import load_quotes, save_quotes
from basis_signal import build_signal_frame
from historical_prescreen import run_historical_prescreen
from factor_strategy import run_factor_prescreen
from factor_collector import analyze_factor_live_costs, collect_factor_live
from tradable_research import run_tradable_research


ROOT = Path(__file__).resolve().parent


def _build(cfg: dict, raw_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    quotes = load_quotes(raw_path)
    aligned = align_quotes(quotes, cfg)
    quality = quality_summary(quotes, aligned)
    signals = build_signal_frame(aligned, cfg)
    return aligned, quality, signals


def _run(
    cfg: dict,
    raw_path: str | Path,
    report_dir: str | Path | None = None,
    processed_dir: str | Path | None = None,
) -> Path:
    aligned, quality, signals = _build(cfg, raw_path)
    processed = (
        Path(processed_dir)
        if processed_dir
        else ROOT / cfg["output"]["processed_dir"]
    )
    processed.mkdir(parents=True, exist_ok=True)
    save_quotes(aligned, processed / "aligned_basis.parquet")
    save_quotes(signals, processed / "signal_frame.parquet")
    research = run_research(signals, cfg, quality)
    output = Path(report_dir) if report_dir else ROOT / cfg["output"]["report_dir"]
    report_path = write_report(research, quality, output)
    print(research["scenarios"].to_string(index=False))
    print(research["checks"].to_string(index=False))
    print(f"Decision: {'GO' if research['go'] else 'FAIL'}")
    print(f"Report: {report_path.resolve()}")
    return report_path


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(
        description="Causal hourly cross-market idiosyncratic reversion research"
    )
    cli.add_argument("--config", default=str(ROOT / "config.yaml"))
    sub = cli.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect-live", help="Collect synchronized Alpaca/Binance L1 quotes")
    collect.add_argument("--duration-hours", type=float, default=0.0, help="0 means run until Ctrl+C")

    build = sub.add_parser("build", help="Validate and align raw quote files")
    build.add_argument("--raw", required=True)
    build.add_argument("--output", default=str(ROOT / "data/processed/aligned_basis.parquet"))

    run = sub.add_parser("run", help="Build, backtest, stress and report")
    run.add_argument("--raw", required=True)
    run.add_argument("--report-dir")

    demo = sub.add_parser("demo", help="Run an end-to-end deterministic plumbing check")
    demo.add_argument("--output", default=str(ROOT / "data/demo/demo_quotes.parquet"))

    historical = sub.add_parser(
        "historical-prescreen",
        help="Download minute bars and run a causal preliminary historical screen",
    )
    historical.add_argument("--start", help="Inclusive UTC date (YYYY-MM-DD)")
    historical.add_argument("--end", help="Inclusive UTC date (YYYY-MM-DD)")
    historical.add_argument("--refresh", action="store_true", help="Ignore matching cache files")
    historical.add_argument("--report-dir")

    factor = sub.add_parser(
        "factor-prescreen",
        help="Run the causal hourly idiosyncratic factor strategy",
    )
    factor.add_argument("--start", help="Inclusive UTC date (YYYY-MM-DD)")
    factor.add_argument("--end", help="Inclusive UTC date (YYYY-MM-DD)")
    factor.add_argument("--refresh", action="store_true", help="Ignore matching cache files")
    factor.add_argument("--report-dir")

    factor_collect = sub.add_parser(
        "collect-factor-live",
        help="Collect Binance L1 books for all factor-strategy target and hedge legs",
    )
    factor_collect.add_argument("--duration-hours", type=float, default=0.0)

    factor_costs = sub.add_parser(
        "analyze-factor-live",
        help="Summarize observed Binance factor-strategy spreads",
    )
    factor_costs.add_argument("--raw", default=str(ROOT / "data/factor_raw"))
    factor_costs.add_argument("--report-dir")

    tradable = sub.add_parser(
        "tradable-research",
        help="Run purged tradable-return research without generating a report",
    )
    tradable.add_argument("--start", help="Inclusive UTC date (YYYY-MM-DD)")
    tradable.add_argument("--end", help="Inclusive UTC date (YYYY-MM-DD)")
    tradable.add_argument("--refresh", action="store_true")
    tradable.add_argument(
        "--category",
        action="append",
        dest="categories",
        help="Target category; repeat to include more than one",
    )
    tradable.add_argument(
        "--symbol",
        action="append",
        dest="symbols",
        help="Canonical target symbol; repeat to include more than one",
    )
    return cli


def main() -> None:
    args = parser().parse_args()
    load_local_env()
    cfg = load_config(args.config)
    if args.command == "collect-live":
        collect_live(cfg, args.duration_hours)
        return
    if args.command == "build":
        aligned, quality, _ = _build(cfg, args.raw)
        save_quotes(aligned, args.output)
        print(quality.to_string(index=False))
        print(f"Aligned output: {Path(args.output).resolve()}")
        return
    if args.command == "run":
        _run(cfg, args.raw, args.report_dir)
        return
    if args.command == "demo":
        raw = generate_demo_quotes(cfg, args.output)
        print(f"Demo raw quotes: {raw.resolve()}")
        _run(
            cfg,
            raw,
            ROOT / "reports/demo",
            ROOT / "data/demo/processed",
        )
        return
    if args.command == "historical-prescreen":
        run_historical_prescreen(
            cfg,
            start_text=args.start,
            end_text=args.end,
            refresh=args.refresh,
            report_dir=args.report_dir,
        )
        return
    if args.command == "factor-prescreen":
        run_factor_prescreen(
            cfg,
            start_text=args.start,
            end_text=args.end,
            refresh=args.refresh,
            report_dir=args.report_dir,
        )
        return
    if args.command == "collect-factor-live":
        collect_factor_live(cfg, args.duration_hours)
        return
    if args.command == "analyze-factor-live":
        analyze_factor_live_costs(cfg, args.raw, args.report_dir)
        return
    if args.command == "tradable-research":
        run_tradable_research(
            cfg,
            start_text=args.start,
            end_text=args.end,
            refresh=args.refresh,
            categories=args.categories,
            symbols=args.symbols,
        )


if __name__ == "__main__":
    main()
