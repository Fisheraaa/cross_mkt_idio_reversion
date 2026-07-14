from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest import performance_metrics, run_backtest
from config import with_cost_multiplier


def _edge_calibration(trades: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    if len(trades) < 10 or trades["predicted_net_edge_bps"].nunique() < 5:
        return pd.DataFrame(), False
    ranked = trades.copy()
    ranked["edge_bucket"] = pd.qcut(
        ranked["predicted_net_edge_bps"], 5, labels=False, duplicates="drop"
    )
    table = ranked.groupby("edge_bucket", as_index=False).agg(
        predicted_edge_bps=("predicted_net_edge_bps", "mean"),
        actual_net_bps=("net_return_bps", "mean"),
        trades=("net_pnl", "size"),
    )
    monotonic = bool(table["actual_net_bps"].is_monotonic_increasing and len(table) == 5)
    return table, monotonic


def _concentration(trades: pd.DataFrame) -> float:
    if trades.empty:
        return np.inf
    by_symbol = trades.groupby("symbol")["net_pnl"].sum().abs()
    denominator = by_symbol.sum()
    return float(by_symbol.max() / denominator) if denominator > 0 else np.inf


def _positive_month_fraction(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    monthly = trades.assign(month=trades["exit_time"].dt.strftime("%Y-%m")).groupby("month")["net_pnl"].sum()
    return float((monthly > 0).mean()) if len(monthly) else 0.0


def run_research(
    signal_frame: pd.DataFrame,
    cfg: dict[str, Any],
    quality: pd.DataFrame | None = None,
) -> dict[str, Any]:
    baseline_trades = run_backtest(signal_frame, cfg)
    baseline_metrics = performance_metrics(baseline_trades, cfg)

    double_cfg = with_cost_multiplier(cfg, 2.0)
    double_trades = run_backtest(signal_frame, double_cfg)
    double_metrics = performance_metrics(double_trades, double_cfg)

    delay_cfg = with_cost_multiplier(cfg, 1.0)
    delay_cfg["execution"]["delay_seconds"] = 30.0
    delay_trades = run_backtest(signal_frame, delay_cfg)
    delay_metrics = performance_metrics(delay_trades, delay_cfg)

    calibration, monotonic = _edge_calibration(baseline_trades)
    concentration = _concentration(baseline_trades)
    positive_months = _positive_month_fraction(baseline_trades)
    thresholds = cfg["acceptance"]
    valid_frame = signal_frame.loc[signal_frame["quality_ok"]]
    observed_days = int(valid_frame["timestamp"].dt.floor("D").nunique()) if len(valid_frame) else 0
    observed_symbols = int(valid_frame["symbol"].nunique()) if len(valid_frame) else 0
    allowed_sources = set(thresholds.get("allowed_cash_sources_for_go", []))
    observed_cash_sources = set(valid_frame.get("cash_source", pd.Series(dtype=str)).dropna())
    cash_source_eligible = bool(observed_cash_sources and observed_cash_sources.issubset(allowed_sources))
    if quality is not None and quality["aligned_rows"].sum() > 0:
        quality_pass_rate = float(
            quality["quality_rows"].sum() / quality["aligned_rows"].sum()
        )
    else:
        quality_pass_rate = 0.0
    checks = pd.DataFrame(
        [
            {"check": "observed_trading_days", "value": observed_days, "threshold": f">= {thresholds['min_observed_trading_days']}", "pass": observed_days >= int(thresholds["min_observed_trading_days"])},
            {"check": "quality_pass_rate", "value": quality_pass_rate, "threshold": f">= {thresholds['min_quality_pass_rate']}", "pass": quality_pass_rate >= float(thresholds["min_quality_pass_rate"])},
            {"check": "observed_symbols", "value": observed_symbols, "threshold": f">= {thresholds['min_symbols']}", "pass": observed_symbols >= int(thresholds["min_symbols"])},
            {"check": "cash_source_executable", "value": ",".join(sorted(observed_cash_sources)) or "none", "threshold": "configured executable source", "pass": cash_source_eligible},
            {"check": "net_pnl_positive", "value": baseline_metrics["net_pnl"], "threshold": "> 0", "pass": baseline_metrics["net_pnl"] > 0},
            {"check": "net_sharpe", "value": baseline_metrics["sharpe"], "threshold": f">= {thresholds['min_net_sharpe']}", "pass": baseline_metrics["sharpe"] >= float(thresholds["min_net_sharpe"])},
            {"check": "minimum_trades", "value": baseline_metrics["trades"], "threshold": f">= {thresholds['min_trades']}", "pass": baseline_metrics["trades"] >= int(thresholds["min_trades"])},
            {"check": "double_cost_net_positive", "value": double_metrics["net_pnl"], "threshold": "> 0", "pass": double_metrics["net_pnl"] > 0},
            {"check": "delay_30s_net_positive", "value": delay_metrics["net_pnl"], "threshold": "> 0", "pass": delay_metrics["net_pnl"] > 0},
            {"check": "symbol_pnl_concentration", "value": concentration, "threshold": f"<= {thresholds['max_symbol_pnl_concentration']}", "pass": concentration <= float(thresholds["max_symbol_pnl_concentration"])},
            {"check": "positive_month_fraction", "value": positive_months, "threshold": f">= {thresholds['min_positive_month_fraction']}", "pass": positive_months >= float(thresholds["min_positive_month_fraction"])},
            {"check": "edge_bucket_monotonicity", "value": monotonic, "threshold": "true", "pass": monotonic if thresholds.get("require_edge_monotonicity", True) else True},
        ]
    )
    scenarios = pd.DataFrame(
        [
            {"scenario": "baseline", **baseline_metrics},
            {"scenario": "double_modelled_cost", **double_metrics},
            {"scenario": "execution_delay_30s", **delay_metrics},
        ]
    )
    return {
        "trades": baseline_trades,
        "scenarios": scenarios,
        "checks": checks,
        "edge_calibration": calibration,
        "go": bool(checks["pass"].all()),
    }


def write_report(
    research: dict[str, Any],
    quality: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    research["trades"].to_csv(output / "trades.csv", index=False)
    research["scenarios"].to_csv(output / "scenarios.csv", index=False)
    research["checks"].to_csv(output / "acceptance_checks.csv", index=False)
    research["edge_calibration"].to_csv(output / "edge_calibration.csv", index=False)
    quality.to_csv(output / "data_quality.csv", index=False)

    baseline = research["scenarios"].iloc[0]
    status = "GO" if research["go"] else "FAIL"
    lines = [
        "# Synchronized Basis V2 Research Report",
        "",
        f"**Decision: {status}**",
        "",
        "This decision is mechanical. A failed check is never overridden by parameter tuning.",
        "",
        "## Baseline",
        "",
        f"- Trades: {int(baseline['trades'])}",
        f"- Mid-price gross PnL: ${baseline.get('mid_gross_pnl', 0.0):,.2f}",
        f"- Net PnL: ${baseline['net_pnl']:,.2f}",
        f"- Net Sharpe: {baseline['sharpe']:.3f}",
        f"- Annual return: {baseline['annual_return']:.3%}",
        f"- Max drawdown: {baseline['max_drawdown']:.3%}",
        "",
        "## Acceptance Checks",
        "",
        "| Check | Value | Threshold | Pass |",
        "|---|---:|---:|:---:|",
    ]
    for _, row in research["checks"].iterrows():
        value = row["value"]
        display = f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value)
        lines.append(f"| {row['check']} | {display} | {row['threshold']} | {'YES' if row['pass'] else 'NO'} |")
    lines.extend(
        [
            "",
            "## Method Controls",
            "",
            "- Cash and perpetual quotes are paired only after both were received.",
            "- Signal mean and volatility use data through t-1 only.",
            "- Entries and exits use future bid/ask quotes after the configured delay.",
            "- Fees, quoted spread, extra slippage, impact, funding, borrow and opportunity cost are included.",
            "- Double-cost and 30-second-delay scenarios rerun the full trade path.",
            "",
            "Detailed CSV files are stored beside this report.",
        ]
    )
    report_path = output / "basis_v2_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
