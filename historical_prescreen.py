from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest import SECONDS_PER_YEAR, performance_metrics
from historical_providers import AlpacaHistoricalBars, BinanceHistoricalMinutes


def _utc_day(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.normalize()


def historical_bounds(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_timestamp = _utc_day(start)
    end_exclusive = _utc_day(end) + pd.Timedelta(days=1)
    if end_exclusive <= start_timestamp:
        raise ValueError("Historical end date must not be earlier than start date")
    return start_timestamp, end_exclusive


def _cache_path(cache_dir: Path, name: str, start: pd.Timestamp, end_exclusive: pd.Timestamp) -> Path:
    start_text = start.strftime("%Y%m%d")
    end_text = (end_exclusive - pd.Timedelta(days=1)).strftime("%Y%m%d")
    return cache_dir / f"{name}_{start_text}_{end_text}.pkl"


def _read_or_fetch(path: Path, refresh: bool, fetch: Any) -> pd.DataFrame:
    if path.exists() and not refresh:
        return pd.read_pickle(path)
    frame = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(path)
    return frame


def download_history(
    cfg: dict[str, Any],
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    refresh: bool = False,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    root = Path(cfg["_root"])
    cache_dir = root / str(cfg["historical"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    start_label = start.strftime("%Y-%m-%d")
    end_label = (end_exclusive - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    errors: list[dict[str, str]] = []

    alpaca_path = _cache_path(cache_dir, "alpaca_cash_1m", start, end_exclusive)
    if alpaca_path.exists() and not refresh:
        cash = pd.read_pickle(alpaca_path)
        print(f"cache Alpaca cash rows={len(cash):,}")
    else:
        print(f"download Alpaca cash 1m {start_label}..{end_label}")
        alpaca = AlpacaHistoricalBars(cfg)
        cash = _read_or_fetch(
            alpaca_path,
            refresh,
            lambda: alpaca.fetch(cfg["symbols"], start, end_exclusive),
        )
        print(f"downloaded Alpaca cash rows={len(cash):,}")

    for canonical in cfg["symbols"]:
        if cash.empty or "symbol" not in cash or not cash["symbol"].eq(canonical).any():
            errors.append(
                {
                    "symbol": canonical,
                    "dataset": "cash",
                    "error": "no rows returned for configured cash symbol",
                }
            )

    binance = BinanceHistoricalMinutes(cfg)
    market: dict[str, dict[str, pd.DataFrame]] = {}
    funding: dict[str, pd.DataFrame] = {}
    for canonical, spec in cfg["symbols"].items():
        provider_symbol = str(spec["perp_symbol"])
        market[canonical] = {}
        for kind in ("perp", "mark", "index"):
            path = _cache_path(
                cache_dir, f"binance_{provider_symbol}_{kind}_1m", start, end_exclusive
            )
            try:
                print(f"{'refresh' if refresh else 'load'} Binance {provider_symbol} {kind}")
                frame = _read_or_fetch(
                    path,
                    refresh,
                    lambda s=provider_symbol, k=kind: binance.fetch_klines(
                        s, k, start, end_exclusive
                    ),
                )
            except Exception as exc:
                frame = pd.DataFrame()
                errors.append({"symbol": canonical, "dataset": kind, "error": str(exc)})
                print(f"warning Binance {provider_symbol} {kind}: {exc}")
            if frame.empty and not any(
                item["symbol"] == canonical and item["dataset"] == kind for item in errors
            ):
                errors.append(
                    {"symbol": canonical, "dataset": kind, "error": "no rows returned"}
                )
            market[canonical][kind] = frame
            print(f"  rows={len(frame):,}")

        path = _cache_path(
            cache_dir, f"binance_{provider_symbol}_funding", start, end_exclusive
        )
        try:
            funding[canonical] = _read_or_fetch(
                path,
                refresh,
                lambda s=provider_symbol: binance.fetch_funding(s, start, end_exclusive),
            )
        except Exception as exc:
            funding[canonical] = pd.DataFrame()
            errors.append({"symbol": canonical, "dataset": "funding", "error": str(exc)})
            print(f"warning Binance {provider_symbol} funding: {exc}")
        if funding[canonical].empty and not any(
            item["symbol"] == canonical and item["dataset"] == "funding" for item in errors
        ):
            errors.append(
                {"symbol": canonical, "dataset": "funding", "error": "no rows returned"}
            )
        print(f"  funding rows={len(funding[canonical]):,}")
    return {"cash": cash, "market": market, "funding": funding}, errors


def _regular_session_mask(timestamp: pd.Series) -> pd.Series:
    local = timestamp.dt.tz_convert("America/New_York")
    minutes = local.dt.hour * 60 + local.dt.minute
    return local.dt.dayofweek.lt(5) & minutes.ge(9 * 60 + 30) & minutes.lt(16 * 60)


def build_historical_frame(
    history: dict[str, Any], cfg: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cash_all = history["cash"]
    frames: list[pd.DataFrame] = []
    quality_rows: list[dict[str, Any]] = []
    for symbol, spec in cfg["symbols"].items():
        cash = cash_all.loc[cash_all.get("symbol", pd.Series(dtype=str)).eq(symbol)].copy()
        cash = cash.loc[_regular_session_mask(cash["timestamp"])] if not cash.empty else cash
        inputs = history["market"].get(symbol, {})
        counts = {"cash_rows": len(cash)}
        renamed: dict[str, pd.DataFrame] = {}
        for kind in ("perp", "mark", "index"):
            source = inputs.get(kind, pd.DataFrame()).copy()
            counts[f"{kind}_rows"] = len(source)
            if not source.empty:
                source = source.loc[_regular_session_mask(source["timestamp"])]
                renamed[kind] = source.rename(
                    columns={column: f"{kind}_{column}" for column in ("open", "high", "low", "close")}
                )
            else:
                renamed[kind] = source
        required = [cash, renamed["perp"], renamed["mark"], renamed["index"]]
        if any(frame.empty for frame in required):
            quality_rows.append(
                {"symbol": symbol, **counts, "aligned_rows": 0, "quality_rows": 0, "trading_days": 0}
            )
            continue
        aligned = cash.rename(
            columns={column: f"cash_{column}" for column in ("open", "high", "low", "close", "volume")}
        )[["timestamp", "cash_open", "cash_high", "cash_low", "cash_close", "cash_volume"]]
        for kind in ("perp", "mark", "index"):
            columns = ["timestamp"] + [f"{kind}_{column}" for column in ("open", "high", "low", "close")]
            aligned = aligned.merge(
                renamed[kind][columns], on="timestamp", how="inner", validate="one_to_one"
            )
        multiplier = float(spec.get("perp_to_cash_multiplier", 1.0))
        aligned["symbol"] = symbol
        aligned["decision_time"] = aligned["timestamp"] + pd.Timedelta(minutes=1)
        aligned["basis"] = np.log(aligned["perp_close"] * multiplier / aligned["cash_close"])
        aligned["basis_open"] = np.log(aligned["perp_open"] * multiplier / aligned["cash_open"])
        aligned["mark_index_basis"] = np.log(aligned["mark_close"] / aligned["index_close"])
        aligned["cash_index_gap"] = np.log(aligned["index_close"] / aligned["cash_close"])
        positive_columns = [
            f"{kind}_{field}"
            for kind in ("cash", "perp", "mark", "index")
            for field in ("open", "close")
        ]
        finite_positive = aligned[positive_columns].gt(0).all(axis=1)
        aligned["quality_ok"] = (
            finite_positive
            & aligned["basis"].abs().mul(10_000).le(float(cfg["alignment"]["max_abs_basis_bps"]))
            & aligned["cash_index_gap"].abs().mul(10_000).le(
                float(cfg["historical"]["max_cash_index_gap_bps"])
            )
        )
        quality_rows.append(
            {
                "symbol": symbol,
                **counts,
                "aligned_rows": len(aligned),
                "quality_rows": int(aligned["quality_ok"].sum()),
                "trading_days": int(aligned.loc[aligned["quality_ok"], "timestamp"].dt.date.nunique()),
            }
        )
        frames.append(aligned)
    if not frames:
        return pd.DataFrame(), pd.DataFrame(quality_rows)
    combined = pd.concat(frames, ignore_index=True).sort_values(["symbol", "timestamp"])
    return combined.reset_index(drop=True), pd.DataFrame(quality_rows)


def historical_costs_bps(cfg: dict[str, Any], multiplier: float = 1.0) -> dict[str, float]:
    execution = cfg["execution"]
    historical = cfg["historical"]
    spread = (
        float(historical["modeled_cash_roundtrip_spread_bps"])
        + float(historical["modeled_perp_roundtrip_spread_bps"])
    ) * multiplier
    fees = 2.0 * (
        float(execution["cash_taker_fee_bps"]) + float(execution["perp_taker_fee_bps"])
    ) * multiplier
    slippage = 2.0 * (
        float(execution["cash_extra_slippage_bps"])
        + float(execution["perp_extra_slippage_bps"])
    ) * multiplier
    impact = 4.0 * float(execution["impact_bps_per_fill"]) * multiplier
    return {
        "modeled_spread_bps": spread,
        "fee_bps": fees,
        "extra_slippage_bps": slippage,
        "impact_bps": impact,
        "total_trading_bps": spread + fees + slippage + impact,
    }


def build_historical_signals(
    aligned: pd.DataFrame, cfg: dict[str, Any], cost_multiplier: float = 1.0
) -> pd.DataFrame:
    if aligned.empty:
        return aligned.copy()
    settings = cfg["signal"]
    cost = historical_costs_bps(cfg, cost_multiplier)
    frames: list[pd.DataFrame] = []
    for _, sample in aligned.groupby("symbol", sort=True):
        sample = sample.sort_values("timestamp").copy()
        valid_basis = sample["basis"].where(sample["quality_ok"])
        history = valid_basis.shift(1)
        mean = history.rolling(
            int(settings["rolling_window"]), min_periods=int(settings["min_history"])
        ).mean()
        std = history.rolling(
            int(settings["rolling_window"]), min_periods=int(settings["min_history"])
        ).std(ddof=1).replace(0.0, np.nan)
        sample["basis_mean_lagged"] = mean
        sample["basis_std_lagged"] = std
        sample["z_score"] = (sample["basis"] - mean) / std
        sample["expected_gross_edge_bps"] = (
            (sample["basis"] - mean).abs() - float(settings["exit_z"]) * std
        ).clip(lower=0.0) * 10_000.0
        sample["cost_hurdle_bps"] = (
            cost["total_trading_bps"] + float(settings["safety_margin_bps"])
        )
        sample["predicted_net_edge_bps"] = (
            sample["expected_gross_edge_bps"] - sample["cost_hurdle_bps"]
        )
        sample["entry_candidate"] = (
            sample["quality_ok"]
            & sample["z_score"].abs().ge(float(settings["entry_z"]))
            & sample["predicted_net_edge_bps"].gt(0.0)
        )
        frames.append(sample)
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "timestamp"])


def _leg_pnl(direction: int, notional: float, entry: float, exit_price: float) -> float:
    return direction * notional / entry * (exit_price - entry)


def _funding_pnl(
    funding: pd.DataFrame,
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    perp_direction: int,
    notional: float,
) -> float:
    if funding.empty:
        return 0.0
    held = funding.loc[
        funding["timestamp"].gt(entry_time) & funding["timestamp"].le(exit_time),
        "funding_rate",
    ]
    return float((-perp_direction * notional * held).sum())


def _close_reason(row: pd.Series, entry_time: pd.Timestamp, cfg: dict[str, Any]) -> str | None:
    if pd.isna(row.get("z_score")):
        return None
    settings = cfg["signal"]
    held_minutes = (row["decision_time"] - entry_time).total_seconds() / 60.0
    if held_minutes < float(settings["min_holding_minutes"]):
        return None
    if abs(float(row["z_score"])) >= float(settings["stop_z"]):
        return "stop"
    if abs(float(row["z_score"])) <= float(settings["exit_z"]):
        return "reversion"
    if held_minutes >= float(settings["max_holding_minutes"]):
        return "timeout"
    return None


def run_historical_backtest(
    signals: pd.DataFrame,
    funding_by_symbol: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
    cost_multiplier: float = 1.0,
    extra_delay_minutes: int = 0,
) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()
    notional = float(cfg["execution"]["notional_per_leg"])
    cost = historical_costs_bps(cfg, cost_multiplier)
    trades: list[dict[str, Any]] = []
    for symbol, symbol_frame in signals.groupby("symbol", sort=True):
        local_dates = symbol_frame["timestamp"].dt.tz_convert("America/New_York").dt.date
        for _, sample in symbol_frame.assign(session_date=local_dates).groupby("session_date", sort=True):
            sample = sample.sort_values("timestamp").reset_index(drop=True)
            index = 0
            while index < len(sample) - 2:
                candidate = sample.iloc[index]
                if not bool(candidate.get("entry_candidate", False)):
                    index += 1
                    continue
                entry_index = index + 1 + int(extra_delay_minutes)
                if entry_index >= len(sample) - 1:
                    break
                expected_entry_time = candidate["decision_time"] + pd.Timedelta(
                    minutes=extra_delay_minutes
                )
                entry = sample.iloc[entry_index]
                if entry["timestamp"] != expected_entry_time or not bool(entry["quality_ok"]):
                    index += 1
                    continue
                perp_direction = -1 if float(candidate["z_score"]) > 0 else 1
                cash_direction = -perp_direction
                exit_signal_index = entry_index
                exit_index: int | None = None
                reason: str | None = None
                while exit_signal_index < len(sample) - 1:
                    exit_signal = sample.iloc[exit_signal_index]
                    reason = _close_reason(exit_signal, entry["timestamp"], cfg)
                    if reason:
                        proposed = exit_signal_index + 1 + int(extra_delay_minutes)
                        if proposed < len(sample):
                            expected = exit_signal["decision_time"] + pd.Timedelta(
                                minutes=extra_delay_minutes
                            )
                            if sample.iloc[proposed]["timestamp"] == expected:
                                exit_index = proposed
                        break
                    exit_signal_index += 1
                if exit_index is None:
                    exit_index = len(sample) - 1
                    exit_signal_index = max(entry_index, exit_index - 1)
                    reason = "end_of_session"
                if exit_index <= entry_index:
                    index = entry_index + 1
                    continue
                exit_row = sample.iloc[exit_index]
                gross = _leg_pnl(
                    cash_direction, notional, float(entry["cash_open"]), float(exit_row["cash_open"])
                ) + _leg_pnl(
                    perp_direction, notional, float(entry["perp_open"]), float(exit_row["perp_open"])
                )
                trading_cost = notional * cost["total_trading_bps"] / 10_000.0
                funding_pnl = _funding_pnl(
                    funding_by_symbol.get(symbol, pd.DataFrame()),
                    entry["timestamp"],
                    exit_row["timestamp"],
                    perp_direction,
                    notional,
                )
                held_seconds = (exit_row["timestamp"] - entry["timestamp"]).total_seconds()
                borrow = (
                    notional
                    * float(cfg["execution"]["cash_borrow_rate_annual"])
                    * held_seconds
                    / SECONDS_PER_YEAR
                    if cash_direction < 0
                    else 0.0
                )
                opportunity = (
                    notional
                    * (1.0 + float(cfg["execution"]["perp_margin_fraction"]))
                    * float(cfg["execution"]["opportunity_cost_annual"])
                    * held_seconds
                    / SECONDS_PER_YEAR
                )
                net = gross - trading_cost + funding_pnl - borrow - opportunity
                trades.append(
                    {
                        "symbol": symbol,
                        "signal_time": candidate["decision_time"],
                        "entry_time": entry["timestamp"],
                        "exit_signal_time": sample.iloc[exit_signal_index]["decision_time"],
                        "exit_time": exit_row["timestamp"],
                        "side": "long_perp_short_cash" if perp_direction > 0 else "short_perp_long_cash",
                        "entry_z": candidate["z_score"],
                        "exit_z": sample.iloc[exit_signal_index]["z_score"],
                        "predicted_net_edge_bps": candidate["predicted_net_edge_bps"],
                        "holding_minutes": held_seconds / 60.0,
                        "exit_reason": reason,
                        "notional_per_leg": notional,
                        "entry_cash_open": entry["cash_open"],
                        "entry_perp_open": entry["perp_open"],
                        "exit_cash_open": exit_row["cash_open"],
                        "exit_perp_open": exit_row["perp_open"],
                        "mid_gross_pnl": gross,
                        "modeled_spread_cost": notional * cost["modeled_spread_bps"] / 10_000.0,
                        "fee_cost": notional * cost["fee_bps"] / 10_000.0,
                        "extra_slippage_cost": notional * cost["extra_slippage_bps"] / 10_000.0,
                        "market_impact_cost": notional * cost["impact_bps"] / 10_000.0,
                        "funding_cost": -funding_pnl,
                        "borrow_cost": borrow,
                        "opportunity_cost": opportunity,
                        "net_pnl": net,
                        "net_return_bps": net / notional * 10_000.0,
                    }
                )
                index = exit_index + 1
    return pd.DataFrame(trades)


def _scenario_metrics(trades: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    metrics = performance_metrics(trades, cfg)
    metrics["gross_pnl"] = float(trades["mid_gross_pnl"].sum()) if not trades.empty else 0.0
    return metrics


def evaluate_historical_screen(
    signals: pd.DataFrame,
    quality: pd.DataFrame,
    funding: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    baseline = run_historical_backtest(signals, funding, cfg)
    double_signals = build_historical_signals(signals, cfg, cost_multiplier=2.0)
    doubled = run_historical_backtest(double_signals, funding, cfg, cost_multiplier=2.0)
    delayed = run_historical_backtest(signals, funding, cfg, extra_delay_minutes=1)
    scenario_rows = [
        {"scenario": "baseline", **_scenario_metrics(baseline, cfg)},
        {"scenario": "double_modeled_cost", **_scenario_metrics(doubled, cfg)},
        {"scenario": "one_minute_extra_delay", **_scenario_metrics(delayed, cfg)},
    ]
    scenarios = pd.DataFrame(scenario_rows)
    valid = signals.loc[signals["quality_ok"]] if not signals.empty else signals
    days = int(valid["timestamp"].dt.date.nunique()) if not valid.empty else 0
    symbols = int(valid["symbol"].nunique()) if not valid.empty else 0
    baseline_metrics = scenario_rows[0]
    doubled_metrics = scenario_rows[1]
    delayed_metrics = scenario_rows[2]
    if baseline.empty:
        pnl_concentration = np.inf
    else:
        symbol_abs_pnl = baseline.groupby("symbol")["net_pnl"].sum().abs()
        pnl_concentration = float(symbol_abs_pnl.max() / symbol_abs_pnl.sum())
    checks = pd.DataFrame(
        [
            {"check": "observed_trading_days", "value": days, "threshold": f">= {cfg['acceptance']['min_observed_trading_days']}", "pass": days >= int(cfg["acceptance"]["min_observed_trading_days"])},
            {"check": "observed_symbols", "value": symbols, "threshold": f">= {cfg['acceptance']['min_symbols']}", "pass": symbols >= int(cfg["acceptance"]["min_symbols"])},
            {"check": "minimum_screen_trades", "value": baseline_metrics["trades"], "threshold": f">= {cfg['historical']['min_screen_trades']}", "pass": baseline_metrics["trades"] >= int(cfg["historical"]["min_screen_trades"])},
            {"check": "symbol_pnl_concentration", "value": pnl_concentration, "threshold": f"<= {cfg['acceptance']['max_symbol_pnl_concentration']}", "pass": pnl_concentration <= float(cfg["acceptance"]["max_symbol_pnl_concentration"])},
            {"check": "gross_pnl_positive", "value": baseline_metrics["gross_pnl"], "threshold": "> 0", "pass": baseline_metrics["gross_pnl"] > 0},
            {"check": "baseline_net_positive", "value": baseline_metrics["net_pnl"], "threshold": "> 0", "pass": baseline_metrics["net_pnl"] > 0},
            {"check": "double_cost_net_positive", "value": doubled_metrics["net_pnl"], "threshold": "> 0", "pass": doubled_metrics["net_pnl"] > 0},
            {"check": "extra_delay_net_positive", "value": delayed_metrics["net_pnl"], "threshold": "> 0", "pass": delayed_metrics["net_pnl"] > 0},
        ]
    )
    signal_diagnostics = (
        signals.groupby("symbol", as_index=False)
        .agg(
            aligned_rows=("basis", "size"),
            quality_rows=("quality_ok", "sum"),
            entry_candidates=("entry_candidate", "sum"),
            basis_abs_p99_bps=("basis", lambda values: values.abs().quantile(0.99) * 10_000.0),
        )
        if not signals.empty
        else pd.DataFrame()
    )
    trade_diagnostics = (
        baseline.groupby("symbol", as_index=False)
        .agg(
            trades=("net_pnl", "size"),
            gross_pnl=("mid_gross_pnl", "sum"),
            net_pnl=("net_pnl", "sum"),
            win_rate=("net_pnl", lambda values: (values > 0).mean()),
            average_holding_minutes=("holding_minutes", "mean"),
        )
        if not baseline.empty
        else pd.DataFrame(columns=["symbol", "trades", "gross_pnl", "net_pnl", "win_rate", "average_holding_minutes"])
    )
    symbol_diagnostics = signal_diagnostics.merge(
        trade_diagnostics, on="symbol", how="left", validate="one_to_one"
    )
    for column in ("trades", "gross_pnl", "net_pnl"):
        if column in symbol_diagnostics:
            symbol_diagnostics[column] = symbol_diagnostics[column].fillna(0)
    exit_diagnostics = (
        baseline.groupby("exit_reason", as_index=False)
        .agg(
            trades=("net_pnl", "size"),
            gross_pnl=("mid_gross_pnl", "sum"),
            net_pnl=("net_pnl", "sum"),
            average_holding_minutes=("holding_minutes", "mean"),
        )
        if not baseline.empty
        else pd.DataFrame()
    )
    return {
        "status": "PRELIMINARY_ONLY",
        "screen_pass": bool(checks["pass"].all()),
        "trades": baseline,
        "scenarios": scenarios,
        "checks": checks,
        "quality": quality,
        "symbol_diagnostics": symbol_diagnostics,
        "exit_diagnostics": exit_diagnostics,
    }


def write_historical_report(
    result: dict[str, Any],
    errors: list[dict[str, str]],
    output_dir: Path,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    cfg: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    result["trades"].to_csv(output_dir / "historical_trades.csv", index=False)
    result["scenarios"].to_csv(output_dir / "historical_scenarios.csv", index=False)
    result["checks"].to_csv(output_dir / "historical_screen_checks.csv", index=False)
    result["quality"].to_csv(output_dir / "historical_quality.csv", index=False)
    result["symbol_diagnostics"].to_csv(
        output_dir / "historical_symbol_diagnostics.csv", index=False
    )
    result["exit_diagnostics"].to_csv(
        output_dir / "historical_exit_diagnostics.csv", index=False
    )
    pd.DataFrame(errors, columns=["symbol", "dataset", "error"]).to_csv(
        output_dir / "historical_download_errors.csv", index=False
    )
    baseline = result["scenarios"].iloc[0]
    costs = historical_costs_bps(cfg)
    lines = [
        "# Historical Basis Pre-screen",
        "",
        "**Status: PRELIMINARY_ONLY**",
        "",
        f"Screen checks: {'PASS' if result['screen_pass'] else 'FAIL'} (this is never a live GO decision).",
        "",
        f"Period: {start.date()} through {(end_exclusive - pd.Timedelta(days=1)).date()}",
        f"Cash source: Alpaca {cfg['historical']['alpaca_feed']} one-minute trade bars",
        "Perpetual source: Binance USD-M contract, mark and index one-minute bars",
        "",
        "## Baseline",
        "",
        f"- Trades: {int(baseline['trades'])}",
        f"- Gross PnL: ${baseline['gross_pnl']:,.2f}",
        f"- Net PnL: ${baseline['net_pnl']:,.2f}",
        f"- Sharpe: {baseline['sharpe']:.3f}",
        f"- Modeled round-trip trading cost: {costs['total_trading_bps']:.1f} bps per trade",
        f"- Entry hurdle including safety margin: {costs['total_trading_bps'] + float(cfg['signal']['safety_margin_bps']):.1f} bps",
        "",
        "## Symbol diagnostics",
        "",
        "| Symbol | Candidates | Trades | Gross PnL | Net PnL |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in result["symbol_diagnostics"].iterrows():
        lines.append(
            f"| {row['symbol']} | {int(row['entry_candidates'])} | {int(row['trades'])} | "
            f"${row['gross_pnl']:,.2f} | ${row['net_pnl']:,.2f} |"
        )
    lines.extend(
        [
        "",
        "## Non-negotiable limitations",
        "",
        "- Binance does not supply historical top-of-book bid/ask through this dataset.",
        "- Alpaca Basic IEX bars are not consolidated SIP executable quotes.",
        "- Historical spreads are fixed conservative assumptions, not observed spreads.",
        "- Therefore this module can reject a weak idea, but cannot approve deployment.",
        "",
        "## Causal controls",
        "",
        "- Each signal uses a completed minute and rolling statistics through the prior minute only.",
        "- Cash, contract, mark and index bars must have the exact same minute timestamp.",
        "- Entry and exit use a later minute open; missing minutes are never forward-filled.",
        "- Tests include doubled modeled costs and one additional full minute of delay.",
        "- Parameters and symbol membership are fixed; this command performs no search or tuning.",
        "",
        "Detailed CSV files are stored beside this report.",
        ]
    )
    path = output_dir / "historical_prescreen_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_historical_prescreen(
    cfg: dict[str, Any],
    start_text: str | None = None,
    end_text: str | None = None,
    refresh: bool = False,
    report_dir: str | Path | None = None,
) -> Path:
    start, end_exclusive = historical_bounds(
        start_text or str(cfg["historical"]["start_date"]),
        end_text or str(cfg["historical"]["end_date"]),
    )
    history, errors = download_history(cfg, start, end_exclusive, refresh)
    aligned, quality = build_historical_frame(history, cfg)
    signals = build_historical_signals(aligned, cfg)
    result = evaluate_historical_screen(signals, quality, history["funding"], cfg)
    download_complete = not errors
    download_check = pd.DataFrame(
        [
            {
                "check": "configured_datasets_complete",
                "value": len(errors),
                "threshold": "= 0 download errors",
                "pass": download_complete,
            }
        ]
    )
    result["checks"] = pd.concat([download_check, result["checks"]], ignore_index=True)
    result["screen_pass"] = bool(result["checks"]["pass"].all())

    root = Path(cfg["_root"])
    processed = root / str(cfg["historical"]["cache_dir"]) / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    key = f"{start.strftime('%Y%m%d')}_{(end_exclusive - pd.Timedelta(days=1)).strftime('%Y%m%d')}"
    aligned.to_pickle(processed / f"aligned_{key}.pkl")
    signals.to_pickle(processed / f"signals_{key}.pkl")
    output = Path(report_dir) if report_dir else root / str(cfg["output"]["report_dir"]) / "historical"
    report = write_historical_report(result, errors, output, start, end_exclusive, cfg)
    print(result["scenarios"].to_string(index=False))
    print(result["checks"].to_string(index=False))
    print("Status: PRELIMINARY_ONLY")
    print(f"Screen checks: {'PASS' if result['screen_pass'] else 'FAIL'}")
    print(f"Report: {report.resolve()}")
    return report
