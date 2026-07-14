from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60


@dataclass(frozen=True)
class FillSet:
    mid: float
    quote: float
    slipped: float
    final: float


def _fill_prices(row: pd.Series, venue: str, side: int, cfg: dict[str, Any]) -> FillSet:
    execution = cfg["execution"]
    mid = float(row[f"{venue}_mid"])
    quote = float(row[f"{venue}_ask"] if side > 0 else row[f"{venue}_bid"])
    slip_bps = float(execution[f"{venue}_extra_slippage_bps"])
    impact_bps = float(execution["impact_bps_per_fill"])
    direction = 1.0 if side > 0 else -1.0
    slipped = quote * (1.0 + direction * slip_bps / 10_000.0)
    final = slipped * (1.0 + direction * impact_bps / 10_000.0)
    return FillSet(mid=mid, quote=quote, slipped=slipped, final=final)


def _leg_pnl(direction: int, notional: float, entry: float, exit_: float) -> float:
    quantity = direction * notional / entry
    return quantity * (exit_ - entry)


def _pair_pnl(
    perp_direction: int,
    notional: float,
    entry_cash: float,
    exit_cash: float,
    entry_perp: float,
    exit_perp: float,
) -> float:
    cash_direction = -perp_direction
    return _leg_pnl(cash_direction, notional, entry_cash, exit_cash) + _leg_pnl(
        perp_direction, notional, entry_perp, exit_perp
    )


def _execution_index(sample: pd.DataFrame, signal_index: int, delay_seconds: float) -> int | None:
    target = sample.iloc[signal_index]["timestamp"] + pd.Timedelta(seconds=delay_seconds)
    timestamps = pd.DatetimeIndex(sample["timestamp"])
    index = int(timestamps.searchsorted(target, side="left"))
    while index < len(sample) and not bool(sample.iloc[index]["quality_ok"]):
        index += 1
    return index if index < len(sample) else None


def _funding_pnl(
    sample: pd.DataFrame,
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    perp_direction: int,
    notional: float,
) -> float:
    funding_times = pd.to_datetime(sample["funding_time"], utc=True, errors="coerce")
    held = sample.loc[
        funding_times.notna()
        & funding_times.gt(entry_time)
        & funding_times.le(exit_time)
    ].copy()
    held["funding_time"] = funding_times.loc[held.index]
    if held.empty:
        return 0.0
    events = held.sort_values("timestamp").drop_duplicates("funding_time", keep="last")
    return float((-perp_direction * notional * events["funding_rate"].fillna(0.0)).sum())


def _close_reason(row: pd.Series, entry_time: pd.Timestamp, cfg: dict[str, Any]) -> str | None:
    if not bool(row.get("signal_observation", True)) or pd.isna(row.get("z_score")):
        return None
    settings = cfg["signal"]
    held_minutes = (row["timestamp"] - entry_time).total_seconds() / 60.0
    if held_minutes < float(settings["min_holding_minutes"]):
        return None
    if abs(float(row["z_score"])) >= float(settings["stop_z"]):
        return "stop"
    if abs(float(row["z_score"])) <= float(settings["exit_z"]):
        return "reversion"
    if held_minutes >= float(settings["max_holding_minutes"]):
        return "timeout"
    return None


def run_backtest(signal_frame: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    if signal_frame.empty:
        return pd.DataFrame()
    execution = cfg["execution"]
    delay = float(execution["delay_seconds"])
    notional = float(execution["notional_per_leg"])
    trades: list[dict] = []

    for symbol, sample in signal_frame.groupby("symbol", sort=True):
        sample = sample.sort_values("timestamp").reset_index(drop=True)
        index = 0
        while index < len(sample):
            candidate = sample.iloc[index]
            if not bool(candidate.get("entry_candidate", False)):
                index += 1
                continue
            entry_index = _execution_index(sample, index, delay)
            if entry_index is None:
                break
            entry = sample.iloc[entry_index]
            perp_direction = -1 if float(candidate["z_score"]) > 0 else 1
            cash_direction = -perp_direction
            exit_signal_index = entry_index + 1
            reason = None
            while exit_signal_index < len(sample):
                row = sample.iloc[exit_signal_index]
                reason = _close_reason(row, entry["timestamp"], cfg)
                if reason:
                    break
                exit_signal_index += 1
            if exit_signal_index >= len(sample):
                exit_signal_index = len(sample) - 1
                reason = "end_of_data"
            exit_index = _execution_index(sample, exit_signal_index, delay)
            if exit_index is None:
                exit_index = len(sample) - 1
                reason = "end_of_data"
            if exit_index <= entry_index:
                index = entry_index + 1
                continue
            exit_row = sample.iloc[exit_index]

            entry_cash = _fill_prices(entry, "cash", cash_direction, cfg)
            entry_perp = _fill_prices(entry, "perp", perp_direction, cfg)
            exit_cash = _fill_prices(exit_row, "cash", -cash_direction, cfg)
            exit_perp = _fill_prices(exit_row, "perp", -perp_direction, cfg)

            mid_gross = _pair_pnl(
                perp_direction, notional, entry_cash.mid, exit_cash.mid, entry_perp.mid, exit_perp.mid
            )
            quote_pnl = _pair_pnl(
                perp_direction, notional, entry_cash.quote, exit_cash.quote, entry_perp.quote, exit_perp.quote
            )
            slipped_pnl = _pair_pnl(
                perp_direction, notional, entry_cash.slipped, exit_cash.slipped, entry_perp.slipped, exit_perp.slipped
            )
            fill_pnl = _pair_pnl(
                perp_direction, notional, entry_cash.final, exit_cash.final, entry_perp.final, exit_perp.final
            )
            cash_fee_rate = float(execution["cash_taker_fee_bps"]) / 10_000.0
            perp_fee_rate = float(execution["perp_taker_fee_bps"]) / 10_000.0
            fees = notional * 2.0 * (cash_fee_rate + perp_fee_rate)
            funding_pnl = _funding_pnl(
                sample, entry["timestamp"], exit_row["timestamp"], perp_direction, notional
            )
            held_seconds = (exit_row["timestamp"] - entry["timestamp"]).total_seconds()
            borrow = (
                notional
                * float(execution["cash_borrow_rate_annual"])
                * held_seconds
                / SECONDS_PER_YEAR
                if cash_direction < 0
                else 0.0
            )
            opportunity = (
                notional
                * (1.0 + float(execution["perp_margin_fraction"]))
                * float(execution["opportunity_cost_annual"])
                * held_seconds
                / SECONDS_PER_YEAR
            )
            net_pnl = fill_pnl - fees + funding_pnl - borrow - opportunity
            trades.append(
                {
                    "symbol": symbol,
                    "signal_time": candidate["timestamp"],
                    "entry_time": entry["timestamp"],
                    "exit_signal_time": sample.iloc[exit_signal_index]["timestamp"],
                    "exit_time": exit_row["timestamp"],
                    "side": "long_perp_short_cash" if perp_direction > 0 else "short_perp_long_cash",
                    "entry_z": candidate["z_score"],
                    "exit_z": sample.iloc[exit_signal_index]["z_score"],
                    "predicted_net_edge_bps": candidate["predicted_net_edge_bps"],
                    "holding_minutes": held_seconds / 60.0,
                    "exit_reason": reason,
                    "notional_per_leg": notional,
                    "mid_gross_pnl": mid_gross,
                    "quoted_spread_cost": mid_gross - quote_pnl,
                    "extra_slippage_cost": quote_pnl - slipped_pnl,
                    "market_impact_cost": slipped_pnl - fill_pnl,
                    "fee_cost": fees,
                    "funding_cost": -funding_pnl,
                    "borrow_cost": borrow,
                    "opportunity_cost": opportunity,
                    "net_pnl": net_pnl,
                    "net_return_bps": net_pnl / notional * 10_000.0,
                }
            )
            index = exit_index + 1
    return pd.DataFrame(trades)


def performance_metrics(trades: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, float]:
    capital = float(cfg["execution"]["initial_capital"])
    if trades.empty:
        return {
            "trades": 0,
            "net_pnl": 0.0,
            "mid_gross_pnl": 0.0,
            "total_cost": 0.0,
            "annual_return": 0.0,
            "sharpe": np.nan,
            "max_drawdown": 0.0,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "average_holding_minutes": np.nan,
        }
    daily = trades.assign(day=trades["exit_time"].dt.floor("D")).groupby("day")["net_pnl"].sum()
    full_days = pd.date_range(daily.index.min(), daily.index.max(), freq="D", tz="UTC")
    daily = daily.reindex(full_days, fill_value=0.0)
    returns = daily / capital
    std = returns.std(ddof=1)
    sharpe = returns.mean() / std * np.sqrt(365.0) if std > 0 else np.nan
    cumulative = daily.cumsum()
    nav = capital + cumulative
    running_peak = nav.cummax().clip(lower=capital)
    drawdown = nav / running_peak - 1.0
    elapsed_years = max((full_days[-1] - full_days[0]).days / 365.25, 1.0 / 365.25)
    total_return = float(daily.sum() / capital)
    annual_return = (1.0 + total_return) ** (1.0 / elapsed_years) - 1.0 if total_return > -1 else -1.0
    wins = trades.loc[trades["net_pnl"] > 0, "net_pnl"].sum()
    losses = -trades.loc[trades["net_pnl"] < 0, "net_pnl"].sum()
    return {
        "trades": int(len(trades)),
        "net_pnl": float(trades["net_pnl"].sum()),
        "mid_gross_pnl": float(trades["mid_gross_pnl"].sum()),
        "total_cost": float(trades["mid_gross_pnl"].sum() - trades["net_pnl"].sum()),
        "annual_return": float(annual_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "win_rate": float((trades["net_pnl"] > 0).mean()),
        "profit_factor": float(wins / losses) if losses > 0 else np.inf,
        "average_holding_minutes": float(trades["holding_minutes"].mean()),
    }
