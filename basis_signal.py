from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _lagged_rolling(series: pd.Series, window: int, min_history: int) -> tuple[pd.Series, pd.Series]:
    history = series.shift(1)
    mean = history.rolling(window, min_periods=min_history).mean()
    std = history.rolling(window, min_periods=min_history).std(ddof=1).replace(0.0, np.nan)
    return mean, std


def build_signal_frame(aligned: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    if aligned.empty:
        return aligned.copy()
    settings = cfg["signal"]
    execution = cfg["execution"]
    frames = []
    for _, sample in aligned.groupby("symbol", sort=True):
        sample = sample.sort_values("timestamp").copy()
        sample["signal_bucket"] = sample["timestamp"].dt.floor(
            f"{int(settings['bar_seconds'])}s"
        )
        signal_rows = (
            sample.loc[sample["quality_ok"]]
            .groupby("signal_bucket", as_index=False, sort=True)
            .tail(1)
            .copy()
        )
        if signal_rows.empty:
            sample["signal_observation"] = False
            sample["entry_candidate"] = False
            frames.append(sample.drop(columns="signal_bucket"))
            continue
        mean, std = _lagged_rolling(
            signal_rows["basis"],
            int(settings["rolling_window"]),
            int(settings["min_history"]),
        )
        signal_rows["basis_mean_lagged"] = mean
        signal_rows["basis_std_lagged"] = std
        signal_rows["z_score"] = (signal_rows["basis"] - mean) / std
        signal_rows["expected_gross_edge_bps"] = (
            (signal_rows["basis"] - mean).abs() - float(settings["exit_z"]) * std
        ).clip(lower=0.0) * 10_000.0
        fee_hurdle = 2.0 * (
            float(execution["cash_taker_fee_bps"])
            + float(execution["perp_taker_fee_bps"])
        )
        extra_hurdle = 2.0 * (
            float(execution["cash_extra_slippage_bps"])
            + float(execution["perp_extra_slippage_bps"])
            + 2.0 * float(execution["impact_bps_per_fill"])
        )
        signal_rows["cost_hurdle_bps"] = (
            signal_rows["cash_spread_bps"]
            + signal_rows["perp_spread_bps"]
            + fee_hurdle
            + extra_hurdle
            + float(settings["safety_margin_bps"])
        )
        signal_rows["predicted_net_edge_bps"] = (
            signal_rows["expected_gross_edge_bps"] - signal_rows["cost_hurdle_bps"]
        )
        signal_rows["entry_candidate"] = (
            signal_rows["z_score"].abs().ge(float(settings["entry_z"]))
            & signal_rows["predicted_net_edge_bps"].gt(0.0)
        )
        signal_columns = [
            "timestamp",
            "basis_mean_lagged",
            "basis_std_lagged",
            "z_score",
            "expected_gross_edge_bps",
            "cost_hurdle_bps",
            "predicted_net_edge_bps",
            "entry_candidate",
        ]
        sample = sample.merge(
            signal_rows[signal_columns],
            on="timestamp",
            how="left",
            validate="one_to_one",
        )
        sample["signal_observation"] = sample["z_score"].notna()
        sample["entry_candidate"] = sample["entry_candidate"].fillna(False).astype(bool)
        frames.append(sample.drop(columns="signal_bucket"))
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "timestamp"])
