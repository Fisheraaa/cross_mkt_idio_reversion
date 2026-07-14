from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest import SECONDS_PER_YEAR
from factor_config import (
    instrument_specs_for_targets,
    research_target_specs,
    session_parameters,
    target_specs,
)
from factor_strategy import (
    _fit_ridge,
    _hourly_bars,
    build_factor_hourly_panel,
    download_factor_market_history,
    instrument_roundtrip_cost_bps,
)
from historical_prescreen import historical_bounds


BASE_MODEL_FEATURES = [
    "z_score",
    "dislocation",
    "residual",
    "model_r2",
    "gross_beta",
    "reversion_slope_lagged",
    "reversion_t_stat_lagged",
    "adf_t_stat_lagged",
    "adf_p_value_lagged",
    "expected_gross_edge_bps",
    "relative_basis_bps",
    "relative_basis_z",
    "long_relative_funding_bps",
    "relative_volume",
    "cross_z_median",
    "cross_z_std",
    "cross_z_min",
    "cross_z_max",
]
LAGGED_MODEL_FEATURES = [
    *[f"residual_lag{lag}" for lag in (1, 2, 4, 8, 24)],
    *[f"z_lag{lag}" for lag in (1, 2, 4, 8, 24)],
    *[f"residual_sum{window}" for window in (4, 8, 24, 72)],
    *[f"residual_vol{window}" for window in (4, 8, 24, 72)],
]


def _research_cache_fingerprints(cfg: dict[str, Any]) -> tuple[str, str]:
    """Bind processed caches to every setting that changes their contents."""
    factor = cfg["factor_strategy"]
    research = cfg["tradable_research"]
    signal_payload = {
        "targets": factor["targets"],
        "session_parameters": factor["session_parameters"],
        "execution_interval": factor["execution_interval"],
        "signal_interval": factor["signal_interval"],
        "signal_horizon_hours": factor["signal_horizon_hours"],
        "ridge_alpha": factor["ridge_alpha"],
        "nonnegative_betas": factor.get("nonnegative_betas", True),
        "minimum_execution_bars_per_hour": factor["minimum_execution_bars_per_hour"],
        "minimum_signal_bars_per_hour": factor["minimum_signal_bars_per_hour"],
        "max_contract_mark_gap_bps": factor["max_contract_mark_gap_bps"],
        "dynamic_beta_half_life_hours": research["dynamic_beta_half_life_hours"],
    }

    def digest(payload: dict[str, Any]) -> str:
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:12]

    signal_fingerprint = digest(signal_payload)
    label_payload = {
        "signal_fingerprint": signal_fingerprint,
        "execution_delay_minutes": factor["execution_delay_minutes"],
        "cost_profiles": factor["cost_profiles"],
        "primary_horizon_hours": research["primary_horizon_hours"],
        "basis_lookback_hours": research["basis_lookback_hours"],
        "basis_min_history_hours": research["basis_min_history_hours"],
        "volume_lookback_hours": research["volume_lookback_hours"],
        "execution": cfg["execution"],
    }
    return signal_fingerprint, digest(label_payload)


def build_dynamic_factor_signals(
    panel: pd.DataFrame, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Estimate beta causally with the recursive Kalman/RLS regression form."""
    if panel.empty:
        return panel.copy()
    settings = cfg["factor_strategy"]
    research = cfg["tradable_research"]
    targets = target_specs(cfg)
    half_life = float(research["dynamic_beta_half_life_hours"])
    forgetting = float(0.5 ** (1.0 / half_life))
    output: list[pd.DataFrame] = []
    for symbol, raw in panel.groupby("symbol", sort=True):
        sample = raw.sort_values("timestamp").reset_index(drop=True).copy()
        target = targets[str(symbol)]
        factors = [str(factor) for factor in target["factors"]]
        x_columns = [f"{factor}_return" for factor in factors]
        parameters = session_parameters(cfg, str(target["session"]))
        regression_lookback = int(parameters["regression_lookback_hours"])
        minimum = int(parameters["regression_min_hours"])
        residual_lookback = int(parameters["residual_lookback_hours"])
        residual_minimum = int(parameters["residual_min_hours"])
        signal_horizon = int(settings["signal_horizon_hours"])
        for column in (
            "intercept",
            "model_r2",
            "residual",
            "dislocation",
            "dislocation_mean_lagged",
            "dislocation_std_lagged",
            "z_score",
        ):
            sample[column] = np.nan
        for factor in factors:
            sample[f"beta_{factor}"] = np.nan
        valid = sample["quality_ok"] & sample[x_columns + ["target_return"]].notna().all(axis=1)
        initial_indices = np.flatnonzero(valid.to_numpy())[:minimum]
        if len(initial_indices) < minimum:
            output.append(sample)
            continue
        initial_end = int(initial_indices[-1])
        initial_x = sample.loc[initial_indices, x_columns].to_numpy(float)
        initial_y = sample.loc[initial_indices, "target_return"].to_numpy(float)
        intercept, beta, _ = _fit_ridge(
            initial_x,
            initial_y,
            float(settings["ridge_alpha"]),
            bool(settings.get("nonnegative_betas", True)),
        )
        theta = np.r_[intercept, beta]
        design = np.column_stack([np.ones(len(initial_x)), initial_x])
        factor_variance = initial_x.std(axis=0, ddof=1) ** 2
        penalty = np.diag(
            np.r_[1e-10, float(settings["ridge_alpha"]) * factor_variance]
        )
        covariance = np.linalg.inv(design.T @ design + penalty)
        initial_residuals = initial_y - design @ theta
        residual_history = list(initial_residuals)
        target_history = list(initial_y)
        timestamp_history = list(sample.loc[initial_indices, "timestamp"])
        session_history = list(sample.loc[initial_indices, "session_date"])
        dislocation_history: list[float] = []
        for index in range(len(initial_residuals)):
            if signal_horizon == 1:
                dislocation_history.append(float(initial_residuals[index]))
                continue
            if index + 1 < signal_horizon:
                continue
            block_timestamps = timestamp_history[index - signal_horizon + 1 : index + 1]
            contiguous = all(
                block_timestamps[position] - block_timestamps[position - 1]
                == pd.Timedelta(hours=1)
                for position in range(1, len(block_timestamps))
            )
            if str(target["session"]) == "us_rth":
                block_sessions = session_history[index - signal_horizon + 1 : index + 1]
                contiguous &= len(set(block_sessions)) == 1
            if contiguous:
                dislocation_history.append(
                    float(sum(initial_residuals[index - signal_horizon + 1 : index + 1]))
                )

        for index in range(initial_end + 1, len(sample)):
            if not bool(valid.iloc[index]):
                continue
            x = np.r_[1.0, sample.loc[index, x_columns].to_numpy(float)]
            target_return = float(sample.loc[index, "target_return"])
            residual = float(target_return - x @ theta)
            trailing_residuals = np.asarray(
                residual_history[-regression_lookback:], dtype=float
            )
            trailing_targets = np.asarray(target_history[-regression_lookback:], dtype=float)
            denominator = float(
                np.square(trailing_targets - trailing_targets.mean()).sum()
            )
            model_r2 = (
                1.0 - float(np.square(trailing_residuals).sum()) / denominator
                if denominator > 0
                else np.nan
            )
            current_dislocation = np.nan
            if signal_horizon == 1:
                current_dislocation = residual
            elif len(residual_history) >= signal_horizon - 1:
                prior_timestamps = timestamp_history[-(signal_horizon - 1) :]
                current_timestamp = pd.Timestamp(sample.loc[index, "timestamp"])
                timestamps = prior_timestamps + [current_timestamp]
                contiguous = all(
                    timestamps[position] - timestamps[position - 1]
                    == pd.Timedelta(hours=1)
                    for position in range(1, len(timestamps))
                )
                if str(target["session"]) == "us_rth":
                    sessions = session_history[-(signal_horizon - 1) :] + [
                        sample.loc[index, "session_date"]
                    ]
                    contiguous &= len(set(sessions)) == 1
                if contiguous:
                    current_dislocation = float(
                        sum(residual_history[-(signal_horizon - 1) :]) + residual
                    )
            prior_dislocations = np.asarray(
                dislocation_history[-residual_lookback:], dtype=float
            )
            if (
                np.isfinite(current_dislocation)
                and len(prior_dislocations) >= residual_minimum
                and prior_dislocations.std(ddof=1) > 0
            ):
                mean = float(prior_dislocations.mean())
                std = float(prior_dislocations.std(ddof=1))
                sample.loc[index, "dislocation"] = current_dislocation
                sample.loc[index, "dislocation_mean_lagged"] = mean
                sample.loc[index, "dislocation_std_lagged"] = std
                sample.loc[index, "z_score"] = (current_dislocation - mean) / std
            sample.loc[index, "intercept"] = theta[0]
            sample.loc[index, "model_r2"] = model_r2
            sample.loc[index, "residual"] = residual
            for factor, value in zip(factors, theta[1:]):
                sample.loc[index, f"beta_{factor}"] = value
            residual_history.append(residual)
            target_history.append(target_return)
            timestamp_history.append(sample.loc[index, "timestamp"])
            session_history.append(sample.loc[index, "session_date"])
            if np.isfinite(current_dislocation):
                dislocation_history.append(float(current_dislocation))
            gain = covariance @ x / (forgetting + float(x @ covariance @ x))
            theta = theta + gain * residual
            if bool(settings.get("nonnegative_betas", True)):
                theta[1:] = np.maximum(theta[1:], 0.0)
            covariance = (
                covariance - np.outer(gain, x) @ covariance
            ) / forgetting
        beta_columns = [f"beta_{factor}" for factor in factors]
        sample["gross_beta"] = sample[beta_columns].abs().sum(axis=1)
        sample["factor_model_type"] = "dynamic_rls"
        output.append(sample)
    return pd.concat(output, ignore_index=True).sort_values(["symbol", "timestamp"])


def history_availability(
    history: dict[str, dict[str, pd.DataFrame]],
    cfg: dict[str, Any],
    selected_targets: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Describe exchange-observed coverage without assuming listing dates."""
    required = instrument_specs_for_targets(cfg, selected_targets)
    rows: list[dict[str, Any]] = []
    for instrument, spec in required.items():
        datasets = history.get(instrument, {})
        starts: list[pd.Timestamp] = []
        ends: list[pd.Timestamp] = []
        row: dict[str, Any] = {
            "instrument": instrument,
            "provider_symbol": str(spec["perp_symbol"]),
            "is_target": instrument in selected_targets,
        }
        for kind in ("perp", "mark", "index", "funding"):
            frame = datasets.get(kind, pd.DataFrame())
            row[f"{kind}_rows"] = len(frame)
            if not frame.empty:
                timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
                first = timestamps.min()
                last = timestamps.max()
                row[f"{kind}_first"] = first
                row[f"{kind}_last"] = last
                if kind != "funding":
                    starts.append(first)
                    ends.append(last)
            else:
                row[f"{kind}_first"] = pd.NaT
                row[f"{kind}_last"] = pd.NaT
        row["common_price_start"] = max(starts) if len(starts) == 3 else pd.NaT
        row["common_price_end"] = min(ends) if len(ends) == 3 else pd.NaT
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["is_target", "instrument"], ascending=[False, True])


def _last_observed_values(
    frame: pd.DataFrame, decision_times: pd.Series, value_column: str
) -> np.ndarray:
    if frame.empty:
        return np.full(len(decision_times), np.nan)
    observed = frame.dropna(subset=["timestamp", value_column]).sort_values("timestamp")
    if observed.empty:
        return np.full(len(decision_times), np.nan)
    observed_times = pd.to_datetime(observed["timestamp"], utc=True).astype("int64").to_numpy()
    query_times = pd.to_datetime(decision_times, utc=True).astype("int64").to_numpy()
    positions = np.searchsorted(observed_times, query_times, side="right") - 1
    values = np.full(len(query_times), np.nan)
    valid = positions >= 0
    values[valid] = observed[value_column].to_numpy(float)[positions[valid]]
    return values


def add_tradable_features(
    signals: pd.DataFrame,
    history: dict[str, dict[str, pd.DataFrame]],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Add only features observable by each completed signal hour."""
    if signals.empty:
        return signals.copy()
    settings = cfg["tradable_research"]
    targets = target_specs(cfg)
    bar_cache: dict[tuple[str, str], pd.DataFrame] = {}

    def basis(instrument: str, session: str) -> pd.Series:
        key = (instrument, session)
        if key not in bar_cache:
            bar_cache[key] = _hourly_bars(history.get(instrument, {}), cfg, session)
        bars = bar_cache[key]
        if bars.empty:
            return pd.Series(dtype=float)
        values = np.log(bars["perp_close"] / bars["index_close"]) * 10_000.0
        return pd.Series(values.to_numpy(float), index=bars["timestamp"])

    output: list[pd.DataFrame] = []
    basis_lookback = int(settings["basis_lookback_hours"])
    basis_minimum = int(settings["basis_min_history_hours"])
    volume_lookback = int(settings["volume_lookback_hours"])
    for symbol, raw in signals.groupby("symbol", sort=True):
        sample = raw.sort_values("timestamp").copy()
        target = targets[str(symbol)]
        factors = [str(factor) for factor in target["factors"]]
        session = str(target["session"])
        sample["relative_basis_bps"] = basis(str(symbol), session).reindex(
            sample["timestamp"]
        ).to_numpy(float)
        target_funding = _last_observed_values(
            history.get(str(symbol), {}).get("funding", pd.DataFrame()),
            sample["decision_time"],
            "funding_rate",
        )
        long_funding = -target_funding * 10_000.0
        for factor in factors:
            beta = sample[f"beta_{factor}"].to_numpy(float)
            factor_basis = basis(factor, session).reindex(sample["timestamp"]).to_numpy(float)
            sample["relative_basis_bps"] -= beta * factor_basis
            factor_funding = _last_observed_values(
                history.get(factor, {}).get("funding", pd.DataFrame()),
                sample["decision_time"],
                "funding_rate",
            )
            long_funding += beta * factor_funding * 10_000.0
        sample["long_relative_funding_bps"] = long_funding
        prior_basis = sample["relative_basis_bps"].shift(1)
        basis_mean = prior_basis.rolling(basis_lookback, min_periods=basis_minimum).mean()
        basis_std = prior_basis.rolling(basis_lookback, min_periods=basis_minimum).std()
        sample["relative_basis_z"] = (
            sample["relative_basis_bps"] - basis_mean
        ) / basis_std
        prior_volume = sample["perp_quote_volume"].shift(1)
        volume_median = prior_volume.rolling(
            volume_lookback, min_periods=volume_lookback
        ).median()
        sample["relative_volume"] = sample["perp_quote_volume"] / volume_median
        for lag in (1, 2, 4, 8, 24):
            sample[f"residual_lag{lag}"] = sample["residual"].shift(lag)
            sample[f"z_lag{lag}"] = sample["z_score"].shift(lag)
        for window in (4, 8, 24, 72):
            sample[f"residual_sum{window}"] = sample["residual"].rolling(
                window, min_periods=window
            ).sum()
            sample[f"residual_vol{window}"] = sample["residual"].rolling(
                window, min_periods=window
            ).std()
        output.append(sample)
    enriched = pd.concat(output, ignore_index=True)
    cross = enriched.groupby("decision_time")["z_score"].agg(
        cross_z_median="median",
        cross_z_std="std",
        cross_z_min="min",
        cross_z_max="max",
    )
    return enriched.join(cross, on="decision_time").sort_values(["symbol", "timestamp"])


def _price_maps(
    history: dict[str, dict[str, pd.DataFrame]], instruments: set[str]
) -> dict[str, pd.Series]:
    maps: dict[str, pd.Series] = {}
    for instrument in instruments:
        frame = history.get(instrument, {}).get("perp", pd.DataFrame())
        if frame.empty:
            maps[instrument] = pd.Series(dtype=float)
            continue
        maps[instrument] = (
            frame.drop_duplicates("timestamp", keep="last")
            .set_index("timestamp")["open"]
            .astype(float)
            .sort_index()
        )
    return maps


def _funding_interval_sums(
    frame: pd.DataFrame, entry_times: pd.Series, exit_times: pd.Series
) -> np.ndarray:
    """Return sum(rate) for events in (entry, exit] for every row."""
    if frame.empty:
        return np.zeros(len(entry_times))
    funding = frame.dropna(subset=["timestamp", "funding_rate"]).sort_values("timestamp")
    if funding.empty:
        return np.zeros(len(entry_times))
    timestamps = pd.to_datetime(funding["timestamp"], utc=True).astype("int64").to_numpy()
    cumulative = np.r_[0.0, np.cumsum(funding["funding_rate"].to_numpy(float))]
    entry_ns = pd.to_datetime(entry_times, utc=True).astype("int64").to_numpy()
    exit_ns = pd.to_datetime(exit_times, utc=True).astype("int64").to_numpy()
    lower = np.searchsorted(timestamps, entry_ns, side="right")
    upper = np.searchsorted(timestamps, exit_ns, side="right")
    return cumulative[upper] - cumulative[lower]


def build_tradable_labels(
    signals: pd.DataFrame,
    history: dict[str, dict[str, pd.DataFrame]],
    cfg: dict[str, Any],
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Build exact delayed perpetual outcomes with beta frozen at decision time."""
    if signals.empty:
        return signals.copy()
    settings = cfg["tradable_research"]
    factor_settings = cfg["factor_strategy"]
    targets = target_specs(cfg)
    horizons = [int(value) for value in (horizons or settings["horizons_hours"])]
    selected = {str(symbol): targets[str(symbol)] for symbol in signals["symbol"].unique()}
    instruments = set(instrument_specs_for_targets(cfg, selected))
    prices = _price_maps(history, instruments)
    delay = pd.Timedelta(minutes=int(factor_settings["execution_delay_minutes"]))
    margin_fraction = float(cfg["execution"]["perp_margin_fraction"])
    opportunity_rate = float(cfg["execution"]["opportunity_cost_annual"])
    output: list[pd.DataFrame] = []
    for symbol, raw in signals.groupby("symbol", sort=True):
        sample = raw.sort_values("timestamp").copy()
        target = targets[str(symbol)]
        factors = [str(factor) for factor in target["factors"]]
        entry_times = pd.to_datetime(sample["decision_time"], utc=True) + delay
        entry_target = prices[str(symbol)].reindex(entry_times).to_numpy(float)
        session = str(target["session"])
        valid_session_points = set()
        if session == "us_rth":
            valid_session_points = set(
                zip(sample["execution_group"], pd.to_datetime(sample["timestamp"], utc=True))
            )
        for horizon in horizons:
            exit_clock = pd.to_datetime(sample["decision_time"], utc=True) + pd.Timedelta(
                hours=horizon
            )
            exit_times = exit_clock + delay
            valid = np.isfinite(entry_target) & (entry_target > 0)
            if session == "us_rth":
                valid &= np.fromiter(
                    (
                        (group, timestamp) in valid_session_points
                        for group, timestamp in zip(sample["execution_group"], exit_clock)
                    ),
                    dtype=bool,
                    count=len(sample),
                )
            exit_target = prices[str(symbol)].reindex(exit_times).to_numpy(float)
            valid &= np.isfinite(exit_target) & (exit_target > 0)
            long_gross_bps = (exit_target / entry_target - 1.0) * 10_000.0
            transaction_cost_bps = np.full(
                len(sample), instrument_roundtrip_cost_bps(cfg, str(symbol))
            )
            long_funding_bps = -_funding_interval_sums(
                history.get(str(symbol), {}).get("funding", pd.DataFrame()),
                entry_times,
                exit_times,
            ) * 10_000.0
            gross_beta = np.zeros(len(sample))
            for factor in factors:
                beta = sample[f"beta_{factor}"].to_numpy(float)
                factor_entry = prices[factor].reindex(entry_times).to_numpy(float)
                factor_exit = prices[factor].reindex(exit_times).to_numpy(float)
                valid &= (
                    np.isfinite(factor_entry)
                    & np.isfinite(factor_exit)
                    & (factor_entry > 0)
                    & (factor_exit > 0)
                    & np.isfinite(beta)
                )
                long_gross_bps -= beta * (factor_exit / factor_entry - 1.0) * 10_000.0
                transaction_cost_bps += np.abs(beta) * instrument_roundtrip_cost_bps(
                    cfg, factor
                )
                gross_beta += np.abs(beta)
                long_funding_bps += beta * _funding_interval_sums(
                    history.get(factor, {}).get("funding", pd.DataFrame()),
                    entry_times,
                    exit_times,
                ) * 10_000.0
            opportunity_bps = (
                (1.0 + gross_beta)
                * margin_fraction
                * opportunity_rate
                * horizon
                * 3600.0
                / SECONDS_PER_YEAR
                * 10_000.0
            )
            labeled = sample.loc[valid].copy()
            labeled["horizon_hours"] = horizon
            labeled["entry_time"] = entry_times.loc[valid].to_numpy()
            labeled["exit_time"] = exit_times.loc[valid].to_numpy()
            labeled["long_gross_bps"] = long_gross_bps[valid]
            labeled["long_funding_bps"] = long_funding_bps[valid]
            labeled["transaction_cost_bps"] = transaction_cost_bps[valid]
            labeled["opportunity_cost_bps"] = opportunity_bps[valid]
            labeled["long_net_bps"] = (
                labeled["long_gross_bps"]
                + labeled["long_funding_bps"]
                - labeled["transaction_cost_bps"]
                - labeled["opportunity_cost_bps"]
            )
            labeled["short_net_bps"] = (
                -labeled["long_gross_bps"]
                - labeled["long_funding_bps"]
                - labeled["transaction_cost_bps"]
                - labeled["opportunity_cost_bps"]
            )
            output.append(labeled)
    if not output:
        return pd.DataFrame()
    return pd.concat(output, ignore_index=True).sort_values(
        ["horizon_hours", "entry_time", "symbol"]
    )


def compact_tradable_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """Drop repeated market-bar columns that are not model inputs or audit fields."""
    if labels.empty:
        return labels.copy()
    identity = [
        "symbol",
        "category",
        "session",
        "timestamp",
        "decision_time",
        "session_date",
        "execution_group",
        "factor_model_type",
        "horizon_hours",
        "entry_time",
        "exit_time",
        "long_gross_bps",
        "long_funding_bps",
        "transaction_cost_bps",
        "opportunity_cost_bps",
        "long_net_bps",
        "short_net_bps",
    ]
    beta_columns = [column for column in labels if column.startswith("beta_")]
    requested = identity + BASE_MODEL_FEATURES + LAGGED_MODEL_FEATURES + beta_columns
    columns = list(dict.fromkeys(column for column in requested if column in labels))
    return labels[columns].copy()


def purged_chronological_partitions(
    labels: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Create train/validation/test partitions and purge crossing outcomes."""
    if labels.empty:
        return {name: labels.copy() for name in ("train", "validation", "test")}, {}
    settings = cfg["tradable_research"]
    dates = sorted(pd.to_datetime(labels["entry_time"], utc=True).dt.normalize().unique())
    if len(dates) < 3:
        empty = labels.iloc[0:0].copy()
        return (
            {"train": labels.copy(), "validation": empty, "test": empty},
            {"calendar_days": len(dates), "purged_rows": 0},
        )
    train_count = int(len(dates) * float(settings["train_fraction"]))
    validation_count = int(len(dates) * float(settings["validation_fraction"]))
    train_count = min(max(train_count, 1), len(dates) - 2)
    validation_count = min(max(validation_count, 1), len(dates) - train_count - 1)
    validation_start = pd.Timestamp(dates[train_count])
    test_start = pd.Timestamp(dates[train_count + validation_count])
    entry = pd.to_datetime(labels["entry_time"], utc=True)
    exit_time = pd.to_datetime(labels["exit_time"], utc=True)
    partitions = {
        "train": labels.loc[entry.lt(validation_start) & exit_time.lt(validation_start)].copy(),
        "validation": labels.loc[
            entry.ge(validation_start) & entry.lt(test_start) & exit_time.lt(test_start)
        ].copy(),
        "test": labels.loc[entry.ge(test_start)].copy(),
    }
    metadata = {
        "validation_start": validation_start,
        "test_start": test_start,
        "calendar_days": len(dates),
        "purged_rows": len(labels) - sum(len(frame) for frame in partitions.values()),
    }
    return partitions, metadata


def _candidate_exposures(
    row: pd.Series, cfg: dict[str, Any], notional: float
) -> dict[str, float]:
    targets = target_specs(cfg)
    symbol = str(row["symbol"])
    direction = int(row["direction"])
    exposures = {symbol: direction * notional}
    for factor in targets[symbol]["factors"]:
        exposures[str(factor)] = exposures.get(str(factor), 0.0) - (
            direction * float(row[f"beta_{factor}"]) * notional
        )
    return {instrument: value for instrument, value in exposures.items() if abs(value) > 1e-9}


def build_model_candidates(
    predicted: pd.DataFrame, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Apply the economic hurdle, no-overlap rule and portfolio limits."""
    if predicted.empty:
        return predicted.copy()
    settings = cfg["tradable_research"]
    sample = predicted.copy()
    sample["direction"] = np.sign(sample["prediction_bps"]).astype(int)
    sample["predicted_edge_bps"] = (
        sample["prediction_bps"].abs()
        - sample["transaction_cost_bps"]
        - sample["opportunity_cost_bps"]
        - float(settings["safety_margin_bps"])
    )
    sample = sample.loc[sample["direction"].ne(0) & sample["predicted_edge_bps"].gt(0)].copy()
    if sample.empty:
        return sample
    non_overlapping: list[pd.Series] = []
    for _, symbol_rows in sample.sort_values(["symbol", "entry_time"]).groupby("symbol"):
        next_free: pd.Timestamp | None = None
        for _, row in symbol_rows.iterrows():
            entry_time = pd.Timestamp(row["entry_time"])
            if next_free is not None and entry_time < next_free:
                continue
            non_overlapping.append(row)
            next_free = pd.Timestamp(row["exit_time"])
    candidates = pd.DataFrame(non_overlapping)
    if candidates.empty:
        return candidates

    limits = cfg["factor_strategy"].get("portfolio_limits", {})
    capital = float(cfg["execution"]["initial_capital"])
    notional = float(cfg["execution"]["notional_per_leg"])
    max_pairs = int(limits.get("max_concurrent_pairs", 10**9))
    max_gross = float(limits.get("max_gross_leverage", np.inf)) * capital
    max_net = float(limits.get("max_instrument_net_fraction", np.inf)) * capital
    accepted: list[pd.Series] = []
    active: list[pd.Series] = []
    ordered = candidates.sort_values(
        ["entry_time", "predicted_edge_bps", "symbol"], ascending=[True, False, True]
    )
    for _, row in ordered.iterrows():
        entry_time = pd.Timestamp(row["entry_time"])
        active = [item for item in active if pd.Timestamp(item["exit_time"]) > entry_time]
        if len(active) >= max_pairs:
            continue
        active_exposures: dict[str, float] = {}
        gross = 0.0
        for item in active:
            exposures = _candidate_exposures(item, cfg, notional)
            gross += sum(abs(value) for value in exposures.values())
            for instrument, value in exposures.items():
                active_exposures[instrument] = active_exposures.get(instrument, 0.0) + value
        new_exposures = _candidate_exposures(row, cfg, notional)
        candidate_gross = sum(abs(value) for value in new_exposures.values())
        if gross + candidate_gross > max_gross:
            continue
        combined = active_exposures.copy()
        for instrument, value in new_exposures.items():
            combined[instrument] = combined.get(instrument, 0.0) + value
        if any(abs(value) > max_net for value in combined.values()):
            continue
        active.append(row)
        accepted.append(row)
    return pd.DataFrame(accepted).sort_values(["entry_time", "symbol"]).reset_index(drop=True)


def run_netted_portfolio(
    candidates: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Net simultaneous instrument deltas before charging execution costs."""
    empty_metrics = {
        "trades": 0,
        "symbols": 0,
        "gross_pnl": 0.0,
        "funding_pnl": 0.0,
        "netted_transaction_cost": 0.0,
        "unnetted_transaction_cost": 0.0,
        "cost_saving": 0.0,
        "opportunity_cost": 0.0,
        "net_pnl": 0.0,
        "stressed_net_pnl": 0.0,
        "break_even_cost_multiplier": np.nan,
        "win_rate_unnetted": np.nan,
    }
    if candidates.empty:
        return empty_metrics, pd.DataFrame()
    notional = float(cfg["execution"]["notional_per_leg"])
    events: list[dict[str, Any]] = []
    gross_pnl = 0.0
    funding_pnl = 0.0
    unnetted_cost = 0.0
    unnetted_wins = 0
    for _, row in candidates.iterrows():
        direction = int(row["direction"])
        gross_bps = direction * float(row["long_gross_bps"])
        funding_bps = direction * float(row["long_funding_bps"])
        gross_pnl += gross_bps * notional / 10_000.0
        funding_pnl += funding_bps * notional / 10_000.0
        pair_cost = float(row["transaction_cost_bps"]) * notional / 10_000.0
        unnetted_cost += pair_cost
        pair_net = (
            gross_bps
            + funding_bps
            - float(row["transaction_cost_bps"])
            - float(row["opportunity_cost_bps"])
        )
        unnetted_wins += int(pair_net > 0)
        for instrument, exposure in _candidate_exposures(row, cfg, notional).items():
            events.append(
                {
                    "timestamp": row["entry_time"],
                    "instrument": instrument,
                    "delta_notional": exposure,
                }
            )
            events.append(
                {
                    "timestamp": row["exit_time"],
                    "instrument": instrument,
                    "delta_notional": -exposure,
                }
            )
    event_frame = (
        pd.DataFrame(events)
        .groupby(["timestamp", "instrument"], as_index=False)["delta_notional"]
        .sum()
        .sort_values(["timestamp", "instrument"])
    )
    event_frame = event_frame.loc[event_frame["delta_notional"].abs().gt(1e-9)].copy()
    event_frame["one_way_cost_bps"] = event_frame["instrument"].map(
        lambda instrument: instrument_roundtrip_cost_bps(cfg, str(instrument)) / 2.0
    )
    event_frame["transaction_cost"] = (
        event_frame["delta_notional"].abs()
        * event_frame["one_way_cost_bps"]
        / 10_000.0
    )
    netted_cost = float(event_frame["transaction_cost"].sum())

    opportunity_cost = 0.0
    active_exposures: dict[str, float] = {}
    previous_time: pd.Timestamp | None = None
    margin_fraction = float(cfg["execution"]["perp_margin_fraction"])
    opportunity_rate = float(cfg["execution"]["opportunity_cost_annual"])
    for timestamp, timestamp_events in event_frame.groupby("timestamp", sort=True):
        timestamp = pd.Timestamp(timestamp)
        if previous_time is not None:
            held_seconds = (timestamp - previous_time).total_seconds()
            gross_notional = sum(abs(value) for value in active_exposures.values())
            opportunity_cost += (
                gross_notional
                * margin_fraction
                * opportunity_rate
                * held_seconds
                / SECONDS_PER_YEAR
            )
        for _, event in timestamp_events.iterrows():
            instrument = str(event["instrument"])
            active_exposures[instrument] = active_exposures.get(instrument, 0.0) + float(
                event["delta_notional"]
            )
        active_exposures = {
            instrument: value
            for instrument, value in active_exposures.items()
            if abs(value) > 1e-9
        }
        previous_time = timestamp
    net_pnl = gross_pnl + funding_pnl - netted_cost - opportunity_cost
    stress_multiplier = float(
        cfg["tradable_research"]["validation_cost_stress_multiplier"]
    )
    stressed_net_pnl = (
        gross_pnl
        + funding_pnl
        - stress_multiplier * netted_cost
        - opportunity_cost
    )
    break_even_cost_multiplier = (
        (gross_pnl + funding_pnl - opportunity_cost) / netted_cost
        if netted_cost > 0
        else np.nan
    )
    metrics = {
        "trades": len(candidates),
        "symbols": int(candidates["symbol"].nunique()),
        "gross_pnl": gross_pnl,
        "funding_pnl": funding_pnl,
        "netted_transaction_cost": netted_cost,
        "unnetted_transaction_cost": unnetted_cost,
        "cost_saving": unnetted_cost - netted_cost,
        "opportunity_cost": opportunity_cost,
        "net_pnl": net_pnl,
        "stressed_net_pnl": stressed_net_pnl,
        "break_even_cost_multiplier": break_even_cost_multiplier,
        "win_rate_unnetted": unnetted_wins / len(candidates),
    }
    return metrics, event_frame.reset_index(drop=True)


def _training_feature_schema(
    train: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    base_features = [
        feature
        for feature in BASE_MODEL_FEATURES + LAGGED_MODEL_FEATURES
        if feature in train and train[feature].notna().any()
    ]
    symbols = sorted(str(symbol) for symbol in train["symbol"].dropna().unique())
    return base_features, symbols


def _model_frame(
    raw: pd.DataFrame, base_features: list[str], symbols: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    frame = raw.copy()
    symbol_features: list[str] = []
    for symbol in symbols:
        column = f"symbol_{symbol}"
        frame[column] = frame["symbol"].eq(symbol).astype(float)
        symbol_features.append(column)
    features = base_features + symbol_features
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=features + ["long_gross_bps"])
    return frame, features


def run_fixed_xgboost_research(
    partitions: dict[str, pd.DataFrame], cfg: dict[str, Any]
) -> dict[str, Any]:
    """Run an expanding monthly model aligned with the reversion hypothesis."""
    import xgboost as xgb
    from xgboost import XGBRegressor

    base_features, symbols = _training_feature_schema(partitions["train"])
    _, features = _model_frame(partitions["train"], base_features, symbols)
    settings = cfg["tradable_research"]
    model_settings = settings["xgboost"]
    entry_z = float(cfg["factor_strategy"]["entry_z"])
    stress_multiplier = float(settings["validation_cost_stress_multiplier"])

    def new_model() -> XGBRegressor:
        return XGBRegressor(
            n_estimators=int(model_settings["n_estimators"]),
            max_depth=int(model_settings["max_depth"]),
            learning_rate=float(model_settings["learning_rate"]),
            min_child_weight=float(model_settings["min_child_weight"]),
            subsample=float(model_settings["subsample"]),
            colsample_bytree=float(model_settings["colsample_bytree"]),
            reg_lambda=float(model_settings["reg_lambda"]),
            reg_alpha=float(model_settings["reg_alpha"]),
            objective="reg:squarederror",
            random_state=int(cfg["project"]["random_seed"]),
            n_jobs=4,
        )

    def walkforward_predictions(
        development: pd.DataFrame, scoring: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if development.empty or scoring.empty:
            return scoring.iloc[0:0].copy(), pd.DataFrame(
                columns=["feature", "mean_abs_shap"]
            )
        score_entry = pd.to_datetime(scoring["entry_time"], utc=True)
        periods = score_entry.dt.tz_localize(None).dt.to_period("M")
        combined = pd.concat([development, scoring], ignore_index=True)
        combined_entry = pd.to_datetime(combined["entry_time"], utc=True)
        combined_exit = pd.to_datetime(combined["exit_time"], utc=True)
        predicted_blocks: list[pd.DataFrame] = []
        shap_totals = np.zeros(len(features), dtype=float)
        shap_rows = 0
        random_seed = int(cfg["project"]["random_seed"])

        for period in sorted(periods.unique()):
            score_mask = periods.eq(period)
            block_raw = scoring.loc[score_mask].copy()
            block_start = pd.Timestamp(period.start_time, tz="UTC")
            block_start = max(
                block_start, score_entry.loc[score_mask].min().normalize()
            )
            train_raw = combined.loc[
                combined_entry.lt(block_start) & combined_exit.lt(block_start)
            ].copy()
            train, _ = _model_frame(train_raw, base_features, symbols)
            block, _ = _model_frame(block_raw, base_features, symbols)
            train = train.loc[train["z_score"].abs().ge(entry_z)].copy()
            block = block.loc[block["z_score"].abs().ge(entry_z)].copy()
            if train.empty or block.empty:
                continue
            train["reversion_gross_bps"] = (
                -np.sign(train["z_score"]) * train["long_gross_bps"]
            )
            model = new_model()
            model.fit(train[features], train["reversion_gross_bps"])
            block["predicted_reversion_gross_bps"] = np.maximum(
                model.predict(block[features]), 0.0
            )
            # build_model_candidates applies the baseline hurdle. Reserve the
            # additional stressed cost here so every entry clears the full stress.
            executable_edge = np.maximum(
                block["predicted_reversion_gross_bps"]
                - (stress_multiplier - 1.0) * block["transaction_cost_bps"],
                0.0,
            )
            block["prediction_bps"] = -np.sign(block["z_score"]) * executable_edge
            predicted_blocks.append(block)

            shap_sample = block[features].sample(
                min(200, len(block)), random_state=random_seed
            )
            contributions = model.get_booster().predict(
                xgb.DMatrix(shap_sample, feature_names=features), pred_contribs=True
            )
            shap_totals += np.abs(contributions[:, :-1]).sum(axis=0)
            shap_rows += len(shap_sample)

        predicted = (
            pd.concat(predicted_blocks, ignore_index=True)
            if predicted_blocks
            else scoring.iloc[0:0].copy()
        )
        shap_importance = pd.DataFrame(
            {
                "feature": features,
                "mean_abs_shap": shap_totals / max(shap_rows, 1),
            }
        ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
        return predicted, shap_importance

    if partitions["train"].empty or partitions["validation"].empty:
        return {
            "status": "INSUFFICIENT_HISTORY",
            "features": features,
            "validation_metrics": run_netted_portfolio(pd.DataFrame(), cfg)[0],
            "test_metrics": None,
            "test_locked": True,
            "shap_importance": pd.DataFrame(columns=["feature", "mean_abs_shap"]),
        }

    validation, shap_importance = walkforward_predictions(
        partitions["train"], partitions["validation"]
    )
    validation_candidates = build_model_candidates(validation, cfg)
    validation_metrics, validation_events = run_netted_portfolio(
        validation_candidates, cfg
    )
    validation_entries = pd.to_datetime(
        partitions["validation"]["entry_time"], utc=True
    )
    validation_midpoint = validation_entries.min() + (
        validation_entries.max() - validation_entries.min()
    ) / 2
    candidate_entries = pd.to_datetime(validation_candidates["entry_time"], utc=True)
    validation_subperiod_metrics = {
        "first_half": run_netted_portfolio(
            validation_candidates.loc[candidate_entries.lt(validation_midpoint)], cfg
        )[0],
        "second_half": run_netted_portfolio(
            validation_candidates.loc[candidate_entries.ge(validation_midpoint)], cfg
        )[0],
    }
    validation_pass = validation_authorizes_test(
        validation_metrics, cfg, validation_subperiod_metrics
    )

    unseen_cutoff = pd.Timestamp(str(settings["unseen_test_not_before"]), tz="UTC")
    test_entries = pd.to_datetime(partitions["test"]["entry_time"], utc=True)
    test_is_unseen = not partitions["test"].empty and test_entries.min() >= unseen_cutoff
    result: dict[str, Any] = {
        "status": (
            "VALIDATION_PASS_FORWARD_TEST_REQUIRED"
            if validation_pass and not test_is_unseen
            else "VALIDATION_PASS"
            if validation_pass
            else "VALIDATION_FAIL"
        ),
        "features": features,
        "validation_metrics": validation_metrics,
        "validation_candidates": validation_candidates,
        "validation_events": validation_events,
        "validation_subperiod_metrics": validation_subperiod_metrics,
        "test_metrics": None,
        "test_candidates": pd.DataFrame(),
        "test_events": pd.DataFrame(),
        "test_locked": True,
        "test_lock_reason": (
            "validation_failed"
            if not validation_pass
            else "previous_test_period_already_observed"
            if not test_is_unseen
            else ""
        ),
        "shap_importance": shap_importance,
    }
    if validation_pass and test_is_unseen:
        development = pd.concat(
            [partitions["train"], partitions["validation"]], ignore_index=True
        )
        test, _ = walkforward_predictions(development, partitions["test"])
        test_candidates = build_model_candidates(test, cfg)
        test_metrics, test_events = run_netted_portfolio(test_candidates, cfg)
        test_pass = validation_authorizes_test(test_metrics, cfg)
        result.update(
            {
                "status": "TEST_PASS" if test_pass else "TEST_FAIL",
                "test_metrics": test_metrics,
                "test_candidates": test_candidates,
                "test_events": test_events,
                "test_locked": False,
            }
        )
    return result


def validation_authorizes_test(
    metrics: dict[str, Any],
    cfg: dict[str, Any],
    subperiod_metrics: dict[str, dict[str, Any]] | None = None,
) -> bool:
    """Fail closed before any final-test predictions are generated."""
    settings = cfg["tradable_research"]
    total_pass = bool(
        int(metrics.get("trades", 0)) >= int(settings["minimum_validation_trades"])
        and float(metrics.get("gross_pnl", 0.0)) > 0.0
        and float(metrics.get("net_pnl", 0.0)) > 0.0
        and float(metrics.get("stressed_net_pnl", 0.0)) > 0.0
    )
    if not total_pass or subperiod_metrics is None:
        return total_pass
    minimum = int(settings["minimum_validation_subperiod_trades"])
    return all(
        int(period.get("trades", 0)) >= minimum
        and float(period.get("gross_pnl", 0.0)) > 0.0
        and float(period.get("net_pnl", 0.0)) > 0.0
        and float(period.get("stressed_net_pnl", 0.0)) > 0.0
        for period in subperiod_metrics.values()
    )


def validate_tradable_research_config(cfg: dict[str, Any]) -> None:
    settings = cfg["tradable_research"]
    if float(settings["dynamic_beta_half_life_hours"]) <= 0:
        raise ValueError("dynamic_beta_half_life_hours must be positive")
    horizons = [int(value) for value in settings["horizons_hours"]]
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("tradable_research horizons must be positive")
    if int(settings["primary_horizon_hours"]) not in horizons:
        raise ValueError("primary_horizon_hours must be in horizons_hours")
    fractions = [
        float(settings["train_fraction"]),
        float(settings["validation_fraction"]),
        float(settings["test_fraction"]),
    ]
    if any(value <= 0 for value in fractions) or not np.isclose(sum(fractions), 1.0):
        raise ValueError("tradable_research split fractions must be positive and sum to 1")
    if int(settings["minimum_validation_trades"]) <= 0:
        raise ValueError("minimum_validation_trades must be positive")
    if int(settings["minimum_validation_subperiod_trades"]) <= 0:
        raise ValueError("minimum_validation_subperiod_trades must be positive")
    if float(settings["validation_cost_stress_multiplier"]) < 1.0:
        raise ValueError("validation_cost_stress_multiplier must be at least 1")
    pd.Timestamp(str(settings["unseen_test_not_before"]), tz="UTC")


def run_tradable_research(
    cfg: dict[str, Any],
    start_text: str | None = None,
    end_text: str | None = None,
    refresh: bool = False,
    categories: list[str] | None = None,
    symbols: list[str] | None = None,
) -> dict[str, Any]:
    """Run the no-report, fail-closed tradable-return research path."""
    validate_tradable_research_config(cfg)
    default_categories = [str(value) for value in cfg["tradable_research"]["target_categories"]]
    effective_categories = (
        categories if categories is not None else ([] if symbols else default_categories)
    )
    selected_targets = research_target_specs(
        cfg,
        symbols=symbols,
        categories=effective_categories,
    )
    start, end_exclusive = historical_bounds(
        start_text or str(cfg["historical"]["factor_start_date"]),
        end_text or str(cfg["historical"]["end_date"]),
    )
    root = Path(cfg["_root"])
    processed = root / str(cfg["historical"]["cache_dir"]) / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    key = (
        f"{start.strftime('%Y%m%d')}_"
        f"{(end_exclusive - pd.Timedelta(days=1)).strftime('%Y%m%d')}"
    )
    scope_parts = effective_categories or default_categories
    if symbols:
        scope_parts = ["symbols", *sorted(symbols)]
    scope = "_".join(str(value) for value in scope_parts)
    signal_fingerprint, label_fingerprint = _research_cache_fingerprints(cfg)
    signal_cache = processed / (
        f"tradable_dynamic_signals_v2_{scope}_{key}_{signal_fingerprint}.pkl"
    )
    label_cache = processed / (
        f"tradable_labels_v3_{scope}_{key}_{label_fingerprint}.pkl"
    )
    history, errors = download_factor_market_history(
        cfg,
        start,
        end_exclusive,
        refresh=refresh,
        target_symbols=list(selected_targets),
        allow_unavailable=True,
    )
    availability = history_availability(history, cfg, selected_targets)
    if signal_cache.exists() and not refresh:
        signals = pd.read_pickle(signal_cache)
        quality = (
            signals.groupby(["symbol", "category", "session"], as_index=False)
            .agg(
                hourly_rows=("timestamp", "size"),
                quality_rows=("quality_ok", "sum"),
                trading_days=("session_date", "nunique"),
            )
        )
    else:
        panel, quality = build_factor_hourly_panel(history, cfg)
        panel = (
            panel.loc[panel["symbol"].isin(selected_targets)].copy()
            if not panel.empty
            else panel
        )
        signals = build_dynamic_factor_signals(panel, cfg)
        signals.to_pickle(signal_cache)
    if label_cache.exists() and not refresh:
        labels = pd.read_pickle(label_cache)
    else:
        features = add_tradable_features(signals, history, cfg)
        labels = build_tradable_labels(
            features,
            history,
            cfg,
            horizons=[int(cfg["tradable_research"]["primary_horizon_hours"])],
        )
        labels = compact_tradable_labels(labels)
        labels.to_pickle(label_cache)
    if labels.empty:
        primary = labels.copy()
    else:
        primary = labels.loc[
            labels["horizon_hours"].eq(
                int(cfg["tradable_research"]["primary_horizon_hours"])
            )
        ].copy()
    partitions, split_metadata = purged_chronological_partitions(primary, cfg)

    minimum_days = {
        "train": int(cfg["tradable_research"]["minimum_train_days"]),
        "validation": int(cfg["tradable_research"]["minimum_validation_days"]),
        "test": int(cfg["tradable_research"]["minimum_test_days"]),
    }
    observed_days = {
        name: int(pd.to_datetime(frame["entry_time"], utc=True).dt.normalize().nunique())
        if not frame.empty
        else 0
        for name, frame in partitions.items()
    }
    history_sufficient = all(observed_days[name] >= minimum_days[name] for name in minimum_days)
    if history_sufficient:
        model_result = run_fixed_xgboost_research(partitions, cfg)
    else:
        model_result = {
            "status": "INSUFFICIENT_HISTORY",
            "validation_metrics": run_netted_portfolio(pd.DataFrame(), cfg)[0],
            "test_metrics": None,
            "test_locked": True,
            "shap_importance": pd.DataFrame(columns=["feature", "mean_abs_shap"]),
        }

    availability_path = processed / f"tradable_availability_{scope}_{key}.csv"
    availability.to_csv(availability_path, index=False)

    print("Exchange-observed availability")
    print(
        availability[
            ["instrument", "is_target", "common_price_start", "common_price_end"]
        ].to_string(index=False)
    )
    print("\nPurged split")
    if split_metadata:
        print(
            f"validation_start={split_metadata.get('validation_start')} "
            f"test_start={split_metadata.get('test_start')} "
            f"purged_rows={split_metadata.get('purged_rows', 0)}"
        )
    print(
        pd.DataFrame(
            [
                {
                    "partition": name,
                    "rows": len(partitions[name]),
                    "observed_days": observed_days[name],
                    "minimum_days": minimum_days[name],
                }
                for name in ("train", "validation", "test")
            ]
        ).to_string(index=False)
    )
    print(f"\nModel status: {model_result['status']}")
    print(pd.DataFrame([model_result["validation_metrics"]]).to_string(index=False))
    subperiod_metrics = model_result.get("validation_subperiod_metrics", {})
    if subperiod_metrics:
        print("\nValidation chronological halves")
        print(
            pd.DataFrame(
                [
                    {"period": name, **metrics}
                    for name, metrics in subperiod_metrics.items()
                ]
            ).to_string(index=False)
        )
    if model_result["test_locked"]:
        print(
            "Test status: LOCKED "
            f"({model_result.get('test_lock_reason', 'validation_failed')})"
        )
    else:
        print("Test status: OPENED_AFTER_VALIDATION_PASS")
        print(pd.DataFrame([model_result["test_metrics"]]).to_string(index=False))
    if not model_result["shap_importance"].empty:
        print("\nValidation TreeSHAP importance")
        print(model_result["shap_importance"].head(15).to_string(index=False))
    print(f"Download errors: {len(errors)}")
    print(f"Processed labels: {label_cache.resolve()}")
    return {
        "availability": availability,
        "quality": quality,
        "labels": labels,
        "partitions": partitions,
        "split_metadata": split_metadata,
        "observed_days": observed_days,
        "errors": errors,
        "model": model_result,
    }
