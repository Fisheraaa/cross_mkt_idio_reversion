from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest import SECONDS_PER_YEAR, performance_metrics
from factor_config import (
    instrument_history_start,
    instrument_cost_profile,
    instrument_specs,
    instrument_specs_for_targets,
    research_target_specs,
    session_parameters,
    target_specs,
    validate_factor_configuration,
)
from historical_prescreen import historical_bounds
from historical_providers import BinanceHistoricalMinutes
from statsmodels.tsa.stattools import mackinnonp


def _cache_path(
    cache_dir: Path,
    provider_symbol: str,
    kind: str,
    interval: str | None,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> Path:
    start_text = start.strftime("%Y%m%d")
    end_text = (end_exclusive - pd.Timedelta(days=1)).strftime("%Y%m%d")
    suffix = "funding" if kind == "funding" else f"{kind}_{interval}"
    return cache_dir / f"binance_{provider_symbol}_{suffix}_{start_text}_{end_text}.pkl"


def _read_or_fetch(path: Path, refresh: bool, fetch: Any) -> pd.DataFrame:
    if path.exists() and not refresh:
        return pd.read_pickle(path)
    frame = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(path)
    return frame


def _interval_delta(interval: str | None) -> pd.Timedelta:
    if interval is None:
        return pd.Timedelta(milliseconds=1)
    minutes = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60}
    if interval not in minutes:
        raise ValueError(f"Unsupported cache interval: {interval}")
    return pd.Timedelta(minutes=minutes[interval])


def _read_or_extend_cache(
    path: Path,
    cache_dir: Path,
    provider_symbol: str,
    kind: str,
    interval: str | None,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    refresh: bool,
    fetch_range: Any,
) -> tuple[pd.DataFrame, str]:
    """Reuse overlapping dated caches and fetch only uncovered boundaries."""
    if path.exists() and not refresh:
        return pd.read_pickle(path), "cache"
    if refresh:
        frame = fetch_range(start, end_exclusive)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_pickle(path)
        return frame, "download"

    suffix = "funding" if kind == "funding" else f"{kind}_{interval}"
    cached_frames: list[pd.DataFrame] = []
    for candidate in cache_dir.glob(f"binance_{provider_symbol}_{suffix}_*.pkl"):
        try:
            cached = pd.read_pickle(candidate)
        except (OSError, ValueError, EOFError):
            continue
        if not cached.empty and "timestamp" in cached:
            cached_frames.append(cached)

    fetched_frames: list[pd.DataFrame] = []
    source = "download"
    if cached_frames:
        combined_cache = (
            pd.concat(cached_frames, ignore_index=True)
            .drop_duplicates("timestamp", keep="last")
            .sort_values("timestamp")
        )
        cached_start = pd.Timestamp(combined_cache["timestamp"].iloc[0])
        cached_end = pd.Timestamp(combined_cache["timestamp"].iloc[-1])
        if start < cached_start:
            fetched_frames.append(fetch_range(start, min(cached_start, end_exclusive)))
        next_timestamp = cached_end + _interval_delta(interval)
        if next_timestamp < end_exclusive:
            fetched_frames.append(fetch_range(max(start, next_timestamp), end_exclusive))
        cached_frames = [combined_cache]
        source = "cache+download" if fetched_frames else "cache-reused"
    else:
        fetched_frames.append(fetch_range(start, end_exclusive))

    frames = [frame for frame in cached_frames + fetched_frames if not frame.empty]
    if frames:
        frame = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates("timestamp", keep="last")
            .sort_values("timestamp")
        )
        timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.loc[timestamps.ge(start) & timestamps.lt(end_exclusive)].reset_index(drop=True)
    else:
        frame = pd.DataFrame()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(path)
    return frame, source


def download_factor_market_history(
    cfg: dict[str, Any],
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    refresh: bool = False,
    target_symbols: list[str] | None = None,
    target_categories: list[str] | None = None,
    allow_unavailable: bool = False,
) -> tuple[dict[str, dict[str, pd.DataFrame]], list[dict[str, str]]]:
    validate_factor_configuration(cfg)
    root = Path(cfg["_root"])
    cache_dir = root / str(cfg["historical"]["cache_dir"])
    selected_targets = research_target_specs(cfg, target_symbols, target_categories)
    if not selected_targets:
        raise ValueError("The requested factor target subset is empty")
    instruments = {
        canonical: str(spec["perp_symbol"])
        for canonical, spec in instrument_specs_for_targets(cfg, selected_targets).items()
    }
    all_specs = instrument_specs(cfg)
    settings = cfg["factor_strategy"]
    execution_interval = str(settings.get("execution_interval", "1m"))
    signal_interval = str(settings.get("signal_interval", execution_interval))
    history: dict[str, dict[str, pd.DataFrame]] = {}
    errors: list[dict[str, str]] = []

    def download_instrument(
        canonical: str, provider_symbol: str
    ) -> tuple[str, dict[str, pd.DataFrame], list[dict[str, str]], list[str]]:
        provider = BinanceHistoricalMinutes(cfg)
        datasets: dict[str, pd.DataFrame] = {}
        instrument_errors: list[dict[str, str]] = []
        messages: list[str] = []
        dataset_start = instrument_history_start(all_specs[canonical], start)
        for kind in ("perp", "mark", "index", "funding"):
            interval = None if kind == "funding" else (
                execution_interval if kind == "perp" else signal_interval
            )
            path = _cache_path(
                cache_dir, provider_symbol, kind, interval, start, end_exclusive
            )
            try:
                if kind == "funding":
                    frame, source = _read_or_extend_cache(
                        path,
                        cache_dir,
                        provider_symbol,
                        kind,
                        interval,
                        dataset_start,
                        end_exclusive,
                        refresh,
                        lambda lower, upper, s=provider_symbol: provider.fetch_funding(
                            s, lower, upper
                        ),
                    )
                else:
                    frame, source = _read_or_extend_cache(
                        path,
                        cache_dir,
                        provider_symbol,
                        kind,
                        interval,
                        dataset_start,
                        end_exclusive,
                        refresh,
                        lambda lower, upper, s=provider_symbol, k=kind: provider.fetch_klines(
                            s, k, lower, upper, interval=str(interval)
                        ),
                    )
            except Exception as exc:
                frame = pd.DataFrame()
                source = "failed"
                instrument_errors.append(
                    {"symbol": canonical, "dataset": kind, "error": str(exc)}
                )
                messages.append(f"warning Binance {provider_symbol} {kind}: {exc}")
            if frame.empty and not any(
                row["symbol"] == canonical and row["dataset"] == kind
                for row in instrument_errors
            ) and not allow_unavailable:
                instrument_errors.append(
                    {"symbol": canonical, "dataset": kind, "error": "no rows returned"}
                )
            datasets[kind] = frame
            interval_text = "" if interval is None else f"/{interval}"
            messages.append(
                f"{source} {canonical}/{kind}{interval_text} rows={len(frame):,}"
            )
        return canonical, datasets, instrument_errors, messages

    workers = max(1, int(cfg["historical"].get("max_parallel_instruments", 1)))
    if workers == 1:
        completed = (
            download_instrument(canonical, provider_symbol)
            for canonical, provider_symbol in instruments.items()
        )
        for canonical, datasets, instrument_errors, messages in completed:
            history[canonical] = datasets
            errors.extend(instrument_errors)
            for message in messages:
                print(message, flush=True)
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(instruments))) as executor:
            futures = {
                executor.submit(download_instrument, canonical, provider_symbol): canonical
                for canonical, provider_symbol in instruments.items()
            }
            for future in as_completed(futures):
                canonical, datasets, instrument_errors, messages = future.result()
                history[canonical] = datasets
                errors.extend(instrument_errors)
                for message in messages:
                    print(message, flush=True)
    return history, errors


def _hourly_bars(
    datasets: dict[str, pd.DataFrame], cfg: dict[str, Any], session: str
) -> pd.DataFrame:
    required = [datasets.get(kind, pd.DataFrame()) for kind in ("perp", "mark", "index")]
    if any(frame.empty for frame in required):
        return pd.DataFrame()
    def bucket(frame: pd.DataFrame) -> pd.DataFrame:
        sample = frame.copy()
        timestamps = pd.to_datetime(sample["timestamp"], utc=True, errors="coerce")
        if session == "us_rth":
            local = timestamps.dt.tz_convert("America/New_York")
            minute_of_day = local.dt.hour * 60 + local.dt.minute
            from_open = minute_of_day - (9 * 60 + 30)
            regular = local.dt.dayofweek.lt(5) & from_open.ge(0) & from_open.lt(6 * 60)
            sample = sample.loc[regular].copy()
            local = timestamps.loc[regular].dt.tz_convert("America/New_York")
            minute_of_day = local.dt.hour * 60 + local.dt.minute
            sample["session_date"] = local.dt.date
            sample["hour_number"] = ((minute_of_day - (9 * 60 + 30)) // 60).astype(int)
            sample["hour_timestamp"] = (
                local.dt.normalize()
                + pd.Timedelta(hours=9, minutes=30)
                + pd.to_timedelta(sample["hour_number"], unit="h")
            ).dt.tz_convert("UTC")
        else:
            sample["session_date"] = timestamps.dt.date
            sample["hour_number"] = timestamps.dt.hour
            sample["hour_timestamp"] = timestamps.dt.floor("h")
        return sample

    def aggregate(frame: pd.DataFrame, prefix: str, count_name: str) -> pd.DataFrame:
        sample = bucket(frame)
        if sample.empty:
            return pd.DataFrame()
        aggregations: dict[str, tuple[str, str]] = {
            count_name: ("timestamp", "size"),
            f"{prefix}_open": ("open", "first"),
            f"{prefix}_high": ("high", "max"),
            f"{prefix}_low": ("low", "min"),
            f"{prefix}_close": ("close", "last"),
        }
        if "quote_volume" in sample:
            aggregations[f"{prefix}_quote_volume"] = ("quote_volume", "sum")
        return (
            sample.groupby(
                ["session_date", "hour_number", "hour_timestamp"], as_index=False
            )
            .agg(**aggregations)
            .rename(columns={"hour_timestamp": "timestamp"})
        )

    perp = aggregate(required[0], "perp", "execution_bar_count")
    mark = aggregate(required[1], "mark", "mark_bar_count")
    index = aggregate(required[2], "index", "index_bar_count")
    if perp.empty or mark.empty or index.empty:
        return pd.DataFrame()
    hourly = perp.merge(
        mark.drop(columns=["session_date", "hour_number"]),
        on="timestamp",
        how="inner",
        validate="one_to_one",
    ).merge(
        index.drop(columns=["session_date", "hour_number"]),
        on="timestamp",
        how="inner",
        validate="one_to_one",
    )
    hourly["maximum_contract_mark_gap_bps"] = (
        np.log(hourly["perp_close"] / hourly["mark_close"]).abs() * 10_000.0
    )
    if "perp_quote_volume" not in hourly:
        hourly["perp_quote_volume"] = np.nan
    settings = cfg["factor_strategy"]
    hourly["quality_ok"] = (
        hourly["execution_bar_count"].ge(
            int(settings.get("minimum_execution_bars_per_hour", 1))
        )
        & hourly["mark_bar_count"].ge(int(settings.get("minimum_signal_bars_per_hour", 1)))
        & hourly["index_bar_count"].ge(int(settings.get("minimum_signal_bars_per_hour", 1)))
        & hourly[
            ["perp_open", "perp_close", "mark_open", "mark_close", "index_open", "index_close"]
        ].gt(0).all(axis=1)
        & hourly["maximum_contract_mark_gap_bps"].le(
            float(settings["max_contract_mark_gap_bps"])
        )
    )
    # The index is the synchronized economic reference. Perpetual trades remain the fill source.
    hourly["return"] = np.log(hourly["index_close"] / hourly["index_open"])
    hourly["decision_time"] = hourly["timestamp"] + pd.Timedelta(hours=1)
    hourly["session"] = session
    hourly["execution_group"] = (
        hourly["session_date"].astype(str) if session == "us_rth" else "continuous"
    )
    return hourly.sort_values("timestamp").reset_index(drop=True)


def build_factor_hourly_panel(
    history: dict[str, dict[str, pd.DataFrame]], cfg: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = target_specs(cfg)
    bars: dict[tuple[str, str], pd.DataFrame] = {}

    def get_bars(instrument: str, session: str) -> pd.DataFrame:
        key = (instrument, session)
        if key not in bars:
            bars[key] = _hourly_bars(history.get(instrument, {}), cfg, session)
        return bars[key]

    panels: list[pd.DataFrame] = []
    quality_rows: list[dict[str, Any]] = []
    for symbol, spec in targets.items():
        session = str(spec["session"])
        factors = [str(value) for value in spec["factors"]]
        target = get_bars(symbol, session)
        factor_bars = {factor: get_bars(factor, session) for factor in factors}
        if target.empty or any(factor_bars[factor].empty for factor in factors):
            quality_rows.append(
                {
                    "symbol": symbol,
                    "category": spec["category"],
                    "session": session,
                    "factors": "+".join(factors),
                    "hourly_rows": 0,
                    "quality_rows": 0,
                    "trading_days": 0,
                    "average_daily_quote_volume": np.nan,
                }
            )
            continue
        panel = target.rename(
            columns={
                "perp_open": "target_open",
                "perp_close": "target_close",
                "mark_close": "target_mark_close",
                "index_close": "target_index_close",
                "return": "target_return",
                "quality_ok": "target_quality_ok",
            }
        )[
            [
                "timestamp", "decision_time", "session_date", "hour_number", "execution_group",
                "target_open", "target_close", "target_mark_close", "target_index_close",
                "target_return", "target_quality_ok", "perp_quote_volume",
            ]
        ]
        for factor in factors:
            factor_frame = factor_bars[factor].rename(
                columns={
                    "perp_open": f"{factor}_open",
                    "perp_close": f"{factor}_close",
                    "return": f"{factor}_return",
                    "quality_ok": f"{factor}_quality_ok",
                }
            )[[
                "timestamp", f"{factor}_open", f"{factor}_close",
                f"{factor}_return", f"{factor}_quality_ok",
            ]]
            panel = panel.merge(factor_frame, on="timestamp", how="inner", validate="one_to_one")
        panel["symbol"] = symbol
        panel["category"] = str(spec["category"])
        panel["session"] = session
        quality_columns = ["target_quality_ok"] + [f"{factor}_quality_ok" for factor in factors]
        panel["quality_ok"] = panel[quality_columns].all(axis=1)
        quality_rows.append(
            {
                "symbol": symbol,
                "category": spec["category"],
                "session": session,
                "factors": "+".join(factors),
                "hourly_rows": len(panel),
                "quality_rows": int(panel["quality_ok"].sum()),
                "trading_days": int(panel.loc[panel["quality_ok"], "session_date"].nunique()),
                "average_daily_quote_volume": float(
                    panel.loc[panel["quality_ok"]]
                    .groupby("session_date")["perp_quote_volume"]
                    .sum(min_count=1)
                    .mean()
                ),
            }
        )
        panels.append(panel)
    if not panels:
        return pd.DataFrame(), pd.DataFrame(quality_rows)
    return (
        pd.concat(panels, ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(drop=True),
        pd.DataFrame(quality_rows),
    )


def _fit_ridge(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    nonnegative: bool,
) -> tuple[float, np.ndarray, float]:
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0, ddof=1)
    if np.any(~np.isfinite(x_std)) or np.any(x_std <= 0):
        raise ValueError("Factor return variance is zero")
    y_mean = float(y.mean())
    standardized = (x - x_mean) / x_std
    centered_y = y - y_mean
    candidates: list[tuple[float, np.ndarray, float]] = []
    subsets = range(1, 1 << x.shape[1]) if nonnegative else [((1 << x.shape[1]) - 1)]
    for mask in subsets:
        active = [index for index in range(x.shape[1]) if mask & (1 << index)]
        selected = standardized[:, active]
        coefficient_scaled = np.linalg.solve(
            selected.T @ selected + alpha * np.eye(len(active)),
            selected.T @ centered_y,
        )
        beta = np.zeros(x.shape[1])
        beta[active] = coefficient_scaled / x_std[active]
        if nonnegative and np.any(beta < -1e-12):
            continue
        beta = np.maximum(beta, 0.0) if nonnegative else beta
        intercept = y_mean - float(x_mean @ beta)
        fitted = intercept + x @ beta
        objective = float(np.square(y - fitted).sum()) + alpha * float(
            np.square(coefficient_scaled).sum()
        )
        candidates.append((intercept, beta, objective))
    if nonnegative:
        zero_intercept = y_mean
        candidates.append((zero_intercept, np.zeros(x.shape[1]), float(np.square(y - y_mean).sum())))
    if not candidates:
        raise ValueError("No feasible Ridge solution")
    intercept, beta, _ = min(candidates, key=lambda candidate: candidate[2])
    fitted = intercept + x @ beta
    denominator = float(np.square(y - y_mean).sum())
    r_squared = 1.0 - float(np.square(y - fitted).sum()) / denominator if denominator > 0 else np.nan
    return intercept, beta, r_squared


def instrument_roundtrip_cost_bps(
    cfg: dict[str, Any], instrument: str, multiplier: float = 1.0
) -> float:
    profile = instrument_cost_profile(cfg, instrument)
    execution = cfg["execution"]
    return multiplier * (
        profile["roundtrip_spread_bps"]
        + 2.0 * float(execution["perp_taker_fee_bps"])
        + 2.0 * profile["extra_slippage_per_fill_bps"]
        + 2.0 * float(execution["impact_bps_per_fill"])
    )


def _continuity_blocks(frame: pd.DataFrame, session: str) -> pd.Series:
    gap = frame["timestamp"].diff().ne(pd.Timedelta(hours=1))
    if session == "us_rth":
        gap |= frame["session_date"].ne(frame["session_date"].shift(1))
    return gap.cumsum()


def _hac_slope_t_stat(
    x: np.ndarray, y: np.ndarray, slope: float, max_lag: int
) -> float:
    design = np.column_stack([np.ones(len(x)), x])
    intercept = float(y.mean() - slope * x.mean())
    errors = y - intercept - slope * x
    scores = design * errors[:, None]
    meat = scores.T @ scores
    for lag in range(1, min(max_lag, len(x) - 1) + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        covariance = scores[lag:].T @ scores[:-lag]
        meat += weight * (covariance + covariance.T)
    try:
        bread = np.linalg.inv(design.T @ design)
    except np.linalg.LinAlgError:
        return np.nan
    robust_covariance = bread @ meat @ bread
    robust_covariance *= len(x) / max(len(x) - design.shape[1], 1)
    slope_variance = float(robust_covariance[1, 1])
    return slope / np.sqrt(slope_variance) if slope_variance > 0 else np.nan


def _adf_with_lags(
    state: pd.DataFrame, residual_lookback: int, lags: int
) -> tuple[float, float]:
    observations: list[list[float]] = []
    outcomes: list[float] = []
    for _, block in state.groupby("block", sort=False):
        values = block["dislocation"].dropna().to_numpy(float)
        if len(values) < lags + 3:
            continue
        differences = np.diff(values)
        for index in range(lags, len(differences)):
            outcomes.append(float(differences[index]))
            observations.append(
                [
                    1.0,
                    float(values[index]),
                    *[
                        float(differences[index - lag])
                        for lag in range(1, lags + 1)
                    ],
                ]
            )
    if len(outcomes) > residual_lookback:
        outcomes = outcomes[-residual_lookback:]
        observations = observations[-residual_lookback:]
    if len(outcomes) <= lags + 3:
        return np.nan, np.nan
    design = np.asarray(observations, dtype=float)
    dependent = np.asarray(outcomes, dtype=float)
    try:
        coefficients, _, _, _ = np.linalg.lstsq(design, dependent, rcond=None)
        errors = dependent - design @ coefficients
        degrees = len(dependent) - design.shape[1]
        covariance = float(errors @ errors / max(degrees, 1)) * np.linalg.inv(
            design.T @ design
        )
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    variance = float(covariance[1, 1])
    statistic = float(coefficients[1] / np.sqrt(variance)) if variance > 0 else np.nan
    p_value = (
        float(mackinnonp(statistic, regression="c", N=1))
        if np.isfinite(statistic)
        else np.nan
    )
    return statistic, p_value


def _model_state_statistics(
    training: pd.DataFrame,
    residuals: np.ndarray,
    current_timestamp: pd.Timestamp,
    current_session_date: Any,
    current_residual: float,
    session: str,
    residual_lookback: int,
    residual_minimum: int,
    signal_horizon: int,
    forecast_horizon: int,
    parameter_count: int,
) -> dict[str, float] | None:
    state = training[["timestamp", "session_date"]].copy()
    state["residual"] = residuals
    state = state.tail(residual_lookback + signal_horizon + forecast_horizon).reset_index(drop=True)
    state["block"] = _continuity_blocks(state, session)
    state["dislocation"] = state.groupby("block", sort=False)["residual"].transform(
        lambda values: values.rolling(signal_horizon, min_periods=signal_horizon).sum()
    )
    future_parts = [
        state.groupby("block", sort=False)["residual"].shift(-step)
        for step in range(1, forecast_horizon + 1)
    ]
    state["forward_residual"] = sum(future_parts)

    dislocations = state["dislocation"].dropna().tail(residual_lookback)
    if len(dislocations) < residual_minimum:
        return None
    mean = float(dislocations.mean())
    std = float(dislocations.std(ddof=1))
    degrees = max(len(training) - parameter_count - 1, 1)
    std *= float(np.sqrt(len(training) / degrees))
    if not np.isfinite(std) or std <= 0:
        return None

    prior = state.iloc[-1]
    contiguous = (
        current_timestamp - prior["timestamp"] == pd.Timedelta(hours=1)
        and (session == "continuous" or current_session_date == prior["session_date"])
    )
    if signal_horizon == 1:
        current_dislocation = current_residual
    elif contiguous:
        tail = state.loc[state["block"].eq(prior["block"]), "residual"].tail(signal_horizon - 1)
        if len(tail) != signal_horizon - 1:
            return None
        current_dislocation = float(tail.sum() + current_residual)
    else:
        return None

    predictive = state.dropna(subset=["dislocation", "forward_residual"]).tail(residual_lookback)
    if len(predictive) < residual_minimum:
        return None
    x = predictive["dislocation"].to_numpy(float)
    y = predictive["forward_residual"].to_numpy(float)
    centered_x = x - x.mean()
    centered_y = y - y.mean()
    sum_xx = float(centered_x @ centered_x)
    if sum_xx <= 0:
        return None
    slope = float(centered_x @ centered_y / sum_xx)
    reversion_t = _hac_slope_t_stat(x, y, slope, max_lag=forecast_horizon)
    adf_t, adf_p = _adf_with_lags(
        state,
        residual_lookback,
        lags=max(signal_horizon - 1, 0),
    )
    return {
        "dislocation": current_dislocation,
        "mean": mean,
        "std": std,
        "z_score": (current_dislocation - mean) / std,
        "reversion_slope": slope,
        "reversion_t": reversion_t,
        "adf_t": adf_t,
        "adf_p": adf_p,
        "expected_gross_edge_bps": abs(slope * (current_dislocation - mean)) * 10_000.0,
        "reversion_observations": float(len(predictive)),
    }


def build_factor_signals(
    panel: pd.DataFrame,
    cfg: dict[str, Any],
    cost_multiplier: float = 1.0,
) -> pd.DataFrame:
    if panel.empty:
        return panel.copy()
    settings = cfg["factor_strategy"]
    targets = target_specs(cfg)
    output: list[pd.DataFrame] = []
    for symbol, sample in panel.groupby("symbol", sort=True):
        sample = sample.sort_values("timestamp").reset_index(drop=True).copy()
        target = targets[str(symbol)]
        factors = [str(value) for value in target["factors"]]
        session = str(target["session"])
        parameters = session_parameters(cfg, session)
        lookback = int(parameters["regression_lookback_hours"])
        minimum = int(parameters["regression_min_hours"])
        residual_lookback = int(parameters["residual_lookback_hours"])
        residual_minimum = int(parameters["residual_min_hours"])
        columns = [
            "intercept", "model_r2", "residual", "dislocation",
            "dislocation_mean_lagged", "dislocation_std_lagged", "z_score",
            "reversion_slope_lagged", "reversion_t_stat_lagged",
            "residual_ar1_lagged", "residual_ar1_t_stat_lagged",
            "adf_t_stat_lagged", "adf_p_value_lagged", "reversion_observations",
            "expected_gross_edge_bps",
        ] + [f"beta_{factor}" for factor in factors]
        for column in columns:
            sample[column] = np.nan
        x_columns = [f"{factor}_return" for factor in factors]
        for index in range(len(sample)):
            start = max(0, index - lookback)
            history = sample.iloc[start:index]
            valid = history["quality_ok"] & history[x_columns + ["target_return"]].notna().all(axis=1)
            training = history.loc[valid]
            if len(training) < minimum or not bool(sample.iloc[index]["quality_ok"]):
                continue
            try:
                intercept, beta, r_squared = _fit_ridge(
                    training[x_columns].to_numpy(float),
                    training["target_return"].to_numpy(float),
                    float(settings["ridge_alpha"]),
                    bool(settings.get("nonnegative_betas", True)),
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            current_x = sample.loc[index, x_columns].to_numpy(float)
            residual = float(sample.loc[index, "target_return"] - intercept - current_x @ beta)
            training_residuals = (
                training["target_return"].to_numpy(float)
                - intercept
                - training[x_columns].to_numpy(float) @ beta
            )
            statistics = _model_state_statistics(
                training,
                training_residuals,
                sample.loc[index, "timestamp"],
                sample.loc[index, "session_date"],
                residual,
                session,
                residual_lookback,
                residual_minimum,
                int(settings["signal_horizon_hours"]),
                int(settings.get("forecast_horizon_hours", settings["signal_horizon_hours"])),
                len(factors),
            )
            sample.loc[index, "intercept"] = intercept
            sample.loc[index, "model_r2"] = r_squared
            sample.loc[index, "residual"] = residual
            for factor, value in zip(factors, beta):
                sample.loc[index, f"beta_{factor}"] = value
            if statistics is not None:
                sample.loc[index, "dislocation"] = statistics["dislocation"]
                sample.loc[index, "dislocation_mean_lagged"] = statistics["mean"]
                sample.loc[index, "dislocation_std_lagged"] = statistics["std"]
                sample.loc[index, "z_score"] = statistics["z_score"]
                sample.loc[index, "reversion_slope_lagged"] = statistics["reversion_slope"]
                sample.loc[index, "reversion_t_stat_lagged"] = statistics["reversion_t"]
                # Keep legacy column names so old analysis notebooks fail gracefully.
                sample.loc[index, "residual_ar1_lagged"] = statistics["reversion_slope"]
                sample.loc[index, "residual_ar1_t_stat_lagged"] = statistics["reversion_t"]
                sample.loc[index, "adf_t_stat_lagged"] = statistics["adf_t"]
                sample.loc[index, "adf_p_value_lagged"] = statistics["adf_p"]
                sample.loc[index, "reversion_observations"] = statistics["reversion_observations"]
                sample.loc[index, "expected_gross_edge_bps"] = statistics[
                    "expected_gross_edge_bps"
                ]
        beta_columns = [f"beta_{factor}" for factor in factors]
        sample["gross_beta"] = sample[beta_columns].abs().sum(axis=1)
        sample["cost_hurdle_bps"] = instrument_roundtrip_cost_bps(
            cfg, str(symbol), cost_multiplier
        )
        for factor in factors:
            sample["cost_hurdle_bps"] += sample[f"beta_{factor}"].abs() * (
                instrument_roundtrip_cost_bps(cfg, factor, cost_multiplier)
            )
        sample["cost_hurdle_bps"] += float(settings["safety_margin_bps"])
        sample["predicted_net_edge_bps"] = (
            sample["expected_gross_edge_bps"] - sample["cost_hurdle_bps"]
        )
        stable_beta = sample[beta_columns].abs().le(float(settings["max_abs_beta"])).all(axis=1)
        sample["model_eligible"] = (
            sample["model_r2"].ge(float(settings["min_r2"]))
            & sample["reversion_t_stat_lagged"].le(
                float(settings["max_reversion_t_stat"])
            )
            & sample["reversion_slope_lagged"].lt(0.0)
            & sample["adf_p_value_lagged"].le(float(settings["max_adf_p_value"]))
            & stable_beta
            & sample["gross_beta"].le(float(settings["max_gross_beta"]))
        )
        sample["entry_candidate"] = (
            sample["quality_ok"]
            & sample["model_eligible"]
            & sample["z_score"].abs().ge(float(settings["entry_z"]))
            & sample["z_score"].abs().lt(float(settings["stop_z"]))
            & sample["predicted_net_edge_bps"].gt(0.0)
        )
        output.append(sample)
    return pd.concat(output, ignore_index=True).sort_values(["symbol", "timestamp"])


def reprice_factor_signals(
    signals: pd.DataFrame,
    cfg: dict[str, Any],
    cost_multiplier: float,
) -> pd.DataFrame:
    if signals.empty:
        return signals.copy()
    settings = cfg["factor_strategy"]
    targets = target_specs(cfg)
    output: list[pd.DataFrame] = []
    for symbol, sample in signals.groupby("symbol", sort=True):
        sample = sample.copy()
        factors = [str(value) for value in targets[str(symbol)]["factors"]]
        sample["cost_hurdle_bps"] = instrument_roundtrip_cost_bps(
            cfg, str(symbol), cost_multiplier
        )
        for factor in factors:
            sample["cost_hurdle_bps"] += sample[f"beta_{factor}"].abs() * (
                instrument_roundtrip_cost_bps(cfg, factor, cost_multiplier)
            )
        sample["cost_hurdle_bps"] += float(settings["safety_margin_bps"])
        sample["predicted_net_edge_bps"] = (
            sample["expected_gross_edge_bps"] - sample["cost_hurdle_bps"]
        )
        sample["entry_candidate"] = (
            sample["quality_ok"]
            & sample["model_eligible"]
            & sample["z_score"].abs().ge(float(settings["entry_z"]))
            & sample["z_score"].abs().lt(float(settings["stop_z"]))
            & sample["predicted_net_edge_bps"].gt(0.0)
        )
        output.append(sample)
    return pd.concat(output, ignore_index=True).sort_values(["symbol", "timestamp"])


def _close_reason(
    row: pd.Series,
    entry_time: pd.Timestamp,
    cfg: dict[str, Any],
    execution_delay_minutes: int = 0,
) -> str | None:
    if pd.isna(row.get("z_score")):
        return None
    settings = cfg["factor_strategy"]
    proposed_exit = row["decision_time"] + pd.Timedelta(minutes=execution_delay_minutes)
    held_hours = (proposed_exit - entry_time).total_seconds() / 3600.0
    if abs(float(row["z_score"])) >= float(settings["stop_z"]):
        return "stop"
    if held_hours < float(settings["min_holding_hours"]):
        return None
    if abs(float(row["z_score"])) <= float(settings["exit_z"]):
        return "reversion"
    if held_hours >= float(settings["max_holding_hours"]):
        return "timeout"
    return None


def _leg_pnl(signed_notional: float, entry: float, exit_price: float) -> float:
    return signed_notional / entry * (exit_price - entry)


def _funding_pnl(
    funding: pd.DataFrame,
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    signed_notional: float,
) -> float:
    if funding.empty:
        return 0.0
    held = funding.loc[
        funding["timestamp"].gt(entry_time) & funding["timestamp"].le(exit_time),
        "funding_rate",
    ]
    return float((-signed_notional * held).sum())


def _trade_exposures(row: pd.Series) -> dict[str, float]:
    exposures = {str(row["symbol"]): float(row["target_signed_notional"])}
    for factor in str(row.get("factor_names", "")).split("+"):
        if factor:
            exposures[factor] = exposures.get(factor, 0.0) + float(
                row.get(f"{factor}_signed_notional", 0.0)
            )
    return exposures


def _apply_portfolio_limits(trades: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    if trades.empty:
        return trades
    limits = cfg["factor_strategy"].get("portfolio_limits", {})
    max_pairs = int(limits.get("max_concurrent_pairs", 10**9))
    capital = float(cfg["execution"]["initial_capital"])
    max_gross = float(limits.get("max_gross_leverage", np.inf)) * capital
    max_net = float(limits.get("max_instrument_net_fraction", np.inf)) * capital
    ordered = trades.sort_values(
        ["entry_time", "predicted_net_edge_bps", "symbol"],
        ascending=[True, False, True],
    )
    active: list[pd.Series] = []
    accepted: list[pd.Series] = []
    for _, candidate in ordered.iterrows():
        entry_time = candidate["entry_time"]
        active = [row for row in active if row["exit_time"] > entry_time]
        if len(active) >= max_pairs:
            continue
        gross = sum(float(row["gross_notional"]) for row in active)
        if gross + float(candidate["gross_notional"]) > max_gross:
            continue
        exposures: dict[str, float] = {}
        for row in active + [candidate]:
            for instrument, value in _trade_exposures(row).items():
                exposures[instrument] = exposures.get(instrument, 0.0) + value
        if any(abs(value) > max_net for value in exposures.values()):
            continue
        candidate = candidate.copy()
        candidate["concurrent_pairs_at_entry"] = len(active) + 1
        candidate["portfolio_gross_at_entry"] = gross + float(candidate["gross_notional"])
        active.append(candidate)
        accepted.append(candidate)
    if not accepted:
        return trades.iloc[0:0].copy()
    return pd.DataFrame(accepted).sort_values(["entry_time", "symbol"]).reset_index(drop=True)


def run_factor_backtest(
    signals: pd.DataFrame,
    history: dict[str, dict[str, pd.DataFrame]],
    cfg: dict[str, Any],
    cost_multiplier: float = 1.0,
    delay_minutes: int | None = None,
) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()
    settings = cfg["factor_strategy"]
    execution = cfg["execution"]
    targets = target_specs(cfg)
    notional = float(execution["notional_per_leg"])
    execution_delay = int(
        settings["execution_delay_minutes"] if delay_minutes is None else delay_minutes
    )
    price_maps: dict[str, pd.Series] = {}
    for instrument in instrument_specs(cfg):
        minute = history.get(instrument, {}).get("perp", pd.DataFrame())
        if minute.empty:
            price_maps[instrument] = pd.Series(dtype=float)
        else:
            price_maps[instrument] = (
                minute.drop_duplicates("timestamp", keep="last")
                .set_index("timestamp")["open"]
                .astype(float)
                .sort_index()
            )
    trades: list[dict[str, Any]] = []
    for symbol, symbol_frame in signals.groupby("symbol", sort=True):
        target_spec = targets[str(symbol)]
        factors = [str(value) for value in target_spec["factors"]]
        group_column = "execution_group" if "execution_group" in symbol_frame else "session_date"
        for _, sample in symbol_frame.groupby(group_column, sort=True):
            sample = sample.sort_values("timestamp").reset_index(drop=True)
            index = 0
            while index < len(sample) - 2:
                candidate = sample.iloc[index]
                if not bool(candidate.get("entry_candidate", False)):
                    index += 1
                    continue
                entry_index = index + 1
                if entry_index >= len(sample) - 1:
                    break
                entry = sample.iloc[entry_index]
                if entry["timestamp"] != candidate["decision_time"] or not bool(entry["quality_ok"]):
                    index += 1
                    continue
                entry_time = entry["timestamp"] + pd.Timedelta(minutes=execution_delay)
                entry_prices = {
                    "target": price_maps[symbol].get(entry_time, np.nan),
                    **{factor: price_maps[factor].get(entry_time, np.nan) for factor in factors},
                }
                if any(not np.isfinite(value) or value <= 0 for value in entry_prices.values()):
                    index += 1
                    continue
                target_direction = -1 if float(candidate["z_score"]) > 0 else 1
                betas = {factor: float(candidate[f"beta_{factor}"]) for factor in factors}
                signed_notionals = {"target": target_direction * notional}
                signed_notionals.update(
                    {factor: -target_direction * beta * notional for factor, beta in betas.items()}
                )
                exit_signal_index = entry_index
                exit_index: int | None = None
                reason: str | None = None
                while exit_signal_index < len(sample) - 1:
                    exit_signal = sample.iloc[exit_signal_index]
                    reason = _close_reason(exit_signal, entry_time, cfg, execution_delay)
                    if reason:
                        proposed = exit_signal_index + 1
                        if proposed < len(sample):
                            if sample.iloc[proposed]["timestamp"] == exit_signal["decision_time"]:
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
                exit_time = exit_row["timestamp"] + pd.Timedelta(minutes=execution_delay)
                exit_prices = {
                    "target": price_maps[symbol].get(exit_time, np.nan),
                    **{factor: price_maps[factor].get(exit_time, np.nan) for factor in factors},
                }
                if any(not np.isfinite(value) or value <= 0 for value in exit_prices.values()):
                    index = exit_index + 1
                    continue
                gross_pnl = _leg_pnl(
                    signed_notionals["target"],
                    float(entry_prices["target"]),
                    float(exit_prices["target"]),
                )
                factor_pnl: dict[str, float] = {}
                for factor in factors:
                    factor_pnl[factor] = _leg_pnl(
                        signed_notionals[factor],
                        float(entry_prices[factor]),
                        float(exit_prices[factor]),
                    )
                    gross_pnl += factor_pnl[factor]
                gross_notional = float(sum(abs(value) for value in signed_notionals.values()))
                leg_instruments = {"target": str(symbol), **{factor: factor for factor in factors}}
                spread_cost = 0.0
                fee_cost = 0.0
                slippage_cost = 0.0
                impact_cost = 0.0
                for leg, instrument in leg_instruments.items():
                    leg_notional = abs(float(signed_notionals[leg]))
                    profile = instrument_cost_profile(cfg, instrument)
                    spread_cost += leg_notional * profile["roundtrip_spread_bps"] * cost_multiplier / 10_000.0
                    fee_cost += leg_notional * 2.0 * float(execution["perp_taker_fee_bps"]) * cost_multiplier / 10_000.0
                    slippage_cost += leg_notional * 2.0 * profile["extra_slippage_per_fill_bps"] * cost_multiplier / 10_000.0
                    impact_cost += leg_notional * 2.0 * float(execution["impact_bps_per_fill"]) * cost_multiplier / 10_000.0
                funding_pnl = _funding_pnl(
                    history.get(symbol, {}).get("funding", pd.DataFrame()),
                    entry_time,
                    exit_time,
                    signed_notionals["target"],
                )
                for factor in factors:
                    funding_pnl += _funding_pnl(
                        history.get(factor, {}).get("funding", pd.DataFrame()),
                        entry_time,
                        exit_time,
                        signed_notionals[factor],
                    )
                held_seconds = (exit_time - entry_time).total_seconds()
                opportunity = (
                    gross_notional
                    * float(execution["perp_margin_fraction"])
                    * float(execution["opportunity_cost_annual"])
                    * held_seconds
                    / SECONDS_PER_YEAR
                )
                net_pnl = (
                    gross_pnl - spread_cost - fee_cost - slippage_cost - impact_cost
                    + funding_pnl - opportunity
                )
                row: dict[str, Any] = {
                    "symbol": symbol,
                    "category": str(target_spec["category"]),
                    "session": str(target_spec["session"]),
                    "factor_names": "+".join(factors),
                    "signal_time": candidate["decision_time"],
                    "signal_session_date": candidate["session_date"],
                    "entry_time": entry_time,
                    "exit_signal_time": sample.iloc[exit_signal_index]["decision_time"],
                    "exit_time": exit_time,
                    "execution_delay_minutes": execution_delay,
                    "side": "long_residual" if target_direction > 0 else "short_residual",
                    "entry_z": candidate["z_score"],
                    "exit_z": sample.iloc[exit_signal_index]["z_score"],
                    "entry_model_r2": candidate["model_r2"],
                    "entry_residual_ar1": candidate["residual_ar1_lagged"],
                    "entry_residual_ar1_t_stat": candidate["residual_ar1_t_stat_lagged"],
                    "entry_reversion_slope": candidate.get("reversion_slope_lagged", np.nan),
                    "entry_reversion_t_stat": candidate.get("reversion_t_stat_lagged", np.nan),
                    "entry_adf_p_value": candidate.get("adf_p_value_lagged", np.nan),
                    "predicted_net_edge_bps": candidate["predicted_net_edge_bps"],
                    "holding_hours": held_seconds / 3600.0,
                    "holding_minutes": held_seconds / 60.0,
                    "exit_reason": reason,
                    "target_notional": notional,
                    "target_signed_notional": signed_notionals["target"],
                    "gross_notional": gross_notional,
                    "mid_gross_pnl": gross_pnl,
                    "modeled_spread_cost": spread_cost,
                    "fee_cost": fee_cost,
                    "extra_slippage_cost": slippage_cost,
                    "market_impact_cost": impact_cost,
                    "funding_cost": -funding_pnl,
                    "opportunity_cost": opportunity,
                    "net_pnl": net_pnl,
                    "net_return_bps": net_pnl / notional * 10_000.0,
                }
                for factor in factors:
                    row[f"beta_{factor}"] = betas[factor]
                    row[f"{factor}_signed_notional"] = signed_notionals[factor]
                    row[f"{factor}_gross_pnl"] = factor_pnl[factor]
                trades.append(row)
                index = exit_index + 1
    raw = pd.DataFrame(trades)
    return _apply_portfolio_limits(raw, cfg)


def _metrics(trades: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    metrics = performance_metrics(trades, cfg)
    metrics["gross_pnl"] = float(trades["mid_gross_pnl"].sum()) if not trades.empty else 0.0
    return metrics


def _signal_funnel(signals: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "symbol", "rows", "quality", "factor_model_ready", "r2_pass", "adf_pass",
        "reversion_pass", "model_eligible", "z_trigger", "inside_stop_band",
        "eligible_entry_band", "cost_cover", "entry_candidate",
    ]
    if signals.empty:
        return pd.DataFrame(columns=columns)
    settings = cfg["factor_strategy"]
    rows: list[dict[str, Any]] = []
    for symbol, sample in signals.groupby("symbol", sort=True):
        z_trigger = sample["z_score"].abs().ge(float(settings["entry_z"]))
        inside_stop = sample["z_score"].abs().lt(float(settings["stop_z"]))
        rows.append(
            {
                "symbol": symbol,
                "rows": len(sample),
                "quality": int(sample["quality_ok"].sum()),
                "factor_model_ready": int(sample["model_r2"].notna().sum()),
                "r2_pass": int(sample["model_r2"].ge(float(settings["min_r2"])).sum()),
                "adf_pass": int(
                    sample["adf_p_value_lagged"].le(float(settings["max_adf_p_value"])).sum()
                ),
                "reversion_pass": int(
                    sample["reversion_t_stat_lagged"]
                    .le(float(settings["max_reversion_t_stat"]))
                    .sum()
                ),
                "model_eligible": int(sample["model_eligible"].sum()),
                "z_trigger": int(z_trigger.sum()),
                "inside_stop_band": int(inside_stop.sum()),
                "eligible_entry_band": int(
                    (sample["model_eligible"] & z_trigger & inside_stop).sum()
                ),
                "cost_cover": int(sample["predicted_net_edge_bps"].gt(0.0).sum()),
                "entry_candidate": int(sample["entry_candidate"].sum()),
            }
        )
    funnel = pd.DataFrame(rows, columns=columns)
    totals = {column: funnel[column].sum() for column in columns if column != "symbol"}
    return pd.concat([funnel, pd.DataFrame([{"symbol": "TOTAL", **totals}])], ignore_index=True)


def _universe_table(quality: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    targets = target_specs(cfg)
    rows = [
        {
            "symbol": symbol,
            "provider_symbol": spec["perp_symbol"],
            "category": spec["category"],
            "session": spec["session"],
            "factors": "+".join(str(value) for value in spec["factors"]),
            "cost_profile": spec["cost_profile"],
            "economic_logic": spec["logic"],
            "invalidation_condition": spec["invalidation"],
        }
        for symbol, spec in targets.items()
    ]
    universe = pd.DataFrame(rows)
    if not quality.empty:
        quality_columns = [
            "symbol", "hourly_rows", "quality_rows", "trading_days",
            "average_daily_quote_volume",
        ]
        universe = universe.merge(quality[quality_columns], on="symbol", how="left")
    return universe


def evaluate_factor_strategy(
    panel: pd.DataFrame,
    signals: pd.DataFrame,
    history: dict[str, dict[str, pd.DataFrame]],
    quality: pd.DataFrame,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    baseline = run_factor_backtest(signals, history, cfg)
    settings = cfg["factor_strategy"]
    ungated_signals = signals.copy()
    if not ungated_signals.empty:
        ungated_signals["entry_candidate"] = (
            ungated_signals["quality_ok"]
            & ungated_signals["model_eligible"]
            & ungated_signals["z_score"].abs().ge(float(settings["entry_z"]))
            & ungated_signals["z_score"].abs().lt(float(settings["stop_z"]))
        )
    ungated = run_factor_backtest(ungated_signals, history, cfg)
    doubled_signals = reprice_factor_signals(signals, cfg, cost_multiplier=2.0)
    doubled = run_factor_backtest(doubled_signals, history, cfg, cost_multiplier=2.0)
    delayed = run_factor_backtest(
        signals,
        history,
        cfg,
        delay_minutes=int(cfg["factor_strategy"]["stress_delay_minutes"]),
    )
    dates = sorted(signals["session_date"].dropna().unique()) if not signals.empty else []
    split_index = int(len(dates) * (1.0 - float(cfg["factor_strategy"]["holdout_fraction"])))
    split_index = min(max(split_index, 0), max(len(dates) - 1, 0))
    holdout_start = dates[split_index] if dates else None
    if holdout_start is None or baseline.empty:
        holdout = baseline.iloc[0:0].copy()
        development = baseline.iloc[0:0].copy()
    else:
        trade_dates = baseline["signal_session_date"]
        holdout = baseline.loc[trade_dates >= holdout_start].copy()
        development = baseline.loc[trade_dates < holdout_start].copy()
    if holdout_start is None or ungated.empty:
        ungated_holdout = ungated.iloc[0:0].copy()
        ungated_development = ungated.iloc[0:0].copy()
    else:
        ungated_dates = ungated["signal_session_date"]
        ungated_holdout = ungated.loc[ungated_dates >= holdout_start].copy()
        ungated_development = ungated.loc[ungated_dates < holdout_start].copy()
    scenarios = pd.DataFrame(
        [
            {"scenario": "full_baseline", **_metrics(baseline, cfg)},
            {"scenario": "full_double_cost", **_metrics(doubled, cfg)},
            {"scenario": "full_30_minute_delay", **_metrics(delayed, cfg)},
            {"scenario": "development_baseline", **_metrics(development, cfg)},
            {"scenario": "late_sample_baseline", **_metrics(holdout, cfg)},
            {"scenario": "diagnostic_ignore_cost_gate", **_metrics(ungated, cfg)},
            {
                "scenario": "diagnostic_development_ignore_cost_gate",
                **_metrics(ungated_development, cfg),
            },
            {
                "scenario": "diagnostic_late_ignore_cost_gate",
                **_metrics(ungated_holdout, cfg),
            },
        ]
    )
    if baseline.empty:
        concentration = np.inf
    else:
        absolute_symbol_pnl = baseline.groupby("symbol")["net_pnl"].sum().abs()
        denominator = float(absolute_symbol_pnl.sum())
        concentration = (
            float(absolute_symbol_pnl.max() / denominator) if denominator > 0 else np.inf
        )
    eligible_symbols = int(
        signals.loc[signals["model_eligible"], "symbol"].nunique()
    ) if not signals.empty else 0
    full_metrics = scenarios.iloc[0]
    double_metrics = scenarios.iloc[1]
    delayed_metrics = scenarios.iloc[2]
    development_metrics = scenarios.iloc[3]
    late_sample_metrics = scenarios.iloc[4]
    targets = target_specs(cfg)
    category_counts = pd.Series(
        [str(spec["category"]) for spec in targets.values()]
    ).value_counts()
    checks = pd.DataFrame(
        [
            {"check": "configured_target_count", "value": len(targets), "threshold": f">= {cfg['factor_strategy']['target_count_minimum']}", "pass": len(targets) >= int(cfg["factor_strategy"]["target_count_minimum"])},
            {"check": "crypto_equity_target_count", "value": int(category_counts.get("crypto_equity", 0)), "threshold": ">= 5", "pass": int(category_counts.get("crypto_equity", 0)) >= 5},
            {"check": "crypto_asset_target_count", "value": int(category_counts.get("crypto_asset", 0)), "threshold": ">= 10", "pass": int(category_counts.get("crypto_asset", 0)) >= 10},
            {"check": "tech_equity_target_count", "value": int(category_counts.get("tech_equity", 0)), "threshold": ">= 5", "pass": int(category_counts.get("tech_equity", 0)) >= 5},
            {"check": "observed_trading_days", "value": len(dates), "threshold": f">= {cfg['acceptance']['min_observed_trading_days']}", "pass": len(dates) >= int(cfg["acceptance"]["min_observed_trading_days"])},
            {"check": "eligible_symbols", "value": eligible_symbols, "threshold": f">= {cfg['acceptance']['min_symbols']}", "pass": eligible_symbols >= int(cfg["acceptance"]["min_symbols"])},
            {"check": "full_minimum_trades", "value": full_metrics["trades"], "threshold": f">= {cfg['historical']['min_screen_trades']}", "pass": full_metrics["trades"] >= int(cfg["historical"]["min_screen_trades"])},
            {"check": "development_minimum_trades", "value": development_metrics["trades"], "threshold": f">= {cfg['factor_strategy']['min_development_trades']}", "pass": development_metrics["trades"] >= int(cfg["factor_strategy"]["min_development_trades"])},
            {"check": "late_sample_minimum_trades", "value": late_sample_metrics["trades"], "threshold": f">= {cfg['factor_strategy']['min_late_sample_trades']}", "pass": late_sample_metrics["trades"] >= int(cfg["factor_strategy"]["min_late_sample_trades"])},
            {"check": "symbol_pnl_concentration", "value": concentration, "threshold": f"<= {cfg['acceptance']['max_symbol_pnl_concentration']}", "pass": concentration <= float(cfg["acceptance"]["max_symbol_pnl_concentration"])},
            {"check": "full_gross_positive", "value": full_metrics["gross_pnl"], "threshold": "> 0", "pass": full_metrics["gross_pnl"] > 0},
            {"check": "full_net_positive", "value": full_metrics["net_pnl"], "threshold": "> 0", "pass": full_metrics["net_pnl"] > 0},
            {"check": "late_sample_gross_positive", "value": late_sample_metrics["gross_pnl"], "threshold": "> 0", "pass": late_sample_metrics["gross_pnl"] > 0},
            {"check": "late_sample_net_positive", "value": late_sample_metrics["net_pnl"], "threshold": "> 0", "pass": late_sample_metrics["net_pnl"] > 0},
            {"check": "double_cost_net_positive", "value": double_metrics["net_pnl"], "threshold": "> 0", "pass": double_metrics["net_pnl"] > 0},
            {"check": "extra_delay_net_positive", "value": delayed_metrics["net_pnl"], "threshold": "> 0", "pass": delayed_metrics["net_pnl"] > 0},
        ]
    )
    model_diagnostics = signals.groupby("symbol", as_index=False).agg(
        hourly_rows=("timestamp", "size"),
        eligible_hours=("model_eligible", "sum"),
        entry_candidates=("entry_candidate", "sum"),
        median_r2=("model_r2", "median"),
        median_reversion_slope=("reversion_slope_lagged", "median"),
        median_reversion_t_stat=("reversion_t_stat_lagged", "median"),
        median_adf_p_value=("adf_p_value_lagged", "median"),
        median_expected_edge_bps=("expected_gross_edge_bps", "median"),
    ) if not signals.empty else pd.DataFrame()
    trade_diagnostics = baseline.groupby("symbol", as_index=False).agg(
        trades=("net_pnl", "size"),
        gross_pnl=("mid_gross_pnl", "sum"),
        net_pnl=("net_pnl", "sum"),
        win_rate=("net_pnl", lambda values: (values > 0).mean()),
    ) if not baseline.empty else pd.DataFrame(columns=["symbol", "trades", "gross_pnl", "net_pnl", "win_rate"])
    diagnostics = model_diagnostics.merge(trade_diagnostics, on="symbol", how="left")
    if not diagnostics.empty:
        diagnostics.insert(
            1, "category", diagnostics["symbol"].map(
                {symbol: spec["category"] for symbol, spec in targets.items()}
            )
        )
        diagnostics.insert(
            2, "factors", diagnostics["symbol"].map(
                {
                    symbol: "+".join(str(value) for value in spec["factors"])
                    for symbol, spec in targets.items()
                }
            )
        )
    for column in ("trades", "gross_pnl", "net_pnl"):
        if column in diagnostics:
            diagnostics[column] = pd.to_numeric(diagnostics[column], errors="coerce").fillna(0.0)
    return {
        "status": "FACTOR_PRELIMINARY",
        "screen_pass": bool(checks["pass"].all()),
        "late_sample_start": holdout_start,
        "trades": baseline,
        "development_trades": development,
        "late_sample_trades": holdout,
        "ungated_diagnostic_trades": ungated,
        "scenarios": scenarios,
        "checks": checks,
        "quality": quality,
        "diagnostics": diagnostics,
        "signal_funnel": _signal_funnel(signals, cfg),
        "universe": _universe_table(quality, cfg),
        "category_attribution": baseline.groupby("category", as_index=False).agg(
            trades=("net_pnl", "size"),
            gross_pnl=("mid_gross_pnl", "sum"),
            net_pnl=("net_pnl", "sum"),
        ) if not baseline.empty else pd.DataFrame(
            columns=["category", "trades", "gross_pnl", "net_pnl"]
        ),
    }


def write_factor_report(
    result: dict[str, Any],
    errors: list[dict[str, str]],
    output_dir: Path,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    cfg: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    trade_headers = [
        "symbol", "category", "session", "factor_names", "signal_time", "entry_time",
        "exit_time", "side", "entry_z", "entry_model_r2", "predicted_net_edge_bps",
        "gross_notional", "mid_gross_pnl", "modeled_spread_cost", "fee_cost",
        "extra_slippage_cost", "market_impact_cost", "funding_cost", "opportunity_cost",
        "net_pnl",
    ]

    def write_csv(frame: pd.DataFrame, filename: str, headers: list[str] | None = None) -> None:
        output = frame if len(frame.columns) else pd.DataFrame(columns=headers or ["no_rows"])
        output.to_csv(output_dir / filename, index=False)

    write_csv(result["trades"], "factor_trades.csv", trade_headers)
    write_csv(result["development_trades"], "factor_development_trades.csv", trade_headers)
    write_csv(result["late_sample_trades"], "factor_late_sample_trades.csv", trade_headers)
    write_csv(
        result["ungated_diagnostic_trades"],
        "factor_ungated_diagnostic_trades.csv",
        trade_headers,
    )
    result["scenarios"].to_csv(output_dir / "factor_scenarios.csv", index=False)
    result["checks"].to_csv(output_dir / "factor_checks.csv", index=False)
    result["quality"].to_csv(output_dir / "factor_data_quality.csv", index=False)
    result["diagnostics"].to_csv(output_dir / "factor_symbol_diagnostics.csv", index=False)
    result["signal_funnel"].to_csv(output_dir / "factor_signal_funnel.csv", index=False)
    result["universe"].to_csv(output_dir / "factor_universe.csv", index=False)
    result["category_attribution"].to_csv(
        output_dir / "factor_category_attribution.csv", index=False
    )
    pd.DataFrame(errors, columns=["symbol", "dataset", "error"]).to_csv(
        output_dir / "factor_download_errors.csv", index=False
    )
    full = result["scenarios"].loc[result["scenarios"]["scenario"].eq("full_baseline")].iloc[0]
    development = result["scenarios"].loc[
        result["scenarios"]["scenario"].eq("development_baseline")
    ].iloc[0]
    late_sample = result["scenarios"].loc[
        result["scenarios"]["scenario"].eq("late_sample_baseline")
    ].iloc[0]
    ungated = result["scenarios"].loc[
        result["scenarios"]["scenario"].eq("diagnostic_ignore_cost_gate")
    ].iloc[0]
    ungated_development = result["scenarios"].loc[
        result["scenarios"]["scenario"].eq("diagnostic_development_ignore_cost_gate")
    ].iloc[0]
    ungated_late = result["scenarios"].loc[
        result["scenarios"]["scenario"].eq("diagnostic_late_ignore_cost_gate")
    ].iloc[0]
    gross_notional = float(result["trades"]["gross_notional"].sum()) if not result["trades"].empty else 0.0
    gross_edge_bps = float(full["gross_pnl"] / gross_notional * 10_000.0) if gross_notional > 0 else 0.0
    factors = ", ".join(cfg["factor_strategy"]["factors"])
    category_counts = result["universe"]["category"].value_counts()
    funnel_total = result["signal_funnel"].loc[
        result["signal_funnel"]["symbol"].eq("TOTAL")
    ].iloc[0]
    lines = [
        "# Hourly Idiosyncratic Factor Strategy",
        "",
        "**Status: FACTOR_PRELIMINARY**",
        "",
        f"Mechanical screen: {'PASS' if result['screen_pass'] else 'FAIL'}.",
        "This historical bar reconstruction is not a live deployment approval.",
        "",
        f"Period: {start.date()} through {(end_exclusive - pd.Timedelta(days=1)).date()}",
        f"Late-sample split starts: {result['late_sample_start']}",
        f"Factors: {factors}",
        f"Configured targets: {len(result['universe'])} "
        f"({int(category_counts.get('crypto_equity', 0))} crypto equities, "
        f"{int(category_counts.get('crypto_asset', 0))} crypto assets, "
        f"{int(category_counts.get('tech_equity', 0))} technology equities)",
        "",
        "## First-principles correction",
        "",
        "The prior module traded cash/perpetual basis. This module restores the project contract:",
        "target perpetual return minus its causal, predeclared economic-factor portfolio return.",
        "US equities use regular-session hours; crypto assets remain continuous across UTC midnight.",
        "",
        "## Results",
        "",
        f"- Full trades: {int(full['trades'])}",
        f"- Full gross PnL: ${full['gross_pnl']:,.2f}",
        f"- Full net PnL: ${full['net_pnl']:,.2f}",
        f"- Full Sharpe: {full['sharpe']:.3f}",
        f"- Gross edge on total gross notional: {gross_edge_bps:.2f} bps",
        f"- Signal funnel candidates before portfolio limits: {int(funnel_total['entry_candidate'])}",
        f"- Diagnostic trades without the cost gate: {int(ungated['trades'])}",
        f"- Diagnostic gross/net PnL: ${ungated['gross_pnl']:,.2f} / ${ungated['net_pnl']:,.2f}",
        f"- Diagnostic development gross/net: ${ungated_development['gross_pnl']:,.2f} / "
        f"${ungated_development['net_pnl']:,.2f}",
        f"- Diagnostic late gross/net: ${ungated_late['gross_pnl']:,.2f} / "
        f"${ungated_late['net_pnl']:,.2f}",
        f"- Development trades/net PnL: {int(development['trades'])} / ${development['net_pnl']:,.2f}",
        f"- Late-sample trades: {int(late_sample['trades'])}",
        f"- Late-sample gross PnL: ${late_sample['gross_pnl']:,.2f}",
        f"- Late-sample net PnL: ${late_sample['net_pnl']:,.2f}",
        "- The late sample is not an untouched holdout because this research iteration inspected it.",
        "",
        "## Symbol diagnostics",
        "",
        "| Symbol | Category | Factors | Eligible hours | Candidates | Trades | Median R2 | Reversion t | Net PnL |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in result["diagnostics"].iterrows():
        lines.append(
            f"| {row['symbol']} | {row['category']} | {row['factors']} | "
            f"{int(row['eligible_hours'])} | {int(row['entry_candidates'])} | "
            f"{int(row['trades'])} | {row['median_r2']:.3f} | "
            f"{row['median_reversion_t_stat']:.3f} | ${row['net_pnl']:,.2f} |"
        )
    lines.extend(
        [
            "",
            "## Causal and economic controls",
            "",
            "- Each beta and intercept use only earlier completed hours.",
            "- Ridge betas are constrained nonnegative for economically directional equity factors.",
            "- Factor combinations are fixed from economic logic before examining strategy PnL.",
            "- A model needs sufficient R2, lag-augmented ADF stationarity and a one-sided 5%",
            "  significant negative two-hour forward-residual slope using Newey-West errors.",
            "- The signal is a two-hour cumulative residual; all legs fill at an exact later 5-minute open.",
            "- Every target and hedge leg pays its own profile's spread and slippage, plus fees and impact.",
            "- Funding and margin opportunity cost use the signed multi-leg position.",
            "- Concurrent gross and per-instrument net exposures are constrained chronologically.",
            "- The last 40% of dates are reported separately, but are marked research-contaminated.",
            "- Double-cost and 30-minute-delay paths are rerun independently.",
            "- The ignore-cost-gate scenario is attribution only and can never authorize trading.",
            "",
            "## Limitations",
            "",
            "- Historical top-of-book is unavailable, so spreads remain conservative assumptions.",
            "- Several TradFi contracts are newly listed, so their usable history can remain short.",
            "- Five-minute execution bars do not reveal historical top-of-book depth or queue position.",
            "- XGBoost/SHAP is not admitted without a nonempty development set and new forward labels.",
            "",
            "Detailed CSV files are stored beside this report.",
        ]
    )
    report = output_dir / "factor_strategy_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_factor_prescreen(
    cfg: dict[str, Any],
    start_text: str | None = None,
    end_text: str | None = None,
    refresh: bool = False,
    report_dir: str | Path | None = None,
) -> Path:
    start, end_exclusive = historical_bounds(
        start_text
        or str(cfg["historical"].get("factor_start_date", cfg["historical"]["start_date"])),
        end_text or str(cfg["historical"]["end_date"]),
    )
    history, errors = download_factor_market_history(cfg, start, end_exclusive, refresh)
    panel, quality = build_factor_hourly_panel(history, cfg)
    signals = build_factor_signals(panel, cfg)
    result = evaluate_factor_strategy(panel, signals, history, quality, cfg)
    complete_check = pd.DataFrame(
        [
            {
                "check": "configured_datasets_complete",
                "value": len(errors),
                "threshold": "= 0 download errors",
                "pass": not errors,
            }
        ]
    )
    result["checks"] = pd.concat([complete_check, result["checks"]], ignore_index=True)
    result["screen_pass"] = bool(result["checks"]["pass"].all())

    root = Path(cfg["_root"])
    key = f"{start.strftime('%Y%m%d')}_{(end_exclusive - pd.Timedelta(days=1)).strftime('%Y%m%d')}"
    processed = root / str(cfg["historical"]["cache_dir"]) / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    panel.to_pickle(processed / f"factor_panel_{key}.pkl")
    signals.to_pickle(processed / f"factor_signals_{key}.pkl")
    output = Path(report_dir) if report_dir else root / str(cfg["output"]["report_dir"]) / "factor"
    report = write_factor_report(result, errors, output, start, end_exclusive, cfg)
    print(result["scenarios"].to_string(index=False))
    print(result["checks"].to_string(index=False))
    print("Status: FACTOR_PRELIMINARY")
    print(f"Screen checks: {'PASS' if result['screen_pass'] else 'FAIL'}")
    print(f"Report: {report.resolve()}")
    return report
