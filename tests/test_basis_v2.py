from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from alignment import align_quotes
from backtest import performance_metrics, run_backtest
from collector import snapshot_health
from config import load_config, load_local_env
from schema import normalize_quotes
from basis_signal import build_signal_frame
from historical_prescreen import (
    build_historical_signals,
    evaluate_historical_screen,
    run_historical_backtest,
)
from historical_providers import AlpacaHistoricalBars, BinanceHistoricalMinutes
from factor_config import (
    instrument_specs_for_targets,
    research_target_specs,
    target_specs,
    validate_factor_configuration,
)
from factor_strategy import (
    _close_reason,
    _fit_ridge,
    _hourly_bars,
    _read_or_extend_cache,
    build_factor_signals,
    instrument_roundtrip_cost_bps,
    reprice_factor_signals,
    run_factor_backtest,
)
from factor_collector import factor_snapshot_health
from tradable_research import (
    _research_cache_fingerprints,
    build_dynamic_factor_signals,
    build_tradable_labels,
    purged_chronological_partitions,
    run_netted_portfolio,
    validation_authorizes_test,
)


def _quotes(skew_seconds: float = 0.2) -> pd.DataFrame:
    base = pd.Timestamp("2026-01-05 14:30:00", tz="UTC")
    rows = []
    for i in range(4):
        timestamp = base + pd.Timedelta(minutes=i)
        rows.extend(
            [
                {
                    "timestamp": timestamp,
                    "received_at": timestamp + pd.Timedelta(seconds=1),
                    "symbol": "NVDA",
                    "venue": "cash",
                    "bid": 99.99 + i,
                    "ask": 100.01 + i,
                    "clock_uncertainty_ms": 0.0,
                },
                {
                    "timestamp": timestamp + pd.Timedelta(seconds=skew_seconds),
                    "received_at": timestamp + pd.Timedelta(seconds=1.2),
                    "symbol": "NVDA",
                    "venue": "perp",
                    "bid": 100.09 + i,
                    "ask": 100.11 + i,
                    "clock_uncertainty_ms": 0.0,
                },
            ]
        )
    return normalize_quotes(pd.DataFrame(rows))


def test_local_env_loads_only_credentials_without_overriding(monkeypatch) -> None:
    env_file = Path("tests/_local_env_test.env")
    try:
        env_file.write_text(
            "APCA_API_KEY_ID=file-key\n"
            "APCA_API_SECRET_KEY='file-secret'\n"
            "UNRELATED=value\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("APCA_API_KEY_ID", "system-key")
        monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
        monkeypatch.delenv("UNRELATED", raising=False)

        loaded = load_local_env(env_file)

        assert loaded == ["APCA_API_SECRET_KEY"]
        assert os.environ["APCA_API_KEY_ID"] == "system-key"
        assert os.environ["APCA_API_SECRET_KEY"] == "file-secret"
        assert "UNRELATED" not in os.environ
    finally:
        env_file.unlink(missing_ok=True)


def test_snapshot_health_distinguishes_api_success_from_stale_cash() -> None:
    cfg = load_config()
    now = pd.Timestamp("2026-07-13 03:30:00", tz="UTC")
    records = [
        {
            "timestamp": (now - pd.Timedelta(days=2)).isoformat(),
            "received_at": now.isoformat(),
            "symbol": "NVDA",
            "venue": "cash",
            "bid": 100.0,
            "ask": 100.1,
        },
        {
            "timestamp": (now - pd.Timedelta(seconds=1)).isoformat(),
            "received_at": now.isoformat(),
            "symbol": "NVDA",
            "venue": "perp",
            "bid": 100.0,
            "ask": 100.1,
        },
    ]

    health = snapshot_health(records, cfg)

    assert health["pairs"] == 1
    assert health["fresh_pairs"] == 0
    assert health["median_cash_age_seconds"] > 100_000


def test_alignment_is_event_time_causal_and_checks_skew() -> None:
    cfg = load_config()
    cfg["symbols"] = {"NVDA": cfg["symbols"]["NVDA"]}
    aligned = align_quotes(_quotes(), cfg)
    assert not aligned.empty
    assert aligned["quality_ok"].all()
    assert (aligned["timestamp"] >= aligned["cash_received_at"]).all()
    assert (aligned["timestamp"] >= aligned["perp_received_at"]).all()

    bad = align_quotes(_quotes(skew_seconds=6.0), cfg)
    assert not bad["quality_ok"].any()


def test_signal_statistics_are_lagged() -> None:
    cfg = load_config()
    cfg["signal"]["rolling_window"] = 3
    cfg["signal"]["min_history"] = 3
    cfg["signal"]["bar_seconds"] = 60
    aligned = align_quotes(_quotes(), cfg)
    baseline = build_signal_frame(aligned, cfg)
    changed = aligned.copy()
    changed.loc[changed.index[-1], "basis"] = 1.0
    revised = build_signal_frame(changed, cfg)
    assert np.isclose(
        baseline.iloc[-1]["basis_mean_lagged"], revised.iloc[-1]["basis_mean_lagged"]
    )


def test_backtest_uses_delayed_bid_ask_and_full_costs() -> None:
    cfg = load_config()
    cfg = deepcopy(cfg)
    cfg["execution"]["delay_seconds"] = 5
    cfg["signal"]["min_holding_minutes"] = 0
    cfg["signal"]["max_holding_minutes"] = 30
    times = pd.date_range("2026-01-05 14:30", periods=4, freq="min", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": times,
            "symbol": "NVDA",
            "quality_ok": True,
            "entry_candidate": [True, False, False, False],
            "signal_observation": [True, True, True, True],
            "z_score": [3.0, 2.0, 0.1, 0.0],
            "predicted_net_edge_bps": [50.0, 0.0, 0.0, 0.0],
            "cash_bid": [99.9, 99.9, 99.9, 99.9],
            "cash_ask": [100.1, 100.1, 100.1, 100.1],
            "cash_mid": [100.0] * 4,
            "perp_bid": [100.9, 100.9, 100.0, 100.0],
            "perp_ask": [101.1, 101.1, 100.2, 100.2],
            "perp_mid": [101.0, 101.0, 100.1, 100.1],
            "funding_rate": [0.0] * 4,
            "funding_time": [pd.NaT] * 4,
        }
    )
    trades = run_backtest(frame, cfg)
    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["entry_time"] == times[1]
    assert trade["exit_time"] == times[3]
    assert trade["quoted_spread_cost"] > 0
    assert trade["fee_cost"] > 0
    assert trade["net_pnl"] < trade["mid_gross_pnl"]


def test_execution_clock_distinguishes_five_and_thirty_second_delay() -> None:
    cfg = deepcopy(load_config())
    cfg["signal"]["min_holding_minutes"] = 0
    times = pd.date_range("2026-01-05 14:30", periods=10, freq="10s", tz="UTC")
    z_score = [3.0, np.nan, np.nan, np.nan, 0.1, np.nan, np.nan, np.nan, np.nan, np.nan]
    signal_observation = [True, False, False, False, True, False, False, False, False, False]
    frame = pd.DataFrame(
        {
            "timestamp": times,
            "symbol": "NVDA",
            "quality_ok": True,
            "entry_candidate": [True] + [False] * 9,
            "signal_observation": signal_observation,
            "z_score": z_score,
            "predicted_net_edge_bps": [50.0] + [np.nan] * 9,
            "cash_bid": [99.99] * 10,
            "cash_ask": [100.01] * 10,
            "cash_mid": [100.0] * 10,
            "perp_bid": np.linspace(101.0, 100.0, 10),
            "perp_ask": np.linspace(101.02, 100.02, 10),
            "perp_mid": np.linspace(101.01, 100.01, 10),
            "funding_rate": [0.0] * 10,
            "funding_time": [pd.NaT] * 10,
        }
    )
    cfg["execution"]["delay_seconds"] = 5
    fast = run_backtest(frame, cfg)
    cfg["execution"]["delay_seconds"] = 30
    slow = run_backtest(frame, cfg)

    assert fast.iloc[0]["entry_time"] == times[1]
    assert slow.iloc[0]["entry_time"] == times[3]
    assert fast.iloc[0]["exit_time"] == times[5]
    assert slow.iloc[0]["exit_time"] == times[7]


def test_historical_provider_payloads_are_parsed_with_utc_timestamps() -> None:
    alpaca = AlpacaHistoricalBars.parse_payload(
        {
            "bars": {
                "NVDA": [
                    {
                        "t": "2026-07-10T13:30:00Z",
                        "o": 100,
                        "h": 101,
                        "l": 99,
                        "c": 100.5,
                        "v": 123,
                    }
                ]
            }
        }
    )
    binance = BinanceHistoricalMinutes.parse_klines(
        [[1783690200000, "100.1", "101", "99", "100.6", "0"]]
    )

    assert alpaca.iloc[0]["provider_symbol"] == "NVDA"
    assert str(alpaca.iloc[0]["timestamp"].tz) == "UTC"
    assert str(binance.iloc[0]["timestamp"].tz) == "UTC"
    assert binance.iloc[0]["close"] == 100.6


def test_historical_signal_statistics_do_not_use_current_or_future_basis() -> None:
    cfg = deepcopy(load_config())
    cfg["signal"]["rolling_window"] = 3
    cfg["signal"]["min_history"] = 3
    times = pd.date_range("2026-07-06 13:30", periods=5, freq="min", tz="UTC")
    aligned = pd.DataFrame(
        {
            "timestamp": times,
            "decision_time": times + pd.Timedelta(minutes=1),
            "symbol": "NVDA",
            "quality_ok": True,
            "basis": [0.0010, 0.0012, 0.0009, 0.0011, 0.0013],
        }
    )
    baseline = build_historical_signals(aligned, cfg)
    changed = aligned.copy()
    changed.loc[changed.index[-1], "basis"] = 0.5
    revised = build_historical_signals(changed, cfg)

    assert np.isclose(
        baseline.iloc[-1]["basis_mean_lagged"], revised.iloc[-1]["basis_mean_lagged"]
    )


def _historical_signal_frame() -> pd.DataFrame:
    times = pd.date_range("2026-07-06 13:30", periods=5, freq="min", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": times,
            "decision_time": times + pd.Timedelta(minutes=1),
            "symbol": "NVDA",
            "quality_ok": True,
            "basis": [0.01, 0.009, 0.001, 0.0, 0.0],
            "basis_mean_lagged": 0.0,
            "basis_std_lagged": 0.001,
            "z_score": [3.0, 0.1, 0.0, 0.0, 0.0],
            "predicted_net_edge_bps": [50.0, 0.0, 0.0, 0.0, 0.0],
            "entry_candidate": [True, False, False, False, False],
            "cash_open": [100.0, 100.0, 100.0, 100.0, 100.0],
            "perp_open": [101.0, 101.0, 100.2, 100.1, 100.1],
        }
    )


def test_historical_backtest_uses_next_contiguous_minute_open() -> None:
    cfg = deepcopy(load_config())
    cfg["signal"]["min_holding_minutes"] = 0
    frame = _historical_signal_frame()

    trades = run_historical_backtest(frame, {"NVDA": pd.DataFrame()}, cfg)

    assert len(trades) == 1
    assert trades.iloc[0]["signal_time"] == frame.iloc[0]["decision_time"]
    assert trades.iloc[0]["entry_time"] == frame.iloc[1]["timestamp"]
    assert trades.iloc[0]["exit_time"] == frame.iloc[2]["timestamp"]
    assert trades.iloc[0]["net_pnl"] < trades.iloc[0]["mid_gross_pnl"]


def test_historical_screen_can_never_return_formal_go() -> None:
    cfg = deepcopy(load_config())
    cfg["signal"]["rolling_window"] = 2
    cfg["signal"]["min_history"] = 2
    result = evaluate_historical_screen(
        _historical_signal_frame(),
        pd.DataFrame([{"symbol": "NVDA", "quality_rows": 5}]),
        {"NVDA": pd.DataFrame()},
        cfg,
    )

    assert result["status"] == "PRELIMINARY_ONLY"
    assert "go" not in result


def test_drawdown_includes_loss_from_initial_capital() -> None:
    cfg = load_config()
    trades = pd.DataFrame(
        {
            "exit_time": [pd.Timestamp("2026-07-06", tz="UTC")],
            "net_pnl": [-300.0],
            "mid_gross_pnl": [-100.0],
            "holding_minutes": [5.0],
        }
    )

    metrics = performance_metrics(trades, cfg)

    expected = -300.0 / float(cfg["execution"]["initial_capital"])
    assert np.isclose(metrics["max_drawdown"], expected)


def test_nonnegative_ridge_solves_constraint_instead_of_clipping() -> None:
    random = np.random.default_rng(42)
    first = random.normal(0.0, 0.01, 200)
    second = 0.9 * first + random.normal(0.0, 0.002, 200)
    x = np.column_stack([first, second])
    y = 1.8 * first + random.normal(0.0, 0.001, 200)

    _, beta, r_squared = _fit_ridge(x, y, alpha=1.0, nonnegative=True)

    assert (beta >= 0).all()
    assert beta.sum() < 3.0
    assert r_squared > 0.9


def test_factor_backtest_uses_later_hour_and_beta_hedge() -> None:
    cfg = deepcopy(load_config())
    cfg["factor_strategy"]["min_holding_hours"] = 0
    cfg["factor_strategy"]["targets"]["NVDA"]["factors"] = ["QQQ", "SPY"]
    times = pd.date_range("2026-07-06 13:30", periods=5, freq="h", tz="UTC")
    signals = pd.DataFrame(
        {
            "timestamp": times,
            "decision_time": times + pd.Timedelta(hours=1),
            "session_date": [pd.Timestamp("2026-07-06").date()] * 5,
            "symbol": "NVDA",
            "quality_ok": True,
            "entry_candidate": [True, False, False, False, False],
            "z_score": [3.0, 0.1, 0.0, 0.0, 0.0],
            "model_r2": 0.8,
            "residual_ar1_lagged": -0.2,
            "residual_ar1_t_stat_lagged": -2.0,
            "predicted_net_edge_bps": [80.0, 0.0, 0.0, 0.0, 0.0],
            "beta_QQQ": 1.0,
            "beta_SPY": 0.0,
            "target_open": [101.0, 101.0, 99.0, 99.0, 99.0],
            "QQQ_open": [100.0, 100.0, 100.0, 100.0, 100.0],
            "SPY_open": [100.0] * 5,
        }
    )
    delay = pd.Timedelta(minutes=int(cfg["factor_strategy"]["execution_delay_minutes"]))
    execution_times = [times[1] + delay, times[2] + delay]
    history = {
        "NVDA": {
            "funding": pd.DataFrame(),
            "perp": pd.DataFrame({"timestamp": execution_times, "open": [101.0, 99.0]}),
        },
        "QQQ": {
            "funding": pd.DataFrame(),
            "perp": pd.DataFrame({"timestamp": execution_times, "open": [100.0, 100.0]}),
        },
        "SPY": {
            "funding": pd.DataFrame(),
            "perp": pd.DataFrame({"timestamp": execution_times, "open": [100.0, 100.0]}),
        },
    }

    trades = run_factor_backtest(signals, history, cfg)

    assert len(trades) == 1
    assert trades.iloc[0]["entry_time"] == execution_times[0]
    assert trades.iloc[0]["exit_time"] == execution_times[1]
    assert trades.iloc[0]["QQQ_signed_notional"] == 100_000.0
    assert trades.iloc[0]["net_pnl"] < trades.iloc[0]["mid_gross_pnl"]


def test_factor_snapshot_health_reports_observed_spread() -> None:
    cfg = load_config()
    now = pd.Timestamp("2026-07-13 14:30:00", tz="UTC")
    records = [
        {
            "symbol": symbol,
            "timestamp": (now - pd.Timedelta(seconds=1)).isoformat(),
            "received_at": now.isoformat(),
            "bid": 99.99,
            "ask": 100.01,
        }
        for symbol in ("NVDA", "QQQ")
    ]

    health = factor_snapshot_health(records, cfg)

    assert health["fresh"] == 2
    assert np.isclose(health["median_spread_bps"], 2.0)


def test_factor_beta_does_not_use_current_target_return() -> None:
    cfg = deepcopy(load_config())
    cfg["factor_strategy"]["targets"]["NVDA"]["factors"] = ["QQQ", "SPY"]
    cfg["factor_strategy"]["session_parameters"]["us_rth"] = {
        "regression_lookback_hours": 12,
        "regression_min_hours": 6,
        "residual_lookback_hours": 8,
        "residual_min_hours": 4,
    }
    times = pd.date_range("2026-07-01 13:30", periods=24, freq="h", tz="UTC")
    qqq = np.linspace(-0.01, 0.01, len(times))
    spy = np.sin(np.arange(len(times))) * 0.003
    panel = pd.DataFrame(
        {
            "timestamp": times,
            "decision_time": times + pd.Timedelta(hours=1),
            "session_date": [times[index // 6].date() for index in range(len(times))],
            "symbol": "NVDA",
            "quality_ok": True,
            "target_return": 1.2 * qqq + 0.4 * spy + 0.001 * np.sin(np.arange(len(times)) * 0.7),
            "QQQ_return": qqq,
            "SPY_return": spy,
        }
    )
    baseline = build_factor_signals(panel, cfg)
    changed = panel.copy()
    changed.loc[changed.index[-1], "target_return"] += 0.5
    revised = build_factor_signals(changed, cfg)

    assert np.isclose(baseline.iloc[-1]["beta_QQQ"], revised.iloc[-1]["beta_QQQ"])
    assert np.isclose(baseline.iloc[-1]["beta_SPY"], revised.iloc[-1]["beta_SPY"])
    assert np.isclose(
        baseline.iloc[-1]["reversion_slope_lagged"],
        revised.iloc[-1]["reversion_slope_lagged"],
    )
    assert np.isclose(
        baseline.iloc[-1]["adf_p_value_lagged"],
        revised.iloc[-1]["adf_p_value_lagged"],
    )


def test_factor_universe_meets_project_category_coverage() -> None:
    cfg = load_config()

    validate_factor_configuration(cfg)
    targets = target_specs(cfg)
    categories = pd.Series([spec["category"] for spec in targets.values()]).value_counts()

    assert len(targets) == 30
    assert categories["crypto_equity"] >= 5
    assert categories["crypto_asset"] >= 10
    assert categories["tech_equity"] >= 5


def test_factor_entry_never_opens_beyond_stop_threshold() -> None:
    cfg = deepcopy(load_config())
    cfg["factor_strategy"]["targets"]["NVDA"]["factors"] = ["QQQ", "SPY"]
    signals = pd.DataFrame(
        {
            "symbol": ["NVDA", "NVDA"],
            "timestamp": pd.date_range("2026-07-01", periods=2, freq="h", tz="UTC"),
            "quality_ok": True,
            "model_eligible": True,
            "z_score": [3.0, 5.0],
            "expected_gross_edge_bps": [100.0, 100.0],
            "beta_QQQ": [1.0, 1.0],
            "beta_SPY": [0.0, 0.0],
        }
    )

    repriced = reprice_factor_signals(signals, cfg, cost_multiplier=1.0)

    assert repriced["entry_candidate"].tolist() == [True, False]


def test_continuous_hourly_bars_do_not_break_at_utc_midnight() -> None:
    cfg = deepcopy(load_config())
    cfg["factor_strategy"]["minimum_execution_bars_per_hour"] = 2
    cfg["factor_strategy"]["minimum_signal_bars_per_hour"] = 2
    times = pd.date_range("2026-07-01 23:00", periods=4, freq="30min", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": times,
            "open": [100.0, 100.1, 100.2, 100.3],
            "high": [100.2, 100.3, 100.4, 100.5],
            "low": [99.9, 100.0, 100.1, 100.2],
            "close": [100.1, 100.2, 100.3, 100.4],
            "quote_volume": [1_000.0] * 4,
        }
    )

    bars = _hourly_bars({"perp": frame, "mark": frame, "index": frame}, cfg, "continuous")

    assert len(bars) == 2
    assert bars["execution_group"].nunique() == 1
    assert bars.iloc[0]["session_date"] != bars.iloc[1]["session_date"]


def test_close_reason_measures_holding_time_at_delayed_exit_fill() -> None:
    cfg = deepcopy(load_config())
    cfg["factor_strategy"]["min_holding_hours"] = 1
    row = pd.Series(
        {
            "decision_time": pd.Timestamp("2026-07-01 15:00", tz="UTC"),
            "z_score": 0.1,
        }
    )

    reason = _close_reason(
        row,
        pd.Timestamp("2026-07-01 14:05", tz="UTC"),
        cfg,
        execution_delay_minutes=5,
    )

    assert reason == "reversion"


def test_stop_loss_is_not_blocked_by_minimum_holding_period() -> None:
    cfg = deepcopy(load_config())
    cfg["factor_strategy"]["min_holding_hours"] = 4
    row = pd.Series(
        {
            "decision_time": pd.Timestamp("2026-07-01 15:00", tz="UTC"),
            "z_score": 5.0,
        }
    )

    reason = _close_reason(
        row,
        pd.Timestamp("2026-07-01 14:05", tz="UTC"),
        cfg,
        execution_delay_minutes=5,
    )

    assert reason == "stop"


def test_crypto_research_subset_downloads_only_required_instruments() -> None:
    cfg = load_config()

    selected = research_target_specs(cfg, categories=["crypto_asset"])
    instruments = instrument_specs_for_targets(cfg, selected)

    assert len(selected) == 13
    assert {"BTC", "ETH", "SOL"}.issubset(instruments)
    assert "QQQ" not in instruments
    assert "MSTR" not in instruments


def test_dated_cache_is_reused_without_redownloading() -> None:
    cache_dir = Path("tests/_cache_range_test")
    cache_dir.mkdir(exist_ok=True)
    times = pd.date_range("2026-07-01", periods=288, freq="5min", tz="UTC")
    cached = pd.DataFrame(
        {
            "timestamp": times,
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
        }
    )
    existing = cache_dir / "binance_TESTUSDT_perp_5m_20260701_20260701.pkl"
    requested = cache_dir / "binance_TESTUSDT_perp_5m_20260701_20260702.pkl"

    def unexpected_fetch(_start: pd.Timestamp, _end: pd.Timestamp) -> pd.DataFrame:
        raise AssertionError("covered cache range must not be fetched")

    try:
        cached.to_pickle(existing)
        frame, source = _read_or_extend_cache(
            requested,
            cache_dir,
            "TESTUSDT",
            "perp",
            "5m",
            pd.Timestamp("2026-07-01", tz="UTC"),
            pd.Timestamp("2026-07-02", tz="UTC"),
            False,
            unexpected_fetch,
        )

        assert source == "cache-reused"
        assert len(frame) == len(cached)
        assert requested.exists()
    finally:
        existing.unlink(missing_ok=True)
        requested.unlink(missing_ok=True)
        cache_dir.rmdir()


def test_tradable_label_uses_delayed_perpetual_prices_and_full_cost() -> None:
    cfg = load_config()
    timestamp = pd.Timestamp("2026-07-01 00:00", tz="UTC")
    delay = pd.Timedelta(minutes=int(cfg["factor_strategy"]["execution_delay_minutes"]))
    entry_time = timestamp + pd.Timedelta(hours=1) + delay
    exit_time = entry_time + pd.Timedelta(hours=2)
    signals = pd.DataFrame(
        {
            "symbol": ["ETH"],
            "session": ["continuous"],
            "timestamp": [timestamp],
            "decision_time": [timestamp + pd.Timedelta(hours=1)],
            "execution_group": ["continuous"],
            "beta_BTC": [0.5],
        }
    )
    history = {
        "ETH": {
            "perp": pd.DataFrame(
                {"timestamp": [entry_time, exit_time], "open": [100.0, 101.0]}
            ),
            "funding": pd.DataFrame(columns=["timestamp", "funding_rate"]),
        },
        "BTC": {
            "perp": pd.DataFrame(
                {"timestamp": [entry_time, exit_time], "open": [100.0, 100.0]}
            ),
            "funding": pd.DataFrame(columns=["timestamp", "funding_rate"]),
        },
    }

    labels = build_tradable_labels(signals, history, cfg, horizons=[2])

    assert len(labels) == 1
    row = labels.iloc[0]
    assert row["entry_time"] == entry_time
    assert row["exit_time"] == exit_time
    assert np.isclose(row["long_gross_bps"], 100.0)
    expected_cost = instrument_roundtrip_cost_bps(cfg, "ETH") + 0.5 * (
        instrument_roundtrip_cost_bps(cfg, "BTC")
    )
    assert np.isclose(row["transaction_cost_bps"], expected_cost)
    assert row["long_net_bps"] < row["long_gross_bps"]


def test_research_cache_fingerprint_tracks_signal_and_execution_settings() -> None:
    cfg = load_config()
    signal_base, label_base = _research_cache_fingerprints(cfg)
    delayed = deepcopy(cfg)
    delayed["factor_strategy"]["execution_delay_minutes"] += 5
    signal_delayed, label_delayed = _research_cache_fingerprints(delayed)
    slower_signal = deepcopy(cfg)
    slower_signal["factor_strategy"]["signal_horizon_hours"] = 8
    signal_slower, label_slower = _research_cache_fingerprints(slower_signal)

    assert signal_delayed == signal_base
    assert label_delayed != label_base
    assert signal_slower != signal_base
    assert label_slower != label_base


def test_purged_split_removes_labels_crossing_boundaries() -> None:
    cfg = load_config()
    entries = pd.date_range("2026-01-01", periods=12, freq="D", tz="UTC")
    exits = entries + pd.Timedelta(hours=1)
    exits = exits.to_series(index=range(len(exits)))
    exits.iloc[5] = pd.Timestamp("2026-01-07 12:00", tz="UTC")
    exits.iloc[8] = pd.Timestamp("2026-01-10 12:00", tz="UTC")
    labels = pd.DataFrame(
        {
            "entry_time": entries,
            "exit_time": exits.to_numpy(),
            "long_gross_bps": 0.0,
        }
    )

    partitions, metadata = purged_chronological_partitions(labels, cfg)

    assert metadata["validation_start"] == pd.Timestamp("2026-01-07", tz="UTC")
    assert metadata["test_start"] == pd.Timestamp("2026-01-10", tz="UTC")
    assert partitions["train"]["exit_time"].lt(metadata["validation_start"]).all()
    assert partitions["validation"]["exit_time"].lt(metadata["test_start"]).all()
    assert metadata["purged_rows"] == 2


def test_portfolio_nets_opposite_factor_fills_before_costing() -> None:
    cfg = load_config()
    entry = pd.Timestamp("2026-07-01 14:35", tz="UTC")
    exit_time = entry + pd.Timedelta(hours=2)
    btc_cost = instrument_roundtrip_cost_bps(cfg, "BTC")
    rows = []
    for symbol, direction, long_gross in (
        ("MSTR", 1, 10.0),
        ("HOOD", -1, -10.0),
    ):
        pair_cost = instrument_roundtrip_cost_bps(cfg, symbol) + btc_cost
        rows.append(
            {
                "symbol": symbol,
                "entry_time": entry,
                "exit_time": exit_time,
                "direction": direction,
                "beta_BTC": 1.0,
                "beta_QQQ": 0.0,
                "long_gross_bps": long_gross,
                "long_funding_bps": 0.0,
                "transaction_cost_bps": pair_cost,
                "opportunity_cost_bps": 0.0,
            }
        )

    metrics, events = run_netted_portfolio(pd.DataFrame(rows), cfg)

    assert metrics["trades"] == 2
    assert metrics["cost_saving"] > 0
    assert metrics["netted_transaction_cost"] < metrics["unnetted_transaction_cost"]
    assert "BTC" not in set(events["instrument"])


def test_failed_validation_keeps_final_test_locked() -> None:
    cfg = load_config()
    minimum = int(cfg["tradable_research"]["minimum_validation_trades"])

    assert not validation_authorizes_test(
        {
            "trades": minimum,
            "gross_pnl": 100.0,
            "net_pnl": -1.0,
            "stressed_net_pnl": -10.0,
        },
        cfg,
    )
    assert not validation_authorizes_test(
        {
            "trades": minimum,
            "gross_pnl": 100.0,
            "net_pnl": 1.0,
            "stressed_net_pnl": -1.0,
        },
        cfg,
    )
    assert validation_authorizes_test(
        {
            "trades": minimum,
            "gross_pnl": 100.0,
            "net_pnl": 1.0,
            "stressed_net_pnl": 0.1,
        },
        cfg,
    )


def test_validation_requires_both_chronological_halves_to_survive_stress() -> None:
    cfg = load_config()
    total = {
        "trades": 100,
        "gross_pnl": 100.0,
        "net_pnl": 50.0,
        "stressed_net_pnl": 10.0,
    }
    passing_half = {
        "trades": int(
            cfg["tradable_research"]["minimum_validation_subperiod_trades"]
        ),
        "gross_pnl": 20.0,
        "net_pnl": 10.0,
        "stressed_net_pnl": 1.0,
    }
    failing_half = {**passing_half, "stressed_net_pnl": -1.0}

    assert validation_authorizes_test(
        total,
        cfg,
        {"first_half": passing_half, "second_half": passing_half},
    )
    assert not validation_authorizes_test(
        total,
        cfg,
        {"first_half": passing_half, "second_half": failing_half},
    )


def test_dynamic_beta_prediction_does_not_use_current_target_return() -> None:
    cfg = deepcopy(load_config())
    cfg["factor_strategy"]["session_parameters"]["continuous"] = {
        "regression_lookback_hours": 12,
        "regression_min_hours": 6,
        "residual_lookback_hours": 8,
        "residual_min_hours": 4,
    }
    times = pd.date_range("2026-07-01", periods=24, freq="h", tz="UTC")
    btc = np.linspace(-0.01, 0.01, len(times))
    panel = pd.DataFrame(
        {
            "timestamp": times,
            "decision_time": times + pd.Timedelta(hours=1),
            "session_date": times.date,
            "execution_group": "continuous",
            "symbol": "ETH",
            "quality_ok": True,
            "target_return": 1.3 * btc + np.sin(np.arange(len(times))) * 0.001,
            "BTC_return": btc,
        }
    )

    baseline = build_dynamic_factor_signals(panel, cfg)
    changed = panel.copy()
    changed.loc[changed.index[-1], "target_return"] += 0.5
    revised = build_dynamic_factor_signals(changed, cfg)

    assert np.isclose(baseline.iloc[-1]["beta_BTC"], revised.iloc[-1]["beta_BTC"])
