from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def generate_demo_quotes(cfg: dict[str, Any], output: str | Path) -> Path:
    rng = np.random.default_rng(int(cfg["project"]["random_seed"]))
    symbols = list(cfg["symbols"])[:3]
    business_days = pd.bdate_range("2026-01-05", periods=25, tz="America/New_York")
    timestamps = []
    for day in business_days:
        start = day.normalize() + pd.Timedelta(hours=9, minutes=30)
        timestamps.extend(pd.date_range(start, periods=390, freq="min").tz_convert("UTC"))
    index = pd.DatetimeIndex(timestamps)
    records: list[dict] = []
    for symbol_index, symbol in enumerate(symbols):
        cash = 100.0 + symbol_index * 30.0
        basis = 0.0
        for i, timestamp in enumerate(index):
            cash *= np.exp(rng.normal(0.0, 0.0007))
            shock = rng.normal(0.0, 0.00018)
            if i % 173 == 0 and i > 500:
                shock += rng.choice([-1.0, 1.0]) * 0.008
            basis = 0.92 * basis + shock
            perp = cash * np.exp(basis)
            cash_half_spread = cash * 0.00010
            perp_half_spread = perp * 0.00015
            cash_source = timestamp + pd.Timedelta(milliseconds=100)
            perp_source = timestamp + pd.Timedelta(milliseconds=300)
            cash_received = timestamp + pd.Timedelta(seconds=1.0)
            perp_received = timestamp + pd.Timedelta(seconds=1.3)
            funding_time = timestamp.ceil("8h")
            records.extend(
                [
                    {
                        "timestamp": cash_source,
                        "received_at": cash_received,
                        "symbol": symbol,
                        "venue": "cash",
                        "bid": cash - cash_half_spread,
                        "ask": cash + cash_half_spread,
                        "bid_size": 1000,
                        "ask_size": 1000,
                        "source": "deterministic_demo",
                        "clock_uncertainty_ms": 0.0,
                        "receive_time_assumed": False,
                    },
                    {
                        "timestamp": perp_source,
                        "received_at": perp_received,
                        "symbol": symbol,
                        "venue": "perp",
                        "bid": perp - perp_half_spread,
                        "ask": perp + perp_half_spread,
                        "bid_size": 1000,
                        "ask_size": 1000,
                        "mark_price": perp,
                        "index_price": cash,
                        "funding_rate": 0.00001,
                        "funding_time": funding_time,
                        "source": "deterministic_demo",
                        "clock_uncertainty_ms": 0.0,
                        "receive_time_assumed": False,
                    },
                ]
            )
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(target, index=False)
    return target
