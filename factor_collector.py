from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from collector import DailyJsonlSink
from factor_config import instrument_specs, validate_factor_configuration
from providers import BinanceFuturesSnapshot
from schema import load_quotes


def factor_mappings(cfg: dict[str, Any]) -> dict[str, dict[str, str]]:
    validate_factor_configuration(cfg)
    return {
        canonical: {"perp_symbol": str(spec["perp_symbol"])}
        for canonical, spec in instrument_specs(cfg).items()
    }


def factor_snapshot_health(
    records: list[dict[str, Any]], cfg: dict[str, Any]
) -> dict[str, float | int]:
    if not records:
        return {"instruments": 0, "fresh": 0, "median_age_seconds": np.nan, "median_spread_bps": np.nan}
    frame = pd.DataFrame(records)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["received_at"] = pd.to_datetime(frame["received_at"], utc=True, errors="coerce")
    frame["bid"] = pd.to_numeric(frame["bid"], errors="coerce")
    frame["ask"] = pd.to_numeric(frame["ask"], errors="coerce")
    frame["age_seconds"] = (frame["received_at"] - frame["timestamp"]).dt.total_seconds()
    mid = (frame["bid"] + frame["ask"]) / 2.0
    frame["spread_bps"] = (frame["ask"] - frame["bid"]) / mid * 10_000.0
    valid = (
        frame["bid"].gt(0)
        & frame["ask"].gt(frame["bid"])
        & frame["age_seconds"].between(-1.0, float(cfg["alignment"]["max_source_age_seconds"]))
    )
    return {
        "instruments": int(frame["symbol"].nunique()),
        "fresh": int(frame.loc[valid, "symbol"].nunique()),
        "median_age_seconds": float(frame["age_seconds"].median()),
        "median_spread_bps": float(frame.loc[valid, "spread_bps"].median()),
    }


def collect_factor_live(cfg: dict[str, Any], duration_hours: float | None = None) -> None:
    mappings = factor_mappings(cfg)
    root = Path(cfg["_root"])
    sink = DailyJsonlSink(root / str(cfg["collector"]["factor_raw_dir"]))
    provider = BinanceFuturesSnapshot(cfg)
    poll_seconds = max(float(cfg["collector"]["poll_seconds"]), 0.25)
    deadline = None if not duration_hours or duration_hours <= 0 else time.monotonic() + duration_hours * 3600
    print(f"Collecting {len(mappings)} Binance factor-strategy L1 books every {poll_seconds:.2f}s")
    print(f"Raw output: {sink.directory}")
    cycles = 0
    failures = 0
    try:
        while deadline is None or time.monotonic() < deadline:
            started = time.monotonic()
            try:
                records = provider.fetch(mappings, uuid.uuid4().hex)
                target = sink.append(records)
                health = factor_snapshot_health(records, cfg)
                cycles += 1
                if cycles == 1 or cycles % 30 == 0:
                    print(
                        f"cycles={cycles} records={len(records)} failures={failures} "
                        f"fresh={health['fresh']}/{health['instruments']} "
                        f"age={health['median_age_seconds']:.1f}s "
                        f"spread={health['median_spread_bps']:.2f}bps file={target.name}"
                    )
            except Exception as exc:
                failures += 1
                print(f"factor collection failure {failures}: {exc}")
            time.sleep(max(0.0, poll_seconds - (time.monotonic() - started)))
    except KeyboardInterrupt:
        print("Collection stopped by user; JSONL data already flushed.")


def analyze_factor_live_costs(
    cfg: dict[str, Any], raw_path: str | Path, output_dir: str | Path | None = None
) -> Path:
    quotes = load_quotes(raw_path)
    frame = quotes.loc[quotes["venue"].eq("perp") & quotes["quote_valid"]].copy()
    if frame.empty:
        raise ValueError("No valid Binance perpetual quotes found")
    frame["age_seconds"] = (frame["received_at"] - frame["timestamp"]).dt.total_seconds()
    frame = frame.loc[
        frame["age_seconds"].between(-1.0, float(cfg["alignment"]["max_source_age_seconds"]))
    ]
    mid = (frame["bid"] + frame["ask"]) / 2.0
    frame["spread_bps"] = (frame["ask"] - frame["bid"]) / mid * 10_000.0
    summary = frame.groupby("symbol", as_index=False).agg(
        observations=("spread_bps", "size"),
        median_spread_bps=("spread_bps", "median"),
        p90_spread_bps=("spread_bps", lambda values: values.quantile(0.90)),
        median_age_seconds=("age_seconds", "median"),
        first_timestamp=("timestamp", "min"),
        last_timestamp=("timestamp", "max"),
    )
    root = Path(cfg["_root"])
    output = Path(output_dir) if output_dir else root / str(cfg["output"]["report_dir"]) / "factor_live"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "factor_live_spreads.csv"
    summary.to_csv(path, index=False)
    print(summary.to_string(index=False))
    print(f"Spread report: {path.resolve()}")
    return path
