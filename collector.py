from __future__ import annotations

import json
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from providers import AlpacaLatestQuotes, BinanceFuturesSnapshot


class DailyJsonlSink:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def append(self, records: list[dict]) -> Path:
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        target = self.directory / f"quotes_{day}.jsonl"
        with target.open("a", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            handle.flush()
        return target


def snapshot_health(records: list[dict], cfg: dict[str, Any]) -> dict[str, float | int]:
    """Summarize quote freshness separately from HTTP collection success."""
    by_symbol: dict[str, dict[str, dict]] = {}
    for record in records:
        by_symbol.setdefault(record["symbol"], {})[record["venue"]] = record

    max_age = float(cfg["alignment"]["max_source_age_seconds"])
    max_skew = float(cfg["alignment"]["max_source_skew_seconds"])
    fresh_pairs = 0
    cash_ages: list[float] = []
    perp_ages: list[float] = []
    for venues in by_symbol.values():
        if "cash" not in venues or "perp" not in venues:
            continue
        cash = venues["cash"]
        perp = venues["perp"]
        cash_time = pd.Timestamp(cash["timestamp"])
        perp_time = pd.Timestamp(perp["timestamp"])
        decision_time = max(pd.Timestamp(cash["received_at"]), pd.Timestamp(perp["received_at"]))
        cash_age = (decision_time - cash_time).total_seconds()
        perp_age = (decision_time - perp_time).total_seconds()
        skew = abs((cash_time - perp_time).total_seconds())
        cash_ages.append(cash_age)
        perp_ages.append(perp_age)
        quotes_valid = (
            float(cash["bid"]) > 0
            and float(cash["ask"]) > float(cash["bid"])
            and float(perp["bid"]) > 0
            and float(perp["ask"]) > float(perp["bid"])
        )
        if quotes_valid and -1.0 <= cash_age <= max_age and -1.0 <= perp_age <= max_age and skew <= max_skew:
            fresh_pairs += 1
    return {
        "pairs": len(by_symbol),
        "fresh_pairs": fresh_pairs,
        "median_cash_age_seconds": statistics.median(cash_ages) if cash_ages else float("nan"),
        "median_perp_age_seconds": statistics.median(perp_ages) if perp_ages else float("nan"),
    }


def collect_live(cfg: dict[str, Any], duration_hours: float | None = None) -> None:
    mappings = cfg["symbols"]
    collector_cfg = cfg["collector"]
    root = Path(cfg["_root"])
    sink = DailyJsonlSink(root / collector_cfg["raw_dir"])
    cash = AlpacaLatestQuotes(cfg)
    perp = BinanceFuturesSnapshot(cfg)
    poll_seconds = max(float(collector_cfg["poll_seconds"]), 0.25)
    deadline = None if not duration_hours or duration_hours <= 0 else time.monotonic() + duration_hours * 3600

    print(f"Collecting {len(mappings)} synchronized pairs every {poll_seconds:.2f}s")
    print(f"Raw output: {sink.directory}")
    cycles = 0
    failures = 0
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            while deadline is None or time.monotonic() < deadline:
                cycle_started = time.monotonic()
                snapshot_id = uuid.uuid4().hex
                cash_future = pool.submit(cash.fetch, mappings, snapshot_id)
                perp_future = pool.submit(perp.fetch, mappings, snapshot_id)
                try:
                    records = cash_future.result() + perp_future.result()
                    target = sink.append(records)
                    health = snapshot_health(records, cfg)
                    cycles += 1
                    if cycles == 1 or cycles % 30 == 0:
                        print(
                            f"cycles={cycles} api_records={len(records)} failures={failures} "
                            f"fresh_pairs={health['fresh_pairs']}/{health['pairs']} "
                            f"cash_age={health['median_cash_age_seconds']:.1f}s "
                            f"perp_age={health['median_perp_age_seconds']:.1f}s "
                            f"file={target.name}"
                        )
                except Exception as exc:
                    failures += 1
                    print(f"collection failure {failures}: {exc}")
                elapsed = time.monotonic() - cycle_started
                time.sleep(max(0.0, poll_seconds - elapsed))
    except KeyboardInterrupt:
        print("Collection stopped by user; JSONL data already flushed.")
