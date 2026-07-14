from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "timestamp",
    "received_at",
    "symbol",
    "venue",
    "bid",
    "ask",
}
NUMERIC_COLUMNS = [
    "bid",
    "ask",
    "bid_size",
    "ask_size",
    "mark_price",
    "index_price",
    "funding_rate",
    "clock_uncertainty_ms",
]
TIME_COLUMNS = ["timestamp", "received_at", "funding_time"]


class QuoteSchemaError(ValueError):
    pass


def _read_one(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise QuoteSchemaError(
                        f"Invalid JSON in {path} at line {line_number}: {exc}"
                    ) from exc
        return pd.DataFrame(records)
    raise QuoteSchemaError(f"Unsupported quote file: {path}")


def _candidate_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if not path.exists():
        raise FileNotFoundError(path)
    for pattern in ("*.parquet", "*.csv", "*.jsonl", "*.ndjson"):
        yield from sorted(path.glob(pattern))


def load_quotes(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    files = list(_candidate_files(source))
    if not files:
        raise QuoteSchemaError(f"No quote files found under {source}")
    frames = [_read_one(file) for file in files]
    frame = pd.concat(frames, ignore_index=True, sort=False)
    return normalize_quotes(frame)


def normalize_quotes(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise QuoteSchemaError(f"Missing required columns: {sorted(missing)}")

    result = frame.copy()
    for column in TIME_COLUMNS:
        if column in result:
            result[column] = pd.to_datetime(result[column], utc=True, errors="coerce")
    for column in NUMERIC_COLUMNS:
        if column not in result:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result["symbol"] = result["symbol"].astype(str).str.upper()
    result["venue"] = result["venue"].astype(str).str.lower()
    if "receive_time_assumed" not in result:
        result["receive_time_assumed"] = False
    result["receive_time_assumed"] = result["receive_time_assumed"].fillna(False).astype(bool)

    result["quote_valid"] = (
        result["timestamp"].notna()
        & result["received_at"].notna()
        & result["venue"].isin(["cash", "perp"])
        & result["bid"].gt(0)
        & result["ask"].gt(result["bid"])
    )
    result = result.sort_values(["received_at", "venue", "symbol"])
    result = result.drop_duplicates(
        ["received_at", "venue", "symbol"], keep="last"
    ).reset_index(drop=True)
    return result


def save_quotes(frame: pd.DataFrame, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() == ".csv":
        frame.to_csv(target, index=False)
    else:
        frame.to_parquet(target, index=False)
    return target
