from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _spread_bps(bid: float, ask: float) -> float:
    midpoint = (bid + ask) / 2.0
    return (ask - bid) / midpoint * 10_000.0 if midpoint > 0 else np.nan


def align_quotes(quotes: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    settings = cfg["alignment"]
    mappings = cfg["symbols"]
    rows: list[dict] = []

    for symbol, pair in mappings.items():
        events = quotes.loc[quotes["symbol"].eq(symbol)].sort_values("received_at")
        if events.empty:
            continue
        latest: dict[str, pd.Series] = {}
        for _, event in events.iterrows():
            venue = event["venue"]
            latest[venue] = event
            if "cash" not in latest or "perp" not in latest:
                continue
            cash = latest["cash"]
            perp = latest["perp"]
            decision_time = max(cash["received_at"], perp["received_at"])
            cash_age = (decision_time - cash["timestamp"]).total_seconds()
            perp_age = (decision_time - perp["timestamp"]).total_seconds()
            cash_receive_lag = (cash["received_at"] - cash["timestamp"]).total_seconds()
            perp_receive_lag = (perp["received_at"] - perp["timestamp"]).total_seconds()
            skew = abs((cash["timestamp"] - perp["timestamp"]).total_seconds())
            cash_bid, cash_ask = float(cash["bid"]), float(cash["ask"])
            perp_bid, perp_ask = float(perp["bid"]), float(perp["ask"])
            multiplier = float(pair.get("perp_to_cash_multiplier", 1.0))
            normalized_perp_bid = perp_bid * multiplier
            normalized_perp_ask = perp_ask * multiplier
            cash_mid = (cash_bid + cash_ask) / 2.0
            perp_mid = (normalized_perp_bid + normalized_perp_ask) / 2.0
            cash_spread = _spread_bps(cash_bid, cash_ask)
            perp_spread = _spread_bps(normalized_perp_bid, normalized_perp_ask)
            assumed = bool(cash["receive_time_assumed"] or perp["receive_time_assumed"])
            clock_uncertainty = max(
                float(cash.get("clock_uncertainty_ms", 0.0) or 0.0),
                float(perp.get("clock_uncertainty_ms", 0.0) or 0.0),
            )
            basis = np.log(perp_mid / cash_mid)
            quality_ok = bool(
                cash["quote_valid"]
                and perp["quote_valid"]
                and cash_age >= -1.0
                and perp_age >= -1.0
                and cash_age <= float(settings["max_source_age_seconds"])
                and perp_age <= float(settings["max_source_age_seconds"])
                and cash_receive_lag <= float(settings["max_receive_lag_seconds"])
                and perp_receive_lag <= float(settings["max_receive_lag_seconds"])
                and cash_receive_lag >= -1.0
                and perp_receive_lag >= -1.0
                and skew <= float(settings["max_source_skew_seconds"])
                and clock_uncertainty <= float(settings["max_clock_uncertainty_ms"])
                and cash_spread <= float(settings["max_cash_spread_bps"])
                and perp_spread <= float(settings["max_perp_spread_bps"])
                and abs(basis) * 10_000.0 <= float(settings["max_abs_basis_bps"])
                and (bool(settings["allow_assumed_receive_time"]) or not assumed)
            )
            rows.append(
                {
                    "timestamp": decision_time,
                    "symbol": symbol,
                    "cash_timestamp": cash["timestamp"],
                    "perp_timestamp": perp["timestamp"],
                    "cash_received_at": cash["received_at"],
                    "perp_received_at": perp["received_at"],
                    "cash_bid": cash_bid,
                    "cash_ask": cash_ask,
                    "cash_mid": cash_mid,
                    "perp_bid": normalized_perp_bid,
                    "perp_ask": normalized_perp_ask,
                    "perp_mid": perp_mid,
                    "mark_price": float(perp["mark_price"]) * multiplier if pd.notna(perp["mark_price"]) else np.nan,
                    "index_price": float(perp["index_price"]) * multiplier if pd.notna(perp["index_price"]) else np.nan,
                    "funding_rate": perp["funding_rate"],
                    "funding_time": perp["funding_time"] if "funding_time" in perp else pd.NaT,
                    "basis": basis,
                    "cash_source": cash.get("source", "unknown"),
                    "perp_source": perp.get("source", "unknown"),
                    "clock_uncertainty_ms": clock_uncertainty,
                    "source_skew_seconds": skew,
                    "cash_age_seconds": cash_age,
                    "perp_age_seconds": perp_age,
                    "cash_receive_lag_seconds": cash_receive_lag,
                    "perp_receive_lag_seconds": perp_receive_lag,
                    "cash_spread_bps": cash_spread,
                    "perp_spread_bps": perp_spread,
                    "receive_time_assumed": assumed,
                    "quality_ok": quality_ok,
                }
            )

    if not rows:
        return pd.DataFrame()
    aligned = pd.DataFrame(rows).sort_values(["symbol", "timestamp"])
    sample_seconds = int(settings["execution_sample_seconds"])
    aligned["sample_bucket"] = aligned["timestamp"].dt.floor(f"{sample_seconds}s")
    aligned = (
        aligned.groupby(["symbol", "sample_bucket"], as_index=False, sort=True)
        .tail(1)
        .drop(columns="sample_bucket")
        .sort_values(["symbol", "timestamp"])
        .reset_index(drop=True)
    )
    return aligned


def quality_summary(quotes: pd.DataFrame, aligned: pd.DataFrame) -> pd.DataFrame:
    rows = []
    symbols = sorted(set(quotes["symbol"]).union(aligned.get("symbol", [])))
    for symbol in symbols:
        raw = quotes.loc[quotes["symbol"].eq(symbol)]
        sample = aligned.loc[aligned["symbol"].eq(symbol)] if not aligned.empty else aligned
        valid = sample.loc[sample["quality_ok"]] if not sample.empty else sample
        rows.append(
            {
                "symbol": symbol,
                "raw_rows": len(raw),
                "aligned_rows": len(sample),
                "quality_rows": len(valid),
                "quality_pass_rate": len(valid) / len(sample) if len(sample) else 0.0,
                "median_skew_seconds": sample["source_skew_seconds"].median() if len(sample) else np.nan,
                "max_skew_seconds": sample["source_skew_seconds"].max() if len(sample) else np.nan,
                "median_cash_age_seconds": sample["cash_age_seconds"].median() if len(sample) else np.nan,
                "median_perp_age_seconds": sample["perp_age_seconds"].median() if len(sample) else np.nan,
                "median_clock_uncertainty_ms": sample["clock_uncertainty_ms"].median() if len(sample) else np.nan,
                "assumed_receive_rows": int(sample["receive_time_assumed"].sum()) if len(sample) else 0,
                "start": sample["timestamp"].min() if len(sample) else pd.NaT,
                "end": sample["timestamp"].max() if len(sample) else pd.NaT,
            }
        )
    return pd.DataFrame(rows)
