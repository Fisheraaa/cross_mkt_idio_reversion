from __future__ import annotations

import os
import time
from typing import Any

import pandas as pd

from providers import HttpProvider


MINUTE_MS = 60_000


def _require_alpaca_credentials() -> tuple[str, str]:
    key = os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError(
            "Set APCA_API_KEY_ID and APCA_API_SECRET_KEY before historical-prescreen"
        )
    return key, secret


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


class AlpacaHistoricalBars(HttpProvider):
    def __init__(self, cfg: dict[str, Any]) -> None:
        collector = cfg["collector"]
        historical = cfg["historical"]
        super().__init__(float(collector["request_timeout_seconds"]))
        self.base_url = str(collector["alpaca_base_url"]).rstrip("/")
        self.feed = str(historical.get("alpaca_feed", collector.get("alpaca_feed", "iex")))
        self.pause = float(historical.get("request_pause_seconds", 0.0))
        key, secret = _require_alpaca_credentials()
        self.session.headers.update(
            {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        )

    @staticmethod
    def parse_payload(payload: dict[str, Any]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for symbol, bars in payload.get("bars", {}).items():
            for bar in bars or []:
                rows.append(
                    {
                        "provider_symbol": symbol,
                        "timestamp": bar.get("t"),
                        "open": bar.get("o"),
                        "high": bar.get("h"),
                        "low": bar.get("l"),
                        "close": bar.get("c"),
                        "volume": bar.get("v"),
                        "trade_count": bar.get("n"),
                        "vwap": bar.get("vw"),
                    }
                )
        columns = [
            "provider_symbol", "timestamp", "open", "high", "low", "close",
            "volume", "trade_count", "vwap",
        ]
        if not rows:
            return pd.DataFrame(columns=columns)
        frame = pd.DataFrame(rows, columns=columns)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = _numeric(
            frame, ["open", "high", "low", "close", "volume", "trade_count", "vwap"]
        )
        return frame.dropna(subset=["timestamp", "open", "close"])

    def fetch(
        self,
        mappings: dict[str, dict[str, Any]],
        start: pd.Timestamp,
        end_exclusive: pd.Timestamp,
    ) -> pd.DataFrame:
        reverse = {str(spec["cash_symbol"]): canonical for canonical, spec in mappings.items()}
        token: str | None = None
        frames: list[pd.DataFrame] = []
        while True:
            params: dict[str, Any] = {
                "symbols": ",".join(reverse),
                "timeframe": "1Min",
                "start": start.isoformat(),
                "end": end_exclusive.isoformat(),
                "limit": 10_000,
                "adjustment": "raw",
                "feed": self.feed,
                "sort": "asc",
            }
            if token:
                params["page_token"] = token
            payload, _, _ = self._get(f"{self.base_url}/v2/stocks/bars", params=params)
            parsed = self.parse_payload(payload)
            if not parsed.empty:
                frames.append(parsed)
            token = payload.get("next_page_token")
            if not token:
                break
            if self.pause:
                time.sleep(self.pause)
        if not frames:
            return pd.DataFrame()
        frame = pd.concat(frames, ignore_index=True)
        frame["symbol"] = frame["provider_symbol"].map(reverse)
        return (
            frame.dropna(subset=["symbol"])
            .drop_duplicates(["symbol", "timestamp"], keep="last")
            .sort_values(["symbol", "timestamp"])
            .reset_index(drop=True)
        )


class BinanceHistoricalMinutes(HttpProvider):
    ENDPOINTS = {
        "perp": ("/fapi/v1/klines", "symbol"),
        "mark": ("/fapi/v1/markPriceKlines", "symbol"),
        "index": ("/fapi/v1/indexPriceKlines", "pair"),
    }

    def __init__(self, cfg: dict[str, Any]) -> None:
        collector = cfg["collector"]
        historical = cfg["historical"]
        super().__init__(float(collector["request_timeout_seconds"]))
        self.base_url = str(collector["binance_base_url"]).rstrip("/")
        self.pause = float(historical.get("request_pause_seconds", 0.0))

    @staticmethod
    def parse_klines(payload: list[list[Any]]) -> pd.DataFrame:
        columns = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]
        rows = [
            {
                "timestamp": item[0],
                "open": item[1],
                "high": item[2],
                "low": item[3],
                "close": item[4],
                "volume": item[5] if len(item) > 5 else None,
                "quote_volume": item[7] if len(item) > 7 else None,
            }
            for item in payload
            if len(item) >= 5
        ]
        if not rows:
            return pd.DataFrame(columns=columns)
        frame = pd.DataFrame(rows, columns=columns)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True, errors="coerce")
        frame = _numeric(frame, ["open", "high", "low", "close", "volume", "quote_volume"])
        return frame.dropna(subset=["timestamp", "open", "close"])

    def fetch_klines(
        self,
        provider_symbol: str,
        kind: str,
        start: pd.Timestamp,
        end_exclusive: pd.Timestamp,
        interval: str = "1m",
    ) -> pd.DataFrame:
        if kind not in self.ENDPOINTS:
            raise ValueError(f"Unsupported Binance kline kind: {kind}")
        path, symbol_parameter = self.ENDPOINTS[kind]
        cursor = int(start.timestamp() * 1000)
        end_ms = int(end_exclusive.timestamp() * 1000)
        frames: list[pd.DataFrame] = []
        while cursor < end_ms:
            payload, _, _ = self._get(
                f"{self.base_url}{path}",
                params={
                    symbol_parameter: provider_symbol,
                    "interval": interval,
                    "startTime": cursor,
                    "endTime": end_ms - 1,
                    "limit": 1500,
                },
            )
            parsed = self.parse_klines(payload)
            if parsed.empty:
                break
            frames.append(parsed)
            interval_ms = {
                "1m": MINUTE_MS,
                "3m": 3 * MINUTE_MS,
                "5m": 5 * MINUTE_MS,
                "15m": 15 * MINUTE_MS,
                "30m": 30 * MINUTE_MS,
                "1h": 60 * MINUTE_MS,
            }.get(interval)
            if interval_ms is None:
                raise ValueError(f"Unsupported Binance interval: {interval}")
            next_cursor = int(parsed["timestamp"].iloc[-1].timestamp() * 1000) + interval_ms
            if next_cursor <= cursor:
                raise RuntimeError(f"Binance pagination did not advance for {provider_symbol} {kind}")
            cursor = next_cursor
            if len(payload) < 1500:
                break
            if self.pause:
                time.sleep(self.pause)
        if not frames:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]
            )
        return (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates("timestamp", keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    @staticmethod
    def parse_funding(payload: list[dict[str, Any]]) -> pd.DataFrame:
        rows = [
            {
                "timestamp": item.get("fundingTime"),
                "funding_rate": item.get("fundingRate"),
            }
            for item in payload
        ]
        if not rows:
            return pd.DataFrame(columns=["timestamp", "funding_rate"])
        frame = pd.DataFrame(rows)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True, errors="coerce")
        frame["funding_rate"] = pd.to_numeric(frame["funding_rate"], errors="coerce")
        return frame.dropna(subset=["timestamp", "funding_rate"])

    def fetch_funding(
        self,
        provider_symbol: str,
        start: pd.Timestamp,
        end_exclusive: pd.Timestamp,
    ) -> pd.DataFrame:
        cursor = int(start.timestamp() * 1000)
        end_ms = int(end_exclusive.timestamp() * 1000)
        frames: list[pd.DataFrame] = []
        while cursor < end_ms:
            payload, _, _ = self._get(
                f"{self.base_url}/fapi/v1/fundingRate",
                params={
                    "symbol": provider_symbol,
                    "startTime": cursor,
                    "endTime": end_ms - 1,
                    "limit": 1000,
                },
            )
            parsed = self.parse_funding(payload)
            if parsed.empty:
                break
            frames.append(parsed)
            next_cursor = int(parsed["timestamp"].iloc[-1].timestamp() * 1000) + 1
            if next_cursor <= cursor:
                raise RuntimeError(f"Binance funding pagination did not advance for {provider_symbol}")
            cursor = next_cursor
            if len(payload) < 1000:
                break
            if self.pause:
                time.sleep(self.pause)
        if not frames:
            return pd.DataFrame(columns=["timestamp", "funding_rate"])
        return (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates("timestamp", keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
