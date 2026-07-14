from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any

import requests


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp_from_ms(value: Any, fallback: datetime) -> datetime:
    try:
        if value is not None and float(value) > 0:
            return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc)
    except (TypeError, ValueError, OverflowError):
        pass
    return fallback


class HttpProvider:
    def __init__(self, timeout: float, session: requests.Session | None = None) -> None:
        self.timeout = timeout
        self.session = session or requests.Session()

    def _get(self, url: str, **kwargs: Any) -> tuple[Any, datetime, float]:
        last_error: Exception | None = None
        for attempt in range(4):
            started = _utc_now()
            try:
                response = self.session.get(url, timeout=self.timeout, **kwargs)
                received = _utc_now()
                response.raise_for_status()
                round_trip_ms = (received - started).total_seconds() * 1000.0
                return response.json(), received, round_trip_ms
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt == 3:
                    break
                response = getattr(exc, "response", None)
                retry_after = response.headers.get("Retry-After") if response is not None else None
                if response is not None and response.status_code == 429:
                    try:
                        delay = max(float(retry_after or 0.0), 5.0 * (2**attempt))
                    except ValueError:
                        delay = 5.0 * (2**attempt)
                else:
                    delay = 0.5 * (2**attempt)
                time.sleep(min(delay, 60.0))
        raise RuntimeError(f"Request failed for {url}: {last_error}")


class AlpacaLatestQuotes(HttpProvider):
    def __init__(self, cfg: dict[str, Any]) -> None:
        collector = cfg["collector"]
        super().__init__(float(collector["request_timeout_seconds"]))
        self.base_url = str(collector["alpaca_base_url"]).rstrip("/")
        self.feed = str(collector.get("alpaca_feed", "iex"))
        key = os.environ.get("APCA_API_KEY_ID")
        secret = os.environ.get("APCA_API_SECRET_KEY")
        if not key or not secret:
            raise RuntimeError(
                "Set APCA_API_KEY_ID and APCA_API_SECRET_KEY before collect-live"
            )
        self.session.headers.update(
            {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        )

    def fetch(self, mappings: dict[str, dict[str, Any]], snapshot_id: str) -> list[dict]:
        reverse = {spec["cash_symbol"]: canonical for canonical, spec in mappings.items()}
        payload, received, round_trip_ms = self._get(
            f"{self.base_url}/v2/stocks/quotes/latest",
            params={"symbols": ",".join(reverse), "feed": self.feed},
        )
        quotes = payload.get("quotes", payload)
        records = []
        for provider_symbol, quote in quotes.items():
            canonical = reverse.get(provider_symbol)
            if canonical is None:
                continue
            records.append(
                {
                    "snapshot_id": snapshot_id,
                    "timestamp": quote.get("t"),
                    "received_at": received.isoformat(),
                    "symbol": canonical,
                    "provider_symbol": provider_symbol,
                    "venue": "cash",
                    "bid": quote.get("bp"),
                    "ask": quote.get("ap"),
                    "bid_size": quote.get("bs"),
                    "ask_size": quote.get("as"),
                    "source": f"alpaca_{self.feed}",
                    "clock_uncertainty_ms": round_trip_ms / 2.0,
                    "receive_time_assumed": False,
                }
            )
        missing = sorted(set(reverse).difference(item["provider_symbol"] for item in records))
        if missing:
            raise RuntimeError(
                "Alpaca did not return configured cash quotes: " + ", ".join(missing)
            )
        return records


class BinanceFuturesSnapshot(HttpProvider):
    def __init__(self, cfg: dict[str, Any]) -> None:
        collector = cfg["collector"]
        super().__init__(float(collector["request_timeout_seconds"]))
        self.base_url = str(collector["binance_base_url"]).rstrip("/")

    def fetch(self, mappings: dict[str, dict[str, Any]], snapshot_id: str) -> list[dict]:
        reverse = {spec["perp_symbol"]: canonical for canonical, spec in mappings.items()}
        book_payload, book_received, book_rtt = self._get(
            f"{self.base_url}/fapi/v1/ticker/bookTicker"
        )
        premium_payload, premium_received, premium_rtt = self._get(
            f"{self.base_url}/fapi/v1/premiumIndex"
        )
        books = book_payload if isinstance(book_payload, list) else [book_payload]
        premiums = premium_payload if isinstance(premium_payload, list) else [premium_payload]
        premium_by_symbol = {item.get("symbol"): item for item in premiums}
        records = []
        for book in books:
            provider_symbol = book.get("symbol")
            canonical = reverse.get(provider_symbol)
            if canonical is None:
                continue
            premium = premium_by_symbol.get(provider_symbol, {})
            received = max(book_received, premium_received)
            event_value = book.get("time", book.get("E", book.get("T")))
            event_time = _timestamp_from_ms(event_value, book_received)
            records.append(
                {
                    "snapshot_id": snapshot_id,
                    "timestamp": event_time.isoformat(),
                    "received_at": received.isoformat(),
                    "symbol": canonical,
                    "provider_symbol": provider_symbol,
                    "venue": "perp",
                    "bid": book.get("bidPrice"),
                    "ask": book.get("askPrice"),
                    "bid_size": book.get("bidQty"),
                    "ask_size": book.get("askQty"),
                    "mark_price": premium.get("markPrice"),
                    "index_price": premium.get("indexPrice"),
                    "funding_rate": premium.get("lastFundingRate"),
                    "funding_time": _timestamp_from_ms(
                        premium.get("nextFundingTime"), received
                    ).isoformat(),
                    "source": "binance_usdm_futures",
                    "clock_uncertainty_ms": max(book_rtt, premium_rtt) / 2.0,
                    "receive_time_assumed": event_value is None,
                }
            )
        missing = sorted(set(reverse).difference(item.get("provider_symbol") for item in records))
        if missing:
            raise RuntimeError(
                "Binance did not return configured contracts: " + ", ".join(missing)
            )
        return records
