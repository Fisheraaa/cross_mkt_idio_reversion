from __future__ import annotations

from typing import Any

import pandas as pd


VALID_SESSIONS = {"us_rth", "continuous"}
REQUIRED_CATEGORIES = {"crypto_equity", "crypto_asset", "tech_equity"}


def factor_specs(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for canonical, raw in cfg["factor_strategy"]["factors"].items():
        spec = dict(raw) if isinstance(raw, dict) else {"perp_symbol": str(raw)}
        spec["perp_symbol"] = str(spec["perp_symbol"])
        specs[str(canonical)] = spec
    return specs


def target_specs(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    settings = cfg["factor_strategy"]
    configured = settings.get("targets")
    if configured:
        return {str(symbol): dict(spec) for symbol, spec in configured.items()}

    factors = list(factor_specs(cfg))
    return {
        str(symbol): {
            "perp_symbol": str(spec["perp_symbol"]),
            "category": "tech_equity",
            "session": "us_rth",
            "factors": factors,
            "cost_profile": "tradfi_large",
            "logic": "legacy factor configuration",
            "invalidation": "legacy configuration only",
        }
        for symbol, spec in cfg["symbols"].items()
    }


def instrument_specs(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    instruments = factor_specs(cfg)
    for canonical, target in target_specs(cfg).items():
        if canonical in instruments:
            if instruments[canonical]["perp_symbol"] != str(target["perp_symbol"]):
                raise ValueError(f"Conflicting provider symbols for {canonical}")
            merged = dict(target)
            merged.update(instruments[canonical])
            instruments[canonical] = merged
        else:
            instruments[canonical] = dict(target)
    return instruments


def research_target_specs(
    cfg: dict[str, Any],
    symbols: list[str] | None = None,
    categories: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return the predeclared target subset requested for one research run."""
    targets = target_specs(cfg)
    symbol_filter = {str(symbol) for symbol in symbols or []}
    category_filter = {str(category) for category in categories or []}
    unknown = sorted(symbol_filter.difference(targets))
    if unknown:
        raise ValueError(f"Unknown factor targets: {', '.join(unknown)}")
    return {
        symbol: spec
        for symbol, spec in targets.items()
        if (not symbol_filter or symbol in symbol_filter)
        and (not category_filter or str(spec.get("category")) in category_filter)
    }


def instrument_specs_for_targets(
    cfg: dict[str, Any], targets: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Return targets and every factor leg required to hedge them."""
    all_instruments = instrument_specs(cfg)
    required = set(targets)
    for spec in targets.values():
        required.update(str(factor) for factor in spec.get("factors", []))
    return {
        canonical: all_instruments[canonical]
        for canonical in all_instruments
        if canonical in required
    }


def instrument_history_start(
    spec: dict[str, Any], requested_start: pd.Timestamp
) -> pd.Timestamp:
    """Apply an optional declared listing bound without inventing one."""
    available_since = spec.get("available_since")
    if not available_since:
        return requested_start
    declared = pd.Timestamp(available_since)
    declared = declared.tz_localize("UTC") if declared.tzinfo is None else declared.tz_convert("UTC")
    return max(requested_start, declared)


def session_parameters(cfg: dict[str, Any], session: str) -> dict[str, Any]:
    settings = cfg["factor_strategy"]
    parameters = dict(settings.get("session_parameters", {}).get(session, {}))
    for key in (
        "regression_lookback_hours",
        "regression_min_hours",
        "residual_lookback_hours",
        "residual_min_hours",
    ):
        if key not in parameters and key in settings:
            parameters[key] = settings[key]
    missing = [key for key in ("regression_lookback_hours", "regression_min_hours") if key not in parameters]
    if missing:
        raise ValueError(f"Missing {session} parameters: {', '.join(missing)}")
    parameters.setdefault("residual_lookback_hours", parameters["regression_lookback_hours"])
    parameters.setdefault("residual_min_hours", parameters["regression_min_hours"])
    return parameters


def validate_factor_configuration(cfg: dict[str, Any]) -> None:
    settings = cfg["factor_strategy"]
    factors = factor_specs(cfg)
    targets = target_specs(cfg)
    profiles = settings.get("cost_profiles", {})
    minimum = int(settings.get("target_count_minimum", 1))
    if not 0 < float(settings["entry_z"]) < float(settings["stop_z"]):
        raise ValueError("factor_strategy requires 0 < entry_z < stop_z")
    interval_minutes = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30}
    execution_interval = str(settings.get("execution_interval", "1m"))
    if execution_interval not in interval_minutes:
        raise ValueError(f"Unsupported execution_interval: {execution_interval}")
    for delay_key in ("execution_delay_minutes", "stress_delay_minutes"):
        if int(settings[delay_key]) % interval_minutes[execution_interval] != 0:
            raise ValueError(f"{delay_key} must align to {execution_interval} bar opens")
    if len(targets) < minimum:
        raise ValueError(f"Factor universe has {len(targets)} targets; at least {minimum} are required")

    categories = {str(spec.get("category", "")) for spec in targets.values()}
    missing_categories = sorted(REQUIRED_CATEGORIES.difference(categories))
    if minimum >= 25 and missing_categories:
        raise ValueError("Factor universe is missing categories: " + ", ".join(missing_categories))

    provider_to_canonical: dict[str, str] = {}
    for canonical, spec in instrument_specs(cfg).items():
        provider_symbol = str(spec.get("perp_symbol", ""))
        if not provider_symbol:
            raise ValueError(f"Missing perp_symbol for {canonical}")
        previous = provider_to_canonical.get(provider_symbol)
        if previous is not None and previous != canonical:
            raise ValueError(
                f"Provider symbol {provider_symbol} is assigned to both {previous} and {canonical}"
            )
        provider_to_canonical[provider_symbol] = canonical
        profile = spec.get("cost_profile")
        if profile and profile not in profiles:
            raise ValueError(f"Unknown cost profile {profile} for {canonical}")
        if spec.get("available_since"):
            try:
                pd.Timestamp(spec["available_since"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid available_since for {canonical}") from exc

    for target, spec in targets.items():
        session = str(spec.get("session", ""))
        if session not in VALID_SESSIONS:
            raise ValueError(f"Unknown session {session!r} for {target}")
        parameters = session_parameters(cfg, session)
        if int(parameters["regression_min_hours"]) > int(parameters["regression_lookback_hours"]):
            raise ValueError(f"Regression minimum exceeds lookback for {session}")
        if int(parameters["residual_min_hours"]) > int(parameters["residual_lookback_hours"]):
            raise ValueError(f"Residual minimum exceeds lookback for {session}")
        target_factors = [str(value) for value in spec.get("factors", [])]
        if not target_factors:
            raise ValueError(f"No factors configured for {target}")
        if target in target_factors:
            raise ValueError(f"Target {target} cannot hedge itself")
        missing = sorted(set(target_factors).difference(factors))
        if missing:
            raise ValueError(f"Unknown factors for {target}: {', '.join(missing)}")
        if len(target_factors) > 4:
            raise ValueError(f"Target {target} has too many factors for constrained subset Ridge")


def instrument_cost_profile(
    cfg: dict[str, Any], canonical: str
) -> dict[str, float]:
    settings = cfg["factor_strategy"]
    spec = instrument_specs(cfg)[canonical]
    profile_name = str(spec.get("cost_profile", "tradfi_large"))
    profile = settings.get("cost_profiles", {}).get(profile_name)
    if profile is None:
        # Backward-compatible path for small unit-test configurations.
        return {
            "roundtrip_spread_bps": float(
                settings.get("modeled_perp_roundtrip_spread_bps", 6.0)
            ),
            "extra_slippage_per_fill_bps": float(
                cfg["execution"].get("perp_extra_slippage_bps", 1.0)
            ),
        }
    return {
        "roundtrip_spread_bps": float(profile["roundtrip_spread_bps"]),
        "extra_slippage_per_fill_bps": float(profile["extra_slippage_per_fill_bps"]),
    }
