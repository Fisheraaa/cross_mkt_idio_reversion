from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.yaml"
LOCAL_ENV_PATH = ROOT / ".env"
ALLOWED_LOCAL_ENV_KEYS = {"APCA_API_KEY_ID", "APCA_API_SECRET_KEY"}


def load_local_env(path: str | Path | None = None) -> list[str]:
    """Load approved credentials without overriding existing environment values."""
    env_path = Path(path) if path else LOCAL_ENV_PATH
    if not env_path.exists():
        return []

    loaded: list[str] = []
    with env_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ValueError(f"Invalid .env line {line_number}: expected KEY=VALUE")
            key, value = (part.strip() for part in line.split("=", 1))
            if key not in ALLOWED_LOCAL_ENV_KEYS:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            if not value or value in {"你的Key", "你的Secret", "your_key", "your_secret"}:
                continue
            if key not in os.environ:
                os.environ[key] = value
                loaded.append(key)
    return loaded


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path).resolve() if path else DEFAULT_CONFIG
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["_config_path"] = str(config_path)
    cfg["_root"] = str(ROOT)
    return cfg


def with_cost_multiplier(cfg: dict[str, Any], multiplier: float) -> dict[str, Any]:
    revised = deepcopy(cfg)
    execution = revised["execution"]
    for key in (
        "cash_taker_fee_bps",
        "perp_taker_fee_bps",
        "cash_extra_slippage_bps",
        "perp_extra_slippage_bps",
        "impact_bps_per_fill",
    ):
        execution[key] = float(execution[key]) * multiplier
    return revised


def resolve_output_path(cfg: dict[str, Any], section: str, filename: str) -> Path:
    root = Path(cfg["_root"])
    path = root / cfg["output"][section] / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
