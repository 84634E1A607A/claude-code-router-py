"""Configuration loading and validation."""

import json
import os
import re
from typing import Any


def _interpolate_env_vars(obj: Any) -> Any:
    """Recursively replace $VAR or ${VAR} with environment variable values."""
    if isinstance(obj, str):
        def replace(match: re.Match) -> str:
            name = match.group(1) or match.group(2)
            return os.environ.get(name, match.group(0))
        return re.sub(r"\$\{(\w+)\}|\$(\w+)", replace, obj)
    if isinstance(obj, dict):
        return {k: _interpolate_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate_env_vars(item) for item in obj]
    return obj


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return _interpolate_env_vars(raw)


def get_provider(config: dict, name: str) -> dict | None:
    for p in config.get("Providers", []):
        if p["name"] == name:
            return p
    return None


def resolve_route(config: dict, scenario: str = "default") -> tuple[str, str] | None:
    """Return (provider_name, model) for the given routing scenario."""
    router = config.get("Router", {})
    target = router.get(scenario) or router.get("default")
    if not target:
        return None
    parts = target.split(",", 1)
    if len(parts) != 2:
        return None
    return parts[0].strip(), parts[1].strip()
