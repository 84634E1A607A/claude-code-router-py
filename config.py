import json
import os
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


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
    return validate_config(_interpolate_env_vars(raw))


def resolve_tokenizer_path(*candidates: str | None) -> str | None:
    """Return the first non-empty tokenizer path from explicit values or env."""
    for value in candidates:
        if isinstance(value, str):
            value = value.strip()
            if value:
                return value
    return None


def build_inline_config(
    *,
    api_base_url: str,
    api_key: str = "",
    model: str = "/model",
    max_retries: int = 3,
    port: int | None = None,
    api_timeout_ms: int = 120_000,
    tokenizer_path: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    budget_tokens: int | None = None,
    dp_routing_enabled: bool = False,
    dp_server_info_ttl_sec: int = 30,
    dp_sticky_mode: str | None = None,
    dp_session_ttl_sec: float | None = None,
) -> dict:
    """Build a minimal validated config from inline flags or environment values."""
    params: dict[str, Any] = {}
    if temperature is not None:
        params["temperature"] = temperature
    if top_p is not None:
        params["top_p"] = top_p
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    if budget_tokens is not None:
        params["reasoning"] = {"budget_tokens": budget_tokens}

    resolved_tokenizer_path = resolve_tokenizer_path(
        tokenizer_path,
        os.environ.get("CCR_TOKENIZER_PATH"),
        os.environ.get("TOKENIZER_PATH"),
    )

    provider: dict[str, Any] = {
        "name": "default",
        "model": model,
        "api_base_url": api_base_url,
        "api_key": api_key or os.environ.get("API_KEY", ""),
        "max_retries": max_retries,
    }
    if params:
        provider["params"] = params
    if resolved_tokenizer_path:
        provider["tokenizer_path"] = resolved_tokenizer_path
    if dp_routing_enabled:
        dp_routing: dict[str, Any] = {
            "enabled": True,
            "server_info_ttl_sec": dp_server_info_ttl_sec,
        }
        if dp_sticky_mode:
            dp_routing["sticky_mode"] = dp_sticky_mode
        if dp_session_ttl_sec is not None:
            dp_routing["session_ttl_sec"] = dp_session_ttl_sec
        provider["dp_routing"] = dp_routing

    cfg: dict[str, Any] = {
        "API_TIMEOUT_MS": api_timeout_ms,
        "Providers": [provider],
        "Router": {"default": model},
    }
    if port is not None:
        cfg["PORT"] = port
    if resolved_tokenizer_path:
        cfg["tokenizer_path"] = resolved_tokenizer_path
    return cfg


def get_provider(config: dict, name: str) -> dict | None:
    for p in config.get("Providers", []):
        if p["name"] == name:
            return p
    return None


def get_providers_for_model(config: dict, model: str) -> list[dict]:
    return [
        provider
        for provider in config.get("Providers", [])
        if isinstance(provider, dict) and provider.get("model") == model
    ]


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    model: str
    api_base_url: str

    @field_validator("name", "model", "api_base_url")
    @classmethod
    def _non_empty(cls, value: str, info) -> str:
        value = value.strip()
        if not value:
            field_name = info.field_name
            if field_name == "model":
                raise ValueError("model is required and must be non-empty")
            raise ValueError(f"{field_name} is required")
        return value


class RouterConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    default: str

    @field_validator("*")
    @classmethod
    def _valid_target(cls, value: str, info) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{info.field_name} must be a non-empty model name")
        value = value.strip()
        if "," in value:
            raise ValueError(
                f"{info.field_name} uses deprecated 'provider,model' syntax; use only the model name"
            )
        return value


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    Providers: list[ProviderConfig] = Field(min_length=1)
    Router: RouterConfig

    @model_validator(mode="after")
    def _validate_cross_refs(self):
        provider_names: set[str] = set()
        provider_models: set[str] = set()
        for provider in self.Providers:
            if provider.name in provider_names:
                raise ValueError(f"Duplicate provider name '{provider.name}'")
            provider_names.add(provider.name)
            provider_models.add(provider.model)

        for scenario, target in self.Router.model_dump().items():
            if target not in provider_models:
                raise ValueError(f"Router.{scenario} model '{target}' has no matching provider")
        return self


def apply_provider_params(provider: dict, req: dict) -> dict:
    """Apply provider-level param defaults/overrides to an OpenAI-format request.

    Supported fields in provider["params"]:
      temperature  – default if not set in request
      top_p        – default if not set in request
      max_tokens   – default, or ceiling if request already has a value
      reasoning    – dict with budget_tokens; enables thinking if not already set
    """
    params = provider.get("params", {})
    if not params:
        return req

    for field in ("temperature", "top_p"):
        if params.get(field) is not None and req.get(field) is None:
            req[field] = params[field]

    if params.get("max_tokens") is not None:
        limit = params["max_tokens"]
        if req.get("max_tokens") is None:
            req["max_tokens"] = limit
        else:
            req["max_tokens"] = min(req["max_tokens"], limit)

    reasoning = params.get("reasoning")
    if reasoning and req.get("thinking") is None and not req.get("tools"):
        budget = reasoning.get("budget_tokens", 8000) if isinstance(reasoning, dict) else 8000
        req["thinking"] = {"type": "enabled", "budget_tokens": budget}

    return req


def resolve_route(config: dict, scenario: str = "default") -> str | None:
    """Return the routed model name for the given routing scenario."""
    router = config.get("Router", {})
    target = router.get(scenario) or router.get("default")
    if not target:
        return None
    if not isinstance(target, str):
        return None
    model = target.strip()
    if not model or "," in model:
        return None
    return model


def validate_config(config: dict) -> dict:
    if not config:
        return config
    try:
        parsed = ConfigModel.model_validate(config)
    except ValidationError as exc:
        first = exc.errors()[0]
        path = ".".join(str(part) for part in first.get("loc", ()))
        msg = first.get("msg", "invalid config")
        if first.get("type") == "missing" and path.endswith(".model"):
            msg = "model is required and must be non-empty"
        raise ValueError(f"{path}: {msg}" if path else msg) from exc
    return parsed.model_dump(mode="python")
