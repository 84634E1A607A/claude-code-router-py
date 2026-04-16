import json
import os
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, RootModel, ValidationError, field_validator, model_validator


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


class ReasoningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    budget_tokens: int = 8000


class ProviderParamsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    reasoning: ReasoningConfig | None = None


class DPRoutingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    server_info_ttl_sec: int = 30
    sticky_mode: str = "session_system"
    session_ttl_sec: float = 10800.0

    @field_validator("sticky_mode")
    @classmethod
    def _validate_sticky_mode(cls, value: str) -> str:
        value = value.strip()
        if value != "session_system":
            raise ValueError("sticky_mode must be 'session_system'")
        return value


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    model: str
    api_base_url: str
    api_key: str = ""
    max_retries: int = 3
    tokenizer_path: str | None = None
    params: ProviderParamsConfig | None = None
    dp_routing: DPRoutingConfig | None = None

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


class RouterConfig(RootModel[dict[str, str]]):
    @field_validator("root")
    @classmethod
    def _valid_routes(cls, routes: dict[str, str]) -> dict[str, str]:
        if "default" not in routes:
            raise ValueError("default is required")
        for field_name, value in routes.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty model name")
            value = value.strip()
            if "," in value:
                raise ValueError(
                    f"{field_name} uses deprecated 'provider,model' syntax; use only the model name"
                )
            routes[field_name] = value
        return routes

    def get(self, key: str, default: str | None = None) -> str | None:
        return self.root.get(key, default)

    def items(self):
        return self.root.items()


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    PORT: int = 3456
    API_TIMEOUT_MS: int = 600_000
    tokenizer_path: str | None = None
    Providers: list[ProviderConfig] = Field(min_length=1)
    Router: RouterConfig

    @field_validator("tokenizer_path")
    @classmethod
    def _normalize_optional_string(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @model_validator(mode="after")
    def _validate_cross_refs(self):
        provider_names: set[str] = set()
        provider_models: set[str] = set()
        for provider in self.Providers:
            if provider.name in provider_names:
                raise ValueError(f"Duplicate provider name '{provider.name}'")
            provider_names.add(provider.name)
            provider_models.add(provider.model)

        for scenario, target in self.Router.items():
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
