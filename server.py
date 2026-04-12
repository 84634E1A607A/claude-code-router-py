"""
FastAPI server — accepts Anthropic /v1/messages requests and forwards them
to an OpenAI-compatible provider, converting formats in both directions.
"""

import json
import hashlib
import inspect
import logging
import os
import re
import threading
import time
import uuid
import zlib
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from batch import anthropic_batch_to_openai_jsonl, openai_batch_to_anthropic, openai_results_line_to_anthropic
from client import ProviderError, ProviderStream, close_shared_client, get_shared_client, open_provider_stream, post_json
from config import apply_provider_params, get_provider, load_config, resolve_route, validate_config
from converter import anthropic_to_openai, openai_to_anthropic, stream_openai_to_anthropic
from debug import check_and_save_nonstreaming, check_and_save_streaming, log_openai_request

logger = logging.getLogger(__name__)

# Populated either by set_config() (single-process) or lifespan (multi-worker).
_config: dict = {}
_dp_size_cache: dict[str, dict[str, float | int]] = {}
_providers_by_model: dict[str, list[dict]] = {}
_available_models: tuple[str, ...] = ()

_CLAUDE_SESSION_HEADER = "X-Claude-Code-Session-Id"
_DP_OVERRIDE_HEADER = "X-Routed-DP-Rank"
_DP_SERVER_INFO_TTL_SEC = 30
_INVALID_DP_RANK_RE = re.compile(r"routed_dp_rank.*out of range", re.IGNORECASE | re.DOTALL)
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class DPRoutingDecision:
    provider_key: str | None = None
    provider_name: str | None = None
    dp_size: int | None = None
    rank: int | None = None
    provider_dp_rank: int | None = None
    source: str | None = None
    sticky_key: str | None = None
    session_id: str | None = None
    subagent_id: str | None = None

    def response_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.rank is not None:
            headers["X-Router-DP-Rank"] = str(self.rank)
        if self.provider_name:
            headers["X-Router-Provider"] = self.provider_name
        if self.provider_dp_rank is not None:
            headers["X-Router-Provider-DP-Rank"] = str(self.provider_dp_rank)
        if self.source in {"session", "session_system"} and self.sticky_key:
            headers["X-Router-Sticky-Key"] = self.sticky_key
        return headers


@dataclass
class StickySlotAllocator:
    """Manages sticky session-to-slot mapping with round-robin allocation.

    Key properties:
    - Sessions are sticky forever (once assigned, never reassigned)
    - Multiple sessions can share the same slot
    - New sessions get the slot with longest idle time
    - Sessions evicted after TTL of inactivity
    """
    slots: list[str]
    ttl_sec: float = 10800.0  # 3 hours default
    sessions: dict[str, str] = field(default_factory=dict)  # sticky_key -> slot_id
    slot_last_used: dict[str, float] = field(default_factory=dict)  # slot_id -> timestamp
    session_activity: dict[str, float] = field(default_factory=dict)  # sticky_key -> timestamp

    def assign(self, sticky_key: str) -> tuple[str, bool]:
        """Assign a slot to a sticky key.

        Returns (slot_id, is_new) where is_new is True if this was a new assignment.
        Performs lazy cleanup of expired sessions before assigning a new one.
        """
        now = time.monotonic()
        valid_slots = set(self.slots)

        current_slot = self.sessions.get(sticky_key)
        if current_slot in valid_slots:
            self.session_activity[sticky_key] = now
            return current_slot, False

        if current_slot is not None:
            self.sessions.pop(sticky_key, None)
            self.session_activity.pop(sticky_key, None)

        evicted = self._cleanup(now)
        if evicted > 0:
            logger.debug("DP lazy cleanup: evicted=%d sessions remaining=%d", evicted, len(self.sessions))

        slot_id = self._select_slot()

        self.sessions[sticky_key] = slot_id
        self.slot_last_used[slot_id] = now
        self.session_activity[sticky_key] = now

        return slot_id, True

    def _cleanup(self, now: float) -> int:
        """Evict sessions inactive for > TTL. Returns count of evicted sessions."""
        to_evict = [
            sid for sid, last_active in self.session_activity.items()
            if now - last_active > self.ttl_sec
        ]

        for sid in to_evict:
            self.sessions.pop(sid, None)
            self.session_activity.pop(sid, None)
        self.slot_last_used = {
            slot_id: last_used
            for slot_id, last_used in self.slot_last_used.items()
            if slot_id in self.slots
        }

        return len(to_evict)

    def _select_slot(self) -> str:
        """Select the best slot for a new sticky key."""
        used_slots = set(self.slot_last_used.keys())
        if len(used_slots) < len(self.slots):
            for slot_id in self.slots:
                if slot_id not in used_slots:
                    return slot_id

        oldest_time = None
        oldest_slot = self.slots[0]
        for slot_id in self.slots:
            last_used = self.slot_last_used.get(slot_id, 0.0)
            if oldest_time is None or last_used < oldest_time:
                oldest_time = last_used
                oldest_slot = slot_id
        return oldest_slot

    def cleanup(self) -> int:
        """Public cleanup method for external use. Evicts expired sessions."""
        return self._cleanup(time.monotonic())

    def stats(self) -> dict:
        """Return allocation statistics for logging/debugging."""
        sessions_per_slot: dict[str, int] = {}
        for slot_id in self.sessions.values():
            sessions_per_slot[slot_id] = sessions_per_slot.get(slot_id, 0) + 1
        return {
            "slot_count": len(self.slots),
            "total_sessions": len(self.sessions),
            "slots_in_use": len(sessions_per_slot),
            "sessions_per_slot": sessions_per_slot,
        }


@dataclass
class DPAllocator:
    """Backward-compatible wrapper for integer DP ranks."""
    dp_size: int
    ttl_sec: float = 10800.0
    _allocator: StickySlotAllocator = field(init=False)

    def __post_init__(self) -> None:
        self._allocator = StickySlotAllocator(
            slots=[str(rank) for rank in range(self.dp_size)],
            ttl_sec=self.ttl_sec,
        )

    @property
    def sessions(self) -> dict[str, int]:
        return {sid: int(rank) for sid, rank in self._allocator.sessions.items()}

    def assign(self, session_id: str) -> tuple[int, bool]:
        slot_id, is_new = self._allocator.assign(session_id)
        return int(slot_id), is_new

    def cleanup(self) -> int:
        return self._allocator.cleanup()

    def stats(self) -> dict:
        base = self._allocator.stats()
        return {
            "dp_size": self.dp_size,
            "total_sessions": base["total_sessions"],
            "ranks_in_use": base["slots_in_use"],
            "sessions_per_rank": {
                int(rank): count for rank, count in base["sessions_per_slot"].items()
            },
        }


# Global allocator state per provider
_dp_allocators: dict[str, DPAllocator] = {}
_model_allocators: dict[str, StickySlotAllocator] = {}


@dataclass
class RequestMetricsContext:
    provider_key: str
    provider_name: str
    dp_rank: int | None
    started_at: float
    is_stream: bool


class RuntimeMetrics:
    """Tracks per-worker runtime metrics for requests, tokens, and DP usage."""

    def __init__(self, throughput_window_sec: float = 60.0):
        self.throughput_window_sec = throughput_window_sec
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.started_at = time.monotonic()
            self.requests_started = 0
            self.requests_completed = 0
            self.requests_failed = 0
            self.streaming_requests_started = 0
            self.non_streaming_requests_started = 0
            self.active_requests = 0
            self.input_tokens = 0
            self.output_tokens = 0
            self.total_latency_sec = 0.0
            self._provider_stats: dict[str, dict] = {}
            self._recent_completions: deque[dict] = deque()

    def _ensure_provider_stats(self, provider_key: str, provider_name: str) -> dict:
        stats = self._provider_stats.get(provider_key)
        if stats is None:
            stats = {
                "provider_name": provider_name,
                "requests_started": 0,
                "requests_completed": 0,
                "requests_failed": 0,
                "streaming_requests_started": 0,
                "non_streaming_requests_started": 0,
                "active_requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_latency_sec": 0.0,
                "per_dp": {},
            }
            self._provider_stats[provider_key] = stats
        return stats

    @staticmethod
    def _ensure_dp_stats(provider_stats: dict, dp_rank: int) -> dict:
        per_dp = provider_stats["per_dp"]
        dp_stats = per_dp.get(dp_rank)
        if dp_stats is None:
            dp_stats = {
                "requests_started": 0,
                "requests_completed": 0,
                "requests_failed": 0,
                "streaming_requests_started": 0,
                "non_streaming_requests_started": 0,
                "active_requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_latency_sec": 0.0,
            }
            per_dp[dp_rank] = dp_stats
        return dp_stats

    def start_request(
        self,
        provider_key: str,
        provider_name: str,
        dp_rank: int | None,
        is_stream: bool,
    ) -> RequestMetricsContext:
        started_at = time.monotonic()
        with self._lock:
            self.requests_started += 1
            self.active_requests += 1
            if is_stream:
                self.streaming_requests_started += 1
            else:
                self.non_streaming_requests_started += 1

            provider_stats = self._ensure_provider_stats(provider_key, provider_name)
            provider_stats["requests_started"] += 1
            provider_stats["active_requests"] += 1
            if is_stream:
                provider_stats["streaming_requests_started"] += 1
            else:
                provider_stats["non_streaming_requests_started"] += 1

            if dp_rank is not None:
                dp_stats = self._ensure_dp_stats(provider_stats, dp_rank)
                dp_stats["requests_started"] += 1
                dp_stats["active_requests"] += 1
                if is_stream:
                    dp_stats["streaming_requests_started"] += 1
                else:
                    dp_stats["non_streaming_requests_started"] += 1

        return RequestMetricsContext(
            provider_key=provider_key,
            provider_name=provider_name,
            dp_rank=dp_rank,
            started_at=started_at,
            is_stream=is_stream,
        )

    def finish_request(
        self,
        ctx: RequestMetricsContext,
        *,
        success: bool,
        usage: dict | None = None,
    ) -> None:
        finished_at = time.monotonic()
        elapsed = max(finished_at - ctx.started_at, 0.0)
        input_tokens = int((usage or {}).get("input_tokens", 0) or 0)
        output_tokens = int((usage or {}).get("output_tokens", 0) or 0)
        total_tokens = input_tokens + output_tokens

        with self._lock:
            self.active_requests = max(self.active_requests - 1, 0)
            provider_stats = self._ensure_provider_stats(ctx.provider_key, ctx.provider_name)
            provider_stats["active_requests"] = max(provider_stats["active_requests"] - 1, 0)

            dp_stats = None
            if ctx.dp_rank is not None:
                dp_stats = self._ensure_dp_stats(provider_stats, ctx.dp_rank)
                dp_stats["active_requests"] = max(dp_stats["active_requests"] - 1, 0)

            if success:
                self.requests_completed += 1
                self.input_tokens += input_tokens
                self.output_tokens += output_tokens
                self.total_latency_sec += elapsed
                provider_stats["requests_completed"] += 1
                provider_stats["input_tokens"] += input_tokens
                provider_stats["output_tokens"] += output_tokens
                provider_stats["total_latency_sec"] += elapsed
                if dp_stats is not None:
                    dp_stats["requests_completed"] += 1
                    dp_stats["input_tokens"] += input_tokens
                    dp_stats["output_tokens"] += output_tokens
                    dp_stats["total_latency_sec"] += elapsed
                self._recent_completions.append({
                    "finished_at": finished_at,
                    "provider_key": ctx.provider_key,
                    "dp_rank": ctx.dp_rank,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                })
                self._prune_locked(finished_at)
            else:
                self.requests_failed += 1
                provider_stats["requests_failed"] += 1
                if dp_stats is not None:
                    dp_stats["requests_failed"] += 1

    def _prune_locked(self, now: float) -> None:
        cutoff = now - self.throughput_window_sec
        while self._recent_completions and self._recent_completions[0]["finished_at"] < cutoff:
            self._recent_completions.popleft()

    def snapshot(self) -> dict:
        now = time.monotonic()
        with self._lock:
            self._prune_locked(now)
            throughput = self._throughput_from_events(self._recent_completions)
            providers: dict[str, dict] = {}
            for provider_key, stats in self._provider_stats.items():
                provider_events = [e for e in self._recent_completions if e["provider_key"] == provider_key]
                provider_snapshot = {
                    "provider_name": stats["provider_name"],
                    "requests_started": stats["requests_started"],
                    "requests_completed": stats["requests_completed"],
                    "requests_failed": stats["requests_failed"],
                    "streaming_requests_started": stats["streaming_requests_started"],
                    "non_streaming_requests_started": stats["non_streaming_requests_started"],
                    "active_requests": stats["active_requests"],
                    "input_tokens": stats["input_tokens"],
                    "output_tokens": stats["output_tokens"],
                    "total_tokens": stats["input_tokens"] + stats["output_tokens"],
                    "avg_latency_sec": (
                        stats["total_latency_sec"] / stats["requests_completed"]
                        if stats["requests_completed"] else 0.0
                    ),
                    "throughput": self._throughput_from_events(provider_events),
                    "per_dp": {},
                }
                for dp_rank, dp_stats in stats["per_dp"].items():
                    dp_events = [
                        e for e in provider_events
                        if e["dp_rank"] == dp_rank
                    ]
                    provider_snapshot["per_dp"][dp_rank] = {
                        "requests_started": dp_stats["requests_started"],
                        "requests_completed": dp_stats["requests_completed"],
                        "requests_failed": dp_stats["requests_failed"],
                        "streaming_requests_started": dp_stats["streaming_requests_started"],
                        "non_streaming_requests_started": dp_stats["non_streaming_requests_started"],
                        "active_requests": dp_stats["active_requests"],
                        "input_tokens": dp_stats["input_tokens"],
                        "output_tokens": dp_stats["output_tokens"],
                        "total_tokens": dp_stats["input_tokens"] + dp_stats["output_tokens"],
                        "avg_latency_sec": (
                            dp_stats["total_latency_sec"] / dp_stats["requests_completed"]
                            if dp_stats["requests_completed"] else 0.0
                        ),
                        "throughput": self._throughput_from_events(dp_events),
                    }
                providers[provider_key] = provider_snapshot

            return {
                "started_at_monotonic": self.started_at,
                "uptime_sec": max(now - self.started_at, 0.0),
                "throughput_window_sec": self.throughput_window_sec,
                "requests_started": self.requests_started,
                "requests_completed": self.requests_completed,
                "requests_failed": self.requests_failed,
                "streaming_requests_started": self.streaming_requests_started,
                "non_streaming_requests_started": self.non_streaming_requests_started,
                "active_requests": self.active_requests,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.input_tokens + self.output_tokens,
                "avg_latency_sec": (
                    self.total_latency_sec / self.requests_completed
                    if self.requests_completed else 0.0
                ),
                "throughput": throughput,
                "providers": providers,
            }

    def _throughput_from_events(self, events) -> dict:
        total_tokens = sum(int(e["total_tokens"]) for e in events)
        output_tokens = sum(int(e["output_tokens"]) for e in events)
        return {
            "requests_in_window": len(events),
            "total_tokens_per_sec": total_tokens / self.throughput_window_sec,
            "output_tokens_per_sec": output_tokens / self.throughput_window_sec,
        }


_runtime_metrics = RuntimeMetrics()


def _get_session_ttl(provider: dict) -> float:
    """Get session TTL from provider config, default 3 hours."""
    cfg = provider.get("dp_routing")
    if not isinstance(cfg, dict):
        return 10800.0
    ttl = cfg.get("session_ttl_sec", 10800.0)
    try:
        return max(float(ttl), 60.0)  # minimum 1 minute
    except (TypeError, ValueError):
        return 10800.0


def _get_dp_sticky_mode(provider: dict) -> str:
    """Get DP sticky-key mode from provider config, defaulting to session."""
    cfg = provider.get("dp_routing")
    if not isinstance(cfg, dict):
        return "session"
    mode = str(cfg.get("sticky_mode", "session")).strip().lower()
    if mode in {"session", "session_system"}:
        return mode
    return "session"


def _extract_anthropic_system_text(req: dict) -> str:
    """Extract text-only content from an Anthropic system prompt."""
    system = req.get("system")
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts: list[str] = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _normalize_system_prompt(text: str) -> str:
    """Normalize prompt text to avoid fragmentation on whitespace-only changes."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def _short_sticky_hash(text: str) -> str:
    """Return a short, lightweight hash for sticky routing identifiers."""
    return f"{zlib.adler32(text.encode('utf-8', 'surrogatepass')) & 0xffff:04x}"


def _extract_subagent_hash_input(req: dict) -> str | None:
    """Extract the content block used to derive a subagent identifier.

    This uses `messages[0].content[1].text` when present.
    """
    messages = req.get("messages")
    if not isinstance(messages, list) or not messages:
        return None

    first_message = messages[0]
    if not isinstance(first_message, dict):
        return None

    content = first_message.get("content")
    if not isinstance(content, list) or len(content) < 2:
        return None

    block = content[1]
    if not isinstance(block, dict):
        return None

    text = block.get("text")
    if not isinstance(text, str):
        return None

    return text


def _derive_sticky_key(session_id: str, sticky_mode: str, anthropic_req: dict) -> tuple[str, str, str | None]:
    """Build the sticky key used for sticky provider/DP routing.

    `session_system` mode prefers a hash of `messages[0].content[1]` when that
    block is present and otherwise falls back to a normalized system-prompt hash.
    """
    if sticky_mode != "session_system":
        return session_id, "session", None

    subagent_hash_input = _extract_subagent_hash_input(anthropic_req)
    if subagent_hash_input:
        subagent_id = _short_sticky_hash(subagent_hash_input)
        return f"{session_id}:{subagent_id}", "session_system", subagent_id

    system_text = _extract_anthropic_system_text(anthropic_req)
    if not system_text:
        return session_id, "session", None

    normalized_system = _normalize_system_prompt(system_text)
    if not normalized_system:
        return session_id, "session", None

    digest = _short_sticky_hash(normalized_system)
    return f"{session_id}:{digest}", "session_system", None


def _derive_dp_sticky_key(session_id: str, provider: dict, anthropic_req: dict) -> tuple[str, str, str | None]:
    """Backward-compatible wrapper around the generic sticky-key helper."""
    return _derive_sticky_key(session_id, _get_dp_sticky_mode(provider), anthropic_req)


def set_config(cfg: dict) -> None:
    global _config, _providers_by_model, _available_models
    _config = validate_config(cfg) if cfg else cfg
    _providers_by_model = {}
    _available_models = ()
    if _config:
        grouped: dict[str, list[dict]] = {}
        for provider in _config.get("Providers", []):
            if not isinstance(provider, dict):
                continue
            model = str(provider.get("model", "")).strip()
            if not model:
                continue
            grouped.setdefault(model, []).append(provider)
        _providers_by_model = grouped
        _available_models = tuple(sorted(grouped.keys()))
    _dp_size_cache.clear()
    _dp_allocators.clear()
    _model_allocators.clear()
    _runtime_metrics.reset()


def _build_config_from_env() -> dict | None:
    """Build config from individual CCR_* env vars (for gunicorn -e usage).
    Returns None if CCR_API_BASE_URL is not set."""
    api_base_url = os.environ.get("CCR_API_BASE_URL")
    if not api_base_url:
        return None

    params: dict = {}
    if (v := os.environ.get("CCR_TEMPERATURE")) is not None:
        params["temperature"] = float(v)
    if (v := os.environ.get("CCR_TOP_P")) is not None:
        params["top_p"] = float(v)
    if (v := os.environ.get("CCR_MAX_TOKENS")) is not None:
        params["max_tokens"] = int(v)
    if (v := os.environ.get("CCR_BUDGET_TOKENS")) is not None:
        params["reasoning"] = {"budget_tokens": int(v)}

    provider: dict = {
        "name": "default",
        "model": os.environ.get("CCR_MODEL", "/model"),
        "api_base_url": api_base_url,
        "api_key": os.environ.get("CCR_API_KEY") or os.environ.get("API_KEY", ""),
        "max_retries": int(os.environ.get("CCR_MAX_RETRIES", "3")),
    }
    if (tokenizer_path := os.environ.get("CCR_TOKENIZER_PATH") or os.environ.get("TOKENIZER_PATH")):
        provider["tokenizer_path"] = tokenizer_path
    if os.environ.get("CCR_DP_ROUTING_ENABLED", "").strip().lower() in ("1", "true", "yes"):
        dp_routing_cfg: dict = {
            "enabled": True,
            "server_info_ttl_sec": int(os.environ.get("CCR_DP_SERVER_INFO_TTL_SEC", str(_DP_SERVER_INFO_TTL_SEC))),
        }
        if (v := os.environ.get("CCR_DP_STICKY_MODE")) is not None:
            dp_routing_cfg["sticky_mode"] = v
        if (v := os.environ.get("CCR_DP_SESSION_TTL_SEC")) is not None:
            dp_routing_cfg["session_ttl_sec"] = float(v)
        provider["dp_routing"] = dp_routing_cfg
    if params:
        provider["params"] = params

    tokenizer_path = os.environ.get("CCR_TOKENIZER_PATH") or os.environ.get("TOKENIZER_PATH")
    if tokenizer_path:
        provider["tokenizer_path"] = tokenizer_path

    cfg: dict = {
        "API_TIMEOUT_MS": int(os.environ.get("CCR_API_TIMEOUT_MS", "850000")),
        "Providers": [provider],
        "Router": {"default": provider["model"]},
    }
    if tokenizer_path:
        cfg["tokenizer_path"] = tokenizer_path
    return cfg


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _config
    if not _config:
        if inline := os.environ.get("CCR_CONFIG_JSON"):
            import json as _json
            try:
                _config = validate_config(_json.loads(inline))
                logger.info("Config loaded from CCR_CONFIG_JSON (worker pid=%d)", os.getpid())
            except Exception as exc:
                logger.error("Failed to parse CCR_CONFIG_JSON: %s", exc)
        elif cfg := _build_config_from_env():
            _config = validate_config(cfg)
            logger.info("Config loaded from CCR_* env vars (worker pid=%d)", os.getpid())
        else:
            path = os.environ.get("CCR_CONFIG", "config.json")
            try:
                _config = load_config(path)
                logger.info("Config loaded from %s (worker pid=%d)", path, os.getpid())
            except Exception as exc:
                logger.error("Failed to load config %s: %s", path, exc)
    if not _config:
        logger.warning("No config loaded — routes are registered but all proxy endpoints will return 500")
    yield
    await close_shared_client()


app = FastAPI(title="Claude Code Router (Python)", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _timeout() -> float:
    """Read API_TIMEOUT_MS from config, handling string or numeric values."""
    return float(_config.get("API_TIMEOUT_MS", 600_000)) / 1000


def _provider_headers(provider: dict) -> dict:
    api_key = provider.get("api_key", "")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


@dataclass(frozen=True)
class ProviderTarget:
    provider: dict
    model: str
    url: str


@dataclass(frozen=True)
class RoutingSlot:
    flat_index: int
    slot_id: str
    provider: dict
    provider_key: str
    provider_name: str
    model: str
    provider_dp_rank: int | None
    dp_size: int | None


def _resolve_routed_model(scenario: str = "default") -> str:
    model = resolve_route(_config, scenario)
    if model is None:
        raise HTTPException(500, f"No route configured for scenario '{scenario}'")
    return model


def _resolve_model_alias(alias: str) -> str | None:
    router = _config.get("Router", {})
    target = router.get(alias)
    if not isinstance(target, str):
        return None
    target = target.strip()
    return target or None


def _providers_for_model(model: str) -> list[dict]:
    providers = _providers_by_model.get(model)
    if not providers:
        raise HTTPException(500, f"No providers configured for model '{model}'")
    return providers


def _resolve_request_model(requested_model: str | None, scenario: str = "default") -> str:
    requested = str(requested_model or "").strip()
    if not requested:
        return _resolve_routed_model(scenario)

    alias_target = _resolve_model_alias(requested)
    if alias_target is not None:
        return alias_target
    if requested in _providers_by_model:
        return requested
    available = ", ".join(_available_models) if _available_models else "<none>"
    raise HTTPException(400, f"Unknown model '{requested}'. Available models: {available}")


def _select_deterministic_provider(model: str) -> ProviderTarget:
    provider = _providers_for_model(model)[0]
    return ProviderTarget(provider=provider, model=model, url=provider["api_base_url"])


def _provider_pool_key(model: str) -> str:
    return f"model:{model}"


def _provider_slot_id(provider: dict, dp_rank: int | None) -> str:
    provider_key = _dp_cache_key(provider)
    if dp_rank is None:
        return provider_key
    return f"{provider_key}:{dp_rank}"


def _slot_belongs_to_provider(slot_id: str, provider_key: str) -> bool:
    return slot_id == provider_key or slot_id.startswith(provider_key + ":")


def _slot_rank_for_provider(slot_id: str, provider_key: str) -> int | None:
    if slot_id == provider_key:
        return None
    prefix = provider_key + ":"
    if not slot_id.startswith(prefix):
        return None
    suffix = slot_id[len(prefix):]
    try:
        return int(suffix)
    except ValueError:
        return None


def _slot_ttl_for_providers(providers: list[dict]) -> float:
    dp_enabled = [provider for provider in providers if _dp_routing_config(provider) is not None]
    if not dp_enabled:
        return 10800.0
    ttl = _get_session_ttl(dp_enabled[0])
    for provider in dp_enabled[1:]:
        if _get_session_ttl(provider) != ttl:
            raise HTTPException(500, "All DP-enabled providers for a model must share session_ttl_sec")
    return ttl


def _slot_sticky_mode_for_providers(providers: list[dict]) -> str:
    dp_enabled = [provider for provider in providers if _dp_routing_config(provider) is not None]
    if not dp_enabled:
        return "session"
    mode = _get_dp_sticky_mode(dp_enabled[0])
    for provider in dp_enabled[1:]:
        if _get_dp_sticky_mode(provider) != mode:
            raise HTTPException(500, "All DP-enabled providers for a model must share sticky_mode")
    return mode


async def _build_routing_slots(model: str, *, force_refresh: bool = False) -> list[RoutingSlot]:
    providers = _providers_for_model(model)
    slots: list[RoutingSlot] = []
    flat_index = 0
    for provider in providers:
        provider_key = _dp_cache_key(provider)
        provider_name = provider.get("name", "")
        synthetic_dp_rank = 0 if len(providers) == 1 else None
        dp_size = None
        if _dp_routing_config(provider) is not None:
            dp_size = await _get_provider_dp_size(provider, force_refresh=force_refresh)
            if dp_size is not None:
                _dp_size_cache[provider_key] = {"dp_size": dp_size, "fetched_at": time.monotonic()}
        if dp_size is not None and dp_size > 1:
            for provider_dp_rank in range(dp_size):
                slots.append(RoutingSlot(
                    flat_index=flat_index,
                    slot_id=_provider_slot_id(provider, provider_dp_rank),
                    provider=provider,
                    provider_key=provider_key,
                    provider_name=provider_name,
                    model=model,
                    provider_dp_rank=provider_dp_rank,
                    dp_size=dp_size,
                ))
                flat_index += 1
            continue
        slots.append(RoutingSlot(
            flat_index=flat_index,
            slot_id=_provider_slot_id(provider, synthetic_dp_rank),
            provider=provider,
            provider_key=provider_key,
            provider_name=provider_name,
            model=model,
            provider_dp_rank=synthetic_dp_rank,
            dp_size=1 if synthetic_dp_rank is not None else dp_size,
        ))
        flat_index += 1
    return slots


def _get_or_create_model_allocator(model: str, slots: list[RoutingSlot]) -> StickySlotAllocator:
    if not slots:
        raise HTTPException(500, f"No routing slots available for model '{model}'")
    allocator_key = _provider_pool_key(model)
    ttl = _slot_ttl_for_providers([slot.provider for slot in slots])
    slot_ids = [slot.slot_id for slot in slots]
    allocator = _model_allocators.get(allocator_key)
    if allocator is None or allocator.ttl_sec != ttl or allocator.slots != slot_ids:
        allocator = StickySlotAllocator(slots=slot_ids, ttl_sec=ttl)
        if allocator_key in _model_allocators:
            allocator.sessions.update(_model_allocators[allocator_key].sessions)
            allocator.session_activity.update(_model_allocators[allocator_key].session_activity)
            allocator.slot_last_used.update({
                slot_id: ts for slot_id, ts in _model_allocators[allocator_key].slot_last_used.items()
                if slot_id in slot_ids
            })
        _model_allocators[allocator_key] = allocator
        logger.info(
            "Created model allocator for model=%s slot_count=%d ttl_sec=%.0f",
            model,
            len(slot_ids),
            ttl,
        )
    return allocator


def _api_base(provider: dict) -> str:
    """Return the base URL (up to and including /v1) for a provider."""
    url = provider["api_base_url"]
    for suffix in ("/chat/completions", "/completions", "/models", "/batches", "/files"):
        if url.endswith(suffix):
            return url[: -len(suffix)]
    if url.rstrip("/").endswith("/v1"):
        return url.rstrip("/")
    return url.rsplit("/", 1)[0]


def _models_url(provider: dict) -> str:
    return _api_base(provider) + "/models"


def _batches_url(provider: dict) -> str:
    return _api_base(provider) + "/batches"


def _files_url(provider: dict) -> str:
    return _api_base(provider) + "/files"


def _dp_routing_config(provider: dict) -> dict | None:
    cfg = provider.get("dp_routing")
    if not isinstance(cfg, dict) or not cfg.get("enabled"):
        return None
    return cfg


def _dp_cache_key(provider: dict) -> str:
    return f"{provider.get('name', '')}:{provider.get('api_base_url', '')}"


def _dp_server_info_url(provider: dict) -> str:
    api_base = _api_base(provider).rstrip("/")
    root = api_base.removesuffix("/v1").removesuffix("/")
    return f"{root}/get_server_info"


def _dp_server_info_ttl(provider: dict) -> int:
    cfg = _dp_routing_config(provider) or {}
    ttl = cfg.get("server_info_ttl_sec", _DP_SERVER_INFO_TTL_SEC)
    try:
        return max(int(ttl), 1)
    except (TypeError, ValueError):
        return _DP_SERVER_INFO_TTL_SEC


async def _get_provider_dp_size(provider: dict, force_refresh: bool = False) -> int | None:
    if _dp_routing_config(provider) is None:
        return None

    cache_key = _dp_cache_key(provider)
    now = time.monotonic()
    cached = _dp_size_cache.get(cache_key)
    ttl = _dp_server_info_ttl(provider)
    if cached and not force_refresh and now - float(cached["fetched_at"]) < ttl:
        dp_size = int(cached["dp_size"])
        logger.debug(
            "DP size cache hit: provider=%s dp_size=%d age=%.1fs ttl=%ds",
            provider.get("name"),
            dp_size,
            now - float(cached["fetched_at"]),
            ttl,
        )
        return dp_size

    logger.debug(
        "DP size cache miss: provider=%s force_refresh=%s",
        provider.get("name"),
        force_refresh,
    )

    url = _dp_server_info_url(provider)
    client = get_shared_client()
    timeout = _timeout()
    try:
        resp = await client.get(url, headers=_provider_headers(provider), timeout=timeout)
        if resp.status_code >= 400:
            _dp_size_cache.pop(cache_key, None)
            logger.warning("DP routing disabled for %s: /get_server_info returned %d", provider.get("name"), resp.status_code)
            return None
        data = resp.json()
        dp_size = int(data["dp_size"])
        if dp_size < 0:
            raise ValueError("dp_size must be non-negative")
    except Exception as exc:
        _dp_size_cache.pop(cache_key, None)
        logger.warning("DP routing disabled for %s: failed to fetch /get_server_info: %s", provider.get("name"), exc)
        return None

    _dp_size_cache[cache_key] = {"dp_size": dp_size, "fetched_at": now}
    logger.info("DP size fetched: provider=%s dp_size=%d", provider.get("name"), dp_size)
    return dp_size


def _get_or_create_allocator(provider: dict, dp_size: int) -> DPAllocator:
    """Get or create a DPAllocator for the given provider."""
    cache_key = _dp_cache_key(provider)
    allocator = _dp_allocators.get(cache_key)
    if allocator is None or allocator.dp_size != dp_size:
        ttl = _get_session_ttl(provider)
        allocator = DPAllocator(dp_size=dp_size, ttl_sec=ttl)
        _dp_allocators[cache_key] = allocator
        logger.info(
            "Created DP allocator for provider=%s dp_size=%d ttl_sec=%.0f",
            provider.get("name"),
            dp_size,
            ttl,
        )
    return allocator


def _rendezvous_rank(sticky_key: str, dp_size: int) -> int:
    """Deprecated: Use DPAllocator.assign() instead.

    Kept for backward compatibility with existing tests.
    """
    best_rank = 0
    best_score: bytes | None = None
    sticky_key_bytes = sticky_key.encode("utf-8", "surrogatepass")
    for rank in range(dp_size):
        digest = hashlib.sha256()
        digest.update(sticky_key_bytes)
        digest.update(b"\0")
        digest.update(str(rank).encode("ascii"))
        score = digest.digest()
        if best_score is None or score > best_score:
            best_score = score
            best_rank = rank
    return best_rank


def _is_invalid_dp_rank_error(exc: ProviderError) -> bool:
    if not exc.body:
        return False
    return bool(_INVALID_DP_RANK_RE.search(exc.body))


def _log_dp_routing(provider: dict, decision: DPRoutingDecision, remapped: bool = False) -> None:
    if decision.rank is None:
        return
    logger.info(
        "DP routing provider=%s dp_size=%s rank=%s source=%s session_id=%s subagent_id=%s sticky_key=%s remapped=%s",
        provider.get("name"),
        decision.dp_size,
        decision.rank,
        decision.source,
        decision.session_id,
        decision.subagent_id,
        decision.sticky_key,
        remapped,
    )


async def _resolve_dp_routing(
    request: Request,
    model: str,
    anthropic_req: dict,
    base_openai_req: dict,
    force_refresh: bool = False,
) -> tuple[RoutingSlot, dict, DPRoutingDecision]:
    openai_req = dict(base_openai_req)
    slots = await _build_routing_slots(model, force_refresh=force_refresh)
    if not slots:
        raise HTTPException(500, f"No routing slots available for model '{model}'")
    slot_by_id = {slot.slot_id: slot for slot in slots}

    override_rank = request.headers.get(_DP_OVERRIDE_HEADER)
    if override_rank is not None:
        try:
            flat_rank = int(override_rank.strip())
        except ValueError:
            raise HTTPException(400, f"{_DP_OVERRIDE_HEADER} must be an integer")
        if flat_rank < 0 or flat_rank >= len(slots):
            raise HTTPException(400, f"{_DP_OVERRIDE_HEADER} must be in range [0, {len(slots)})")
        selected_slot = slots[flat_rank]
        if selected_slot.provider_dp_rank is not None:
            openai_req["routed_dp_rank"] = selected_slot.provider_dp_rank
        decision = DPRoutingDecision(
            provider_key=selected_slot.provider_key,
            provider_name=selected_slot.provider_name,
            dp_size=selected_slot.dp_size,
            rank=selected_slot.flat_index,
            provider_dp_rank=selected_slot.provider_dp_rank,
            source="override",
        )
        return selected_slot, openai_req, decision

    session_id = request.headers.get(_CLAUDE_SESSION_HEADER)
    if not session_id:
        session_id = str(uuid.uuid4())

    sticky_mode = _slot_sticky_mode_for_providers([slot.provider for slot in slots])
    sticky_key, sticky_source, subagent_id = _derive_sticky_key(session_id, sticky_mode, anthropic_req)

    allocator = _get_or_create_model_allocator(model, slots)
    slot_id, is_new = allocator.assign(sticky_key)
    selected_slot = slot_by_id.get(slot_id)
    if selected_slot is None:
        allocator.sessions.pop(sticky_key, None)
        allocator.session_activity.pop(sticky_key, None)
        slot_id, is_new = allocator.assign(sticky_key)
        selected_slot = slot_by_id[slot_id]

    if selected_slot.provider_dp_rank is not None:
        openai_req["routed_dp_rank"] = selected_slot.provider_dp_rank
    decision = DPRoutingDecision(
        provider_key=selected_slot.provider_key,
        provider_name=selected_slot.provider_name,
        dp_size=selected_slot.dp_size,
        rank=selected_slot.flat_index,
        provider_dp_rank=selected_slot.provider_dp_rank,
        source=sticky_source,
        sticky_key=sticky_key,
        session_id=session_id,
        subagent_id=subagent_id,
    )

    logger.info(
        "Routing assign: model=%s provider=%s session=%s subagent_id=%s sticky_source=%s slot=%d provider_dp_rank=%s is_new=%s total_sessions=%d",
        model,
        selected_slot.provider_name,
        session_id[:8] + "..." if len(session_id) > 8 else session_id,
        subagent_id,
        sticky_source,
        selected_slot.flat_index,
        selected_slot.provider_dp_rank,
        is_new,
        len(allocator.sessions),
    )

    return selected_slot, openai_req, decision


async def _retry_message_request_for_invalid_rank(
    request: Request,
    model: str,
    anthropic_req: dict,
    base_openai_req: dict,
    old_decision: DPRoutingDecision,
) -> tuple[RoutingSlot, dict, DPRoutingDecision]:
    selected_slot, openai_req, new_decision = await _resolve_dp_routing(
        request,
        model,
        anthropic_req,
        base_openai_req,
        force_refresh=True,
    )
    if old_decision.source == "override" and new_decision.rank is None:
        raise HTTPException(400, f"{_DP_OVERRIDE_HEADER} is no longer valid after refreshing routing slots")
    openai_req = apply_provider_params(selected_slot.provider, openai_req)
    _log_dp_routing(selected_slot.provider, new_decision, remapped=(new_decision.rank != old_decision.rank))
    log_openai_request(openai_req)
    return selected_slot, openai_req, new_decision


_tokenizer_cache: dict[str, object] = {}  # tokenizer_path → tokenizer


def _get_tokenizer(tokenizer_path: str):
    if tokenizer_path not in _tokenizer_cache:
        from transformers import AutoTokenizer
        _tokenizer_cache[tokenizer_path] = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
        )
        logger.info("Loaded tokenizer from %s", tokenizer_path)
    return _tokenizer_cache[tokenizer_path]


def _extract_text_for_counting(req: dict) -> str:
    """Flatten all text content in an OpenAI-format request to a single string."""
    parts: list[str] = []
    for msg in req.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, dict) and block.get("type") == "tool_calls":
                    parts.append(json.dumps(block, ensure_ascii=False))
        # tool_calls at message level
        for tc in msg.get("tool_calls") or []:
            parts.append(json.dumps(tc.get("function", {}), ensure_ascii=False))
    for tool in req.get("tools", []):
        parts.append(json.dumps(tool.get("function", {}), ensure_ascii=False))
    return "\n".join(parts)


def _count_tokens_in_openai_req(req: dict, tokenizer_path: str) -> int:
    tok = _get_tokenizer(tokenizer_path)
    text: str | None = None
    apply_chat_template = getattr(tok, "apply_chat_template", None)

    if callable(apply_chat_template):
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": False,
        }
        try:
            sig = inspect.signature(apply_chat_template)
            if "tools" in sig.parameters and req.get("tools"):
                kwargs["tools"] = req["tools"]
            text = apply_chat_template(req.get("messages", []), **kwargs)
        except Exception:
            logger.debug("Falling back to flattened text token counting", exc_info=True)

    if not isinstance(text, str):
        text = _extract_text_for_counting(req)
    return len(tok.encode(text))


def _resolve_tokenizer_path(provider: dict) -> str | None:
    return (
        provider.get("tokenizer_path")
        or _config.get("tokenizer_path")
        or os.environ.get("CCR_TOKENIZER_PATH")
        or os.environ.get("TOKENIZER_PATH")
    )


def _normalize_openai_usage(usage: dict | None) -> dict:
    usage = usage or {}
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    prompt_details = usage.get("prompt_tokens_details") or {}
    cache_read = int(prompt_details.get("cached_tokens", 0) or 0)
    cache_creation = int(prompt_details.get("cache_creation_tokens", 0) or 0)
    input_tokens = max(prompt_tokens - cache_read - cache_creation, 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": completion_tokens,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_creation,
    }


def _usage_from_anthropic_message(message: dict | None) -> dict | None:
    if not isinstance(message, dict):
        return None
    usage = message.get("usage")
    if not isinstance(usage, dict):
        return None
    return {
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
        "cache_read_input_tokens": int(usage.get("cache_read_input_tokens", 0) or 0),
        "cache_creation_input_tokens": int(usage.get("cache_creation_input_tokens", 0) or 0),
    }


def _sglang_tokenize_url(provider: dict) -> str:
    api_base = _api_base(provider).rstrip("/")
    root = api_base.removesuffix("/v1").removesuffix("/")
    return f"{root}/tokenize"


async def _count_tokens_via_sglang(provider: dict, model: str, prompt: str) -> int | None:
    client = get_shared_client()
    timeout = _timeout()
    payload = {
        "model": model,
        "prompt": prompt,
        "add_special_tokens": False,
    }
    try:
        resp = await client.post(
            _sglang_tokenize_url(provider),
            headers=_provider_headers(provider),
            json=payload,
            timeout=timeout,
        )
    except Exception:
        logger.debug("SGLang /tokenize request failed", exc_info=True)
        return None

    if resp.status_code in (404, 405, 501):
        return None
    if resp.status_code >= 400:
        logger.debug("SGLang /tokenize returned %d: %s", resp.status_code, resp.text)
        return None

    try:
        data = resp.json()
        count = data.get("count")
        if count is None:
            return None
        return int(count)
    except Exception:
        logger.debug("Invalid SGLang /tokenize response", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------

@app.post("/v1/messages")
async def messages(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    try:
        model = _resolve_request_model((body or {}).get("model"))
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Routing error")
        raise HTTPException(500, str(exc))

    # Override model in the original request so converter uses the routed model
    body = dict(body)
    body["model"] = model

    # Anthropic → OpenAI, then apply provider param defaults
    base_openai_req = anthropic_to_openai(body)
    selected_slot, openai_req, dp_decision = await _resolve_dp_routing(request, model, body, base_openai_req)
    provider = selected_slot.provider
    url = provider["api_base_url"]
    base_openai_req = apply_provider_params(provider, base_openai_req)
    openai_req = apply_provider_params(provider, openai_req)
    _log_dp_routing(provider, dp_decision)
    log_openai_request(openai_req)

    headers = _provider_headers(provider)
    max_retries: int = provider.get("max_retries", 3)
    timeout = _timeout()

    is_stream = openai_req.get("stream", False)
    metrics_ctx = _runtime_metrics.start_request(
        provider_key=_dp_cache_key(provider),
        provider_name=provider.get("name", ""),
        dp_rank=dp_decision.provider_dp_rank,
        is_stream=is_stream,
    )

    if is_stream:
        # Eagerly connect to check provider status before committing to HTTP 200
        try:
            stream = await open_provider_stream(url, headers, openai_req, timeout, max_retries)
        except ProviderError as exc:
            if _is_invalid_dp_rank_error(exc) and dp_decision.rank is not None:
                selected_slot, openai_req, dp_decision = await _retry_message_request_for_invalid_rank(
                    request,
                    model,
                    body,
                    base_openai_req,
                    dp_decision,
                )
                provider = selected_slot.provider
                url = provider["api_base_url"]
                headers = _provider_headers(provider)
                try:
                    stream = await open_provider_stream(url, headers, openai_req, timeout, max_retries)
                    metrics_ctx.provider_key = _dp_cache_key(provider)
                    metrics_ctx.provider_name = provider.get("name", "")
                    metrics_ctx.dp_rank = dp_decision.provider_dp_rank
                except ProviderError as retry_exc:
                    _runtime_metrics.finish_request(metrics_ctx, success=False)
                    raise HTTPException(retry_exc.status or 502, retry_exc.body or str(retry_exc))
            else:
                _runtime_metrics.finish_request(metrics_ctx, success=False)
                raise HTTPException(exc.status or 502, exc.body or str(exc))
        except Exception as exc:
            logger.exception("Stream connection error")
            _runtime_metrics.finish_request(metrics_ctx, success=False)
            raise HTTPException(502, str(exc))

        return StreamingResponse(
            _stream_response(openai_req, stream, model, metrics_ctx=metrics_ctx),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                **dp_decision.response_headers(),
            },
        )

    # Non-streaming
    try:
        openai_resp = await post_json(
            url, headers, openai_req, timeout=timeout, max_retries=max_retries
        )
    except ProviderError as exc:
        if _is_invalid_dp_rank_error(exc) and dp_decision.rank is not None:
            selected_slot, openai_req, dp_decision = await _retry_message_request_for_invalid_rank(
                request,
                model,
                body,
                base_openai_req,
                dp_decision,
            )
            provider = selected_slot.provider
            url = provider["api_base_url"]
            headers = _provider_headers(provider)
            try:
                metrics_ctx.provider_key = _dp_cache_key(provider)
                metrics_ctx.provider_name = provider.get("name", "")
                metrics_ctx.dp_rank = dp_decision.provider_dp_rank
                openai_resp = await post_json(
                    url, headers, openai_req, timeout=timeout, max_retries=max_retries
                )
            except ProviderError as retry_exc:
                logger.error("Provider error after DP refresh: %s", retry_exc)
                _runtime_metrics.finish_request(metrics_ctx, success=False)
                raise HTTPException(retry_exc.status or 502, retry_exc.body or str(retry_exc))
        else:
            logger.error("Provider error: %s", exc)
            _runtime_metrics.finish_request(metrics_ctx, success=False)
            raise HTTPException(exc.status or 502, exc.body or str(exc))
    except Exception as exc:
        logger.exception("Unexpected error calling provider")
        _runtime_metrics.finish_request(metrics_ctx, success=False)
        raise HTTPException(502, str(exc))

    check_and_save_nonstreaming(openai_req, openai_resp)
    anthropic_resp = openai_to_anthropic(openai_resp, model)
    _runtime_metrics.finish_request(
        metrics_ctx,
        success=True,
        usage=_usage_from_anthropic_message(anthropic_resp) or _normalize_openai_usage(openai_resp.get("usage")),
    )
    return JSONResponse(content=anthropic_resp, headers=dp_decision.response_headers())


async def _stream_response(
    openai_req: dict,
    stream: ProviderStream,
    model: str,
    metrics_ctx: RequestMetricsContext | None = None,
) -> AsyncIterator[str]:
    import json as _json
    from debug import is_enabled as _debug_enabled

    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    _text_buf: list[str] | None = [] if _debug_enabled() else None
    usage: dict | None = None
    completed = False

    try:
        async for event in stream_openai_to_anthropic(stream, message_id, model):
            for line in event.split("\n"):
                if not line.startswith("data: "):
                    continue
                try:
                    parsed = _json.loads(line[6:])
                except _json.JSONDecodeError:
                    continue
                if parsed.get("type") == "message_delta" and isinstance(parsed.get("usage"), dict):
                    usage = _usage_from_anthropic_message(parsed)
            # Accumulate text for debug check (zero cost when CCR_DEBUG is off)
            if _text_buf is not None:
                for line in event.split("\n"):
                    if line.startswith("data: "):
                        try:
                            parsed = _json.loads(line[6:])
                            delta = parsed.get("delta", {})
                            if delta.get("type") in ("text_delta", "thinking_delta"):
                                _text_buf.append(delta.get("text") or delta.get("thinking") or "")
                        except _json.JSONDecodeError:
                            pass
            yield event
        completed = True
    except Exception as exc:
        logger.exception("Streaming error")
        error_event = {
            "type": "error",
            "error": {"type": "api_error", "message": str(exc)},
        }
        yield f"event: error\ndata: {_json.dumps(error_event)}\n\n"
        return
    finally:
        await stream.aclose()
        if _text_buf is not None:
            check_and_save_streaming(openai_req, "".join(_text_buf))
        if metrics_ctx is not None:
            _runtime_metrics.finish_request(metrics_ctx, success=completed, usage=usage)


# ---------------------------------------------------------------------------
# Token counting  POST /v1/messages/count_tokens
# ---------------------------------------------------------------------------

@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    try:
        model = _resolve_request_model((body or {}).get("model"))
        target = _select_deterministic_provider(model)
        provider = target.provider
    except HTTPException:
        raise

    body = dict(body)
    body["model"] = model

    openai_req = anthropic_to_openai(body)
    openai_req = apply_provider_params(provider, openai_req)

    prompt_text = _extract_text_for_counting(openai_req)
    backend_count = await _count_tokens_via_sglang(provider, model, prompt_text)
    if backend_count is not None:
        return {"input_tokens": backend_count}

    tokenizer_path = _resolve_tokenizer_path(provider)
    if not tokenizer_path:
        return {"input_tokens": 0}
    try:
        input_tokens = _count_tokens_in_openai_req(openai_req, tokenizer_path)
    except Exception as exc:
        logger.exception("Token counting error")
        raise HTTPException(500, f"Token counting failed: {exc}")

    return {"input_tokens": input_tokens}


# ---------------------------------------------------------------------------
# Proxy POST /tokens/clear to upstream adapter
# ---------------------------------------------------------------------------

@app.post("/tokens/clear")
async def tokens_clear(request: Request):
    """Proxy /tokens/clear to the upstream chat_to_generate_adapter."""
    body = await request.json()
    model = _resolve_routed_model()
    provider = _select_deterministic_provider(model).provider
    api_base = _api_base(provider)
    # Strip /v1 (with or without trailing slash) to get the adapter root
    adapter_root = api_base.removesuffix("/v1").removesuffix("/")
    url = f"{adapter_root}/tokens/clear"
    api_key = provider.get("api_key", "")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
            resp = await client.post(
                url,
                json=body,
                headers=headers,
            )
            return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")
    except Exception as exc:
        logger.exception("tokens/clear proxy error")
        raise HTTPException(502, f"Upstream /tokens/clear failed: {exc}")


# ---------------------------------------------------------------------------
# Models  GET /v1/models  and  GET /v1/models/{model_id}
# ---------------------------------------------------------------------------

def _openai_model_to_anthropic(m: dict) -> dict:
    import datetime
    ts = m.get("created", 0)
    try:
        created_at = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).isoformat()
    except Exception:
        created_at = "1970-01-01T00:00:00+00:00"
    return {
        "type": "model",
        "id": m.get("id", ""),
        "display_name": m.get("id", ""),
        "created_at": created_at,
    }


@app.get("/v1/models")
async def list_models(before_id: str | None = None,
                      after_id: str | None = None,
                      limit: int = 20):
    try:
        model = _resolve_routed_model()
        provider = _select_deterministic_provider(model).provider
    except HTTPException:
        raise

    url = _models_url(provider)
    headers = _provider_headers(provider)
    timeout = _timeout()
    params: dict = {"limit": limit}
    if before_id:
        params["before"] = before_id
    if after_id:
        params["after"] = after_id

    try:
        client = get_shared_client()
        r = await client.get(url, headers=headers, params=params, timeout=timeout)
        if r.status_code >= 400:
            raise HTTPException(r.status_code, r.text)
        data = r.json()
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error fetching models")
        raise HTTPException(502, str(exc))

    models = [_openai_model_to_anthropic(m) for m in data.get("data", [])]
    return {
        "data": models,
        "has_more": data.get("has_more", False),
        "first_id": models[0]["id"] if models else None,
        "last_id":  models[-1]["id"] if models else None,
    }


@app.get("/v1/models/{model_id:path}")
async def get_model(model_id: str):
    try:
        model = _resolve_routed_model()
        provider = _select_deterministic_provider(model).provider
    except HTTPException:
        raise

    url = _models_url(provider) + f"/{model_id}"
    headers = _provider_headers(provider)
    timeout = _timeout()

    try:
        client = get_shared_client()
        r = await client.get(url, headers=headers, timeout=timeout)
        if r.status_code >= 400:
            raise HTTPException(r.status_code, r.text)
        return _openai_model_to_anthropic(r.json())
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error fetching model %s", model_id)
        raise HTTPException(502, str(exc))


# ---------------------------------------------------------------------------
# Legacy Text Completions  POST /v1/complete
# ---------------------------------------------------------------------------

def _parse_legacy_prompt(prompt: str) -> list[dict]:
    """
    Convert a legacy \\n\\nHuman: / \\n\\nAssistant: prompt string into a messages list.
    Falls back to a single user message if the format is not recognized.
    """
    import re

    has_human = '\n\nHuman: ' in prompt
    has_assistant_start = prompt.lstrip().startswith('\n\nAssistant: ')

    if not has_human and not has_assistant_start:
        return [{"role": "user", "content": prompt.strip()}]

    messages = []
    # Split on delimiters, keeping track of which delimiter was matched
    parts = re.split(r'(\n\nHuman: |\n\nAssistant: )', prompt)
    current_role = "user" if has_human else "assistant"
    i = 0
    while i < len(parts):
        text = parts[i]
        if text in ('\n\nHuman: ', '\n\nAssistant: '):
            current_role = "user" if 'Human' in text else "assistant"
            i += 1
            continue
        text = text.strip()
        if text:
            messages.append({"role": current_role, "content": text})
        i += 1

    return messages or [{"role": "user", "content": prompt.strip()}]


@app.post("/v1/complete")
async def legacy_complete(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    try:
        model = _resolve_request_model(body.get("model"))
        target = _select_deterministic_provider(model)
        provider = target.provider
        url = target.url
    except HTTPException:
        raise

    messages = _parse_legacy_prompt(body.get("prompt", ""))
    openai_req: dict = {
        "model": model,
        "messages": messages,
        "stream": False,
        "max_tokens": body.get("max_tokens_to_sample", 1024),
    }
    for field in ("temperature", "top_p", "top_k"):
        if body.get(field) is not None:
            openai_req[field] = body[field]
    if body.get("stop_sequences"):
        openai_req["stop"] = body["stop_sequences"]

    openai_req = apply_provider_params(provider, openai_req)
    headers = _provider_headers(provider)
    timeout = _timeout()

    try:
        openai_resp = await post_json(url, headers, openai_req, timeout=timeout)
    except ProviderError as exc:
        raise HTTPException(exc.status or 502, exc.body or str(exc))
    except Exception as exc:
        logger.exception("Legacy completion error")
        raise HTTPException(502, str(exc))

    choice = openai_resp["choices"][0]
    finish = choice.get("finish_reason", "stop")
    stop_reason = "max_tokens" if finish == "length" else "stop_sequence"
    text = (choice.get("message") or {}).get("content") or ""

    return {
        "id": openai_resp.get("id", f"compl_{uuid.uuid4().hex[:24]}"),
        "type": "completion",
        "completion": text,
        "stop_reason": stop_reason,
        "model": model,
    }


# ---------------------------------------------------------------------------
# Batch API
# ---------------------------------------------------------------------------

async def _httpx_get(url: str, headers: dict, timeout: float, **kwargs) -> dict:
    client = get_shared_client()
    r = await client.get(url, headers=headers, timeout=timeout, **kwargs)
    if r.status_code >= 400:
        raise HTTPException(r.status_code, r.text)
    return r.json()


async def _httpx_delete(url: str, headers: dict, timeout: float) -> dict:
    client = get_shared_client()
    r = await client.delete(url, headers=headers, timeout=timeout)
    if r.status_code >= 400:
        raise HTTPException(r.status_code, r.text)
    return r.json() if r.text else {}


def _self_base_url(request: Request) -> str:
    return str(request.base_url).rstrip("/")


@app.post("/v1/messages/batches")
async def create_batch(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    try:
        model = _resolve_routed_model()
        provider = _select_deterministic_provider(model).provider
    except HTTPException:
        raise

    requests_list = body.get("requests", [])
    if not requests_list:
        raise HTTPException(400, "requests list is empty")

    # Convert to OpenAI JSONL and upload as a file
    jsonl_content = anthropic_batch_to_openai_jsonl(requests_list, model)
    jsonl_bytes = jsonl_content.encode()

    headers = _provider_headers(provider)
    timeout = _timeout()
    files_url = _files_url(provider)
    batches_url = _batches_url(provider)

    try:
        # Upload input file (use a fresh client for multipart upload)
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            r = await client.post(
                files_url,
                headers={k: v for k, v in headers.items() if k != "Content-Type"},
                files={"file": ("batch_input.jsonl", jsonl_bytes, "application/jsonl")},
                data={"purpose": "batch"},
            )
        if r.status_code >= 400:
            raise HTTPException(r.status_code, r.text)
        file_id = r.json()["id"]

        # Create the batch
        batch_resp = await post_json(batches_url, headers, {
            "input_file_id": file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        }, timeout=timeout)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Batch create error")
        raise HTTPException(502, str(exc))

    return openai_batch_to_anthropic(batch_resp, _self_base_url(request))


@app.get("/v1/messages/batches")
async def list_batches(request: Request,
                       before_id: str | None = None,
                       after_id: str | None = None,
                       limit: int = 20):
    try:
        model = _resolve_routed_model()
        provider = _select_deterministic_provider(model).provider
    except HTTPException:
        raise

    headers = _provider_headers(provider)
    timeout = _timeout()
    url = _batches_url(provider)
    params = {"limit": limit}
    if before_id:
        params["before"] = before_id
    if after_id:
        params["after"] = after_id

    try:
        data = await _httpx_get(url, headers, timeout, params=params)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Batch list error")
        raise HTTPException(502, str(exc))

    base = _self_base_url(request)
    batches = [openai_batch_to_anthropic(b, base) for b in data.get("data", [])]
    return {
        "data": batches,
        "has_more": data.get("has_more", False),
        "first_id": batches[0]["id"] if batches else None,
        "last_id":  batches[-1]["id"] if batches else None,
    }


@app.get("/v1/messages/batches/{batch_id}")
async def get_batch(batch_id: str, request: Request):
    try:
        model = _resolve_routed_model()
        provider = _select_deterministic_provider(model).provider
    except HTTPException:
        raise

    headers = _provider_headers(provider)
    timeout = _timeout()

    try:
        data = await _httpx_get(_batches_url(provider) + f"/{batch_id}", headers, timeout)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Batch get error")
        raise HTTPException(502, str(exc))

    return openai_batch_to_anthropic(data, _self_base_url(request))


@app.post("/v1/messages/batches/{batch_id}/cancel")
async def cancel_batch(batch_id: str, request: Request):
    try:
        model = _resolve_routed_model()
        provider = _select_deterministic_provider(model).provider
    except HTTPException:
        raise

    headers = _provider_headers(provider)
    timeout = _timeout()

    try:
        client = get_shared_client()
        r = await client.post(
            _batches_url(provider) + f"/{batch_id}/cancel",
            headers=headers,
            timeout=timeout,
        )
        if r.status_code >= 400:
            raise HTTPException(r.status_code, r.text)
        data = r.json()
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Batch cancel error")
        raise HTTPException(502, str(exc))

    return openai_batch_to_anthropic(data, _self_base_url(request))


@app.delete("/v1/messages/batches/{batch_id}")
async def delete_batch(batch_id: str):
    try:
        model = _resolve_routed_model()
        provider = _select_deterministic_provider(model).provider
    except HTTPException:
        raise

    headers = _provider_headers(provider)
    timeout = _timeout()

    try:
        await _httpx_delete(_batches_url(provider) + f"/{batch_id}", headers, timeout)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Batch delete error")
        raise HTTPException(502, str(exc))

    return {"id": batch_id, "type": "message_batch_deleted"}


@app.get("/v1/messages/batches/{batch_id}/results")
async def batch_results(batch_id: str, request: Request):
    """Stream batch results as Anthropic JSONL (one result per line)."""
    try:
        model = _resolve_routed_model()
        provider = _select_deterministic_provider(model).provider
    except HTTPException:
        raise

    headers = _provider_headers(provider)
    timeout = _timeout()

    # Get the batch to find output_file_id
    try:
        batch = await _httpx_get(_batches_url(provider) + f"/{batch_id}", headers, timeout)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, str(exc))

    file_id = batch.get("output_file_id")
    if not file_id:
        status = batch.get("status", "unknown")
        raise HTTPException(404, f"Batch results not available yet (status: {status})")

    async def _stream_results():
        url = _files_url(provider) + f"/{file_id}/content"
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout), trust_env=False) as client:
            async with client.stream("GET", url, headers=headers) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    yield json.dumps({
                        "type": "error",
                        "error": {"type": "api_error", "message": body.decode()},
                    }) + "\n"
                    return
                buffer = ""
                async for chunk in resp.aiter_text():
                    buffer += chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        converted = openai_results_line_to_anthropic(line, model)
                        if converted:
                            yield converted + "\n"
                if buffer.strip():
                    converted = openai_results_line_to_anthropic(buffer, model)
                    if converted:
                        yield converted + "\n"

    return StreamingResponse(
        _stream_results(),
        media_type="application/x-jsonlines",
    )


# ---------------------------------------------------------------------------
# Metrics + Health check
# ---------------------------------------------------------------------------

@app.get("/metrics")
async def metrics():
    snapshot = _runtime_metrics.snapshot()
    providers: list[dict] = []
    configured_providers = {
        _dp_cache_key(provider): provider
        for provider in _config.get("Providers", [])
        if isinstance(provider, dict)
    }
    sessions_by_provider: dict[str, int] = {}
    sessions_by_provider_rank: dict[str, dict[int, int]] = {}
    for allocator in _model_allocators.values():
        allocator.cleanup()
        for slot_id, count in allocator.stats()["sessions_per_slot"].items():
            for provider_key in configured_providers:
                if not _slot_belongs_to_provider(slot_id, provider_key):
                    continue
                sessions_by_provider[provider_key] = sessions_by_provider.get(provider_key, 0) + int(count)
                rank = _slot_rank_for_provider(slot_id, provider_key)
                if rank is not None:
                    per_rank = sessions_by_provider_rank.setdefault(provider_key, {})
                    per_rank[rank] = per_rank.get(rank, 0) + int(count)
                break

    for provider_key in sorted(set(configured_providers.keys()) | set(snapshot["providers"].keys())):
        provider_stats = snapshot["providers"].get(provider_key, {
            "provider_name": configured_providers.get(provider_key, {}).get("name", ""),
            "requests_started": 0,
            "requests_completed": 0,
            "requests_failed": 0,
            "streaming_requests_started": 0,
            "non_streaming_requests_started": 0,
            "active_requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "avg_latency_sec": 0.0,
            "throughput": {
                "requests_in_window": 0,
                "total_tokens_per_sec": 0.0,
                "output_tokens_per_sec": 0.0,
            },
            "per_dp": {},
        })
        provider_cfg = configured_providers.get(provider_key, {})
        sessions_per_rank: dict[int, int] = sessions_by_provider_rank.get(provider_key, {})
        total_sessions = sessions_by_provider.get(provider_key, 0)
        dp_size = None
        if (cached := _dp_size_cache.get(provider_key)) is not None:
            dp_size = int(cached["dp_size"])
        elif provider_stats["per_dp"]:
            dp_size = max(int(rank) for rank in provider_stats["per_dp"].keys()) + 1

        per_dp: list[dict] = []
        known_ranks = sorted(set(provider_stats["per_dp"].keys()) | set(sessions_per_rank.keys()))
        for rank in known_ranks:
            dp_stats = provider_stats["per_dp"].get(rank, {})
            throughput = dp_stats.get("throughput", {})
            per_dp.append({
                "rank": rank,
                "sessions": sessions_per_rank.get(rank, 0),
                "active_requests": dp_stats.get("active_requests", 0),
                "requests_started": dp_stats.get("requests_started", 0),
                "requests_completed": dp_stats.get("requests_completed", 0),
                "requests_failed": dp_stats.get("requests_failed", 0),
                "streaming_requests_started": dp_stats.get("streaming_requests_started", 0),
                "non_streaming_requests_started": dp_stats.get("non_streaming_requests_started", 0),
                "input_tokens": dp_stats.get("input_tokens", 0),
                "output_tokens": dp_stats.get("output_tokens", 0),
                "total_tokens": dp_stats.get("total_tokens", 0),
                "avg_latency_sec": dp_stats.get("avg_latency_sec", 0.0),
                "throughput": {
                    "requests_in_window": throughput.get("requests_in_window", 0),
                    "total_tokens_per_sec": throughput.get("total_tokens_per_sec", 0.0),
                    "output_tokens_per_sec": throughput.get("output_tokens_per_sec", 0.0),
                },
            })

        providers.append({
            "provider_key": provider_key,
            "provider_name": provider_stats["provider_name"],
            "api_base_url": provider_cfg.get("api_base_url"),
            "dp_routing_enabled": _dp_routing_config(provider_cfg) is not None,
            "dp_size": dp_size,
            "total_sessions": total_sessions,
            "active_requests": provider_stats["active_requests"],
            "requests_started": provider_stats["requests_started"],
            "requests_completed": provider_stats["requests_completed"],
            "requests_failed": provider_stats["requests_failed"],
            "streaming_requests_started": provider_stats["streaming_requests_started"],
            "non_streaming_requests_started": provider_stats["non_streaming_requests_started"],
            "input_tokens": provider_stats["input_tokens"],
            "output_tokens": provider_stats["output_tokens"],
            "total_tokens": provider_stats["total_tokens"],
            "avg_latency_sec": provider_stats["avg_latency_sec"],
            "throughput": provider_stats["throughput"],
            "per_dp": per_dp,
        })

    return {
        "status": "ok",
        "pid": os.getpid(),
        "uptime_sec": snapshot["uptime_sec"],
        "throughput_window_sec": snapshot["throughput_window_sec"],
        "totals": {
            "active_requests": snapshot["active_requests"],
            "requests_started": snapshot["requests_started"],
            "requests_completed": snapshot["requests_completed"],
            "requests_failed": snapshot["requests_failed"],
            "streaming_requests_started": snapshot["streaming_requests_started"],
            "non_streaming_requests_started": snapshot["non_streaming_requests_started"],
            "input_tokens": snapshot["input_tokens"],
            "output_tokens": snapshot["output_tokens"],
            "total_tokens": snapshot["total_tokens"],
            "avg_latency_sec": snapshot["avg_latency_sec"],
            "throughput": snapshot["throughput"],
        },
        "providers": providers,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8891, timeout_keep_alive=1500, reload=False)
