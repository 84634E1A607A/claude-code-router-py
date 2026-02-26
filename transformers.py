"""
Transformer pipeline — runs on the OpenAI-format request before it is sent
to the provider.

Config format inside a provider:
    "transformer": {
        "use": [
            "flattenMessage",
            ["maxtoken",  {"max_tokens": 16384}],
            ["sampling",  {"temperature": 1.0, "top_p": 1.0}],
            ["reasoning"]
        ]
    }

Each entry is either a transformer name string or a [name, options] pair.
Transformers are applied left-to-right.
"""

import json
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Individual transformers
# Each receives (openai_request: dict, options: dict) and returns the
# (possibly mutated) request dict.
# ---------------------------------------------------------------------------

def _reasoning(req: dict, opts: dict) -> dict:
    """
    Forward Anthropic thinking config to the provider as-is.
    Most OpenAI-compatible providers that support extended thinking
    accept the same `thinking` field.
    If the provider uses a different field (e.g. `enable_thinking`),
    add provider-specific logic here.
    """
    # thinking is already attached by converter.py; nothing extra needed
    # unless provider-specific remapping is required
    return req


def _maxtoken(req: dict, opts: dict) -> dict:
    """
    Apply a max_tokens ceiling/default from config.
    - If the request already has max_tokens, cap it at opts["max_tokens"].
    - If the request has no max_tokens, set it from opts.
    """
    limit = opts.get("max_tokens")
    if limit is None:
        return req
    if req.get("max_tokens") is None:
        req["max_tokens"] = limit
    else:
        req["max_tokens"] = min(req["max_tokens"], limit)
    return req


def _sampling(req: dict, opts: dict) -> dict:
    """
    Override temperature / top_p only when the request does not specify them.
    """
    for field in ("temperature", "top_p"):
        if opts.get(field) is not None and req.get(field) is None:
            req[field] = opts[field]
    return req


def _flatten_message(req: dict, opts: dict) -> dict:
    """
    Convert single-item content arrays to plain strings for providers that
    do not support the array form of message content.
    """
    new_messages = []
    for msg in req.get("messages", []):
        content = msg.get("content")
        if isinstance(content, list):
            # Flatten if every part is a plain text block
            if all(
                isinstance(p, dict) and p.get("type") == "text"
                for p in content
            ):
                msg = dict(msg)  # shallow copy to avoid mutating original
                msg["content"] = "".join(p["text"] for p in content)
        new_messages.append(msg)
    req["messages"] = new_messages
    return req


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Callable[[dict, dict], dict]] = {
    "reasoning":     _reasoning,
    "maxtoken":      _maxtoken,
    "sampling":      _sampling,
    "flattenMessage": _flatten_message,
    "flatten-message": _flatten_message,  # alias
}


# ---------------------------------------------------------------------------
# Pipeline builder & runner
# ---------------------------------------------------------------------------

def build_pipeline(use_list: list) -> list[tuple[Callable, dict]]:
    """
    Parse the 'use' list from provider config into a list of
    (transformer_fn, options) pairs.
    """
    pipeline: list[tuple[Callable, dict]] = []
    for entry in use_list:
        if isinstance(entry, str):
            name, opts = entry, {}
        elif isinstance(entry, list) and len(entry) >= 1:
            name = entry[0]
            opts = entry[1] if len(entry) > 1 and isinstance(entry[1], dict) else {}
        else:
            logger.warning("Unknown transformer entry: %r", entry)
            continue

        fn = _REGISTRY.get(name)
        if fn is None:
            logger.warning("Unknown transformer: %r — skipping", name)
            continue
        pipeline.append((fn, opts))
    return pipeline


def apply_pipeline(pipeline: list[tuple[Callable, dict]], req: dict) -> dict:
    for fn, opts in pipeline:
        req = fn(req, opts)
    return req
