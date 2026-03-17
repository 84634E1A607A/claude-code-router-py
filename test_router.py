"""
Comprehensive tests for the Python Claude Code Router.

Run:
    python test_router.py            # all tests (unit + integration if reachable)
    python test_router.py unit       # unit tests only (no network)
    python test_router.py integration
"""

import asyncio
import json
import os
import sys
import unittest

from converter import anthropic_to_openai, openai_to_anthropic, stream_openai_to_anthropic
from config import apply_provider_params, load_config, resolve_route, get_provider
from batch import (
    anthropic_batch_to_openai_jsonl,
    openai_batch_to_anthropic,
    openai_results_line_to_anthropic,
)

PROVIDER_URL   = os.environ.get("PROVIDER_URL",   "http://172.27.10.48:8000/v1/chat/completions")
PROVIDER_KEY   = os.environ.get("PROVIDER_KEY",   "EMPTY")
PROVIDER_MODEL = os.environ.get("PROVIDER_MODEL", "/model")
CONFIG_PATH    = os.environ.get("CCR_CONFIG",     "config.json")


# ============================================================================
# Helpers
# ============================================================================

def _check_provider_reachable():
    import socket
    from urllib.parse import urlparse
    p = urlparse(PROVIDER_URL)
    try:
        with socket.create_connection((p.hostname, p.port or 80), timeout=2):
            return True
    except OSError:
        return False


def _requires_provider(test):
    import functools
    @functools.wraps(test)
    async def wrapper(self, *a, **kw):
        if not _check_provider_reachable():
            self.skipTest(f"Provider not reachable: {PROVIDER_URL}")
        return await test(self, *a, **kw)
    return wrapper


# ============================================================================
# Unit — converter: Anthropic → OpenAI
# ============================================================================

class TestAnthropicToOpenAI(unittest.TestCase):

    # ── basic ────────────────────────────────────────────────────────────────

    def test_simple_text(self):
        out = anthropic_to_openai({"model": "m", "max_tokens": 1024,
                                   "messages": [{"role": "user", "content": "Hi"}]})
        self.assertEqual(out["messages"], [{"role": "user", "content": "Hi"}])
        self.assertEqual(out["max_tokens"], 1024)

    def test_system_string(self):
        out = anthropic_to_openai({"model": "m", "messages": [], "system": "Be helpful."})
        self.assertEqual(out["messages"][0], {"role": "system", "content": "Be helpful."})

    def test_system_array(self):
        out = anthropic_to_openai({"model": "m", "messages": [], "system": [
            {"type": "text", "text": "Part 1"}, {"type": "text", "text": "Part 2"}]})
        self.assertEqual(out["messages"][0]["content"], "Part 1\nPart 2")

    def test_stop_sequences(self):
        out = anthropic_to_openai({"model": "m", "messages": [],
                                   "stop_sequences": ["STOP", "END"]})
        self.assertEqual(out["stop"], ["STOP", "END"])

    def test_sampling_passthrough(self):
        out = anthropic_to_openai({"model": "m", "messages": [],
                                   "temperature": 0.5, "top_p": 0.9,
                                   "top_k": 40, "max_tokens": 512})
        self.assertEqual(out["temperature"], 0.5)
        self.assertEqual(out["top_p"], 0.9)
        self.assertEqual(out["top_k"], 40)
        self.assertEqual(out["max_tokens"], 512)

    def test_stream_options_added(self):
        out = anthropic_to_openai({"model": "m", "messages": [], "stream": True})
        self.assertTrue(out["stream"])
        self.assertEqual(out["stream_options"], {"include_usage": True})

    def test_no_stream_options_without_stream(self):
        out = anthropic_to_openai({"model": "m", "messages": []})
        self.assertNotIn("stream_options", out)

    # ── content blocks ───────────────────────────────────────────────────────

    def test_tool_use_in_assistant(self):
        out = anthropic_to_openai({"model": "m", "messages": [
            {"role": "assistant", "content": [
                {"type": "text", "text": "Let me check"},
                {"type": "tool_use", "id": "tu_1", "name": "fn", "input": {"x": 1}},
            ]}
        ]})
        msg = out["messages"][0]
        self.assertEqual(msg["content"], "Let me check")
        tc = msg["tool_calls"][0]
        self.assertEqual(tc["id"], "tu_1")
        self.assertEqual(json.loads(tc["function"]["arguments"]), {"x": 1})

    def test_tool_result_in_user(self):
        out = anthropic_to_openai({"model": "m", "messages": [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": "Result"}
            ]}
        ]})
        msg = out["messages"][0]
        self.assertEqual(msg["role"], "tool")
        self.assertEqual(msg["tool_call_id"], "tu_1")
        self.assertEqual(msg["content"], "Result")

    def test_mixed_user_content(self):
        out = anthropic_to_openai({"model": "m", "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "See result:"},
                {"type": "tool_result", "tool_use_id": "tu_1", "content": "42"},
            ]}
        ]})
        roles = [m["role"] for m in out["messages"]]
        self.assertIn("user", roles)
        self.assertIn("tool", roles)

    def test_thinking_skipped_in_history(self):
        out = anthropic_to_openai({"model": "m", "messages": [
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "internal", "signature": "s"},
                {"type": "text", "text": "Answer"},
            ]}
        ]})
        msg = out["messages"][0]
        self.assertEqual(msg["content"], "Answer")
        self.assertNotIn("tool_calls", msg)

    def test_image_base64(self):
        out = anthropic_to_openai({"model": "m", "messages": [
            {"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                                             "media_type": "image/png", "data": "abc=="}}
            ]}
        ]})
        part = out["messages"][0]["content"][0]
        self.assertEqual(part["type"], "image_url")
        self.assertEqual(part["image_url"]["url"], "data:image/png;base64,abc==")

    def test_image_url(self):
        out = anthropic_to_openai({"model": "m", "messages": [
            {"role": "user", "content": [
                {"type": "image", "source": {"type": "url", "url": "https://x.com/img.png"}}
            ]}
        ]})
        self.assertEqual(out["messages"][0]["content"][0]["image_url"]["url"],
                         "https://x.com/img.png")

    # ── tools ────────────────────────────────────────────────────────────────

    def test_tools_conversion(self):
        out = anthropic_to_openai({"model": "m", "messages": [], "tools": [
            {"name": "search", "description": "Web search",
             "input_schema": {"type": "object",
                              "properties": {"q": {"type": "string"}},
                              "required": ["q"]}}
        ]})
        fn = out["tools"][0]["function"]
        self.assertEqual(fn["name"], "search")
        self.assertIn("properties", fn["parameters"])

    def test_tool_strict_true(self):
        out = anthropic_to_openai({"model": "m", "messages": [], "tools": [
            {"name": "fn", "description": "", "input_schema": {"type": "object"}, "strict": True}
        ]})
        self.assertTrue(out["tools"][0]["function"]["strict"])

    def test_tool_strict_absent_omitted(self):
        out = anthropic_to_openai({"model": "m", "messages": [], "tools": [
            {"name": "fn", "description": "", "input_schema": {"type": "object"}}
        ]})
        self.assertNotIn("strict", out["tools"][0]["function"])

    # ── tool_choice ──────────────────────────────────────────────────────────

    def test_tool_choice_auto(self):
        out = anthropic_to_openai({"model": "m", "messages": [],
                                   "tool_choice": {"type": "auto"}})
        self.assertEqual(out["tool_choice"], "auto")

    def test_tool_choice_any(self):
        out = anthropic_to_openai({"model": "m", "messages": [],
                                   "tool_choice": {"type": "any"}})
        self.assertEqual(out["tool_choice"], "required")

    def test_tool_choice_tool(self):
        out = anthropic_to_openai({"model": "m", "messages": [],
                                   "tool_choice": {"type": "tool", "name": "fn"}})
        self.assertEqual(out["tool_choice"]["function"]["name"], "fn")

    def test_tool_choice_none(self):
        out = anthropic_to_openai({"model": "m", "messages": [],
                                   "tool_choice": {"type": "none"}})
        self.assertEqual(out["tool_choice"], "none")

    def test_disable_parallel_tool_use(self):
        out = anthropic_to_openai({"model": "m", "messages": [],
                                   "tool_choice": {"type": "auto",
                                                   "disable_parallel_tool_use": True}})
        self.assertFalse(out["parallel_tool_calls"])

    def test_parallel_tool_use_not_set_by_default(self):
        out = anthropic_to_openai({"model": "m", "messages": [],
                                   "tool_choice": {"type": "auto"}})
        self.assertNotIn("parallel_tool_calls", out)

    # ── thinking ─────────────────────────────────────────────────────────────

    def test_thinking_enabled(self):
        out = anthropic_to_openai({"model": "m", "messages": [],
                                   "thinking": {"type": "enabled", "budget_tokens": 5000}})
        self.assertEqual(out["thinking"], {"type": "enabled", "budget_tokens": 5000})

    def test_thinking_adaptive(self):
        out = anthropic_to_openai({"model": "m", "messages": [],
                                   "thinking": {"type": "adaptive"}})
        self.assertEqual(out["thinking"], {"type": "adaptive"})

    def test_thinking_disabled_omitted(self):
        out = anthropic_to_openai({"model": "m", "messages": [],
                                   "thinking": {"type": "disabled"}})
        self.assertNotIn("thinking", out)

    # ── output_config ────────────────────────────────────────────────────────

    def test_effort_low(self):
        out = anthropic_to_openai({"model": "m", "messages": [],
                                   "output_config": {"effort": "low"}})
        self.assertEqual(out["reasoning_effort"], "low")

    def test_effort_medium(self):
        out = anthropic_to_openai({"model": "m", "messages": [],
                                   "output_config": {"effort": "medium"}})
        self.assertEqual(out["reasoning_effort"], "medium")

    def test_effort_high(self):
        out = anthropic_to_openai({"model": "m", "messages": [],
                                   "output_config": {"effort": "high"}})
        self.assertEqual(out["reasoning_effort"], "high")

    def test_effort_max_to_xhigh(self):
        out = anthropic_to_openai({"model": "m", "messages": [],
                                   "output_config": {"effort": "max"}})
        self.assertEqual(out["reasoning_effort"], "xhigh")

    def test_output_config_json_schema(self):
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        out = anthropic_to_openai({"model": "m", "messages": [],
                                   "output_config": {"format": {"type": "json_schema",
                                                                 "schema": schema}}})
        self.assertEqual(out["response_format"]["type"], "json_schema")
        self.assertEqual(out["response_format"]["json_schema"], schema)

    def test_output_config_absent(self):
        out = anthropic_to_openai({"model": "m", "messages": []})
        self.assertNotIn("reasoning_effort", out)
        self.assertNotIn("response_format", out)

    # ── metadata ─────────────────────────────────────────────────────────────

    def test_metadata_user_id(self):
        out = anthropic_to_openai({"model": "m", "messages": [],
                                   "metadata": {"user_id": "user-abc"}})
        self.assertEqual(out["user"], "user-abc")

    def test_metadata_absent(self):
        out = anthropic_to_openai({"model": "m", "messages": []})
        self.assertNotIn("user", out)


# ============================================================================
# Unit — converter: OpenAI → Anthropic (non-streaming)
# ============================================================================

class TestOpenAIToAnthropic(unittest.TestCase):

    def _wrap(self, message, finish_reason="stop", usage=None):
        return {
            "id": "chatcmpl-test",
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5},
        }

    def test_simple_text(self):
        out = openai_to_anthropic(self._wrap({"role": "assistant", "content": "Hi!"}), "m")
        self.assertEqual(out["type"], "message")
        self.assertEqual(out["role"], "assistant")
        self.assertEqual(out["content"][0], {"type": "text", "text": "Hi!"})
        self.assertEqual(out["stop_reason"], "end_turn")

    def test_response_schema_complete(self):
        out = openai_to_anthropic(self._wrap({"role": "assistant", "content": "x"}), "mymodel")
        for f in ("id", "type", "role", "model", "content",
                  "stop_reason", "stop_sequence", "usage"):
            self.assertIn(f, out)
        self.assertEqual(out["model"], "mymodel")
        self.assertIsNone(out["stop_sequence"])

    def test_tool_call(self):
        out = openai_to_anthropic(self._wrap({
            "role": "assistant", "content": None,
            "tool_calls": [{"id": "call_1", "type": "function",
                            "function": {"name": "search",
                                         "arguments": '{"query":"python"}'}}],
        }, finish_reason="tool_calls"), "m")
        self.assertEqual(out["stop_reason"], "tool_use")
        b = out["content"][0]
        self.assertEqual(b["type"], "tool_use")
        self.assertEqual(b["id"], "call_1")
        self.assertEqual(b["input"], {"query": "python"})

    def test_finish_reason_all(self):
        for fr, expected in [("stop", "end_turn"), ("length", "max_tokens"),
                              ("tool_calls", "tool_use"), ("content_filter", "stop_sequence"),
                              ("unknown_val", "end_turn")]:
            out = openai_to_anthropic(
                self._wrap({"role": "assistant", "content": "x"}, finish_reason=fr), "m")
            self.assertEqual(out["stop_reason"], expected, fr)

    def test_usage_basic(self):
        out = openai_to_anthropic(
            self._wrap({"role": "assistant", "content": "x"},
                       usage={"prompt_tokens": 100, "completion_tokens": 50}), "m")
        self.assertEqual(out["usage"]["input_tokens"], 100)
        self.assertEqual(out["usage"]["output_tokens"], 50)
        self.assertEqual(out["usage"]["cache_read_input_tokens"], 0)
        self.assertEqual(out["usage"]["cache_creation_input_tokens"], 0)

    def test_usage_cache_read(self):
        out = openai_to_anthropic(self._wrap(
            {"role": "assistant", "content": "x"},
            usage={"prompt_tokens": 100, "completion_tokens": 50,
                   "prompt_tokens_details": {"cached_tokens": 40}}), "m")
        self.assertEqual(out["usage"]["input_tokens"], 60)
        self.assertEqual(out["usage"]["cache_read_input_tokens"], 40)

    def test_usage_cache_creation(self):
        out = openai_to_anthropic(self._wrap(
            {"role": "assistant", "content": "x"},
            usage={"prompt_tokens": 100, "completion_tokens": 50,
                   "prompt_tokens_details": {
                       "cached_tokens": 20, "cache_creation_tokens": 30}}), "m")
        self.assertEqual(out["usage"]["input_tokens"], 50)
        self.assertEqual(out["usage"]["cache_read_input_tokens"], 20)
        self.assertEqual(out["usage"]["cache_creation_input_tokens"], 30)

    def test_reasoning_content(self):
        out = openai_to_anthropic(self._wrap(
            {"role": "assistant", "content": "Answer",
             "reasoning_content": "thinking..."}), "m")
        blocks = {b["type"]: b for b in out["content"]}
        self.assertIn("thinking", blocks)
        self.assertEqual(blocks["thinking"]["thinking"], "thinking...")

    def test_thinking_object(self):
        out = openai_to_anthropic(self._wrap(
            {"role": "assistant", "content": "Answer",
             "thinking": {"content": "deep thought", "signature": "sig123"}}), "m")
        blocks = {b["type"]: b for b in out["content"]}
        self.assertEqual(blocks["thinking"]["signature"], "sig123")

    def test_invalid_tool_args_fallback(self):
        out = openai_to_anthropic(self._wrap(
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c", "type": "function",
                             "function": {"name": "fn", "arguments": "bad-json"}}]},
            finish_reason="tool_calls"), "m")
        self.assertIsInstance(out["content"][0]["input"], dict)


# ============================================================================
# Unit — streaming converter
# ============================================================================

class TestStreamConverter(unittest.IsolatedAsyncioTestCase):

    async def _collect(self, chunks):
        async def fake_stream():
            for c in chunks:
                yield c
        events = []
        async for sse in stream_openai_to_anthropic(fake_stream(), "msg_test", "m"):
            for line in sse.strip().split("\n"):
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
        return events

    def _chunk(self, delta, finish_reason=None, usage=None):
        d = {"id": "x", "choices": [{"index": 0, "delta": delta,
                                      "finish_reason": finish_reason}]}
        if usage:
            d["usage"] = usage
        return b"data: " + json.dumps(d).encode()

    async def test_simple_text(self):
        events = await self._collect([
            self._chunk({"content": "Hello"}),
            self._chunk({"content": " world"}),
            self._chunk({}, finish_reason="stop",
                        usage={"prompt_tokens": 10, "completion_tokens": 5}),
            b"data: [DONE]",
        ])
        types = [e["type"] for e in events]
        for t in ("message_start", "content_block_start", "content_block_delta",
                  "content_block_stop", "message_delta", "message_stop"):
            self.assertIn(t, types)
        text = "".join(e["delta"]["text"] for e in events
                       if e["type"] == "content_block_delta"
                       and e["delta"].get("type") == "text_delta")
        self.assertEqual(text, "Hello world")

    async def test_tool_call_stream(self):
        events = await self._collect([
            self._chunk({"tool_calls": [{"index": 0, "id": "call_abc", "type": "function",
                                          "function": {"name": "search", "arguments": ""}}]}),
            self._chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"q"'}}]}),
            self._chunk({"tool_calls": [{"index": 0, "function": {"arguments": ':"hi"}'}}]}),
            self._chunk({}, finish_reason="tool_calls",
                        usage={"prompt_tokens": 20, "completion_tokens": 10}),
            b"data: [DONE]",
        ])
        tool_start = next(e for e in events
                          if e["type"] == "content_block_start"
                          and e["content_block"]["type"] == "tool_use")
        self.assertEqual(tool_start["content_block"]["id"], "call_abc")
        args = "".join(e["delta"]["partial_json"] for e in events
                       if e["type"] == "content_block_delta"
                       and e["delta"].get("type") == "input_json_delta")
        self.assertEqual(json.loads(args), {"q": "hi"})

    async def test_thinking_stream(self):
        events = await self._collect([
            self._chunk({"thinking": {"content": "Let me think..."}}),
            self._chunk({"thinking": {"signature": "sig_abc"}}),
            self._chunk({"content": "Answer"}),
            self._chunk({}, finish_reason="stop"),
            b"data: [DONE]",
        ])
        block_types = [e["content_block"]["type"]
                       for e in events if e["type"] == "content_block_start"]
        self.assertIn("thinking", block_types)
        self.assertIn("text", block_types)
        sigs = [e for e in events
                if e["type"] == "content_block_delta"
                and e["delta"].get("type") == "signature_delta"]
        self.assertEqual(sigs[0]["delta"]["signature"], "sig_abc")

    async def test_reasoning_content_stream(self):
        events = await self._collect([
            self._chunk({"reasoning_content": "thinking step"}),
            self._chunk({"content": "final answer"}),
            self._chunk({}, finish_reason="stop"),
            b"data: [DONE]",
        ])
        block_types = [e["content_block"]["type"]
                       for e in events if e["type"] == "content_block_start"]
        self.assertIn("thinking", block_types)
        self.assertIn("text", block_types)

    async def test_stream_protocol_order(self):
        events = await self._collect([
            self._chunk({"content": "hi"}),
            self._chunk({}, finish_reason="stop",
                        usage={"prompt_tokens": 5, "completion_tokens": 2}),
            b"data: [DONE]",
        ])
        types = [e["type"] for e in events]
        self.assertEqual(types[0], "message_start")
        self.assertLess(types.index("content_block_start"), types.index("content_block_stop"))
        self.assertLess(types.index("content_block_stop"), types.index("message_delta"))
        self.assertEqual(types[-1], "message_stop")

    async def test_stream_usage_fields(self):
        events = await self._collect([
            self._chunk({"content": "hi"}),
            self._chunk({}, finish_reason="stop",
                        usage={"prompt_tokens": 100, "completion_tokens": 42,
                               "prompt_tokens_details": {
                                   "cached_tokens": 20, "cache_creation_tokens": 10}}),
            b"data: [DONE]",
        ])
        u = next(e for e in events if e["type"] == "message_delta")["usage"]
        self.assertEqual(u["input_tokens"], 70)
        self.assertEqual(u["output_tokens"], 42)
        self.assertEqual(u["cache_read_input_tokens"], 20)
        self.assertEqual(u["cache_creation_input_tokens"], 10)

    async def test_max_tokens_stop_reason(self):
        events = await self._collect([
            self._chunk({"content": "truncated"}),
            self._chunk({}, finish_reason="length"),
            b"data: [DONE]",
        ])
        msg_delta = next(e for e in events if e["type"] == "message_delta")
        self.assertEqual(msg_delta["delta"]["stop_reason"], "max_tokens")

    async def test_malformed_line_skipped(self):
        events = await self._collect([
            b"data: not-json",
            self._chunk({"content": "ok"}),
            self._chunk({}, finish_reason="stop"),
            b"data: [DONE]",
        ])
        self.assertIn("message_stop", [e["type"] for e in events])

    async def test_empty_stream(self):
        events = await self._collect([
            self._chunk({}, finish_reason="stop",
                        usage={"prompt_tokens": 5, "completion_tokens": 0}),
            b"data: [DONE]",
        ])
        types = [e["type"] for e in events]
        self.assertIn("message_start", types)
        self.assertIn("message_stop", types)


# ============================================================================
# Unit — config + apply_provider_params
# ============================================================================

class TestConfig(unittest.TestCase):

    def test_resolve_route_default(self):
        self.assertEqual(resolve_route({"Router": {"default": "p,m"}}), ("p", "m"))

    def test_resolve_route_scenario(self):
        cfg = {"Router": {"default": "p,m1", "think": "p,m2"}}
        self.assertEqual(resolve_route(cfg, "think"), ("p", "m2"))

    def test_resolve_route_fallback(self):
        self.assertEqual(resolve_route({"Router": {"default": "p,m"}}, "missing"), ("p", "m"))

    def test_resolve_route_no_router(self):
        self.assertIsNone(resolve_route({}))

    def test_get_provider_found(self):
        p = get_provider({"Providers": [{"name": "foo", "api_key": "k"}]}, "foo")
        self.assertEqual(p["api_key"], "k")

    def test_get_provider_missing(self):
        self.assertIsNone(get_provider({"Providers": []}, "x"))


class TestApplyProviderParams(unittest.TestCase):

    def _p(self, params):
        return {"params": params}

    def test_temperature_default(self):
        out = apply_provider_params(self._p({"temperature": 0.7}), {"model": "m", "messages": []})
        self.assertEqual(out["temperature"], 0.7)

    def test_temperature_not_overridden(self):
        out = apply_provider_params(self._p({"temperature": 0.7}),
                                    {"model": "m", "messages": [], "temperature": 1.0})
        self.assertEqual(out["temperature"], 1.0)

    def test_max_tokens_default(self):
        out = apply_provider_params(self._p({"max_tokens": 4096}), {"model": "m", "messages": []})
        self.assertEqual(out["max_tokens"], 4096)

    def test_max_tokens_cap(self):
        out = apply_provider_params(self._p({"max_tokens": 4096}),
                                    {"model": "m", "messages": [], "max_tokens": 16384})
        self.assertEqual(out["max_tokens"], 4096)

    def test_max_tokens_respects_smaller(self):
        out = apply_provider_params(self._p({"max_tokens": 4096}),
                                    {"model": "m", "messages": [], "max_tokens": 512})
        self.assertEqual(out["max_tokens"], 512)

    def test_reasoning_injects_thinking(self):
        out = apply_provider_params(self._p({"reasoning": {"budget_tokens": 5000}}),
                                    {"model": "m", "messages": []})
        self.assertEqual(out["thinking"]["type"], "enabled")
        self.assertEqual(out["thinking"]["budget_tokens"], 5000)

    def test_reasoning_no_override(self):
        out = apply_provider_params(
            self._p({"reasoning": {"budget_tokens": 5000}}),
            {"model": "m", "messages": [],
             "thinking": {"type": "enabled", "budget_tokens": 1000}})
        self.assertEqual(out["thinking"]["budget_tokens"], 1000)

    def test_empty_params_noop(self):
        req = {"model": "m", "messages": [], "temperature": 0.5}
        out = apply_provider_params({"params": {}}, req)
        self.assertEqual(out["temperature"], 0.5)


# ============================================================================
# Unit — batch conversion
# ============================================================================

class TestBatchConversion(unittest.TestCase):

    def _batch_requests(self):
        return [
            {"custom_id": "req-1", "params": {
                "model": "claude-3-5-sonnet", "max_tokens": 128,
                "messages": [{"role": "user", "content": "Hello"}],
            }},
            {"custom_id": "req-2", "params": {
                "model": "claude-3-5-sonnet", "max_tokens": 64,
                "messages": [{"role": "user", "content": "World"}],
                "system": "Be brief.",
            }},
        ]

    def test_anthropic_batch_to_openai_jsonl(self):
        jsonl = anthropic_batch_to_openai_jsonl(self._batch_requests(), "/model")
        lines = [json.loads(l) for l in jsonl.strip().split("\n")]
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["custom_id"], "req-1")
        self.assertEqual(lines[0]["method"], "POST")
        self.assertEqual(lines[0]["url"], "/v1/chat/completions")
        self.assertEqual(lines[0]["body"]["model"], "/model")
        # system in second request
        msgs = lines[1]["body"]["messages"]
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[0]["content"], "Be brief.")

    def test_batch_no_stream_field(self):
        jsonl = anthropic_batch_to_openai_jsonl(self._batch_requests(), "/model")
        for line in jsonl.strip().split("\n"):
            item = json.loads(line)
            self.assertNotIn("stream", item["body"])

    def test_openai_batch_to_anthropic_in_progress(self):
        ob = {
            "id": "batch_abc",
            "status": "in_progress",
            "created_at": 1700000000,
            "expires_at": 1700086400,
            "request_counts": {"total": 10, "completed": 3, "failed": 1},
        }
        out = openai_batch_to_anthropic(ob, "http://localhost:3456")
        self.assertEqual(out["id"], "batch_abc")
        self.assertEqual(out["type"], "message_batch")
        self.assertEqual(out["processing_status"], "in_progress")
        self.assertIsNone(out["results_url"])
        self.assertEqual(out["request_counts"]["processing"], 6)
        self.assertEqual(out["request_counts"]["succeeded"], 3)
        self.assertEqual(out["request_counts"]["errored"], 1)

    def test_openai_batch_to_anthropic_completed(self):
        ob = {
            "id": "batch_xyz",
            "status": "completed",
            "created_at": 1700000000,
            "expires_at": 1700086400,
            "completed_at": 1700003600,
            "output_file_id": "file_abc",
            "request_counts": {"total": 5, "completed": 5, "failed": 0},
        }
        out = openai_batch_to_anthropic(ob, "http://localhost:3456")
        self.assertEqual(out["processing_status"], "ended")
        self.assertIsNotNone(out["results_url"])
        self.assertIn("batch_xyz", out["results_url"])
        self.assertIsNotNone(out["ended_at"])

    def test_openai_batch_to_anthropic_cancelling(self):
        ob = {"id": "b", "status": "cancelling", "created_at": 0, "request_counts": {}}
        out = openai_batch_to_anthropic(ob, "http://localhost")
        self.assertEqual(out["processing_status"], "canceling")

    def test_results_line_succeeded(self):
        openai_line = json.dumps({
            "id": "r1", "custom_id": "req-1",
            "response": {
                "status_code": 200,
                "body": {
                    "id": "chatcmpl-1",
                    "choices": [{"index": 0,
                                 "message": {"role": "assistant", "content": "Hi!"},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2},
                },
            },
            "error": None,
        })
        out = json.loads(openai_results_line_to_anthropic(openai_line, "/model"))
        self.assertEqual(out["custom_id"], "req-1")
        self.assertEqual(out["result"]["type"], "succeeded")
        self.assertEqual(out["result"]["message"]["type"], "message")

    def test_results_line_error(self):
        openai_line = json.dumps({
            "id": "r2", "custom_id": "req-2",
            "response": {"status_code": 500, "body": {"error": {"message": "Internal error"}}},
            "error": None,
        })
        out = json.loads(openai_results_line_to_anthropic(openai_line, "/model"))
        self.assertEqual(out["result"]["type"], "errored")
        self.assertIn("error", out["result"])

    def test_results_line_network_error(self):
        openai_line = json.dumps({
            "id": "r3", "custom_id": "req-3",
            "response": None,
            "error": "connection timeout",
        })
        out = json.loads(openai_results_line_to_anthropic(openai_line, "/model"))
        self.assertEqual(out["result"]["type"], "errored")

    def test_results_line_empty(self):
        self.assertIsNone(openai_results_line_to_anthropic("", "/model"))
        self.assertIsNone(openai_results_line_to_anthropic("   ", "/model"))

    def test_results_line_bad_json(self):
        self.assertIsNone(openai_results_line_to_anthropic("not-json", "/model"))


# ============================================================================
# Unit — legacy completions prompt parsing
# ============================================================================

class TestLegacyPromptParsing(unittest.TestCase):

    def _parse(self, prompt):
        from server import _parse_legacy_prompt
        return _parse_legacy_prompt(prompt)

    def test_single_human_turn(self):
        msgs = self._parse("\n\nHuman: Hello!\n\nAssistant:")
        self.assertEqual(msgs[0]["role"], "user")
        self.assertIn("Hello!", msgs[0]["content"])

    def test_multi_turn(self):
        msgs = self._parse(
            "\n\nHuman: Hi\n\nAssistant: Hello\n\nHuman: How are you?\n\nAssistant:"
        )
        roles = [m["role"] for m in msgs]
        self.assertEqual(roles[0], "user")
        self.assertEqual(roles[1], "assistant")
        self.assertEqual(roles[2], "user")

    def test_plain_fallback(self):
        msgs = self._parse("Just a plain prompt")
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["role"], "user")
        self.assertEqual(msgs[0]["content"], "Just a plain prompt")


# ============================================================================
# Unit — URL derivation helpers
# ============================================================================

class TestURLHelpers(unittest.TestCase):

    def _prov(self, url):
        return {"api_base_url": url}

    def _base(self, url):
        from server import _api_base
        return _api_base(self._prov(url))

    def test_chat_completions_url(self):
        from server import _batches_url, _models_url, _files_url
        p = self._prov("http://host:8000/v1/chat/completions")
        self.assertEqual(_models_url(p), "http://host:8000/v1/models")
        self.assertEqual(_batches_url(p), "http://host:8000/v1/batches")
        self.assertEqual(_files_url(p), "http://host:8000/v1/files")

    def test_completions_url(self):
        from server import _models_url
        p = self._prov("http://host:8000/v1/completions")
        self.assertEqual(_models_url(p), "http://host:8000/v1/models")

    def test_already_models_url(self):
        from server import _batches_url
        p = self._prov("http://host:8000/v1/models")
        self.assertEqual(_batches_url(p), "http://host:8000/v1/batches")


# ============================================================================
# Integration tests
# ============================================================================

class IntegrationBase(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        import server as srv_mod
        srv_mod.set_config({
            "API_TIMEOUT_MS": 60000,
            "Providers": [{
                "name": "test",
                "api_base_url": PROVIDER_URL,
                "api_key": PROVIDER_KEY,
                "max_retries": 1,
            }],
            "Router": {"default": f"test,{PROVIDER_MODEL}"},
        })
        from httpx import ASGITransport, AsyncClient
        self.client = AsyncClient(
            transport=ASGITransport(app=srv_mod.app),
            base_url="http://test",
            timeout=60.0,
        )

    async def asyncTearDown(self):
        await self.client.aclose()

    def _req(self, extra=None):
        r = {"model": "ignored", "max_tokens": 128,
             "messages": [{"role": "user", "content": "Reply with exactly: PONG"}]}
        if extra:
            r.update(extra)
        return r

    async def _stream_events(self, req):
        resp = await self.client.post(
            "/v1/messages", json={**req, "stream": True},
            headers={"Accept": "text/event-stream"})
        events = []
        for line in resp.text.splitlines():
            if line.startswith("data: "):
                try:
                    events.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass
        return events, resp.status_code


class TestIntegrationMessages(IntegrationBase):

    @_requires_provider
    async def test_basic(self):
        resp = await self.client.post("/v1/messages", json=self._req())
        self.assertEqual(resp.status_code, 200, resp.text)
        data = resp.json()
        self.assertEqual(data["type"], "message")
        self.assertEqual(data["role"], "assistant")
        text = " ".join(b["text"] for b in data["content"] if b["type"] == "text")
        self.assertIn("PONG", text)

    @_requires_provider
    async def test_response_schema(self):
        data = (await self.client.post("/v1/messages", json=self._req())).json()
        for f in ("id", "type", "role", "model", "content", "stop_reason", "usage"):
            self.assertIn(f, data)
        for f in ("input_tokens", "output_tokens",
                  "cache_read_input_tokens", "cache_creation_input_tokens"):
            self.assertIn(f, data["usage"])
        self.assertGreater(data["usage"]["input_tokens"], 0)
        self.assertGreater(data["usage"]["output_tokens"], 0)

    @_requires_provider
    async def test_system_prompt(self):
        data = (await self.client.post("/v1/messages", json=self._req(
            {"system": "Always reply: HELLO_WORLD",
             "messages": [{"role": "user", "content": "Hi"}]}))).json()
        text = " ".join(b["text"] for b in data["content"] if b["type"] == "text")
        self.assertIn("HELLO_WORLD", text)

    @_requires_provider
    async def test_multi_turn(self):
        data = (await self.client.post("/v1/messages", json={
            "model": "ignored", "max_tokens": 128,
            "messages": [
                {"role": "user", "content": "Say: FIRST"},
                {"role": "assistant", "content": "FIRST"},
                {"role": "user", "content": "Now say: SECOND"},
            ]})).json()
        text = " ".join(b["text"] for b in data["content"] if b["type"] == "text")
        self.assertIn("SECOND", text)

    @_requires_provider
    async def test_tool_use(self):
        req = {"model": "ignored", "max_tokens": 256,
               "tools": [{"name": "get_number", "description": "Returns a number",
                           "input_schema": {"type": "object",
                                            "properties": {"n": {"type": "integer"}},
                                            "required": ["n"]}}],
               "tool_choice": {"type": "any"},
               "messages": [{"role": "user", "content": "Call get_number with n=42"}]}
        data = (await self.client.post("/v1/messages", json=req)).json()
        tool_blocks = [b for b in data["content"] if b["type"] == "tool_use"]
        self.assertTrue(len(tool_blocks) > 0)
        self.assertEqual(tool_blocks[0]["name"], "get_number")
        self.assertEqual(tool_blocks[0]["input"].get("n"), 42)

    @_requires_provider
    async def test_tool_choice_none(self):
        req = {"model": "ignored", "max_tokens": 128,
               "tools": [{"name": "fn", "description": "A tool",
                           "input_schema": {"type": "object"}}],
               "tool_choice": {"type": "none"},
               "messages": [{"role": "user", "content": "Hello"}]}
        data = (await self.client.post("/v1/messages", json=req)).json()
        self.assertEqual([b for b in data["content"] if b["type"] == "tool_use"], [])

    @_requires_provider
    async def test_sampling_params(self):
        resp = await self.client.post("/v1/messages",
                                      json=self._req({"temperature": 0.0, "top_p": 1.0}))
        self.assertEqual(resp.status_code, 200)

    @_requires_provider
    async def test_stop_sequences(self):
        req = {"model": "ignored", "max_tokens": 64,
               "stop_sequences": ["STOP"],
               "messages": [{"role": "user", "content": "Count: 1 2 3 STOP 4 5"}]}
        resp = await self.client.post("/v1/messages", json=req)
        self.assertEqual(resp.status_code, 200)

    @_requires_provider
    async def test_metadata_user_id(self):
        req = self._req({"metadata": {"user_id": "test-user-001"}})
        resp = await self.client.post("/v1/messages", json=req)
        self.assertEqual(resp.status_code, 200)


class TestIntegrationStreaming(IntegrationBase):

    @_requires_provider
    async def test_basic_streaming(self):
        events, status = await self._stream_events(self._req())
        self.assertEqual(status, 200)
        for t in ("message_start", "content_block_start", "content_block_delta",
                  "content_block_stop", "message_delta", "message_stop"):
            self.assertIn(t, [e["type"] for e in events])

    @_requires_provider
    async def test_stream_text(self):
        events, status = await self._stream_events(self._req())
        text = "".join(e["delta"]["text"] for e in events
                       if e["type"] == "content_block_delta"
                       and e["delta"].get("type") == "text_delta")
        self.assertIn("PONG", text)

    @_requires_provider
    async def test_stream_usage(self):
        events, _ = await self._stream_events(self._req())
        u = next(e for e in events if e["type"] == "message_delta")["usage"]
        for f in ("input_tokens", "output_tokens",
                  "cache_read_input_tokens", "cache_creation_input_tokens"):
            self.assertIn(f, u)
        self.assertGreater(u["output_tokens"], 0)

    @_requires_provider
    async def test_stream_stop_reason(self):
        events, _ = await self._stream_events(self._req())
        md = next(e for e in events if e["type"] == "message_delta")
        self.assertEqual(md["delta"]["stop_reason"], "end_turn")

    @_requires_provider
    async def test_stream_protocol_order(self):
        events, status = await self._stream_events(self._req())
        types = [e["type"] for e in events]
        self.assertEqual(types[0], "message_start")
        self.assertLess(types.index("content_block_start"), types.index("content_block_stop"))
        self.assertLess(types.index("content_block_stop"), types.index("message_delta"))
        self.assertEqual(types[-1], "message_stop")

    @_requires_provider
    async def test_stream_tool_use(self):
        req = {"model": "ignored", "max_tokens": 256,
               "tools": [{"name": "get_number", "description": "Returns a number",
                           "input_schema": {"type": "object",
                                            "properties": {"n": {"type": "integer"}},
                                            "required": ["n"]}}],
               "tool_choice": {"type": "any"},
               "messages": [{"role": "user", "content": "Call get_number with n=7"}]}
        events, status = await self._stream_events(req)
        self.assertEqual(status, 200)
        tool_starts = [e for e in events
                       if e["type"] == "content_block_start"
                       and e["content_block"]["type"] == "tool_use"]
        self.assertTrue(len(tool_starts) > 0)
        args = "".join(e["delta"]["partial_json"] for e in events
                       if e["type"] == "content_block_delta"
                       and e["delta"].get("type") == "input_json_delta")
        self.assertEqual(json.loads(args).get("n"), 7)

    @_requires_provider
    async def test_stream_multi_turn(self):
        req = {"model": "ignored", "max_tokens": 128, "messages": [
            {"role": "user", "content": "Say: FIRST"},
            {"role": "assistant", "content": "FIRST"},
            {"role": "user", "content": "Now say: SECOND"},
        ]}
        events, status = await self._stream_events(req)
        text = "".join(e["delta"]["text"] for e in events
                       if e["type"] == "content_block_delta"
                       and e["delta"].get("type") == "text_delta")
        self.assertIn("SECOND", text)


class TestIntegrationModels(IntegrationBase):

    @_requires_provider
    async def test_list_models(self):
        resp = await self.client.get("/v1/models")
        self.assertEqual(resp.status_code, 200, resp.text)
        data = resp.json()
        self.assertIn("data", data)
        self.assertIn("has_more", data)
        if data["data"]:
            m = data["data"][0]
            for f in ("id", "type", "display_name", "created_at"):
                self.assertIn(f, m)
            self.assertEqual(m["type"], "model")

    @_requires_provider
    async def test_get_model(self):
        # First get list to find a valid model id
        list_resp = await self.client.get("/v1/models")
        models = list_resp.json().get("data", [])
        if not models:
            self.skipTest("No models returned by provider")
        model_id = models[0]["id"]
        resp = await self.client.get(f"/v1/models/{model_id}")
        self.assertEqual(resp.status_code, 200, resp.text)
        data = resp.json()
        self.assertEqual(data["id"], model_id)
        self.assertEqual(data["type"], "model")


class TestIntegrationLegacyCompletions(IntegrationBase):

    @_requires_provider
    async def test_legacy_complete(self):
        resp = await self.client.post("/v1/complete", json={
            "model": "ignored",
            "prompt": "\n\nHuman: Reply with exactly: PONG\n\nAssistant:",
            "max_tokens_to_sample": 64,
        })
        self.assertEqual(resp.status_code, 200, resp.text)
        data = resp.json()
        self.assertEqual(data["type"], "completion")
        self.assertIn("completion", data)
        self.assertIn("stop_reason", data)
        self.assertIn("PONG", data["completion"])

    @_requires_provider
    async def test_legacy_response_schema(self):
        resp = await self.client.post("/v1/complete", json={
            "model": "ignored",
            "prompt": "\n\nHuman: Hi\n\nAssistant:",
            "max_tokens_to_sample": 32,
            "temperature": 0.0,
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        for f in ("id", "type", "completion", "stop_reason", "model"):
            self.assertIn(f, data)


class TestIntegrationHealth(IntegrationBase):

    async def test_health(self):
        resp = await self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ok")

    async def test_invalid_json(self):
        resp = await self.client.post(
            "/v1/messages",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        self.assertEqual(resp.status_code, 400)


# ============================================================================
# Runner
# ============================================================================

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    if mode in ("unit", "all"):
        for cls in (TestAnthropicToOpenAI, TestOpenAIToAnthropic,
                    TestStreamConverter, TestConfig, TestApplyProviderParams,
                    TestBatchConversion, TestLegacyPromptParsing, TestURLHelpers):
            suite.addTests(loader.loadTestsFromTestCase(cls))

    if mode in ("integration", "all"):
        for cls in (TestIntegrationMessages, TestIntegrationStreaming,
                    TestIntegrationModels, TestIntegrationLegacyCompletions,
                    TestIntegrationHealth):
            suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
