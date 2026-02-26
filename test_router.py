"""
Comprehensive tests for the Python Claude Code Router.

Run:
    python test_router.py                    # all tests
    python test_router.py unit               # unit tests only (no network)
    python test_router.py integration        # integration tests (needs provider)
"""

import asyncio
import json
import sys
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# ─── unit-testable imports ───────────────────────────────────────────────────
from converter import (
    anthropic_to_openai,
    openai_to_anthropic,
    stream_openai_to_anthropic,
)
from transformers import apply_pipeline, build_pipeline
from config import load_config, resolve_route, get_provider


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAnthropicToOpenAI(unittest.TestCase):

    def test_simple_text(self):
        req = {
            "model": "claude-3-5-sonnet",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello!"}],
        }
        out = anthropic_to_openai(req)
        self.assertEqual(out["messages"], [{"role": "user", "content": "Hello!"}])
        self.assertEqual(out["max_tokens"], 1024)

    def test_system_string(self):
        req = {
            "model": "m",
            "messages": [],
            "system": "You are helpful.",
        }
        out = anthropic_to_openai(req)
        self.assertEqual(out["messages"][0], {"role": "system", "content": "You are helpful."})

    def test_system_array(self):
        req = {
            "model": "m",
            "messages": [],
            "system": [
                {"type": "text", "text": "Part 1"},
                {"type": "text", "text": "Part 2"},
            ],
        }
        out = anthropic_to_openai(req)
        self.assertEqual(out["messages"][0]["content"], "Part 1\nPart 2")

    def test_tool_use_in_assistant(self):
        req = {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me check"},
                        {
                            "type": "tool_use",
                            "id": "tu_123",
                            "name": "get_weather",
                            "input": {"city": "NYC"},
                        },
                    ],
                }
            ],
        }
        out = anthropic_to_openai(req)
        msg = out["messages"][0]
        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(msg["content"], "Let me check")
        self.assertEqual(len(msg["tool_calls"]), 1)
        tc = msg["tool_calls"][0]
        self.assertEqual(tc["id"], "tu_123")
        self.assertEqual(tc["function"]["name"], "get_weather")
        self.assertEqual(json.loads(tc["function"]["arguments"]), {"city": "NYC"})

    def test_tool_result_in_user(self):
        req = {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_123",
                            "content": "Sunny, 72°F",
                        }
                    ],
                }
            ],
        }
        out = anthropic_to_openai(req)
        msg = out["messages"][0]
        self.assertEqual(msg["role"], "tool")
        self.assertEqual(msg["tool_call_id"], "tu_123")
        self.assertEqual(msg["content"], "Sunny, 72°F")

    def test_tool_result_mixed_with_text(self):
        req = {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Here is the result:"},
                        {"type": "tool_result", "tool_use_id": "tu_1", "content": "42"},
                    ],
                }
            ],
        }
        out = anthropic_to_openai(req)
        # Should produce a user msg + a tool msg
        roles = [m["role"] for m in out["messages"]]
        self.assertIn("user", roles)
        self.assertIn("tool", roles)

    def test_thinking_skipped_in_assistant_history(self):
        req = {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "internal", "signature": "sig"},
                        {"type": "text", "text": "Answer"},
                    ],
                }
            ],
        }
        out = anthropic_to_openai(req)
        msg = out["messages"][0]
        self.assertEqual(msg["content"], "Answer")
        self.assertNotIn("tool_calls", msg)

    def test_image_base64(self):
        req = {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "abc123==",
                            },
                        }
                    ],
                }
            ],
        }
        out = anthropic_to_openai(req)
        part = out["messages"][0]["content"][0]
        self.assertEqual(part["type"], "image_url")
        self.assertEqual(part["image_url"]["url"], "data:image/png;base64,abc123==")

    def test_tools_conversion(self):
        req = {
            "model": "m",
            "messages": [],
            "tools": [
                {
                    "name": "search",
                    "description": "Web search",
                    "input_schema": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                }
            ],
        }
        out = anthropic_to_openai(req)
        tool = out["tools"][0]
        self.assertEqual(tool["type"], "function")
        self.assertEqual(tool["function"]["name"], "search")
        self.assertIn("properties", tool["function"]["parameters"])

    def test_tool_choice_auto(self):
        req = {"model": "m", "messages": [], "tool_choice": {"type": "auto"}}
        out = anthropic_to_openai(req)
        self.assertEqual(out["tool_choice"], "auto")

    def test_tool_choice_any(self):
        req = {"model": "m", "messages": [], "tool_choice": {"type": "any"}}
        out = anthropic_to_openai(req)
        self.assertEqual(out["tool_choice"], "required")

    def test_tool_choice_specific(self):
        req = {
            "model": "m",
            "messages": [],
            "tool_choice": {"type": "tool", "name": "search"},
        }
        out = anthropic_to_openai(req)
        self.assertEqual(out["tool_choice"]["function"]["name"], "search")

    def test_stop_sequences(self):
        req = {"model": "m", "messages": [], "stop_sequences": ["STOP", "END"]}
        out = anthropic_to_openai(req)
        self.assertEqual(out["stop"], ["STOP", "END"])

    def test_thinking_param(self):
        req = {
            "model": "m",
            "messages": [],
            "thinking": {"type": "enabled", "budget_tokens": 5000},
        }
        out = anthropic_to_openai(req)
        self.assertIn("thinking", out)
        self.assertEqual(out["thinking"]["budget_tokens"], 5000)

    def test_stream_options_added(self):
        req = {"model": "m", "messages": [], "stream": True}
        out = anthropic_to_openai(req)
        self.assertTrue(out.get("stream"))
        self.assertEqual(out.get("stream_options"), {"include_usage": True})

    def test_no_stream_options_when_not_streaming(self):
        req = {"model": "m", "messages": []}
        out = anthropic_to_openai(req)
        self.assertNotIn("stream_options", out)


class TestOpenAIToAnthropic(unittest.TestCase):

    def _wrap(self, message: dict, finish_reason="stop", usage=None) -> dict:
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5},
        }

    def test_simple_text(self):
        resp = self._wrap({"role": "assistant", "content": "Hello!"})
        out = openai_to_anthropic(resp, "claude-3-5-sonnet")
        self.assertEqual(out["type"], "message")
        self.assertEqual(out["role"], "assistant")
        self.assertEqual(out["content"][0], {"type": "text", "text": "Hello!"})
        self.assertEqual(out["stop_reason"], "end_turn")

    def test_tool_call(self):
        resp = self._wrap({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"query":"python"}'},
                }
            ],
        }, finish_reason="tool_calls")
        out = openai_to_anthropic(resp, "m")
        self.assertEqual(out["stop_reason"], "tool_use")
        block = out["content"][0]
        self.assertEqual(block["type"], "tool_use")
        self.assertEqual(block["id"], "call_abc")
        self.assertEqual(block["name"], "search")
        self.assertEqual(block["input"], {"query": "python"})

    def test_finish_reason_mapping(self):
        cases = [
            ("stop", "end_turn"),
            ("length", "max_tokens"),
            ("tool_calls", "tool_use"),
            ("content_filter", "stop_sequence"),
        ]
        for fr, expected in cases:
            resp = self._wrap({"role": "assistant", "content": "x"}, finish_reason=fr)
            out = openai_to_anthropic(resp, "m")
            self.assertEqual(out["stop_reason"], expected, f"Failed for {fr}")

    def test_usage_mapping(self):
        resp = self._wrap(
            {"role": "assistant", "content": "x"},
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )
        out = openai_to_anthropic(resp, "m")
        self.assertEqual(out["usage"]["input_tokens"], 100)
        self.assertEqual(out["usage"]["output_tokens"], 50)

    def test_cached_tokens_subtracted(self):
        resp = self._wrap(
            {"role": "assistant", "content": "x"},
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "prompt_tokens_details": {"cached_tokens": 40},
            },
        )
        out = openai_to_anthropic(resp, "m")
        self.assertEqual(out["usage"]["input_tokens"], 60)
        self.assertEqual(out["usage"]["cache_read_input_tokens"], 40)

    def test_reasoning_content(self):
        resp = self._wrap({
            "role": "assistant",
            "content": "Answer",
            "reasoning_content": "My thinking...",
        })
        out = openai_to_anthropic(resp, "m")
        blocks = {b["type"]: b for b in out["content"]}
        self.assertIn("thinking", blocks)
        self.assertEqual(blocks["thinking"]["thinking"], "My thinking...")
        self.assertIn("text", blocks)

    def test_thinking_object(self):
        resp = self._wrap({
            "role": "assistant",
            "content": "Answer",
            "thinking": {"content": "deep thought", "signature": "sig123"},
        })
        out = openai_to_anthropic(resp, "m")
        blocks = {b["type"]: b for b in out["content"]}
        self.assertIn("thinking", blocks)
        self.assertEqual(blocks["thinking"]["signature"], "sig123")

    def test_invalid_tool_args_fallback(self):
        resp = self._wrap({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_x",
                    "type": "function",
                    "function": {"name": "fn", "arguments": "not-valid-json"},
                }
            ],
        }, finish_reason="tool_calls")
        out = openai_to_anthropic(resp, "m")
        block = out["content"][0]
        self.assertEqual(block["type"], "tool_use")
        # Should not crash; input contains fallback
        self.assertIsInstance(block["input"], dict)


class TestStreamConverter(unittest.IsolatedAsyncioTestCase):

    async def _collect(self, chunks: list[bytes]) -> list[dict]:
        """Run stream converter over mock chunks and return parsed events."""
        async def fake_stream():
            for c in chunks:
                yield c

        events = []
        async for sse_text in stream_openai_to_anthropic(fake_stream(), "msg_test", "m"):
            for line in sse_text.strip().split("\n"):
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
        return events

    def _chunk(self, delta: dict, finish_reason=None, usage=None) -> bytes:
        d: dict = {
            "id": "chatcmpl-x",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if usage:
            d["usage"] = usage
        return b"data: " + json.dumps(d).encode()

    async def test_simple_text_stream(self):
        chunks = [
            self._chunk({"role": "assistant", "content": ""}),
            self._chunk({"content": "Hello"}),
            self._chunk({"content": " world"}),
            self._chunk({}, finish_reason="stop", usage={"prompt_tokens": 10, "completion_tokens": 5}),
            b"data: [DONE]",
        ]
        events = await self._collect(chunks)

        types = [e["type"] for e in events]
        self.assertIn("message_start", types)
        self.assertIn("content_block_start", types)
        self.assertIn("content_block_delta", types)
        self.assertIn("content_block_stop", types)
        self.assertIn("message_delta", types)
        self.assertIn("message_stop", types)

        text_deltas = [
            e["delta"]["text"]
            for e in events
            if e["type"] == "content_block_delta"
            and e.get("delta", {}).get("type") == "text_delta"
        ]
        self.assertEqual("".join(text_deltas), "Hello world")

        msg_delta = next(e for e in events if e["type"] == "message_delta")
        self.assertEqual(msg_delta["delta"]["stop_reason"], "end_turn")
        self.assertEqual(msg_delta["usage"]["output_tokens"], 5)

    async def test_tool_call_stream(self):
        chunks = [
            self._chunk({"role": "assistant", "content": None}),
            self._chunk({"tool_calls": [{"index": 0, "id": "call_abc", "type": "function",
                                          "function": {"name": "search", "arguments": ""}}]}),
            self._chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"q"'}}]}),
            self._chunk({"tool_calls": [{"index": 0, "function": {"arguments": ':"hi"}'}}]}),
            self._chunk({}, finish_reason="tool_calls", usage={"prompt_tokens": 20, "completion_tokens": 10}),
            b"data: [DONE]",
        ]
        events = await self._collect(chunks)

        starts = [e for e in events if e["type"] == "content_block_start"]
        self.assertTrue(any(e["content_block"]["type"] == "tool_use" for e in starts))

        tool_start = next(e for e in starts if e["content_block"]["type"] == "tool_use")
        self.assertEqual(tool_start["content_block"]["id"], "call_abc")
        self.assertEqual(tool_start["content_block"]["name"], "search")

        json_deltas = [
            e["delta"]["partial_json"]
            for e in events
            if e["type"] == "content_block_delta"
            and e.get("delta", {}).get("type") == "input_json_delta"
        ]
        assembled = "".join(json_deltas)
        self.assertEqual(json.loads(assembled), {"q": "hi"})

        msg_delta = next(e for e in events if e["type"] == "message_delta")
        self.assertEqual(msg_delta["delta"]["stop_reason"], "tool_use")

    async def test_thinking_stream(self):
        chunks = [
            self._chunk({"thinking": {"content": "Let me think..."}}),
            self._chunk({"thinking": {"content": " more"}}),
            self._chunk({"thinking": {"signature": "sig_abc"}}),
            self._chunk({"content": "Answer"}),
            self._chunk({}, finish_reason="stop"),
            b"data: [DONE]",
        ]
        events = await self._collect(chunks)

        starts = [e for e in events if e["type"] == "content_block_start"]
        block_types = [e["content_block"]["type"] for e in starts]
        self.assertIn("thinking", block_types)
        self.assertIn("text", block_types)

        thinking_deltas = [
            e for e in events
            if e["type"] == "content_block_delta"
            and e.get("delta", {}).get("type") == "thinking_delta"
        ]
        thinking_text = "".join(e["delta"]["thinking"] for e in thinking_deltas)
        self.assertIn("Let me think", thinking_text)

        sig_deltas = [
            e for e in events
            if e["type"] == "content_block_delta"
            and e.get("delta", {}).get("type") == "signature_delta"
        ]
        self.assertTrue(len(sig_deltas) > 0)
        self.assertEqual(sig_deltas[0]["delta"]["signature"], "sig_abc")

    async def test_empty_stream(self):
        chunks = [
            self._chunk({}, finish_reason="stop", usage={"prompt_tokens": 5, "completion_tokens": 0}),
            b"data: [DONE]",
        ]
        events = await self._collect(chunks)
        types = [e["type"] for e in events]
        self.assertIn("message_start", types)
        self.assertIn("message_stop", types)

    async def test_max_tokens_finish_reason(self):
        chunks = [
            self._chunk({"content": "truncated"}),
            self._chunk({}, finish_reason="length"),
            b"data: [DONE]",
        ]
        events = await self._collect(chunks)
        msg_delta = next(e for e in events if e["type"] == "message_delta")
        self.assertEqual(msg_delta["delta"]["stop_reason"], "max_tokens")


class TestTransformers(unittest.TestCase):

    def test_maxtoken_sets_default(self):
        pipeline = build_pipeline([["maxtoken", {"max_tokens": 8192}]])
        req = {"model": "m", "messages": []}
        out = apply_pipeline(pipeline, req)
        self.assertEqual(out["max_tokens"], 8192)

    def test_maxtoken_caps_existing(self):
        pipeline = build_pipeline([["maxtoken", {"max_tokens": 8192}]])
        req = {"model": "m", "messages": [], "max_tokens": 16384}
        out = apply_pipeline(pipeline, req)
        self.assertEqual(out["max_tokens"], 8192)

    def test_maxtoken_respects_smaller(self):
        pipeline = build_pipeline([["maxtoken", {"max_tokens": 8192}]])
        req = {"model": "m", "messages": [], "max_tokens": 1024}
        out = apply_pipeline(pipeline, req)
        self.assertEqual(out["max_tokens"], 1024)

    def test_sampling_sets_temperature(self):
        pipeline = build_pipeline([["sampling", {"temperature": 0.7}]])
        req = {"model": "m", "messages": []}
        out = apply_pipeline(pipeline, req)
        self.assertEqual(out["temperature"], 0.7)

    def test_sampling_does_not_override_existing(self):
        pipeline = build_pipeline([["sampling", {"temperature": 0.7}]])
        req = {"model": "m", "messages": [], "temperature": 0.9}
        out = apply_pipeline(pipeline, req)
        self.assertEqual(out["temperature"], 0.9)

    def test_flatten_message(self):
        pipeline = build_pipeline(["flattenMessage"])
        req = {
            "model": "m",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hi"}, {"type": "text", "text": "!"}]}
            ],
        }
        out = apply_pipeline(pipeline, req)
        self.assertEqual(out["messages"][0]["content"], "Hi!")

    def test_flatten_message_skips_multimodal(self):
        pipeline = build_pipeline(["flattenMessage"])
        req = {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Look:"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
                    ],
                }
            ],
        }
        out = apply_pipeline(pipeline, req)
        # Should NOT flatten when mixed content
        self.assertIsInstance(out["messages"][0]["content"], list)

    def test_unknown_transformer_skipped(self):
        pipeline = build_pipeline(["nonexistent"])
        self.assertEqual(pipeline, [])

    def test_pipeline_order(self):
        # maxtoken then sampling
        pipeline = build_pipeline([
            ["maxtoken", {"max_tokens": 4096}],
            ["sampling", {"temperature": 1.0, "top_p": 0.95}],
        ])
        req = {"model": "m", "messages": [], "max_tokens": 8192}
        out = apply_pipeline(pipeline, req)
        self.assertEqual(out["max_tokens"], 4096)
        self.assertEqual(out["temperature"], 1.0)
        self.assertEqual(out["top_p"], 0.95)


class TestConfig(unittest.TestCase):

    def test_resolve_route(self):
        cfg = {"Router": {"default": "myprovider,mymodel"}}
        result = resolve_route(cfg)
        self.assertEqual(result, ("myprovider", "mymodel"))

    def test_resolve_route_scenario(self):
        cfg = {"Router": {"default": "p,m1", "think": "p,m2"}}
        result = resolve_route(cfg, "think")
        self.assertEqual(result, ("p", "m2"))

    def test_resolve_route_fallback_to_default(self):
        cfg = {"Router": {"default": "p,m1"}}
        result = resolve_route(cfg, "think")
        self.assertEqual(result, ("p", "m1"))

    def test_get_provider(self):
        cfg = {"Providers": [{"name": "foo", "api_key": "k"}]}
        p = get_provider(cfg, "foo")
        self.assertIsNotNone(p)
        self.assertEqual(p["api_key"], "k")

    def test_get_provider_missing(self):
        cfg = {"Providers": []}
        self.assertIsNone(get_provider(cfg, "nothere"))


# ─────────────────────────────────────────────────────────────────────────────
# Integration tests (requires live provider from demo-config.json)
# ─────────────────────────────────────────────────────────────────────────────

class IntegrationBase(unittest.IsolatedAsyncioTestCase):
    """Start an in-process server and make real HTTP calls."""

    CONFIG_PATH = "/playground/config.json"

    @classmethod
    def setUpClass(cls):
        import os
        if not os.path.exists(cls.CONFIG_PATH):
            raise unittest.SkipTest(f"Config not found: {cls.CONFIG_PATH}")

    async def asyncSetUp(self):
        import server as srv_mod
        from config import load_config
        cfg = load_config(self.CONFIG_PATH)
        srv_mod.set_config(cfg)
        self.app = srv_mod.app

        # ASGI test client
        from httpx import ASGITransport, AsyncClient
        self.client = AsyncClient(
            transport=ASGITransport(app=self.app),
            base_url="http://test",
            timeout=120.0,
        )

    async def asyncTearDown(self):
        await self.client.aclose()

    def _base_request(self, extra: dict | None = None) -> dict:
        req = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 512,
            "messages": [{"role": "user", "content": "Reply with exactly: PONG"}],
        }
        if extra:
            req.update(extra)
        return req


class TestIntegrationNonStreaming(IntegrationBase):

    async def test_basic_response(self):
        resp = await self.client.post("/v1/messages", json=self._base_request())
        self.assertEqual(resp.status_code, 200, resp.text)
        data = resp.json()
        self.assertEqual(data["type"], "message")
        self.assertEqual(data["role"], "assistant")
        # Content may include thinking + text blocks
        all_text = " ".join(
            b.get("text", "") or b.get("thinking", "")
            for b in data["content"]
        )
        self.assertIn("PONG", all_text)

    async def test_sampling_params(self):
        req = self._base_request({"temperature": 0.0, "top_p": 1.0})
        resp = await self.client.post("/v1/messages", json=req)
        self.assertEqual(resp.status_code, 200, resp.text)
        data = resp.json()
        self.assertEqual(data["type"], "message")

    async def test_system_prompt(self):
        req = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 512,
            "system": "Always respond with exactly: HELLO_WORLD",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        resp = await self.client.post("/v1/messages", json=req)
        self.assertEqual(resp.status_code, 200, resp.text)
        data = resp.json()
        all_text = " ".join(
            b.get("text", "") or b.get("thinking", "")
            for b in data["content"]
        )
        self.assertIn("HELLO_WORLD", all_text)

    async def test_tool_use(self):
        req = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 256,
            "tools": [
                {
                    "name": "get_number",
                    "description": "Returns a number",
                    "input_schema": {
                        "type": "object",
                        "properties": {"n": {"type": "integer"}},
                        "required": ["n"],
                    },
                }
            ],
            "tool_choice": {"type": "any"},
            "messages": [{"role": "user", "content": "Call get_number with n=42"}],
        }
        resp = await self.client.post("/v1/messages", json=req)
        self.assertEqual(resp.status_code, 200, resp.text)
        data = resp.json()
        tool_blocks = [b for b in data["content"] if b["type"] == "tool_use"]
        self.assertTrue(len(tool_blocks) > 0, "Expected tool_use block")
        self.assertEqual(tool_blocks[0]["name"], "get_number")

    async def test_usage_fields_present(self):
        resp = await self.client.post("/v1/messages", json=self._base_request())
        data = resp.json()
        self.assertIn("usage", data)
        self.assertIn("input_tokens", data["usage"])
        self.assertIn("output_tokens", data["usage"])
        self.assertGreater(data["usage"]["input_tokens"], 0)
        self.assertGreater(data["usage"]["output_tokens"], 0)

    async def test_multi_turn(self):
        req = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 512,
            "messages": [
                {"role": "user", "content": "Say: FIRST"},
                {"role": "assistant", "content": "FIRST"},
                {"role": "user", "content": "Now say: SECOND"},
            ],
        }
        resp = await self.client.post("/v1/messages", json=req)
        self.assertEqual(resp.status_code, 200, resp.text)
        data = resp.json()
        # Collect all text from text blocks and thinking blocks
        all_text = " ".join(
            b.get("text", "") or b.get("thinking", "")
            for b in data["content"]
        )
        self.assertIn("SECOND", all_text)

    async def test_health_endpoint(self):
        resp = await self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ok")


class TestIntegrationStreaming(IntegrationBase):

    async def _stream_collect(self, req: dict) -> tuple[list[dict], int]:
        """Returns (events, http_status)."""
        resp = await self.client.post(
            "/v1/messages",
            json={**req, "stream": True},
            headers={"Accept": "text/event-stream"},
        )
        if resp.status_code != 200:
            return [], resp.status_code

        events = []
        for line in resp.text.splitlines():
            if line.startswith("data: "):
                try:
                    events.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass
        return events, resp.status_code

    async def test_stream_basic(self):
        events, status = await self._stream_collect(self._base_request())
        self.assertEqual(status, 200)

        types = [e["type"] for e in events]
        self.assertIn("message_start", types)
        self.assertIn("content_block_start", types)
        self.assertIn("content_block_delta", types)
        self.assertIn("content_block_stop", types)
        self.assertIn("message_delta", types)
        self.assertIn("message_stop", types)

    async def test_stream_text_assembly(self):
        events, status = await self._stream_collect(self._base_request())
        self.assertEqual(status, 200)
        # Collect text from both text_delta and thinking_delta blocks
        text_deltas = [
            e["delta"].get("text", "") or e["delta"].get("thinking", "")
            for e in events
            if e["type"] == "content_block_delta"
            and e.get("delta", {}).get("type") in ("text_delta", "thinking_delta")
        ]
        full_text = "".join(text_deltas)
        self.assertIn("PONG", full_text)

    async def test_stream_usage(self):
        events, status = await self._stream_collect(self._base_request())
        self.assertEqual(status, 200)
        msg_delta = next((e for e in events if e["type"] == "message_delta"), None)
        self.assertIsNotNone(msg_delta)
        self.assertIn("usage", msg_delta)
        self.assertGreater(msg_delta["usage"]["output_tokens"], 0)

    async def test_stream_stop_reason(self):
        events, status = await self._stream_collect(self._base_request())
        self.assertEqual(status, 200)
        msg_delta = next(e for e in events if e["type"] == "message_delta")
        self.assertEqual(msg_delta["delta"]["stop_reason"], "end_turn")

    async def test_stream_tool_use(self):
        req = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 256,
            "tools": [
                {
                    "name": "get_number",
                    "description": "Returns a number",
                    "input_schema": {
                        "type": "object",
                        "properties": {"n": {"type": "integer"}},
                        "required": ["n"],
                    },
                }
            ],
            "tool_choice": {"type": "any"},
            "messages": [{"role": "user", "content": "Call get_number with n=7"}],
        }
        events, status = await self._stream_collect(req)
        self.assertEqual(status, 200)
        starts = [e for e in events if e["type"] == "content_block_start"]
        tool_starts = [e for e in starts if e["content_block"]["type"] == "tool_use"]
        self.assertTrue(len(tool_starts) > 0)

        json_deltas = [
            e["delta"]["partial_json"]
            for e in events
            if e["type"] == "content_block_delta"
            and e.get("delta", {}).get("type") == "input_json_delta"
        ]
        args = json.loads("".join(json_deltas))
        self.assertEqual(args.get("n"), 7)

    async def test_stream_order_invariants(self):
        """Validate Anthropic SSE protocol ordering."""
        events, status = await self._stream_collect(self._base_request())
        self.assertEqual(status, 200)
        types = [e["type"] for e in events]

        self.assertEqual(types[0], "message_start")
        self.assertLess(types.index("message_start"), types.index("content_block_start"))
        self.assertLess(types.index("content_block_start"), types.index("content_block_stop"))
        self.assertLess(types.index("content_block_stop"), types.index("message_delta"))
        last_two = types[-2:]
        self.assertIn("message_stop", last_two)


class TestIntegrationReasoning(IntegrationBase):
    """Test thinking / reasoning (only runs if the model supports it)."""

    async def test_reasoning_nonstreaming(self):
        req = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "thinking": {"type": "enabled", "budget_tokens": 2000},
            "messages": [{"role": "user", "content": "What is 2+2? Show your thinking."}],
        }
        resp = await self.client.post("/v1/messages", json=req)
        # If provider doesn't support thinking it may return 400 — that's OK to skip
        if resp.status_code == 400:
            self.skipTest("Provider does not support thinking")
        self.assertEqual(resp.status_code, 200, resp.text)
        data = resp.json()
        # Should have text content at minimum
        text_blocks = [b for b in data["content"] if b["type"] == "text"]
        self.assertTrue(len(text_blocks) > 0)

    async def test_reasoning_streaming(self):
        req = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "thinking": {"type": "enabled", "budget_tokens": 2000},
            "stream": True,
            "messages": [{"role": "user", "content": "What is 3+3?"}],
        }
        resp = await self.client.post("/v1/messages", json=req)
        if resp.status_code == 400:
            self.skipTest("Provider does not support thinking")
        self.assertEqual(resp.status_code, 200)


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    if mode in ("unit", "all"):
        for cls in [
            TestAnthropicToOpenAI,
            TestOpenAIToAnthropic,
            TestStreamConverter,
            TestTransformers,
            TestConfig,
        ]:
            suite.addTests(loader.loadTestsFromTestCase(cls))

    if mode in ("integration", "all"):
        for cls in [
            TestIntegrationNonStreaming,
            TestIntegrationStreaming,
            TestIntegrationReasoning,
        ]:
            suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
