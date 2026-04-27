import importlib
import json
import unittest
from unittest.mock import AsyncMock, Mock, patch


with patch("transformers.AutoTokenizer.from_pretrained", return_value=Mock()):
    chat_to_generate_adapter = importlib.import_module("chat_to_generate_adapter")


class TestMainKeyResolution(unittest.TestCase):
    def setUp(self):
        self.adapter = chat_to_generate_adapter.ChatToGenerateAdapter(
            use_generate_api=False,
            use_completions_for_chat=False,
        )

    def test_explicit_main_key_wins(self):
        main_key = self.adapter._resolve_main_key(
            {
                "main_key": "explicit-main-key",
                "metadata": {"user_id": {"session_id": "session-from-metadata"}},
            }
        )
        self.assertEqual(main_key, "explicit-main-key")

    def test_session_id_fallback_from_metadata_user_id(self):
        main_key = self.adapter._resolve_main_key(
            {
                "metadata": {"user_id": {"session_id": "claude-session-123"}},
            }
        )
        self.assertEqual(main_key, "claude-session-123")

    def test_session_id_fallback_from_user_dict(self):
        main_key = self.adapter._resolve_main_key(
            {
                "user": {"session_id": "claude-session-user-dict"},
            }
        )
        self.assertEqual(main_key, "claude-session-user-dict")

    def test_session_id_fallback_from_user_json_string(self):
        main_key = self.adapter._resolve_main_key(
            {
                "user": "{\"device_id\":\"dev-1\",\"session_id\":\"claude-session-user-json\"}",
            }
        )
        self.assertEqual(main_key, "claude-session-user-json")

    def test_build_generate_request_uses_session_id_as_main_key(self):
        self.adapter.tokenizer = Mock()
        self.adapter.tokenizer.apply_chat_template.return_value = "prompt"

        request = self.adapter._build_generate_request(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "metadata": {"user_id": {"session_id": "claude-session-456"}},
            }
        )

        self.assertEqual(request["main_key"], "claude-session-456")

class TestStreamingPaths(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.adapter = chat_to_generate_adapter.ChatToGenerateAdapter(
            use_generate_api=False,
            use_completions_for_chat=False,
        )

    async def test_process_chat_via_generate_preserves_stream(self):
        self.adapter._build_generate_request = Mock(return_value={"model": "m", "text": "prompt"})
        self.adapter._mock_generate_to_chat_stream = AsyncMock(return_value="stream-response")

        out = await self.adapter._process_chat_via_generate({"stream": True}, {})

        self.assertEqual(out, "stream-response")
        self.assertNotIn("stream", self.adapter._build_generate_request.return_value)
        self.adapter._mock_generate_to_chat_stream.assert_awaited_once()

    async def test_process_chat_via_completions_preserves_stream(self):
        self.adapter.tokenizer = Mock()
        self.adapter.tokenizer.apply_chat_template.return_value = "prompt"
        self.adapter._stream_completions_to_chat = AsyncMock(return_value="stream-response")

        out = await self.adapter._process_chat_via_completions(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
            {},
        )

        self.assertEqual(out, "stream-response")
        streamed_request = self.adapter._stream_completions_to_chat.await_args.args[0]
        self.assertTrue(streamed_request["stream"])

    async def test_process_completions_via_v1_preserves_stream(self):
        self.adapter._stream_raw_provider_response = AsyncMock(return_value="stream-response")

        out = await self.adapter._process_completions_via_v1(
            {"model": "m", "prompt": "hi", "stream": True},
            {},
        )

        self.assertEqual(out, "stream-response")
        streamed_request = self.adapter._stream_raw_provider_response.await_args.args[2]
        self.assertTrue(streamed_request["stream"])

    async def test_mock_generate_stream_raises_on_empty_adapted_response(self):
        self.adapter.forward_request = AsyncMock(return_value=Mock(
            status_code=200,
            json=Mock(return_value={"text": "   ", "meta_info": {}}),
        ))

        with self.assertRaisesRegex(RuntimeError, "empty response"):
            response = await self.adapter._mock_generate_to_chat_stream(
                {"model": "m", "text": "prompt"},
                {},
                {"tools": []},
            )
            async for _ in response.body_iterator:
                pass

    async def test_stream_completions_with_tools_keeps_plain_text(self):
        class FakeStream:
            def __init__(self, events):
                self._events = events

            def __aiter__(self):
                return self._iter()

            async def _iter(self):
                for event in self._events:
                    yield event

            async def aclose(self):
                return None

        events = [
            b'data: {"choices":[{"text":"plain answer","finish_reason":null}]}',
            b'data: {"choices":[{"text":"","finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}',
            b"data: [DONE]",
        ]
        with patch.object(chat_to_generate_adapter, "open_provider_stream", new=AsyncMock(return_value=FakeStream(events))):
            response = await self.adapter._stream_completions_to_chat(
                {
                    "model": "m",
                    "prompt": "p",
                    "stream": True,
                    "_chat_tools": [{"name": "Read", "parameters": {"type": "object", "properties": {}}}],
                },
                {},
            )
            chunks = [chunk async for chunk in response.body_iterator]

        payloads = [chunk for chunk in chunks if chunk.startswith("data: {")]
        self.assertTrue(any('"content": "plain answer"' in chunk for chunk in payloads))
        self.assertTrue(any('"finish_reason": "stop"' in chunk for chunk in payloads))

    async def test_stream_completions_with_tools_raises_on_empty_response(self):
        class FakeStream:
            def __init__(self, events):
                self._events = events

            def __aiter__(self):
                return self._iter()

            async def _iter(self):
                for event in self._events:
                    yield event

            async def aclose(self):
                return None

        events = [
            b'data: {"choices":[{"text":"   ","finish_reason":null}]}',
            b'data: {"choices":[{"text":"","finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":0,"total_tokens":1}}',
            b"data: [DONE]",
        ]
        with patch.object(chat_to_generate_adapter, "open_provider_stream", new=AsyncMock(return_value=FakeStream(events))):
            response = await self.adapter._stream_completions_to_chat(
                {
                    "model": "m",
                    "prompt": "p",
                    "stream": True,
                    "_chat_tools": [{"name": "Read", "parameters": {"type": "object", "properties": {}}}],
                },
                {},
            )
            with self.assertRaisesRegex(RuntimeError, "empty response"):
                async for _ in response.body_iterator:
                    pass

    async def test_process_chat_via_completions_preserves_plain_text_with_tools(self):
        self.adapter.tokenizer = Mock()
        self.adapter.tokenizer.apply_chat_template.return_value = "prompt"
        self.adapter._process_completions_via_v1 = AsyncMock(
            return_value={
                "id": "cmpl-1",
                "created": 1,
                "model": "m",
                "choices": [{"text": "plain answer", "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            }
        )

        out = await self.adapter._process_chat_via_completions(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "Read", "parameters": {"type": "object", "properties": {}}}}],
            },
            {},
        )

        self.assertEqual(out["choices"][0]["message"]["content"], "plain answer")


if __name__ == "__main__":
    unittest.main()
