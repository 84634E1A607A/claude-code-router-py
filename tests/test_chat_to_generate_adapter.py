import importlib
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


if __name__ == "__main__":
    unittest.main()
