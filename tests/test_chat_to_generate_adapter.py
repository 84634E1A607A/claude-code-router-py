import importlib
import unittest
from unittest.mock import Mock, patch


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


if __name__ == "__main__":
    unittest.main()
