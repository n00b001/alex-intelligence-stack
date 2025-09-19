import asyncio
import unittest
from unittest import TestCase, mock

import pytest
import tiktoken
from tiktoken import Encoding

import condenser
from consts import Config


class Test(TestCase):
    cfg = Config()
    cfg.log_level = "DEBUG"

    def setUp(self):
        self.encoder = tiktoken.get_encoding("cl100k_base")

    # ---------- Helpers ----------
    async def fake_proxy_request(self, endpoint, payload, streaming, enable_condensation=True):
        return {"content": "condensed"}

    async def run_call_summarizer(self, summarizer):
        return await condenser._call_summarizer_with_retries(
            summarizer, "prompt text", cfg=self.cfg, model="gpt-4"
        )

    async def run_summarize_text(self, summarizer):
        return await condenser._summarize_text_using_summarizer(
            summarizer, "hello", cfg=self.cfg, model="gpt-4"
        )

    # ---------- Tests ----------
    def test__get_encoder_for_model(self):
        enc = condenser._get_encoder_for_model("gpt-4")
        self.assertIsInstance(enc, Encoding)
        enc2 = condenser._get_encoder_for_model("nonexistent-model")
        self.assertIsInstance(enc2, Encoding)

    def test__num_tokens(self):
        self.assertGreater(condenser._num_tokens("hello world", self.encoder), 0)
        self.assertEqual(condenser._num_tokens("", self.encoder), 0)
        self.assertIsInstance(condenser._num_tokens("text", object()), int)

    def test__is_textual(self):
        self.assertTrue(condenser._is_textual("string"))
        self.assertFalse(condenser._is_textual(123))
        self.assertTrue(condenser._is_textual({"role": "user"}))

    def test__extract_text_from_part(self):
        self.assertEqual(condenser._extract_text_from_part("text"), "text")
        self.assertEqual(
            condenser._extract_text_from_part({"text": {"value": "abc"}}),
            '{"value": "abc"}',
        )
        self.assertEqual(
            condenser._extract_text_from_part({"role": "user", "content": "msg"}),
            "msg",
        )

    def test__normalize_to_message_list(self):
        history = ["hello", {"role": "assistant", "content": "hi"}]
        norm = condenser._normalize_to_message_list(history)
        self.assertIsInstance(norm, list)
        self.assertIn("content", norm[0])

    def test_dict_that_looks_like_text(self):
        history = [{"role": "user", "content": {"type": "text", "text": "hello"}}]
        norm = condenser._normalize_to_message_list(history)
        self.assertEqual(norm[0]["role"], "user")
        self.assertIsInstance(norm[0]["content"], list)
        self.assertEqual(norm[0]["content"][0]["text"], "hello")
        self.assertTrue(norm[0]["is_text"])

    def test_fallback_extract_text_from_content_key(self):
        history = [{"role": "assistant", "content": {"content": "hello from nested"}}]
        norm = condenser._normalize_to_message_list(history)
        self.assertEqual(norm[0]["content"], "hello from nested")
        self.assertTrue(norm[0]["is_text"])

    def test_fallback_extract_text_from_text_key(self):
        history = [{"role": "assistant", "content": {"text": "text inside dict"}}]
        norm = condenser._normalize_to_message_list(history)
        self.assertEqual(norm[0]["content"], "text inside dict")
        self.assertTrue(norm[0]["is_text"])

    def test_case_4_fallback_number(self):
        history = [{"role": "user", "content": 12345}]
        norm = condenser._normalize_to_message_list(history)
        self.assertEqual(norm[0]["content"], "12345")
        self.assertTrue(norm[0]["is_text"])

    def test_history_is_none(self):
        history = None
        norm = condenser._normalize_to_message_list(history)
        self.assertEqual(norm, [])

    def test_openai_chat_style(self):
        history = {
            "messages": [
                {"role": "system", "content": "You are a bot."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"}
            ]
        }
        norm = condenser._normalize_to_message_list(history)
        self.assertEqual(len(norm), 3)
        self.assertEqual(norm[0]["role"], "system")
        self.assertEqual(norm[1]["content"], "Hi")
        self.assertEqual(norm[2]["role"], "assistant")

    def test_responses_completion_style_input(self):
        history = {"input": "Tell me a joke"}
        norm = condenser._normalize_to_message_list(history)
        self.assertEqual(len(norm), 1)
        self.assertEqual(norm[0]["role"], "user")
        self.assertEqual(norm[0]["content"], "Tell me a joke")

    def test_responses_completion_style_prompt_list(self):
        history = {"prompt": ["Say this", "Now say that"]}
        norm = condenser._normalize_to_message_list(history)
        self.assertEqual(len(norm), 2)
        self.assertEqual(norm[0]["content"], "Say this")
        self.assertEqual(norm[1]["content"], "Now say that")

    def test_plain_string_history(self):
        history = "Just a plain string"
        norm = condenser._normalize_to_message_list(history)
        self.assertEqual(len(norm), 1)
        self.assertEqual(norm[0]["role"], "user")
        self.assertEqual(norm[0]["content"], "Just a plain string")

    def test_fallback_nonstandard_object(self):
        history = {"unexpected": "structure"}
        norm = condenser._normalize_to_message_list(history)
        self.assertEqual(len(norm), 1)
        self.assertEqual(norm[0]["content"], "")
        self.assertFalse(norm[0]["is_text"])

    def test_nested_content(self):
        history = [{"role": "assistant", "content": [{"type": "text", "text": "hi"}]}]
        norm = condenser._normalize_to_message_list(history)
        self.assertIsInstance(norm[0]["content"], list)
        self.assertEqual(norm[0]["content"][0]["type"], "text")

    def test_ignore_extra_fields(self):
        history = [{"role": "user", "content": "hello", "timestamp": 12345}]
        norm = condenser._normalize_to_message_list(history)
        self.assertEqual(norm[0]["role"], "user")
        self.assertEqual(norm[0]["content"], "hello")

    def test_multiple_messages(self):
        history = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there!"}
        ]
        norm = condenser._normalize_to_message_list(history)
        self.assertEqual(len(norm), 3)
        self.assertEqual(norm[0]["role"], "system")
        self.assertEqual(norm[1]["content"], "hello")
        self.assertEqual(norm[2]["role"], "assistant")

    def test_structured_text_and_image(self):
        history = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/cat.png"
                        }
                    }
                ]
            }
        ]
        norm = condenser._normalize_to_message_list(history)
        self.assertIsInstance(norm[0]["content"], list)
        self.assertEqual(norm[0]["content"][0]["type"], "text")
        self.assertEqual(norm[0]["content"][1]["type"], "image_url")

    def test_structured_with_base64_image(self):
        history = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this picture"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA..."
                        }
                    }
                ]
            }
        ]
        norm = condenser._normalize_to_message_list(history)
        self.assertEqual(norm[0]["content"][1]["type"], "image_url")
        self.assertTrue(
            norm[0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64")
        )

    def test_structured_with_audio(self):
        history = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "transcribe this audio"},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": "base64audiodata...",
                            "format": "wav"
                        }
                    }
                ]
            }
        ]
        norm = condenser._normalize_to_message_list(history)
        self.assertEqual(norm[0]["content"][1]["type"], "input_audio")
        self.assertEqual(norm[0]["content"][1]["input_audio"]["format"], "wav")

    def test_assistant_structured_response(self):
        history = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Here is a description of the image"},
                    {
                        "type": "output_audio",
                        "output_audio": {
                            "data": "base64encodedoutputaudio...",
                            "format": "mp3"
                        }
                    }
                ]
            }
        ]
        norm = condenser._normalize_to_message_list(history)
        self.assertEqual(norm[0]["role"], "assistant")
        self.assertEqual(norm[0]["content"][0]["type"], "text")
        self.assertEqual(norm[0]["content"][1]["type"], "output_audio")

    def test__split_text_into_pieces(self):
        text = "short text"
        pieces = condenser._split_text_into_pieces(text, 1000, self.encoder)
        self.assertEqual(pieces, [text])

    def test__split_text_into_pieces2(self):
        text = "1" * 2_000
        pieces = condenser._split_text_into_pieces(text, 1000, self.encoder)
        self.assertEqual(pieces, [text])

    def test__split_text_into_pieces3(self):
        text = ["1"] * 2_000
        text = " ".join(text)
        pieces = condenser._split_text_into_pieces(text, 1000, self.encoder)
        split_text = text.replace(" ", " <<<").split("<<<")
        self.assertEqual(pieces[0], split_text[0])
        self.assertEqual(pieces[-1], split_text[-1])
        self.assertEqual(len(pieces), len(split_text))

    def test__split_text_into_pieces4(self):
        text = ["1"] * 2_000
        text = "\n".join(text)
        pieces = condenser._split_text_into_pieces(text, 1000, self.encoder)
        split_text = text.replace("\n", "\n<<<").split("<<<")
        self.assertEqual(pieces[0], split_text[0])
        self.assertEqual(pieces[-1], split_text[-1])
        self.assertEqual(len(pieces), len(split_text))

    def test__split_text_into_pieces5(self):
        text = ["1"] * 10_000
        text = "\n\n".join(text)
        pieces = condenser._split_text_into_pieces(text, 1000, self.encoder)
        split_text = text.replace("\n\n", "\n\n<<<").split("<<<")
        self.assertEqual(pieces[0], split_text[0])
        self.assertEqual(pieces[-1], split_text[-1])
        self.assertEqual(len(pieces), len(split_text))

    def test__make_retry_decorator(self):
        decorator = condenser._make_retry_decorator(attempts=3)
        self.assertTrue(callable(decorator))

    def test__call_summarizer_with_retries(self):
        async def fake_summary(endpoint, payload, streaming, enable_condensation):
            return {"content": "summary"}

        result = asyncio.run(self.run_call_summarizer(fake_summary))
        self.assertIsInstance(result, str)
        self.assertIn("summary", result)

    def test__summarize_text_using_summarizer(self):
        result = asyncio.run(self.run_summarize_text(self.fake_proxy_request))
        self.assertIsInstance(result, str)
        self.assertIn("condensed", result)

    def test__rebuild_history_from_messages(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        # The first argument is self or the actual messages object depending on condenser
        rebuilt = condenser._rebuild_history_from_messages(None, msgs)
        self.assertIsInstance(rebuilt, list)
        self.assertEqual(len(rebuilt), 2)

    def test_dict_with_messages_key(self):
        original = {"messages": [{"role": "user", "content": "hi"}], "extra": "keepme"}
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        rebuilt = condenser._rebuild_history_from_messages(original, msgs)
        # structure should be preserved
        self.assertIn("messages", rebuilt)
        self.assertEqual(len(rebuilt["messages"]), 2)
        self.assertEqual(rebuilt["messages"][0]["content"], "hello")
        self.assertEqual(rebuilt["extra"], "keepme")

    def test_default_branch_with_raw_object(self):
        original = []  # triggers the default branch
        raw_obj = {"role": "user", "content": {"type": "text", "text": "kept raw"}}
        msgs = [
            {"role": "user", "content": "ignored", "raw": raw_obj}
        ]
        rebuilt = condenser._rebuild_history_from_messages(original, msgs)
        # should preserve the raw object
        self.assertEqual(len(rebuilt), 1)
        self.assertEqual(rebuilt[0], raw_obj)

    def test_dict_with_prompt_key(self):
        original = {"prompt": "ignored old prompt"}
        msgs = [
            {"role": "user", "content": "foo"},
            {"role": "assistant", "content": "bar"},
        ]
        rebuilt = condenser._rebuild_history_from_messages(original, msgs)
        # prompt key should be rebuilt as joined string
        self.assertIn("prompt", rebuilt)
        self.assertIn("foo", rebuilt["prompt"])
        self.assertIn("bar", rebuilt["prompt"])

    def test_dict_with_inputs_key(self):
        original = {"inputs": ["previous"]}
        msgs = [
            {"role": "user", "content": "new1"},
            {"role": "assistant", "content": "new2"},
        ]
        rebuilt = condenser._rebuild_history_from_messages(original, msgs)
        # inputs should be a list of contents
        self.assertEqual(rebuilt["inputs"], ["new1", "new2"])

    def test_original_is_string(self):
        original = "start text"
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
        ]
        rebuilt = condenser._rebuild_history_from_messages(original, msgs)
        self.assertIsInstance(rebuilt, str)
        self.assertIn("first", rebuilt)
        self.assertIn("second", rebuilt)

    def test_raw_objects_are_preserved(self):
        original = {"messages": []}
        raw_obj = {"role": "assistant", "content": {"type": "image_url", "image_url": {"url": "x.png"}}}
        msgs = [
            {"role": "assistant", "content": "placeholder", "raw": raw_obj}
        ]
        rebuilt = condenser._rebuild_history_from_messages(original, msgs)
        self.assertEqual(rebuilt["messages"][0], raw_obj)

    def test_condense(self):
        history = [
            {"role": "user", "content": f"hello world {i} " * 10}
            for i in range(2500)
        ]
        result = asyncio.run(
            condenser.condense(
                history,
                target_tokens=40000,
                proxy_request_callback=self.fake_proxy_request,
                model="gpt-4",
                cfg=self.cfg,
            )
        )
        self.assertIsInstance(result, list)
        # condense returns messages, so check the content
        self.assertTrue(any("condensed" in m["content"] for m in result))

    def test_condense2(self):
        history = [
            {"role": "user", "content": f"hello world {i} "}
            for i in range(10_000)
        ]
        result = asyncio.run(
            condenser.condense(
                history,
                target_tokens=40000,
                proxy_request_callback=self.fake_proxy_request,
                model="gpt-4",
                cfg=self.cfg,
            )
        )
        self.assertIsInstance(result, list)
        # condense returns messages, so check the content
        self.assertTrue(any("condensed" in m["content"] for m in result))

    def test_condense3(self):
        history = [
            {"role": "user", "content": f"hello world {i} " * 10_000}
            for i in range(1)
        ]
        result = asyncio.run(
            condenser.condense(
                history,
                target_tokens=40000,
                proxy_request_callback=self.fake_proxy_request,
                model="gpt-4",
                cfg=self.cfg,
            )
        )
        self.assertIsInstance(result, list)
        # condense returns messages, so check the content
        self.assertTrue(any("condensed" in m["content"] for m in result))

    def test_condense4(self):
        history = [
            {"role": "user", "content": f"hello world {i} " * 100_000}
            for i in range(1)
        ]
        result = asyncio.run(
            condenser.condense(
                history,
                target_tokens=40000,
                proxy_request_callback=self.fake_proxy_request,
                model="gpt-4",
                cfg=self.cfg,
            )
        )
        self.assertIsInstance(result, list)
        # condense returns messages, so check the content
        self.assertTrue(any("condensed" in m["content"] for m in result))

    def test_condense5(self):
        history = [
            {"role": "user", "content": "1" * 1_000_000}
            for _ in range(1)
        ]
        result = asyncio.run(
            condenser.condense(
                history,
                target_tokens=40000,
                proxy_request_callback=self.fake_proxy_request,
                model="gpt-4",
                cfg=self.cfg,
            )
        )
        self.assertIsInstance(result, list)
        # condense returns messages, so check the content
        self.assertTrue(any("condensed" in m["content"] for m in result))

    def test_condense6(self):
        history = [
            {"role": "user", "content": "1" * 1_000_000}
            for _ in range(1)
        ]
        cfg = Config()
        cfg.keep_first_n = 0
        cfg.keep_last_n = 0

        result = asyncio.run(
            condenser.condense(
                history,
                target_tokens=40000,
                proxy_request_callback=self.fake_proxy_request,
                model="gpt-4",
                cfg=cfg,
            )
        )
        self.assertIsInstance(result, list)
        # condense returns messages, so check the content
        self.assertTrue(any("condensed" in m["content"] for m in result))

    def test_condense7(self):
        history = [
            {"role": "user", "content": "1" * 1_000_000}
            for _ in range(1)
        ]
        cfg = Config()
        cfg.keep_first_n = 0
        cfg.keep_last_n = 100

        result = asyncio.run(
            condenser.condense(
                history,
                target_tokens=40000,
                proxy_request_callback=self.fake_proxy_request,
                model="gpt-4",
                cfg=cfg,
            )
        )
        self.assertIsInstance(result, list)
        # condense returns messages, so check the content
        self.assertTrue(any("condensed" in m["content"] for m in result))

    def test__get_encoder_for_model_fallback_failure(self):
        """Test fallback when both encoding_for_model and get_encoding fail."""
        # We can't easily make tiktoken fail, so we mock it.
        with mock.patch('tiktoken.get_encoding', side_effect=Exception("Mock failure")):
            enc = condenser._get_encoder_for_model("any-model")
            self.assertIsNone(enc)

    def test__num_tokens_fallback_on_encoder_failure(self):
        """Test the char-based fallback when encoder.encode fails."""
        # Create a mock encoder that raises an exception
        mock_encoder = unittest.mock.Mock()
        mock_encoder.encode.side_effect = Exception("Encoding failed")

        token_count = condenser._num_tokens("Hello, world!", mock_encoder)
        # Should fall back to len(text) // 4
        self.assertEqual(token_count, len("Hello, world!") // 4)

    @pytest.mark.asyncio
    async def test_condense_summarizer_failure_leads_to_fallback(self):
        """Test that if the summarizer consistently fails, condense falls back to truncation."""

        async def failing_summarizer(endpoint, payload, streaming, enable_condensation):
            raise RuntimeError("Summarizer is down")

        history = [{"role": "user", "content": "This is a very long message that needs condensing. " * 100}]
        cfg = Config()
        cfg.condense_construction_attempts = 1  # Reduce attempts for faster test
        cfg.condense_retry_attempts = 1

        result = await condenser.condense(
            history,
            target_tokens=10,  # Force condensation
            proxy_request_callback=failing_summarizer,
            model="gpt-4",
            cfg=cfg,
        )

        # The result should contain the fallback message
        self.assertIn(cfg.truncation_message, str(result))

    @pytest.mark.asyncio
    async def test_condense_final_summary_failure_leads_to_fallback(self):
        """Test that if only the final summary step fails, condense still falls back."""
        call_count = 0

        async def flaky_summarizer(endpoint, payload, streaming, enable_condensation):
            nonlocal call_count
            call_count += 1
            # Fail only on the final summary call (which would be after chunk summaries)
            if call_count > 2:  # Assuming 2 chunk summaries + 1 final
                raise RuntimeError("Final summarizer failed")
            return {"content": f"summary_{call_count}"}

        history = [
            {"role": "user", "content": "Chunk 1 content. " * 50},
            {"role": "user", "content": "Chunk 2 content. " * 50},
        ]
        cfg = Config()
        cfg.condense_construction_attempts = 1
        cfg.condense_retry_attempts = 1
        cfg.keep_first_n = 0
        cfg.keep_last_n = 0

        result = await condenser.condense(
            history,
            target_tokens=10,  # Force condensation
            proxy_request_callback=flaky_summarizer,
            model="gpt-4",
            cfg=cfg,
        )

        # The result should contain the fallback message due to final summary failure
        self.assertIn(cfg.truncation_message, str(result))

    def test__split_text_into_pieces_with_none_encoder(self):
        """Test _split_text_into_pieces when encoder is None, forcing char-count fallback."""
        text = "This is a sample text to be split into pieces." * 20  # Make it long
        pieces = condenser._split_text_into_pieces(text, max_tokens=50, encoder=None)

        # It should still return a list of strings
        self.assertIsInstance(pieces, list)
        self.assertTrue(all(isinstance(p, str) for p in pieces))
        # Since encoder is None, it uses char-count. We can't predict exact splits,
        # but we can check that no piece is excessively long.
        max_chars_per_piece = 50 * 4  # max_tokens * chars_per_token heuristic
        for piece in pieces:
            self.assertLessEqual(len(piece), max_chars_per_piece * 2)  # Allow some buffer

    def test_condense_with_none_encoder(self):
        """Test condense function when _get_encoder_for_model returns None."""

        async def fake_summarizer(endpoint, payload, streaming, enable_condensation):
            return {"content": "condensed"}

        with unittest.mock.patch('condenser._get_encoder_for_model', return_value=None):
            history = [{"role": "user", "content": "Message 1"}, {"role": "user", "content": "Message 2"}]
            result = asyncio.run(condenser.condense(
                history,
                target_tokens=1,  # Force condensation
                proxy_request_callback=fake_summarizer,
                model="unknown-model",
                cfg=self.cfg,
            ))
            # It should still complete, using the char-count fallback for token estimation
            self.assertIsInstance(result, list)
