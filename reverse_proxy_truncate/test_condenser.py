import asyncio
from unittest import TestCase

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
            summarizer, "prompt text", max_tokens=50, cfg=self.cfg, model="gpt-4"
        )

    async def run_summarize_text(self, summarizer):
        return await condenser._summarize_text_using_summarizer(
            summarizer, "hello", max_tokens=50, cfg=self.cfg, model="gpt-4"
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

    def test__split_text_into_pieces(self):
        text = "short text"
        pieces = condenser._split_text_into_pieces(text, 1000, self.encoder)
        self.assertEqual(pieces, [text])

    def test__split_text_into_pieces2(self):
        text = "1" * 2_000
        pieces = condenser._split_text_into_pieces(text, 1000, self.encoder)
        self.assertEqual(pieces, [text])

    def test__split_text_into_pieces3(self):
        text = "1 " * 2_000
        pieces = condenser._split_text_into_pieces(text, 1000, self.encoder)
        self.assertEqual(pieces, [text])

    def test__split_text_into_pieces4(self):
        text = "1\n" * 2_000
        pieces = condenser._split_text_into_pieces(text, 1000, self.encoder)
        self.assertEqual(pieces, [text])

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

    def test_condense(self):
        history = [
            {"role": "user", "content": f"hello world {i} " * 10}
            for i in range(2500)
        ]
        result = asyncio.run(
            condenser.condense(
                history,
                target_tokens=40000,
                summarizer=self.fake_proxy_request,
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
                summarizer=self.fake_proxy_request,
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
                summarizer=self.fake_proxy_request,
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
                summarizer=self.fake_proxy_request,
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
                summarizer=self.fake_proxy_request,
                model="gpt-4",
                cfg=self.cfg,
            )
        )
        self.assertIsInstance(result, list)
        # condense returns messages, so check the content
        self.assertTrue(any("condensed" in m["content"] for m in result))

    def test_condense6(self):
        history = [
            {"role": "user", "content": "1" * 2_000}
            for _ in range(1)
        ]
        cfg = Config()
        cfg.keep_first_n = 0
        cfg.keep_last_n = 0

        result = asyncio.run(
            condenser.condense(
                history,
                target_tokens=40000,
                summarizer=self.fake_proxy_request,
                model="gpt-4",
                cfg=cfg,
            )
        )
        self.assertIsInstance(result, list)
        # condense returns messages, so check the content
        self.assertTrue(any("condensed" in m["content"] for m in result))
