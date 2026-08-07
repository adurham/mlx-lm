# Copyright © 2024 Apple Inc.

import unittest
from pathlib import Path

from huggingface_hub import snapshot_download

from mlx_lm.tokenizer_utils import (
    BPEStreamingDetokenizer,
    NaiveStreamingDetokenizer,
    SPMStreamingDetokenizer,
    TokenizerWrapper,
)
from mlx_lm.utils import load_tokenizer


class TestTokenizers(unittest.TestCase):

    def check_tokenizer(self, tokenizer):
        def check(tokens):
            expected_text = tokenizer.decode(tokens)
            detokenizer = tokenizer.detokenizer
            detokenizer.reset()
            text = ""
            for e, t in enumerate(tokens):
                detokenizer.add_token(t)
                seg = detokenizer.last_segment
                text += seg
                self.assertEqual(detokenizer.tokens, tokens[: e + 1])
            detokenizer.finalize()
            text += detokenizer.last_segment
            self.assertEqual(text, expected_text)

        tokens = tokenizer.encode("こんにちは！私の名前はAI")
        check(tokens)

        tokens = tokenizer.encode("⊕ ⊻ ∧ ¬")
        check(tokens)

        tokens = tokenizer.encode("a ,b")
        check(tokens)

        tokens = tokenizer.encode('{"why_its_funny" :"a_joke_explainer" ,"rating":3.5}')
        check(tokens)

        tokens = tokenizer.encode("3 3")
        check(tokens)

        tokens = tokenizer.encode("import 'package:flutter/material.dart';")
        check(tokens)

        tokens = tokenizer.encode("hello\nworld")
        check(tokens)

    def test_tokenizers(self):
        tokenizer_repos = [
            ("mlx-community/Qwen1.5-0.5B-Chat-4bit", BPEStreamingDetokenizer),
            ("mlx-community/Mistral-7B-v0.2-4bit", SPMStreamingDetokenizer),
            ("mlx-community/Phi-3.5-mini-instruct-4bit", SPMStreamingDetokenizer),
            ("mlx-community/Mistral-7B-Instruct-v0.3", SPMStreamingDetokenizer),
            ("mlx-community/Llama-3.2-1B-Instruct-4bit", BPEStreamingDetokenizer),
            ("mlx-community/Falcon3-7B-Instruct-4bit", BPEStreamingDetokenizer),
        ]
        for tokenizer_repo, expected_detokenizer in tokenizer_repos:
            with self.subTest(tokenizer=tokenizer_repo):
                tokenizer = load_tokenizer(tokenizer_repo)
                tokenizer.decode([0, 1, 2])
                self.assertTrue(isinstance(tokenizer.detokenizer, expected_detokenizer))
                self.check_tokenizer(tokenizer)

        # Try one with a naive detokenizer
        tokenizer = load_tokenizer("mlx-community/Llama-3.2-1B-Instruct-4bit")
        tokenizer._detokenizer = NaiveStreamingDetokenizer(tokenizer)
        self.check_tokenizer(tokenizer)

    def test_bpe_multibyte_split_across_tokens(self):
        """A multi-byte UTF-8 char whose bytes span two BPE tokens, followed by
        an unrelated complete token, must NOT corrupt to U+FFFD.

        Regression for the DSv4 narrow-no-break-space (U+202F) bug: the model
        emits U+202F as two byte-level tokens ('âĢ' = e2,80 then '¯' = af). When
        the incomplete 'âĢ' half was buffered and the next word token arrived,
        the old add_token flushed the still-incomplete prefix as '�'
        (rendering 'Chain, or' as 'Chain,�or'). The detokenizer must instead
        retain the incomplete trailing bytes and only flush complete UTF-8.
        """
        from mlx_lm.tokenizer_utils import BPEStreamingDetokenizer

        # Build a minimal fake HF-like tokenizer exposing just what the
        # BPEStreamingDetokenizer __init__ needs: vocab (id->char) + the
        # clean_up flag. We map ids to the byte-level BPE chars for the three
        # bytes of U+202F (e2,80,af) plus a couple of ASCII chars.
        BPEStreamingDetokenizer.make_byte_decoder()
        inv = {b: c for c, b in BPEStreamingDetokenizer._byte_decoder.items()}
        ch_e2, ch_80, ch_af = inv[0xE2], inv[0x80], inv[0xAF]
        # token 0 -> 'âĢ' (e2,80), token 1 -> '¯' (af), 2 -> 'o', 3 -> 'r', 4 -> ','
        vocab = {
            ch_e2 + ch_80: 0,
            ch_af: 1,
            "o": 2,
            "r": 3,
            ",": 4,
        }

        class _FakeTok:
            def __init__(self, vocab):
                self.vocab = vocab
                self.clean_up_tokenization_spaces = False

        det = BPEStreamingDetokenizer(_FakeTok(vocab))

        # Stream: ',' then U+202F (two tokens) then 'o','r' -> ",\u202for"
        det.reset()
        out = ""
        for tid in [4, 0, 1, 2, 3]:
            det.add_token(tid)
            out += det.last_segment
        det.finalize()
        out += det.last_segment

        self.assertNotIn("\ufffd", out)
        self.assertEqual(out, ",\u202for")

        # The incomplete-half must be held back (empty segment), not flushed.
        det.reset()
        det.add_token(0)  # 'âĢ' — incomplete e2,80
        self.assertEqual(det.last_segment, "")
        det.add_token(1)  # '¯' — completes U+202F
        self.assertEqual(det.last_segment, "\u202f")

    def test_special_tokens(self):
        tokenizer_repo = "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx"
        tokenizer = load_tokenizer(tokenizer_repo)

        detokenizer = tokenizer.detokenizer
        detokenizer.reset()
        detokenizer.add_token(tokenizer.eos_token_id)
        detokenizer.finalize()

        self.assertEqual(detokenizer.last_segment, tokenizer.eos_token)

    def test_tool_calling(self):
        tokenizer_repo = "mlx-community/Qwen3-4B-4bit"
        tokenizer = load_tokenizer(tokenizer_repo)
        self.assertTrue(tokenizer.has_tool_calling)
        self.assertEqual(tokenizer.tool_call_start, "<tool_call>")
        self.assertEqual(tokenizer.tool_call_end, "</tool_call>")

        tokenizer_repo = "mlx-community/Llama-3.2-1B-Instruct-4bit"
        tokenizer = load_tokenizer(tokenizer_repo)
        self.assertFalse(tokenizer.has_tool_calling)

    def test_thinking(self):
        tokenizer_repo = "mlx-community/Qwen3-4B-4bit"
        tokenizer = load_tokenizer(tokenizer_repo)
        self.assertTrue(tokenizer.has_thinking)
        self.assertEqual(tokenizer.think_start, "<think>")
        self.assertEqual(tokenizer.think_end, "</think>")

        tokenizer_repo = "mlx-community/Llama-3.2-1B-Instruct-4bit"
        tokenizer = load_tokenizer(tokenizer_repo)
        self.assertFalse(tokenizer.has_thinking)
        self.assertIsNone(tokenizer.think_start)
        self.assertIsNone(tokenizer.think_end)
        self.assertIsNone(tokenizer.think_start_id)
        self.assertIsNone(tokenizer.think_end_id)

    def test_find_token(self):
        # Check that _find returns a valid index when
        # searching for a think token in short system prompts
        HI, THINK_START, THINK_END = 200, 100, 101
        find = TokenizerWrapper._find
        prompt = [HI]
        start = len(prompt) - 11
        self.assertEqual(find(prompt, [THINK_START], start=start), -1)
        self.assertEqual(find(prompt, [THINK_START], start=start, reverse=True), -1)
        prompt = [HI, THINK_START, THINK_END, THINK_START]
        self.assertEqual(find(prompt, [THINK_START], start=0), 1)
        self.assertEqual(find(prompt, [THINK_START], start=0, reverse=True), 3)


if __name__ == "__main__":
    unittest.main()
