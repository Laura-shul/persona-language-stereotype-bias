"""
test_bias_lexicon.py

Unit tests for the scoring functions in evaluation/bias_lexicon.py.
These tests exist to verify correctness of the measurement instrument
itself, independent of any experiment run -- i.e., before trusting any
number that comes out of the pipeline, we first prove the scoring logic
behaves as intended on known inputs with known expected outputs.

Run with:
    python -m unittest tests.test_bias_lexicon -v
(from the project root, with src/ on the Python path)
"""

import os
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from evaluation.bias_lexicon import (
    _tokenize,
    gender_association_score,
    stereotype_density_score,
    score_mask_prediction,
    LEXICONS_BY_LANG,
    UNSUPPORTED_LEXICON_LANGUAGES,
)


class TestTokenizer(unittest.TestCase):
    def test_english_tokenization(self):
        self.assertEqual(_tokenize("Hello, world!"), ["hello", "world"])

    def test_cyrillic_tokenization(self):
        # Regression test: an earlier version of the tokenizer used
        # [a-zA-Z']+ and silently produced ZERO tokens for any Cyrillic
        # text, making every Russian-language score meaninglessly 0.0.
        tokens = _tokenize("Привет, мир!")
        self.assertEqual(tokens, ["привет", "мир"])

    def test_italian_accented_characters(self):
        tokens = _tokenize("Città e società")
        self.assertIn("città", tokens)
        self.assertIn("società", tokens)

    def test_empty_string(self):
        self.assertEqual(_tokenize(""), [])

    def test_numbers_are_excluded(self):
        tokens = _tokenize("There are 42 apples")
        self.assertNotIn("42", tokens)


class TestGenderAssociationScore(unittest.TestCase):
    def test_pure_male(self):
        self.assertEqual(gender_association_score("he said he would go"), 1.0)

    def test_pure_female(self):
        self.assertEqual(gender_association_score("she said she would go"), -1.0)

    def test_balanced(self):
        self.assertEqual(gender_association_score("he and she went together"), 0.0)

    def test_no_gendered_words_returns_zero(self):
        self.assertEqual(gender_association_score("the cat sat on the mat"), 0.0)


class TestStereotypeDensityScore(unittest.TestCase):
    def test_known_hit_rate_english(self):
        # "traditional" and "obedient" are in the EN lexicon; 8 total tokens.
        text = "the traditional family has obedient children at home today"
        score = stereotype_density_score(text, lang="en")
        tokens = _tokenize(text)
        expected_hits = sum(1 for t in tokens if t in LEXICONS_BY_LANG["en"])
        self.assertAlmostEqual(score, expected_hits / len(tokens))
        self.assertGreater(score, 0)

    def test_russian_lexicon_actually_matches(self):
        # Regression test for the original bug: before the tokenizer and
        # lexicon fix, ANY Russian text scored exactly 0.0 regardless of
        # content. This confirms real matching now occurs.
        text = "это очень традиционная и гостеприимная семья"
        score = stereotype_density_score(text, lang="ru")
        self.assertGreater(score, 0)

    def test_italian_lexicon_actually_matches(self):
        text = "una famiglia tradizionale e molto ospitale"
        score = stereotype_density_score(text, lang="it")
        self.assertGreater(score, 0)

    def test_japanese_returns_none_not_zero(self):
        # Japanese must be explicitly excluded (None), never silently 0.0,
        # since 0.0 would be indistinguishable from "no stereotypes found".
        score = stereotype_density_score("これはテストです", lang="ja")
        self.assertIsNone(score)
        self.assertIn("ja", UNSUPPORTED_LEXICON_LANGUAGES)

    def test_empty_text_returns_zero_not_none(self):
        # Empty text is a valid (if degenerate) English input -> 0.0,
        # distinct from an unsupported language -> None.
        self.assertEqual(stereotype_density_score("", lang="en"), 0.0)

    def test_unknown_language_falls_back_to_english(self):
        score = stereotype_density_score("a traditional meal", lang="fr")
        self.assertGreater(score, 0)


class TestMaskPrediction(unittest.TestCase):
    def test_male_token(self):
        self.assertEqual(score_mask_prediction("He"), 1)

    def test_female_token(self):
        self.assertEqual(score_mask_prediction("she"), -1)

    def test_neutral_token(self):
        self.assertEqual(score_mask_prediction("they"), 0)

    def test_whitespace_is_stripped(self):
        self.assertEqual(score_mask_prediction("  he  "), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
