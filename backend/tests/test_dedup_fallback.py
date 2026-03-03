import os
# Import the class (assuming path is correct in the actual run)
# We will use this file inside the repo
import sys
import unittest
from unittest.mock import MagicMock

sys.path.append(os.getcwd())
# Adjust import path based on where we run it.
# If running from root:
from services.optimizer.core import PromptOptimizer  # noqa: E402


class TestDedupFallback(unittest.TestCase):
    def test_dedup_fallback_preserves_content_with_extra_sentences(self):
        optimizer = PromptOptimizer()
        optimizer.semantic_deduplication_enabled = True
        optimizer.semantic_similarity_threshold = 0.75

        # Force fallback by mocking _get_nlp_model to return None
        optimizer._get_nlp_model = MagicMock(return_value=None)

        # Create input sentences.
        # S1: Base sentence.
        # S2: Base sentence + extra content (that counts as a minimal sentence/phrase).
        # We need high token overlap so Jaccard > 0.75.
        # Words in S1: 10 words.
        # Words in S2: 10 words + 1 word (11).
        # Jaccard = 10 / 11 = 0.90.
        # This SHOULD trigger deduplication with current logic (0.90 > 0.75).
        # But we want it NOT to trigger because sentence structure/count implies significant difference?
        # Actually, if sentence count is the same (1 vs 1), ratio is 1.0.
        # Then (0.9*0.5 + 0.9*0.25 + 1.0*0.25) = 0.45 + 0.225 + 0.25 = 0.925. Still removed.

        # We need S2 to have MORE SENTENCES.
        # S1: "This is a long sentence for testing overlap." (1 sent)
        # S2: "This is a long sentence for testing overlap. Different." (2 sents?)
        # NOTE: The Splitter must split "Different." as part of S2 if it wasn't split by the main splitter.
        # But `_deduplicate_content` splits by `[.!?]`.
        # So input to `_deduplicate_content` is the FULL TEXT string.
        # Then it splits into sentences.
        # So "S1. S2." would be split into ["S1.", "S2."].

        # Wait. If I pass S1 and S2 as separate sentences to the logic.
        # The logic iterates over `unique_sentences`.
        # If I have:
        # Sent A: "A B C D E F."
        # Sent B: "A B C D E F. G."
        # If the splitter splits B into "A B C D E F." and "G.".
        # Then "A B C D E F." is duplicate of Sent A. G is new.
        # Then we are fine.

        # The issue arises when the splitter FAILS to split Sent B, OR if Sent B is legitimately one sentence
        # but structurally different?
        # "A B C D E F, G." (comma).
        # Sentence count 1 vs 1.

        # User says "add sentence-count ratio".
        # This implies comparing the count of internal sentences?
        # Maybe `_lexical_similarity` uses `[.!?]` count.
        # "A B C. D E F." -> 2 sentences.

        # Scenario:
        # Text: "Part one. Part two. Part one. Part two. Extra."
        # Sentences: ["Part one.", "Part two.", "Part one.", "Part two.", "Extra."]
        # Unique: ["Part one.", "Part two.", "Extra."] (Exact dedup handles rest).

        # Near DUP scenario:
        # A: "This is a base sentence."
        # B: "This is a base sentence! And more."
        # If B is treated as ONE element in `unique_sentences` (because `re.split` didn't split it?).
        # `unique_sentences` comes from `re.split(r"(?:\n+|(?<=[.!?])\s+)", text)`.
        # It splits on Newline OR Space after Punctuation.
        # "This is a base sentence! And more." (Space after !) -> Splits into "This is a base sentence!", "And more."
        # So it splits properly usually.

        # When does it NOT split?
        # "This is a base sentence!And more." (No space).
        # Then B is "This is a base sentence!And more."
        # Token overlap high.
        # Sentence count: A has ".", B has "!" and ".". (2 counts).
        # Match!

        # So test case:
        s1 = "This is a very long sentence used for testing deduplication fallback logic."
        s2 = "This is a very long sentence used for testing deduplication fallback logic!AndExtra."

        # Construct `unique_sentences` list manually and invoke the logic?
        # Or mock the `sentences` variable inside the method?
        # I can't mock local variables.
        # I have to pass a text that generates this.
        text = f"{s1}\n{s2}"

        # Expectation: s2 should NOT be removed (replaced/cleaned).
        # Current logic (Jaccard):
        # Tokens s1: ~10. s2: ~12. Jaccard high.
        # It will likely remove s2.

        result = optimizer._deduplicate_content(text)

        # If deduplicated, result will miss s2 (or s2's content).
        print(f"Result: {result}")
        if s2 in result:
            print("S2 Preserved!")
        else:
            print("S2 Removed!")

        # With current Jaccard, likely S2 Removed.
        # With new logic, S2 Preserved.

        # Assertion
        # We want S2 to be preserved.
        if s2 not in result:
            raise AssertionError("S2 was deduplicated incorrectly!")
