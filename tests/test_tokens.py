import unittest
from unittest.mock import patch

from src.utils.tokens import get_encoding_for_model, num_tokens_from_string


class TokenTests(unittest.TestCase):
    def test_unknown_gpt5_snapshot_falls_back_to_o200k(self):
        with patch("src.utils.tokens.tiktoken.encoding_for_model", side_effect=KeyError):
            encoding = get_encoding_for_model("gpt-5.4-mini-2026-03-17")
        self.assertEqual(encoding.name, "o200k_base")
        self.assertGreater(num_tokens_from_string("citation retrieval", "gpt-4o"), 0)


if __name__ == "__main__":
    unittest.main()
