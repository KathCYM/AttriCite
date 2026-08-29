import unittest

from training_data_collection.build_year_splits import (
    Candidate,
    diverse_sample,
    make_rl_train_validation_split,
    normalize_text,
    normalize_title,
)


def candidate(index: int, source: str, target: str) -> Candidate:
    return Candidate(
        row={
            "id": str(index),
            "excerpt": f"Example {index} [CITATION].",
            "target_paper_title": target,
            "target_paper_url": "",
            "source_paper_title": source,
            "source_paper_url": "",
            "year": "2024",
            "split": "train",
        },
        venue="ACL",
        year=2024,
        source_file="acl2024_candidates.csv",
    )


class BuildYearSplitsTests(unittest.TestCase):
    def test_normalization_is_case_and_punctuation_insensitive(self):
        self.assertEqual(normalize_title("A Paper: Test!"), "a paper test")
        self.assertEqual(normalize_text("  A\n  sentence  "), "a sentence")

    def test_diverse_sample_is_deterministic_and_prefers_distinct_sources(self):
        rows = [
            candidate(1, "Source A", "Target A"),
            candidate(2, "Source A", "Target B"),
            candidate(3, "Source B", "Target C"),
            candidate(4, "Source C", "Target D"),
        ]
        first = diverse_sample(rows, quota=3, seed=17)
        second = diverse_sample(rows, quota=3, seed=17)
        self.assertEqual([row.row["id"] for row in first], [row.row["id"] for row in second])
        self.assertEqual(len({row.source_key for row in first}), 3)

    def test_diverse_sample_rejects_an_impossible_quota(self):
        with self.assertRaises(ValueError):
            diverse_sample([candidate(1, "Source", "Target")], quota=2, seed=17)

    def test_rl_validation_split_is_balanced_and_disjoint(self):
        rows = []
        venues = ("ACL", "CVPR", "ICLR", "ICML", "NeurIPS")
        for venue_index, venue in enumerate(venues):
            for index in range(4):
                unique_index = venue_index * 10 + index
                row = candidate(unique_index, f"{venue} Source {index}", f"{venue} Target {index}")
                rows.append(
                    Candidate(
                        row=row.row,
                        venue=venue,
                        year=row.year,
                        source_file=row.source_file,
                    )
                )
        train, validation = make_rl_train_validation_split(rows, 1, seed=23)
        self.assertEqual(len(train), 15)
        self.assertEqual(len(validation), 5)
        self.assertEqual({row.excerpt_key for row in train} & {row.excerpt_key for row in validation}, set())
        self.assertEqual({row.source_key for row in train} & {row.source_key for row in validation}, set())
        self.assertEqual({row.target_key for row in train} & {row.target_key for row in validation}, set())


if __name__ == "__main__":
    unittest.main()
