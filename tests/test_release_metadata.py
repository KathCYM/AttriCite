import hashlib
import json

from training_data_collection.collect_numeric_cs_candidates import normalize_spaces
from training_data_collection.export_release_metadata import PRIVATE_FIELDS


def test_private_release_fields_cover_third_party_text_and_paths():
    assert {
        "raw_sentence",
        "raw_reference",
        "excerpt",
        "pdf_path",
        "discoverability_query",
    } <= PRIVATE_FIELDS


def test_passage_fingerprint_is_normalized_and_stable():
    left = "A  citation\n sentence."
    right = "A citation sentence."
    digest = lambda value: hashlib.sha256(normalize_spaces(value).encode("utf-8")).hexdigest()
    assert digest(left) == digest(right)


def test_public_example_contains_no_passage_text():
    public = {
        "id": 1,
        "source_paper_url": "https://example.org/paper",
        "passage_sha256": "0" * 64,
        "body_sentence_index": 3,
    }
    serialized = json.dumps(public)
    assert all(field not in serialized for field in PRIVATE_FIELDS)
