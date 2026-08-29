import json

from training_data_collection.audit_full_dataset import (
    Decision,
    decide,
    strong_acceptance_evidence,
    structural_failures,
)


def base_row(**updates):
    row = {
        "id": "1",
        "source_paper_title": "Source",
        "source_paper_url": "https://example.org/source",
        "citation_marker": "(Smith, 2020)",
        "raw_sentence": "We use the method introduced by Smith (Smith, 2020).",
        "raw_reference": "Smith. A useful method. 2020.",
        "resolved_paper_id": "abc",
        "resolved_title": "A useful method",
        "resolved_year": 2020,
        "sentence_citation_count": 1,
        "resolution_score": 0.9,
    }
    row.update(updates)
    return row


def test_clean_record_is_accepted():
    assert decide(base_row(), "sample.csv", {}).action == "accept"


def test_bibliography_fragment_is_rejected():
    row = base_row(raw_sentence="Smith, J. [CITATION] A useful method. arXiv:2001.00001, 2020.")
    assert "bibliography_entry_used_as_passage" in structural_failures(row)


def test_external_revision_has_precedence():
    adjudications = {
        ("sample.csv", "1"): {
            "action": "revise",
            "reasons": ["verified_target"],
            "revision": {"resolved_title": "Correct title"},
        }
    }
    decision = decide(base_row(), "sample.csv", adjudications)
    assert decision == Decision(
        "revise",
        ("verified_target",),
        {"resolved_title": "Correct title"},
        "external_adjudication",
    )


def test_missing_target_is_rejected():
    decision = decide(base_row(resolved_paper_id=""), "sample.csv", {})
    assert decision.action == "reject"
    assert "missing_resolved_paper_id" in decision.reasons


def test_exact_title_in_reference_clears_spillover_warning():
    row = base_row(
        raw_reference=(
            "Smith. A useful method. Proceedings of Example, 2020. "
            + "Other bibliography content. " * 100
        )
    )
    assert "exact_resolved_title_in_reference_prefix" in strong_acceptance_evidence(row)
    assert decide(row, "sample.csv", {}).action == "accept"
