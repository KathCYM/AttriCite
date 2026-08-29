"""Reconstruct retrieval metrics from saved result JSON files.

The saved paper buffers cover relevance and citation-count search. Raw history
messages additionally preserve snippet-search observations, so this script
parses their returned titles and reports both structured-only and full-search
retrieval.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from difflib import SequenceMatcher
from pathlib import Path


def title_similarity(left: str, right: str) -> float:
    return SequenceMatcher(a=left.lower(), b=right.lower()).ratio()


def flatten_paper_titles(row: dict) -> list[str]:
    titles: list[str] = []
    for search_results in row.get("papers") or []:
        if not isinstance(search_results, list):
            continue
        for paper in search_results:
            if isinstance(paper, dict) and isinstance(paper.get("title"), str):
                titles.append(paper["title"])
    return titles


def flatten_snippet_titles(row: dict) -> list[str]:
    """Extract titles returned immediately after snippet-search actions."""
    titles: list[str] = []
    history = row.get("history") or []
    for index, message in enumerate(history[:-1]):
        if message.get("role") != "assistant":
            continue
        try:
            action = json.loads(message.get("content") or "").get("action", {})
        except (json.JSONDecodeError, AttributeError):
            continue
        if action.get("name") != "search_text_snippet":
            continue
        observation = history[index + 1]
        if observation.get("role") != "user":
            continue
        content = observation.get("content") or ""
        titles.extend(re.findall(r"(?m)^\s*Title:\s*(.+?)\s*$", content))
    return titles


def reconstructed_rank_from_titles(
    row: dict, candidates: list[str], threshold: float, target_titles: list[str] | None = None
) -> int | None:
    if not candidates:
        return None

    best_rank: int | None = None
    best_score = threshold
    for target in target_titles if target_titles is not None else (row.get("cited_paper_titles") or []):
        if not isinstance(target, str):
            continue
        for rank, candidate in enumerate(candidates):
            score = title_similarity(target, candidate)
            # Match run_main.py exactly: similarity must be strictly greater
            # than the threshold. Keep the first maximum, as numpy.argmax does.
            if score > best_score:
                best_score = score
                best_rank = rank
    return best_rank


def reconstructed_rank(row: dict, threshold: float, target_titles: list[str] | None = None) -> int | None:
    return reconstructed_rank_from_titles(row, flatten_paper_titles(row), threshold, target_titles)


def full_reconstructed_rank(row: dict, threshold: float, target_titles: list[str] | None = None) -> int | None:
    candidates = flatten_paper_titles(row) + flatten_snippet_titles(row)
    return reconstructed_rank_from_titles(row, candidates, threshold, target_titles)


def pct(numerator: int, denominator: int) -> str:
    return f"{100 * numerator / denominator:.2f}%" if denominator else "n/a"


def load_gold(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {str(row["id"]): row["target_paper_title"] for row in csv.DictReader(handle)}


def selected_is_correct(row: dict, gold: dict[str, str], threshold: float) -> bool:
    title = (row.get("selected") or {}).get("title")
    return bool(title) and title_similarity(title, gold[str(row["id"])]) > threshold


def analyze(path: Path, default_threshold: float, gold: dict[str, str] | None = None) -> dict[str, int | float | str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("results", []) if isinstance(payload, dict) else payload
    if gold is not None:
        rows = [row for row in rows if str(row["id"]) in gold]
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    threshold = float(metadata.get("threshold", default_threshold))

    retrieved = 0
    full_retrieved = 0
    snippet_only_retrieved = 0
    correct = 0
    errors = 0
    stored_comparable = 0
    stored_agreements = 0
    recovered_on_errors = 0

    for row in rows:
        target_titles = [gold[str(row["id"])]] if gold is not None else None
        rank = reconstructed_rank(row, threshold, target_titles)
        full_rank = full_reconstructed_rank(row, threshold, target_titles)
        was_retrieved = rank is not None
        was_fully_retrieved = full_rank is not None
        is_error = row.get("status") == "error" or bool(row.get("error"))
        retrieved += was_retrieved
        full_retrieved += was_fully_retrieved
        snippet_only_retrieved += was_fully_retrieved and not was_retrieved
        correct += selected_is_correct(row, gold, threshold) if gold is not None else row.get("is_correct") is True
        errors += is_error
        recovered_on_errors += is_error and was_retrieved

        # Only successful rows have an intentionally computed stored value.
        if not is_error:
            stored_comparable += 1
            stored_agreements += (row.get("is_in_search") is not None) == was_retrieved

    return {
        "file": str(path),
        "examples": len(rows),
        "correct": correct,
        "retrieved": retrieved,
        "full_retrieved": full_retrieved,
        "snippet_only_retrieved": snippet_only_retrieved,
        "retrieval_recall": retrieved / len(rows) if rows else 0.0,
        "selection_given_retrieval": correct / retrieved if retrieved else 0.0,
        "selection_given_full_retrieval": correct / full_retrieved if full_retrieved else 0.0,
        "errors": errors,
        "retrieved_on_errors": recovered_on_errors,
        "stored_agreements": stored_agreements,
        "stored_comparable": stored_comparable,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json", type=Path, nargs="+")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--gold-csv", type=Path)
    args = parser.parse_args()

    gold = load_gold(args.gold_csv) if args.gold_csv else None
    summaries = [analyze(path, args.threshold, gold) for path in args.result_json]
    for summary in summaries:
        print(f"File: {summary['file']}")
        print(
            "  Structured retrieval: "
            f"{summary['retrieved']}/{summary['examples']} = "
            f"{pct(int(summary['retrieved']), int(summary['examples']))}"
        )
        print(
            "  Full retrieval incl. snippet history: "
            f"{summary['full_retrieved']}/{summary['examples']} = "
            f"{pct(int(summary['full_retrieved']), int(summary['examples']))} "
            f"(snippet-only: {summary['snippet_only_retrieved']})"
        )
        print(
            "  Correct given retrieval: "
            f"{summary['correct']}/{summary['retrieved']} = "
            f"{pct(int(summary['correct']), int(summary['retrieved']))}"
        )
        print(
            "  Retrieved errored trajectories: "
            f"{summary['retrieved_on_errors']}/{summary['errors']}"
        )
        print(
            "  Agreement with stored field on successful rows: "
            f"{summary['stored_agreements']}/{summary['stored_comparable']}"
        )

    if len(summaries) > 1:
        examples = sum(int(item["examples"]) for item in summaries)
        correct = sum(int(item["correct"]) for item in summaries)
        retrieved = sum(int(item["retrieved"]) for item in summaries)
        full_retrieved = sum(int(item["full_retrieved"]) for item in summaries)
        snippet_only = sum(int(item["snippet_only_retrieved"]) for item in summaries)
        print("Pooled")
        print(f"  Structured retrieval: {retrieved}/{examples} = {pct(retrieved, examples)}")
        print(
            f"  Full retrieval incl. snippet history: {full_retrieved}/{examples} = "
            f"{pct(full_retrieved, examples)} (snippet-only: {snippet_only})"
        )
        print(
            f"  Correct given full retrieval: {correct}/{full_retrieved} = "
            f"{pct(correct, full_retrieved)}"
        )


if __name__ == "__main__":
    main()
