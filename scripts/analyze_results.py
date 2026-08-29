"""Print a compact diagnostic summary for a CiteGuard result JSON."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def error_category(message: str) -> str:
    text = message.casefold()
    if "maximum context length" in text or "context length" in text:
        return "context_overflow"
    if "no complete json block" in text:
        return "invalid_json"
    if "json" in text and ("decode" in text or "valid" in text or "parse" in text):
        return "invalid_json"
    if "rate limit" in text or "status code: 429" in text or "error code: 429" in text:
        return "rate_limit"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "connection" in text or "connecterror" in text:
        return "connection"
    if "paper with id" in text and "not found" in text:
        return "invalid_paper_id"
    return "other"


def assistant_actions(item: dict) -> list[str]:
    actions: list[str] = []
    for message in item.get("history") or []:
        if message.get("role") != "assistant":
            continue
        content = message.get("content") or ""
        matches = re.findall(r'["\']name["\']\s*:\s*["\']([^"\']+)', content)
        actions.extend(matches[:1] or ["unparsed_assistant_action"])
    return actions


def pct(numerator: int, denominator: int) -> str:
    return f"{100 * numerator / denominator:.2f}%" if denominator else "n/a"


def target_retrieved(item: dict) -> bool:
    """Return whether structured search logged the gold target.

    ``is_in_search`` is a zero-based matched rank, not a Boolean.  In
    particular, rank 0 is a successful retrieval and must not be treated as
    false.  ``None`` represents no structured match (or an unavailable value).
    """
    return item.get("is_in_search") is not None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json", type=Path)
    parser.add_argument("--top-errors", type=int, default=10)
    args = parser.parse_args()

    payload = json.loads(args.result_json.read_text(encoding="utf-8"))
    rows = payload.get("results", []) if isinstance(payload, dict) else payload
    total = len(rows)
    errors = [row for row in rows if row.get("status") == "error" or row.get("error")]
    successful = [row for row in rows if row not in errors]
    correct = sum(row.get("is_correct") is True for row in rows)
    wrong = sum(row.get("is_correct") is False for row in rows)
    in_search = sum(target_retrieved(row) for row in rows)
    selected_given_retrieval = sum(
        row.get("is_correct") is True and target_retrieved(row) for row in rows
    )

    print(f"File: {args.result_json}")
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    print(f"Model: {metadata.get('model', 'unknown')}")
    print(f"Examples: {total}")
    print(f"Correct: {correct}/{total} = {pct(correct, total)}")
    print(f"Wrong: {wrong}; Errors: {len(errors)}; Successful: {len(successful)}")
    print(f"Accuracy among successful: {correct}/{len(successful)} = {pct(correct, len(successful))}")
    print(
        f"Gold found in structured search: {in_search}/{total} = "
        f"{pct(in_search, total)}"
    )
    print(
        "Correct given structured retrieval: "
        f"{selected_given_retrieval}/{in_search} = "
        f"{pct(selected_given_retrieval, in_search)}"
    )

    categories = Counter(error_category(str(row.get("error") or "")) for row in errors)
    print("\nError categories")
    if not categories:
        print("  none")
    for category, count in categories.most_common():
        print(f"  {count:4d}  {category}")

    exact = Counter(str(row.get("error") or "").replace("\n", " ") for row in errors)
    if exact:
        print("\nMost common exact errors")
        for message, count in exact.most_common(args.top_errors):
            clipped = message if len(message) <= 240 else message[:237] + "..."
            print(f"  {count:4d}  {clipped}")

    action_types: Counter[str] = Counter()
    by_count: dict[int, Counter[str]] = defaultdict(Counter)
    for row in rows:
        actions = assistant_actions(row)
        action_types.update(actions)
        outcome = "error" if row in errors else "correct" if row.get("is_correct") else "wrong"
        by_count[len(actions)][outcome] += 1

    print("\nActions | All | Correct | Wrong | Error")
    for count in sorted(by_count):
        values = by_count[count]
        all_count = sum(values.values())
        print(
            f"{count:7d} | {all_count:3d} | {values['correct']:7d} | "
            f"{values['wrong']:5d} | {values['error']:5d}"
        )

    print("\nAction types")
    for action, count in action_types.most_common():
        print(f"  {count:4d}  {action}")

    search_actions = {
        "search_relevance",
        "search_citation_count",
        "search_text_snippet",
    }
    inspection_actions = {"read", "find_in_text"}
    action_total = sum(action_types.values())
    search_total = sum(action_types[action] for action in search_actions)
    inspection_total = sum(action_types[action] for action in inspection_actions)
    print("\nMean tool use per example")
    print(f"  searches:    {search_total / total:.3f}" if total else "  searches:    n/a")
    print(
        f"  inspections: {inspection_total / total:.3f}"
        if total
        else "  inspections: n/a"
    )
    print(f"  all actions: {action_total / total:.3f}" if total else "  all actions: n/a")


if __name__ == "__main__":
    main()
