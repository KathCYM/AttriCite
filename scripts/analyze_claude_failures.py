"""Audit provider-format and action-budget failures in the Claude run."""

from __future__ import annotations

import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


PATH = Path(__file__).resolve().parents[1] / "results_v2/results/anthropic_temperature_sampling/claude_haiku45_test2025_temp0p7_run1.json"


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def title_match(left: str, right: str) -> bool:
    return SequenceMatcher(None, normalize(left), normalize(right)).ratio() > 0.8


def papers_by_id(row: dict) -> dict[str, str]:
    output = {}
    for group in row.get("papers") or []:
        if not isinstance(group, list):
            continue
        for paper in group:
            if isinstance(paper, dict):
                paper_id = paper.get("paperId") or paper.get("paper_id") or paper.get("id")
                if paper_id and paper.get("title"):
                    output[str(paper_id)] = str(paper["title"])
    return output


def retrieved(row: dict) -> bool:
    targets = row.get("cited_paper_titles") or []
    return any(title_match(target, title) for target in targets for title in papers_by_id(row).values())


def category(error: str) -> str:
    if error == "Max actions reached":
        return "max_actions"
    if "No complete JSON block" in error:
        return "no_json"
    if "Failed to parse Output" in error:
        return "schema_parse"
    return "other"


def intended_action(error: str) -> dict | None:
    if "Failed to parse Output from completion " not in error:
        return None
    raw = error.split("Failed to parse Output from completion ", 1)[1].split(". Got:", 1)[0]
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if value.get("type") == "object" and isinstance(value.get("properties"), dict):
        return value["properties"].get("action")
    if isinstance(value.get("action"), dict):
        return value["action"]
    return value


def main() -> None:
    payload = json.loads(PATH.read_text(encoding="utf-8"))
    rows = payload["results"]
    errors = [row for row in rows if row.get("status") == "error" or row.get("error")]
    categories = Counter(category(str(row.get("error") or "")) for row in errors)
    retrieved_by_category = Counter()
    actions = Counter()
    intended_selects = 0
    intended_correct_selects = 0

    for row in errors:
        cat = category(str(row.get("error") or ""))
        retrieved_by_category[cat] += retrieved(row)
        action = intended_action(str(row.get("error") or ""))
        if not action:
            continue
        name = str(action.get("name") or "unknown")
        actions[name] += 1
        if name != "select":
            continue
        intended_selects += 1
        selected_title = papers_by_id(row).get(str(action.get("paper_id")))
        if selected_title and any(title_match(target, selected_title) for target in row.get("cited_paper_titles") or []):
            intended_correct_selects += 1

    print(f"Errors: {len(errors)}/{len(rows)}")
    for cat, count in categories.most_common():
        print(f"  {cat}: {count}; target retrieved in {retrieved_by_category[cat]}")
    print("Intended actions in parse failures:")
    for name, count in actions.most_common():
        print(f"  {name}: {count}")
    print(f"Intended selects matching a retrieved target: {intended_correct_selects}/{intended_selects}")


if __name__ == "__main__":
    main()
