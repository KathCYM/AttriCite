"""Convert CiteGuard CSV splits to veRL's tool-agent parquet format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset

from src.retriever.prompt_config import CITATION_HUMAN_INTRO
from src.utils.entity_matcher import normalize_title


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA = ROOT / "release" / "citealign_metadata.jsonl"


JSON_FORMAT_INSTRUCTIONS = """Return exactly one JSON object with this structure:
{
  "reason": "a short explanation of the decision",
  "action": {
    "name": "one of search_relevance, search_citation_count, search_text_snippet, read, find_in_text, ask_for_more_context, select",
    "query": "required for searches, find_in_text, and ask_for_more_context",
    "paper_id": "required for read, find_in_text, and select",
    "paper_title": "required for ask_for_more_context"
  }
}
Include only fields required by the chosen action. Do not invent a paper ID. You have at most
5 actions. `read` and `find_in_text` may use only a paper from the latest paper search;
`select` may use a paper from any prior paper search. A valid `select` is terminal."""


def _system_prompt() -> str:
    prompt_path = (
        Path(__file__).resolve().parents[2]
        / "src/retriever/prompt_templates/few_shot_tool.txt"
    )
    prompt = prompt_path.read_text(encoding="utf-8")
    if "<FORMAT_INSTRUCTIONS>" not in prompt:
        raise ValueError(f"Missing <FORMAT_INSTRUCTIONS> in {prompt_path}")
    return prompt.replace("<FORMAT_INSTRUCTIONS>", JSON_FORMAT_INSTRUCTIONS)


SYSTEM_PROMPT = _system_prompt()


def _target_ids(metadata_path: Path) -> dict[tuple[str, str], str]:
    targets: dict[tuple[str, str], str] = {}
    with metadata_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["id"]), normalize_title(str(row.get("source_paper_title") or "")))
            target_id = str(row.get("resolved_paper_id") or "").strip()
            if target_id:
                targets[key] = target_id
    return targets


def _records(csv_path: Path, metadata_path: Path = DEFAULT_METADATA) -> list[dict]:
    source = Dataset.from_csv(str(csv_path))
    has_embedded_ids = "target_paper_id" in source.column_names
    target_ids = {} if has_embedded_ids else _target_ids(metadata_path)
    rows: list[dict] = []
    for index, row in enumerate(source):
        excerpt = str(row["excerpt"])
        year = int(row["year"])
        source_title = str(row.get("source_paper_title") or "")
        target_title = str(row["target_paper_title"])
        collection_id = str(row.get("original_id") or row.get("id", index))
        target_id = str(row.get("target_paper_id") or "").strip()
        if not target_id:
            target_id = target_ids.get((collection_id, normalize_title(source_title)), "")
        if not target_id:
            raise ValueError(
                f"No canonical target identifier for split row {row.get('id', index)} "
                f"(collection id {collection_id})."
            )
        rows.append(
            {
                "data_source": "citeguard",
                "agent_name": "citeguard_agent",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        # Match LLMSelfAskAgentPydantic.__call__. The year and
                        # source title constrain tools but are not shown here.
                        "content": f"{CITATION_HUMAN_INTRO}\n\n{excerpt}",
                    },
                ],
                "ability": "citation_retrieval",
                "reward_model": {"style": "rule", "ground_truth": target_id},
                "extra_info": {
                    "index": index,
                    "row_id": str(row.get("id", index)),
                    "venue": str(row.get("venue") or ""),
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        "citeguard": {
                            "create_kwargs": {
                                "year": str(year),
                                "source_title": source_title,
                                "target_title": target_title,
                                "target_id": target_id,
                            }
                        }
                    },
                },
            }
        )
    return rows


def convert(csv_path: Path, output_path: Path, metadata_path: Path = DEFAULT_METADATA) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(_records(csv_path, metadata_path)).to_parquet(str(output_path))
    print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path)
    parser.add_argument("--metadata-jsonl", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training/verl/data"),
    )
    args = parser.parse_args()

    convert(args.train_csv, args.output_dir / "train.parquet", args.metadata_jsonl)
    convert(args.validation_csv, args.output_dir / "validation.parquet", args.metadata_jsonl)
    if args.test_csv is not None:
        convert(args.test_csv, args.output_dir / "test_2025.parquet", args.metadata_jsonl)


if __name__ == "__main__":
    main()
