"""Fail fast on common AttriCite public-release mistakes."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEIGHT_SUFFIXES = {".bin", ".ckpt", ".pt", ".pth", ".safetensors"}
FORBIDDEN_TRACKED_PARTS = {"__pycache__", "cache", "rollouts", "validation_rollouts"}
SECRET_PATTERN = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?token|secret)[ \t]*[:=][ \t]*['\"]?[A-Za-z0-9_-]{16,}"
)
PRIVATE_PATH_PATTERN = re.compile(r"(?i)(?:/home/|/scratch/|/gpfs/|[A-Z]:\\Users\\)")


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    # A release-preparation checkout can contain tracked files that have been
    # deleted but not committed yet. Validate the prospective tree, not stale
    # index entries for paths that no longer exist on disk.
    return [
        ROOT / line
        for line in result.stdout.splitlines()
        if line and (ROOT / line).is_file()
    ]


def main() -> int:
    errors: list[str] = []
    required = [
        "LICENSE",
        "LICENSE_DATASET",
        "MODEL_CARD.md",
        "DATASET_CARD.md",
        "RELEASE.md",
        "training_data_collection/export_release_metadata.py",
        "training_data_collection/finalize_release_dataset.py",
        "training_data_collection/reconstruct_passages.py",
        "training/verl/run_grpo_qwen3_4b.sh",
        "release/citealign_metadata.jsonl",
        "release/validation_report.json",
    ]
    for relative in required:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required artifact: {relative}")

    model_card = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
    if "TODO_RELEASE_" in model_card:
        errors.append("MODEL_CARD.md still contains TODO_RELEASE_* placeholders")
    dataset_card = (ROOT / "DATASET_CARD.md").read_text(encoding="utf-8")
    if "TODO_RELEASE_" in dataset_card:
        errors.append("DATASET_CARD.md still contains TODO_RELEASE_* placeholders")

    for path in tracked_files():
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in WEIGHT_SUFFIXES:
            errors.append(f"model weight is in the Git release: {relative}")
        if any(part in FORBIDDEN_TRACKED_PARTS for part in relative.parts):
            errors.append(f"generated/private directory is in the Git release: {relative}")
        if relative.as_posix() == "scripts/validate_release.py":
            continue
        if path.suffix.lower() not in {".py", ".sh", ".md", ".toml", ".yaml", ".yml", ".txt", ".example"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if SECRET_PATTERN.search(text):
            errors.append(f"possible embedded credential: {relative}")
        if PRIVATE_PATH_PATTERN.search(text):
            errors.append(f"machine-specific absolute path: {relative}")

    if errors:
        print("Release validation failed:")
        for error in sorted(set(errors)):
            print(f"- {error}")
        return 1
    print("Release validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
