"""Reconstruct CiteAlign passages locally and verify their fingerprints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

import requests

from training_data_collection.collect_numeric_cs_candidates import (
    extract_pdf_text,
    normalize_body_text,
    normalize_spaces,
    split_body_and_references,
    split_sentences,
)


def _download(url: str, path: Path) -> None:
    response = requests.get(url, timeout=60, headers={"User-Agent": "AttriCite-reconstruction/1.0"})
    response.raise_for_status()
    if not response.content.startswith(b"%PDF"):
        raise ValueError("source did not return a PDF")
    path.write_bytes(response.content)


def reconstruct(metadata: Path, output_dir: Path) -> tuple[int, int]:
    pdf_dir = output_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "passages.jsonl"
    cache: dict[str, list[str]] = {}
    ok = failed = 0
    with metadata.open(encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as output:
        for line in source:
            row = json.loads(line)
            record_id = str(row["id"])
            try:
                url = row.get("source_pdf_url")
                index = row.get("body_sentence_index")
                if not url:
                    raise ValueError("missing source_pdf_url")
                suffix = Path(urlparse(url).path).suffix or ".pdf"
                pdf_path = pdf_dir / f"{record_id}{suffix}"
                if not pdf_path.exists():
                    _download(url, pdf_path)
                cache_key = str(pdf_path)
                if cache_key not in cache:
                    body, _ = split_body_and_references(extract_pdf_text(pdf_path))
                    cache[cache_key] = split_sentences(normalize_body_text(body))
                expected_digest = row["passage_sha256"]
                if index is None:
                    matches = [
                        sentence for sentence in cache[cache_key]
                        if hashlib.sha256(normalize_spaces(sentence).encode("utf-8")).hexdigest()
                        == expected_digest
                    ]
                    if len(matches) != 1:
                        raise ValueError(f"fingerprint scan found {len(matches)} matching sentences")
                    sentence = matches[0]
                else:
                    sentence = cache[cache_key][int(index)]
                    digest = hashlib.sha256(normalize_spaces(sentence).encode("utf-8")).hexdigest()
                    if digest != expected_digest:
                        raise ValueError("passage fingerprint mismatch")
                marker = str(row["citation_marker"])
                if marker not in sentence:
                    raise ValueError("citation marker not found in reconstructed sentence")
                result = {"id": row["id"], "excerpt": sentence.replace(marker, "[CITATION]", 1), "status": "ok"}
                ok += 1
            except Exception as exc:
                result = {"id": row.get("id"), "status": "error", "error": str(exc)}
                failed += 1
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
    return ok, failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    ok, failed = reconstruct(args.metadata, args.output_dir)
    print(f"Reconstructed {ok}; failed {failed}. See {args.output_dir / 'passages.jsonl'}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
