"""Identifier utilities shared by evaluation and training rewards."""

from __future__ import annotations

import re
import unicodedata


def normalize_title(title: str) -> str:
    text = unicodedata.normalize("NFKC", title).casefold()
    return " ".join(re.findall(r"[\w]+", text, flags=re.UNICODE))


def identifier_matches(paper: dict, target_id: str) -> bool:
    """Match S2 or namespaced external identifiers without consulting titles."""
    target = str(target_id or "").strip()
    if not target:
        return False
    paper_id = str(paper.get("paperId") or paper.get("paper_id") or "").strip()
    if paper_id and paper_id.casefold() == target.casefold():
        return True
    namespace, separator, value = target.partition(":")
    if not separator:
        return False
    external = {
        str(key).casefold(): str(item).strip()
        for key, item in (paper.get("externalIds") or paper.get("external_ids") or {}).items()
    }
    key = {
        "arxiv": "arxiv", "doi": "doi", "corpusid": "corpusid",
        "mag": "mag", "pubmed": "pubmed", "acl": "acl",
    }.get(namespace.casefold())
    if key and external.get(key, "").casefold() == value.strip().casefold():
        return True
    if namespace.casefold() == "url":
        return str(paper.get("url") or "").rstrip("/").casefold() == value.rstrip("/").casefold()
    return False
