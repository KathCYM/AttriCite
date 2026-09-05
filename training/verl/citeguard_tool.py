"""CiteGuard's full inference action environment for veRL ToolAgentLoop."""

from __future__ import annotations

import asyncio
import io
import re
import uuid
from difflib import SequenceMatcher
from typing import Any

import requests
from PyPDF2 import PdfReader
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import ToolResponse

from src.retriever.observation_limits import MAX_ABSTRACT_CHARS, bound_observation
from src.retriever.search_provider import SemanticScholarSearchProvider
from src.utils.entity_matcher import identifier_matches


class _SilentConsole:
    def log(self, *_args: Any, **_kwargs: Any) -> None:
        pass


def _extra_fields(agent_data: Any) -> dict:
    if agent_data is None:
        return {}
    if isinstance(agent_data, dict):
        return agent_data.setdefault("extra_fields", {})
    fields = getattr(agent_data, "extra_fields", None)
    if fields is None:
        fields = {}
        setattr(agent_data, "extra_fields", fields)
    return fields


def _paper_dict(paper: Any) -> dict:
    return paper.model_dump() if hasattr(paper, "model_dump") else dict(paper)


def _pdf_url(paper: dict) -> str:
    return str((paper.get("openAccessPdf") or {}).get("url") or "")


def _download_pdf_text(paper: dict) -> str:
    url = _pdf_url(paper)
    if not url:
        return "This paper does not have an open access PDF."
    headers = {"User-Agent": "Mozilla/5.0"}
    if "aaai" in url.casefold():
        headers["Referer"] = url.replace("/download/", "/view/").rsplit("/", 1)[0]
    try:
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        reader = PdfReader(io.BytesIO(response.content))
        text = "".join((page.extract_text() or "") for page in reader.pages)
        return bound_observation(text) if text else "There was an error reading the PDF. Please try a different paper."
    except Exception as exc:
        return f"There was an error reading the PDF. Please try a different paper. {exc}"


def _format_search(papers: list[dict]) -> str:
    if not papers:
        return "No papers were found for the given search query. Please use a different query."
    chunks = []
    for paper in papers:
        chunks.append(
            f"- Paper ID: {paper.get('paperId')}\n"
            f"   Title: {paper.get('title')}\n"
            f"   Abstract: {(paper.get('abstract') or '')[:MAX_ABSTRACT_CHARS]}\n"
            f"   Citation Count: {paper.get('citationCount')}"
        )
    return bound_observation("\n\n".join(chunks))


class CiteGuardTool(BaseTool):
    """The seven actions and paper-buffer rules used by the inference agent."""

    def __init__(self, config: dict, tool_schema: Any):
        super().__init__(config, tool_schema)
        self.search_limit = int(config.get("search_limit", 10))
        self._instances: dict[str, dict] = {}
        self.provider = SemanticScholarSearchProvider(limit=self.search_limit, console=_SilentConsole())

    async def create(self, instance_id: str | None = None, **kwargs: Any):
        instance_id = instance_id or str(uuid.uuid4())
        # Current veRL passes the per-example values under a create_kwargs key.
        create_kwargs = kwargs.get("create_kwargs", kwargs)
        self._instances[instance_id] = dict(create_kwargs or {})
        return instance_id, ToolResponse(text="CiteGuard is ready.")

    async def execute(self, instance_id: str, parameters: dict, **kwargs: Any):
        initial = self._instances.get(instance_id, {})
        state = _extra_fields(kwargs.get("agent_data")).setdefault(
            "citeguard_state",
            {
                "year": initial.get("year"),
                "source_title": initial.get("source_title", ""),
                "target_title": initial.get("target_title", ""),
                "target_id": str(initial.get("target_id") or ""),
                "all_papers": {},
                "latest_ids": [],
                "actions": 0,
                "selected": False,
            },
        )
        state["actions"] += 1
        if state["actions"] > 5:
            return ToolResponse(text="Max actions reached."), 0.0, {"budget_exceeded": 1}
        if state["selected"]:
            return ToolResponse(text="A paper was already selected; stop."), 0.0, {}

        action = str(parameters.get("action") or "")
        query = str(parameters.get("query") or "").strip()
        paper_id = str(parameters.get("paper_id") or "")
        skip = [state["source_title"]] if state["source_title"] else []

        if action in {"search_relevance", "search_citation_count"}:
            if not query:
                return ToolResponse(text="A non-empty query is required."), 0.0, {"invalid_action": 1}
            try:
                if action == "search_citation_count":
                    found = await asyncio.to_thread(
                        self.provider.citation_count_search, query, state["year"], 100, skip
                    )
                else:
                    found = await asyncio.to_thread(self.provider, query, state["year"], skip)
            except Exception as exc:
                return ToolResponse(text=f"Search service error: {type(exc).__name__}: {exc}"), 0.0, {"infrastructure_error": 1}
            papers = [_paper_dict(paper) for paper in found]
            if state["source_title"]:
                papers = [
                    paper
                    for paper in papers
                    if SequenceMatcher(
                        None,
                        state["source_title"].casefold(),
                        str(paper.get("title") or "").casefold(),
                    ).ratio()
                    <= 0.8
                ]
            state["latest_ids"] = [str(paper.get("paperId") or "") for paper in papers]
            state["all_papers"].update(
                {str(paper["paperId"]): paper for paper in papers if paper.get("paperId")}
            )
            return ToolResponse(text=_format_search(papers)), 0.0, {"search_results": len(papers)}

        if action == "search_text_snippet":
            if not query:
                return ToolResponse(text="A non-empty query is required."), 0.0, {"invalid_action": 1}
            try:
                text = await asyncio.to_thread(
                    self.provider.snippet_search, query, state["year"], state["source_title"], skip
                )
                return ToolResponse(text=bound_observation(text)), 0.0, {}
            except Exception as exc:
                return ToolResponse(text=f"Snippet search error: {type(exc).__name__}: {exc}"), 0.0, {"infrastructure_error": 1}

        if action == "ask_for_more_context":
            paper_title = str(parameters.get("paper_title") or "").strip()
            if not query or not paper_title:
                return ToolResponse(text="Both query and paper_title are required."), 0.0, {"invalid_action": 1}
            return ToolResponse(
                text=(
                    "No additional context was provided by the user. "
                    f"Paper title: {paper_title}. Focus query: {query}. "
                    "Continue using only the original excerpt."
                )
            ), 0.0, {}

        if action in {"read", "find_in_text"}:
            if paper_id not in state["latest_ids"]:
                return ToolResponse(text=f"Paper {paper_id} not found in the latest search buffer."), 0.0, {"invalid_paper_id": 1}
            if action == "find_in_text" and not query:
                return ToolResponse(text="A non-empty query is required."), 0.0, {"invalid_action": 1}
            paper = state["all_papers"][paper_id]
            text = await asyncio.to_thread(_download_pdf_text, paper)
            if action == "find_in_text" and not text.startswith(("There was an error", "This paper does not")):
                sentences = re.split(r"(?<=[.!?])\s+", text.strip())
                matches = [sentence for sentence in sentences if query.casefold() in sentence.casefold()]
                text = "\n".join(matches) if matches else f"No sentence found containing '{query}'."
            return ToolResponse(text=bound_observation(text)), 0.0, {}

        if action == "select":
            paper = state["all_papers"].get(paper_id)
            if paper is None:
                return ToolResponse(text=f"Paper {paper_id} not found in the search buffer."), 0.0, {"invalid_paper_id": 1}
            state["selected"] = True
            correct = identifier_matches(paper, state["target_id"])
            state["correct_selection"] = bool(correct)
            return (
                ToolResponse(text=f"Selection recorded: {paper.get('title')}. Stop now."),
                float(correct),
                {"correct_selection": int(correct), "actions": state["actions"]},
            )

        return ToolResponse(text="Unknown action."), 0.0, {"invalid_action": 1}

    async def calc_reward(self, instance_id: str, **kwargs: Any) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs: Any) -> None:
        self._instances.pop(instance_id, None)
