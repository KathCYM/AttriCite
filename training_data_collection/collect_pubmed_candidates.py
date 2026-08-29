from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.semantic_scholar import SemanticScholarAPI
from training_data_collection.collect_cs_conference_candidates import write_manifest_csv
from training_data_collection.collect_numeric_cs_candidates import (
    DEFAULT_MAX_REQUEST_RETRIES,
    DEFAULT_REQUEST_RETRY_BACKOFF_S,
    ManifestRow,
    default_resume_state_path,
    iter_rows,
    load_resume_state,
    normalize_spaces,
    run_with_retries,
)


EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EUROPE_PMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def ncbi_params(email: str | None, api_key: str | None) -> dict[str, str]:
    params = {"tool": "CiteGuard"}
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    return params


def build_pmc_query(year: int, query: str | None = None) -> str:
    required = f"{year}[PDAT] AND open access[filter]"
    return f"({query}) AND {required}" if query else required


def build_europe_pmc_query(year: int, query: str | None = None) -> str:
    required = f"PUB_YEAR:{year} AND OPEN_ACCESS:Y AND HAS_PDF:Y"
    return f"({query}) AND {required}" if query else required


def fetch_europe_pmc_summaries(
    query: str,
    max_articles: int,
    session: requests.Session,
    email: str | None = None,
    request_pause_s: float = 0.34,
    max_request_retries: int = DEFAULT_MAX_REQUEST_RETRIES,
    request_retry_backoff_s: float = DEFAULT_REQUEST_RETRY_BACKOFF_S,
) -> tuple[list[dict[str, Any]], int]:
    summaries: list[dict[str, Any]] = []
    cursor_mark = "*"
    total_matches = 0
    while len(summaries) < max_articles:
        page_size = min(1000, max_articles - len(summaries))
        params = {
            "query": query,
            "format": "json",
            "resultType": "core",
            "pageSize": str(page_size),
            "cursorMark": cursor_mark,
        }
        if email:
            params["email"] = email
        response = run_with_retries(
            action_label="Searching Europe PMC for indexed open-access PDFs",
            func=lambda params=params: session.get(
                EUROPE_PMC_SEARCH_URL, params=params, timeout=60
            ),
            max_retries=max_request_retries,
            retry_backoff_s=request_retry_backoff_s,
        )
        response.raise_for_status()
        payload = response.json()
        total_matches = int(payload.get("hitCount", 0))
        page = payload.get("resultList", {}).get("result", [])
        summaries.extend(page)
        next_cursor = payload.get("nextCursorMark")
        if not page or not next_cursor or next_cursor == cursor_mark:
            break
        cursor_mark = next_cursor
        if request_pause_s > 0:
            time.sleep(request_pause_s)
    return summaries[:max_articles], total_matches


def fetch_pmc_ids(
    query: str,
    max_articles: int,
    session: requests.Session,
    email: str | None = None,
    api_key: str | None = None,
    max_request_retries: int = DEFAULT_MAX_REQUEST_RETRIES,
    request_retry_backoff_s: float = DEFAULT_REQUEST_RETRY_BACKOFF_S,
) -> tuple[list[str], int]:
    params = {
        **ncbi_params(email, api_key),
        "db": "pmc",
        "term": query,
        "retmode": "json",
        "retmax": str(max_articles),
    }
    response = run_with_retries(
        action_label="Searching the PMC Open Access subset",
        func=lambda: session.get(f"{EUTILS_BASE_URL}/esearch.fcgi", params=params, timeout=60),
        max_retries=max_request_retries,
        retry_backoff_s=request_retry_backoff_s,
    )
    response.raise_for_status()
    result = response.json().get("esearchresult", {})
    return [str(value) for value in result.get("idlist", [])], int(result.get("count", 0))


def fetch_pmc_summaries(
    pmc_ids: list[str],
    session: requests.Session,
    email: str | None = None,
    api_key: str | None = None,
    batch_size: int = 200,
    request_pause_s: float = 0.34,
    max_request_retries: int = DEFAULT_MAX_REQUEST_RETRIES,
    request_retry_backoff_s: float = DEFAULT_REQUEST_RETRY_BACKOFF_S,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for start in range(0, len(pmc_ids), batch_size):
        batch = pmc_ids[start : start + batch_size]
        params = {
            **ncbi_params(email, api_key),
            "db": "pmc",
            "id": ",".join(batch),
            "retmode": "json",
        }
        response = run_with_retries(
            action_label=f"Fetching PMC metadata for {len(batch)} article(s)",
            func=lambda params=params: session.get(
                f"{EUTILS_BASE_URL}/esummary.fcgi", params=params, timeout=60
            ),
            max_retries=max_request_retries,
            retry_backoff_s=request_retry_backoff_s,
        )
        response.raise_for_status()
        result = response.json().get("result", {})
        summaries.extend(result[uid] for uid in result.get("uids", []) if uid in result)
        if start + batch_size < len(pmc_ids) and request_pause_s > 0:
            time.sleep(request_pause_s)
    return summaries


def pmcid_from_summary(summary: dict[str, Any]) -> str | None:
    for article_id in summary.get("articleids", []):
        if article_id.get("idtype") == "pmcid" and article_id.get("value"):
            value = str(article_id["value"])
            return value if value.upper().startswith("PMC") else f"PMC{value}"
    uid = str(summary.get("uid") or "")
    return f"PMC{uid}" if uid.isdigit() else None


def manifest_rows_from_pmc_summaries(
    summaries: list[dict[str, Any]],
    year: int,
    domain: str = "Biomedical Research",
) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for summary in summaries:
        pmcid = pmcid_from_summary(summary)
        title = normalize_spaces(str(summary.get("title") or ""))
        if not pmcid or not title:
            continue
        source_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
        rows.append(
            ManifestRow(
                source_paper_title=title,
                source_paper_url=source_url,
                year=year,
                pdf_url=f"{source_url}pdf/",
                pdf_path=None,
                venue=normalize_spaces(str(summary.get("fulljournalname") or "PubMed Central")),
                domain=domain,
            )
        )
    return rows


def manifest_rows_from_europe_pmc_summaries(
    summaries: list[dict[str, Any]],
    year: int,
    domain: str = "Biomedical Research",
) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for summary in summaries:
        pmcid = str(summary.get("pmcid") or "")
        title = normalize_spaces(str(summary.get("title") or ""))
        pdf_urls = [
            item.get("url")
            for item in summary.get("fullTextUrlList", {}).get("fullTextUrl", [])
            if item.get("availabilityCode") == "OA" and item.get("documentStyle") == "pdf"
        ]
        if not pmcid or not title or not pdf_urls:
            continue
        journal = summary.get("journalInfo", {}).get("journal", {}).get("title")
        rows.append(
            ManifestRow(
                source_paper_title=title,
                source_paper_url=f"https://europepmc.org/articles/{pmcid}",
                year=year,
                pdf_url=str(pdf_urls[0]),
                pdf_path=None,
                venue=normalize_spaces(str(journal or "Europe PMC")),
                domain=domain,
            )
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect CiteGuard candidates from the PubMed Central Open Access subset."
    )
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument(
        "--query",
        help='Optional PMC query added to the year and Open Access filters, e.g. "cancer".',
    )
    parser.add_argument("--max_articles", type=int, default=100)
    parser.add_argument("--domain", default="Biomedical Research")
    parser.add_argument("--fields_of_study", default="Medicine")
    parser.add_argument("--email", default=os.environ.get("NCBI_EMAIL"))
    parser.add_argument("--ncbi_api_key", default=os.environ.get("NCBI_API_KEY"))
    parser.add_argument("--manifest_out", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--audit_jsonl", required=True)
    parser.add_argument("--pdf_cache_dir", default="training_data_collection/cache/pdfs_pubmed")
    parser.add_argument("--split", default="train")
    parser.add_argument("--starting_id", type=int, default=100000)
    parser.add_argument("--citation_mode", choices=["single", "multi", "any"], default="single")
    parser.add_argument("--citation_style", choices=["numeric", "author_year"], default="numeric")
    parser.add_argument("--request_pause_s", type=float, default=0.34)
    parser.add_argument("--resume_state_path")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--max_request_retries", type=int, default=DEFAULT_MAX_REQUEST_RETRIES)
    parser.add_argument(
        "--request_retry_backoff_s", type=float, default=DEFAULT_REQUEST_RETRY_BACKOFF_S
    )
    parser.add_argument("--progress_every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session = requests.Session()
    session.headers.update({"User-Agent": "CiteGuard PubMed collector"})
    query = build_europe_pmc_query(args.year, args.query)
    summaries, total_matches = fetch_europe_pmc_summaries(
        query=query,
        max_articles=args.max_articles,
        session=session,
        email=args.email,
        request_pause_s=args.request_pause_s,
        max_request_retries=args.max_request_retries,
        request_retry_backoff_s=args.request_retry_backoff_s,
    )
    manifest_rows = manifest_rows_from_europe_pmc_summaries(
        summaries, args.year, args.domain
    )
    write_manifest_csv(Path(args.manifest_out), manifest_rows)
    print(
        f"Europe PMC query matched {total_matches} article(s); selected {len(summaries)} and built "
        f"{len(manifest_rows)} manifest row(s)"
    )

    output_csv_path = Path(args.output_csv)
    audit_jsonl_path = Path(args.audit_jsonl)
    resume_state_path = (
        Path(args.resume_state_path)
        if args.resume_state_path
        else default_resume_state_path(output_csv_path)
    )
    iter_rows(
        manifest_rows=manifest_rows,
        api=SemanticScholarAPI(session=session),
        cache_dir=Path(args.pdf_cache_dir),
        split=args.split,
        starting_id=args.starting_id,
        request_pause_s=args.request_pause_s,
        session=session,
        citation_mode=args.citation_mode,
        citation_style=args.citation_style,
        progress_every=args.progress_every if args.progress_every > 0 else None,
        output_csv_path=output_csv_path,
        audit_jsonl_path=audit_jsonl_path,
        resume_state_path=resume_state_path,
        restart=args.restart,
        max_request_retries=args.max_request_retries,
        request_retry_backoff_s=args.request_retry_backoff_s,
        fields_of_study=args.fields_of_study or None,
    )
    state = load_resume_state(resume_state_path)
    print(f"Wrote {state.get('dataset_row_count', 0)} dataset rows to {output_csv_path}")
    print(f"Wrote {state.get('audit_row_count', 0)} audit rows to {audit_jsonl_path}")


if __name__ == "__main__":
    main()
