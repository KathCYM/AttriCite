from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.semantic_scholar import SemanticScholarAPI
from training_data_collection.collect_numeric_cs_candidates import (
    DEFAULT_MAX_REQUEST_RETRIES,
    DEFAULT_REQUEST_RETRY_BACKOFF_S,
    ManifestRow,
    default_resume_state_path,
    iter_rows,
    load_resume_state,
    make_csv_dict_writer,
    normalize_spaces,
    run_with_retries,
    write_audit_jsonl,
    write_dataset_csv,
)


MANIFEST_COLUMNS = [
    "source_paper_title",
    "source_paper_url",
    "year",
    "pdf_url",
    "venue",
    "domain",
]

ANCHOR_PATTERN = re.compile(
    r"<a\b(?P<attrs>[^>]*)href\s*=\s*(?:['\"](?P<href_quoted>[^'\"]+)['\"]|(?P<href_unquoted>[^\s>]+))[^>]*>(?P<text>.*?)</a>",
    flags=re.IGNORECASE | re.DOTALL,
)
TAG_PATTERN = re.compile(r"<[^>]+>")
PMLR_PAPER_BLOCK_PATTERN = re.compile(
    r"<div\b[^>]*class\s*=\s*['\"][^'\"]*\bpaper\b[^'\"]*['\"][^>]*>(?P<body>.*?)</div>",
    flags=re.IGNORECASE | re.DOTALL,
)
PMLR_TITLE_PATTERN = re.compile(
    r"<p\b[^>]*class\s*=\s*['\"][^'\"]*\btitle\b[^'\"]*['\"][^>]*>(?P<title>.*?)</p>",
    flags=re.IGNORECASE | re.DOTALL,
)
SKIP_PDF_TERMS = {
    "abstract",
    "appendix",
    "poster",
    "slides",
    "supp",
    "supplemental",
    "video",
}
GENERIC_TITLE_TERMS = {
    "abstract",
    "bibtex",
    "code",
    "dataset",
    "details",
    "github",
    "html",
    "openreview",
    "pdf",
    "poster",
    "slides",
    "supp",
    "supplemental",
    "video",
}
NEURIPS_ABSTRACT_PATH_PATTERN = re.compile(
    r"^(?P<prefix>/paper_files/paper/\d+/)hash/(?P<hash>[^/]+)-Abstract-(?P<track>[^./]+)\.html/?$",
    flags=re.IGNORECASE,
)
CVF_ABSTRACT_PATH_PATTERN = re.compile(
    r"^(?P<prefix>/content/[^/]+)/html/(?P<name>[^/]+)\.html/?$",
    flags=re.IGNORECASE,
)


@dataclass
class Anchor:
    href: str
    raw_href: str
    text: str
    start: int


@dataclass
class ProceedingsPaper:
    title: str
    source_paper_url: str
    pdf_url: str


@dataclass
class ProceedingsCollectionResult:
    papers: list[ProceedingsPaper]
    fetched_page_urls: list[str]
    listing_page_urls: list[str]


def clean_anchor_text(raw_text: str) -> str:
    no_tags = TAG_PATTERN.sub(" ", raw_text)
    return normalize_spaces(unescape(no_tags))


def extract_anchors(html: str, base_url: str) -> list[Anchor]:
    anchors: list[Anchor] = []
    for match in ANCHOR_PATTERN.finditer(html):
        raw_href = unescape((match.group("href_quoted") or match.group("href_unquoted") or "").strip())
        href = urljoin(base_url, raw_href)
        text = clean_anchor_text(match.group("text"))
        anchors.append(Anchor(href=href, raw_href=raw_href, text=text, start=match.start()))
    return anchors


def is_pdf_url(href: str) -> bool:
    return urlparse(href).path.lower().endswith(".pdf")


def is_placeholder_href(href: str) -> bool:
    stripped = href.strip().lower()
    return stripped in {"", "#"} or stripped.startswith("javascript:")


def is_probable_paper_pdf(anchor: Anchor) -> bool:
    if not is_pdf_url(anchor.href):
        return False
    combined = f"{anchor.text} {anchor.href}".lower()
    return not any(term in combined for term in SKIP_PDF_TERMS)


def is_probable_title_anchor(anchor: Anchor) -> bool:
    if not anchor.text or len(anchor.text) < 8:
        return False
    if is_pdf_url(anchor.href):
        return False
    if is_placeholder_href(anchor.raw_href):
        return False
    lowered = anchor.text.lower()
    if lowered.startswith("http://") or lowered.startswith("https://"):
        return False
    if lowered.startswith("proceedings of "):
        return False
    if lowered in GENERIC_TITLE_TERMS:
        return False
    if any(lowered.startswith(term + " ") for term in GENERIC_TITLE_TERMS):
        return False
    return True


def normalize_link_target(href: str) -> str:
    path = urlparse(href).path.rstrip("/").lower()
    if path.endswith(".pdf"):
        path = path[: -len(".pdf")]
    return path


def derive_pdf_url_from_neurips_abstract_url(href: str) -> str | None:
    parsed = urlparse(href)
    match = NEURIPS_ABSTRACT_PATH_PATTERN.match(parsed.path)
    if match is None:
        return None
    pdf_path = (
        f"{match.group('prefix')}file/"
        f"{match.group('hash')}-Paper-{match.group('track')}.pdf"
    )
    return parsed._replace(path=pdf_path, params="", query="", fragment="").geturl()


def derive_pdf_url_from_cvf_abstract_url(href: str) -> str | None:
    parsed = urlparse(href)
    match = CVF_ABSTRACT_PATH_PATTERN.match(parsed.path)
    if match is None:
        return None
    pdf_path = f"{match.group('prefix')}/papers/{match.group('name')}.pdf"
    return parsed._replace(path=pdf_path, params="", query="", fragment="").geturl()


def extract_pmlr_proceedings_papers_from_html(html: str, proceedings_url: str) -> list[ProceedingsPaper]:
    papers: list[ProceedingsPaper] = []
    seen_source_urls: set[str] = set()

    for block_match in PMLR_PAPER_BLOCK_PATTERN.finditer(html):
        block_html = block_match.group("body")
        title_match = PMLR_TITLE_PATTERN.search(block_html)
        if title_match is None:
            continue
        title = clean_anchor_text(title_match.group("title"))
        if not title:
            continue

        abs_url: str | None = None
        pdf_url: str | None = None
        for anchor in extract_anchors(block_html, proceedings_url):
            lowered_text = anchor.text.lower()
            if lowered_text == "abs":
                abs_url = anchor.href
            elif is_probable_paper_pdf(anchor):
                pdf_url = anchor.href

        if not abs_url or not pdf_url or abs_url in seen_source_urls:
            continue

        papers.append(
            ProceedingsPaper(
                title=title,
                source_paper_url=abs_url,
                pdf_url=pdf_url,
            )
        )
        seen_source_urls.add(abs_url)

    return papers


def find_title_anchor(anchors: list[Anchor], pdf_index: int, max_anchor_gap: int = 8, max_char_gap: int = 1500) -> Anchor | None:
    pdf_anchor = anchors[pdf_index]
    pdf_target = normalize_link_target(pdf_anchor.href)
    saw_same_target_anchor = False

    for offset in range(1, max_anchor_gap + 1):
        candidate_indices = [pdf_index - offset, pdf_index + offset]
        for candidate_index in candidate_indices:
            if candidate_index < 0 or candidate_index >= len(anchors):
                continue
            candidate = anchors[candidate_index]
            if abs(candidate.start - pdf_anchor.start) > max_char_gap:
                continue
            if normalize_link_target(candidate.href) != pdf_target:
                continue
            saw_same_target_anchor = True
            if is_probable_title_anchor(candidate):
                return candidate

    if saw_same_target_anchor:
        return None

    for offset in range(1, max_anchor_gap + 1):
        candidate_indices = [pdf_index - offset, pdf_index + offset]
        for candidate_index in candidate_indices:
            if candidate_index < 0 or candidate_index >= len(anchors):
                continue
            candidate = anchors[candidate_index]
            if abs(candidate.start - pdf_anchor.start) > max_char_gap:
                continue
            if is_probable_title_anchor(candidate):
                return candidate
    return None


def extract_proceedings_papers_from_html(html: str, proceedings_url: str) -> list[ProceedingsPaper]:
    pmlr_papers = extract_pmlr_proceedings_papers_from_html(html, proceedings_url)
    if pmlr_papers:
        return pmlr_papers

    anchors = extract_anchors(html, proceedings_url)
    cvf_papers: list[ProceedingsPaper] = []
    seen_cvf_urls: set[str] = set()
    for anchor in anchors:
        if not is_probable_title_anchor(anchor):
            continue
        pdf_url = derive_pdf_url_from_cvf_abstract_url(anchor.href)
        if pdf_url is None or anchor.href in seen_cvf_urls:
            continue
        cvf_papers.append(
            ProceedingsPaper(
                title=anchor.text,
                source_paper_url=anchor.href,
                pdf_url=pdf_url,
            )
        )
        seen_cvf_urls.add(anchor.href)
    if cvf_papers:
        return cvf_papers

    papers: list[ProceedingsPaper] = []
    seen_pdf_urls: set[str] = set()

    for index, anchor in enumerate(anchors):
        if not is_probable_paper_pdf(anchor):
            continue
        if anchor.href in seen_pdf_urls:
            continue
        title_anchor = find_title_anchor(anchors, index)
        if title_anchor is None:
            continue
        papers.append(
            ProceedingsPaper(
                title=title_anchor.text,
                source_paper_url=title_anchor.href,
                pdf_url=anchor.href,
            )
        )
        seen_pdf_urls.add(anchor.href)

    if papers:
        return papers

    seen_source_urls: set[str] = set()
    for anchor in anchors:
        if not is_probable_title_anchor(anchor):
            continue
        pdf_url = derive_pdf_url_from_neurips_abstract_url(anchor.href)
        if pdf_url is None:
            continue
        if anchor.href in seen_source_urls or pdf_url in seen_pdf_urls:
            continue
        papers.append(
            ProceedingsPaper(
                title=anchor.text,
                source_paper_url=anchor.href,
                pdf_url=pdf_url,
            )
        )
        seen_source_urls.add(anchor.href)
        seen_pdf_urls.add(pdf_url)

    return papers


def dedupe_urls(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        deduped.append(url)
    return deduped


def extract_listing_page_urls(html: str, proceedings_url: str) -> list[str]:
    anchors = extract_anchors(html, proceedings_url)
    preferred_urls: list[str] = []
    fallback_urls: list[str] = []

    for anchor in anchors:
        if is_pdf_url(anchor.href):
            continue
        lowered_text = anchor.text.lower()
        lowered_href = anchor.href.lower()
        if "all papers" in lowered_text or "day=all" in lowered_href:
            preferred_urls.append(anchor.href)
            continue
        if lowered_text.startswith("day ") or "day=" in lowered_href:
            fallback_urls.append(anchor.href)

    return dedupe_urls(preferred_urls or fallback_urls)


def fetch_proceedings_html(
    proceedings_url: str,
    session: requests.Session,
    max_request_retries: int = DEFAULT_MAX_REQUEST_RETRIES,
    request_retry_backoff_s: float = DEFAULT_REQUEST_RETRY_BACKOFF_S,
) -> str:
    response = run_with_retries(
        action_label=f"Fetching proceedings page: {proceedings_url}",
        func=lambda: session.get(proceedings_url, timeout=60),
        max_retries=max_request_retries,
        retry_backoff_s=request_retry_backoff_s,
    )
    response.raise_for_status()
    return response.text


def collect_proceedings_papers(
    proceedings_url: str,
    session: requests.Session,
    max_request_retries: int = DEFAULT_MAX_REQUEST_RETRIES,
    request_retry_backoff_s: float = DEFAULT_REQUEST_RETRY_BACKOFF_S,
) -> ProceedingsCollectionResult:
    fetched_page_urls = [proceedings_url]
    html = fetch_proceedings_html(
        proceedings_url,
        session,
        max_request_retries=max_request_retries,
        request_retry_backoff_s=request_retry_backoff_s,
    )
    papers = extract_proceedings_papers_from_html(html, proceedings_url)
    if papers:
        return ProceedingsCollectionResult(
            papers=papers,
            fetched_page_urls=fetched_page_urls,
            listing_page_urls=[],
        )

    listing_urls = extract_listing_page_urls(html, proceedings_url)
    collected: list[ProceedingsPaper] = []
    seen_pdf_urls: set[str] = set()
    for listing_url in listing_urls:
        fetched_page_urls.append(listing_url)
        listing_html = fetch_proceedings_html(
            listing_url,
            session,
            max_request_retries=max_request_retries,
            request_retry_backoff_s=request_retry_backoff_s,
        )
        for paper in extract_proceedings_papers_from_html(listing_html, listing_url):
            if paper.pdf_url in seen_pdf_urls:
                continue
            collected.append(paper)
            seen_pdf_urls.add(paper.pdf_url)
    return ProceedingsCollectionResult(
        papers=collected,
        fetched_page_urls=dedupe_urls(fetched_page_urls),
        listing_page_urls=listing_urls,
    )


def manifest_rows_from_proceedings_papers(
    papers: list[ProceedingsPaper],
    year: int,
    venue: str,
    domain: str,
    max_papers: int | None = None,
) -> list[ManifestRow]:
    if max_papers is not None:
        papers = papers[:max_papers]
    return [
        ManifestRow(
            source_paper_title=paper.title,
            source_paper_url=paper.source_paper_url,
            year=year,
            pdf_url=paper.pdf_url,
            pdf_path=None,
            venue=venue,
            domain=domain,
        )
        for paper in papers
    ]


def build_manifest_rows_from_proceedings(
    proceedings_url: str,
    year: int,
    venue: str,
    domain: str,
    session: requests.Session,
    max_papers: int | None = None,
    max_request_retries: int = DEFAULT_MAX_REQUEST_RETRIES,
    request_retry_backoff_s: float = DEFAULT_REQUEST_RETRY_BACKOFF_S,
) -> list[ManifestRow]:
    collection = collect_proceedings_papers(
        proceedings_url,
        session,
        max_request_retries=max_request_retries,
        request_retry_backoff_s=request_retry_backoff_s,
    )
    return manifest_rows_from_proceedings_papers(
        collection.papers,
        year=year,
        venue=venue,
        domain=domain,
        max_papers=max_papers,
    )


def write_manifest_csv(path: Path, rows: list[ManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = make_csv_dict_writer(handle, MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "source_paper_title": row.source_paper_title,
                    "source_paper_url": row.source_paper_url,
                    "year": row.year,
                    "pdf_url": row.pdf_url,
                    "venue": row.venue,
                    "domain": row.domain,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crawl CS conference proceedings pages, download paper PDFs, and collect CiteGuard training rows."
    )
    parser.add_argument(
        "--proceedings_url",
        action="append",
        required=True,
        help="Proceedings page URL with direct paper PDF links. Repeat for multiple pages.",
    )
    parser.add_argument("--year", required=True, type=int, help="Publication year for all crawled papers.")
    parser.add_argument("--venue", required=True, help="Venue label to store in the audit manifest.")
    parser.add_argument("--domain", default="Computer Science", help="Domain label to store in the audit manifest.")
    parser.add_argument(
        "--fields_of_study",
        default="Computer Science",
        help=(
            "Semantic Scholar field filter used for reference resolution and discoverability "
            '(for example, "Medicine" or "Biology"). Use an empty string to disable the filter.'
        ),
    )
    parser.add_argument("--output_csv", required=True, help="Output CSV path.")
    parser.add_argument("--audit_jsonl", required=True, help="Output JSONL audit path.")
    parser.add_argument(
        "--manifest_out",
        help="Optional manifest CSV path for the crawled proceedings papers.",
    )
    parser.add_argument(
        "--pdf_cache_dir",
        default="training_data_collection/cache/pdfs",
        help="Directory to reuse/download PDFs.",
    )
    parser.add_argument("--split", default="train", help="Split label to write into the output CSV.")
    parser.add_argument("--starting_id", type=int, default=100000, help="Starting integer ID for new rows.")
    parser.add_argument(
        "--citation_mode",
        choices=["single", "multi", "any"],
        default="any",
        help="Keep only single-citation sentences, multi-citation sentences, or any citation count.",
    )
    parser.add_argument(
        "--citation_style",
        choices=["numeric", "author_year"],
        default="numeric",
        help="Citation style to extract from the crawled papers.",
    )
    parser.add_argument(
        "--max_papers_per_url",
        type=int,
        help="Optional cap on the number of crawled papers from each proceedings page.",
    )
    parser.add_argument(
        "--request_pause_s",
        type=float,
        default=0.0,
        help="Sleep duration between Semantic Scholar API calls.",
    )
    parser.add_argument(
        "--resume_state_path",
        help="Optional checkpoint state path. Defaults to <output_csv>.state.json.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Delete existing output/checkpoint files and start this collection run from scratch.",
    )
    parser.add_argument(
        "--max_request_retries",
        type=int,
        default=DEFAULT_MAX_REQUEST_RETRIES,
        help="Number of retries for transient proceedings, PDF, and Semantic Scholar request failures.",
    )
    parser.add_argument(
        "--request_retry_backoff_s",
        type=float,
        default=DEFAULT_REQUEST_RETRY_BACKOFF_S,
        help="Base backoff in seconds for request retries.",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=100,
        help="Print one example source paper every N processed papers. Use 0 to disable.",
    )
    parser.add_argument(
        "--max_dataset_rows",
        type=int,
        help="Optional cap on dataset rows written in this run; use 1 for a smoke test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session = requests.Session()
    session.headers.update({"User-Agent": "CiteGuard cs-conference collector"})
    output_csv_path = Path(args.output_csv)
    audit_jsonl_path = Path(args.audit_jsonl)
    resume_state_path = (
        Path(args.resume_state_path)
        if args.resume_state_path
        else default_resume_state_path(output_csv_path)
    )

    manifest_rows: list[ManifestRow] = []
    total_fetched_pages = 0
    for proceedings_url in args.proceedings_url:
        collection = collect_proceedings_papers(
            proceedings_url,
            session,
            max_request_retries=args.max_request_retries,
            request_retry_backoff_s=args.request_retry_backoff_s,
        )
        selected_manifest_rows = manifest_rows_from_proceedings_papers(
            collection.papers,
            year=args.year,
            venue=args.venue,
            domain=args.domain,
            max_papers=args.max_papers_per_url,
        )
        manifest_rows.extend(selected_manifest_rows)
        total_fetched_pages += len(collection.fetched_page_urls)
        found_papers = len(collection.papers)
        selected_papers = len(selected_manifest_rows)
        print(
            f"{proceedings_url}: fetched {len(collection.fetched_page_urls)} page(s), "
            f"found {found_papers} paper(s), using {selected_papers}"
        )
        if collection.listing_page_urls:
            print(
                f"  Followed {len(collection.listing_page_urls)} listing page(s); "
                f"first listing page: {collection.listing_page_urls[0]}"
            )

    print(
        f"Collected {len(manifest_rows)} source paper(s) from "
        f"{total_fetched_pages} fetched proceedings page(s)"
    )

    if args.manifest_out:
        write_manifest_csv(Path(args.manifest_out), manifest_rows)

    api = SemanticScholarAPI(session=session)
    iter_rows(
        manifest_rows=manifest_rows,
        api=api,
        cache_dir=Path(args.pdf_cache_dir),
        split=args.split,
        starting_id=args.starting_id,
        request_pause_s=args.request_pause_s,
        session=session,
        citation_mode=args.citation_mode,
        citation_style=args.citation_style,
        progress_every=(args.progress_every if args.progress_every > 0 else None),
        output_csv_path=output_csv_path,
        audit_jsonl_path=audit_jsonl_path,
        resume_state_path=resume_state_path,
        restart=args.restart,
        max_request_retries=args.max_request_retries,
        request_retry_backoff_s=args.request_retry_backoff_s,
        fields_of_study=args.fields_of_study or None,
        max_dataset_rows=args.max_dataset_rows,
    )
    final_state = load_resume_state(resume_state_path)
    print(f"Crawled {len(manifest_rows)} papers from {len(args.proceedings_url)} proceedings page(s)")
    if args.manifest_out:
        print(f"Wrote manifest CSV to {args.manifest_out}")
    print(f"Wrote {final_state.get('dataset_row_count', 0)} dataset rows to {args.output_csv}")
    print(f"Wrote {final_state.get('audit_row_count', 0)} audit rows to {args.audit_jsonl}")
    print(f"Checkpoint state saved at {resume_state_path}")


if __name__ == "__main__":
    main()
