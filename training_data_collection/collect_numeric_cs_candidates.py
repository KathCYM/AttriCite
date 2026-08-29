from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable

import requests
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.semantic_scholar import SemanticScholarAPI


DATASET_COLUMNS = [
    "id",
    "excerpt",
    "target_paper_title",
    "target_paper_url",
    "source_paper_title",
    "source_paper_url",
    "year",
    "split",
]

NUMERIC_CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*[-,;]\s*\d+)*)\]")
AUTHOR_YEAR_PARENTHETICAL_PATTERN = re.compile(r"\(([^()]{3,200}?\b(?:19|20)\d{2}[a-z]?\b[^()]*)\)")
AUTHOR_YEAR_NARRATIVE_PATTERN = re.compile(
    r"\b([A-Z][A-Za-z'`-]*(?:\s+[A-Z][A-Za-z'`-]*)*(?:\s+et\s+al\.)?(?:\s+(?:and|&)\s+[A-Z][A-Za-z'`-]+)?)\s*\((\d{4}[a-z]?)\)"
)
YEAR_TOKEN_PATTERN = re.compile(r"\b(?:19|20)\d{2}[a-z]?\b", flags=re.IGNORECASE)
REFERENCE_HEADING_PATTERN = re.compile(
    r"\n\s*(?:\d+(?:\.\d+)*)?\s*(references|bibliography)\s*\n",
    flags=re.IGNORECASE,
)
REFERENCE_ENTRY_PATTERNS = [
    re.compile(r"^\[(\d+)\]\s*(.*)$"),
    re.compile(r"^(\d+)\.\s+(.*)$"),
]
AUTHOR_YEAR_REFERENCE_START_PATTERN = re.compile(
    r"^[A-Z][^\n]{0,160}?\b(?:19|20)\d{2}[a-z]?\b"
)
AUTHOR_YEAR_INLINE_BOUNDARY_PATTERN = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z][A-Za-z'`’\-]+(?:\s+[A-Z][A-Za-z'`’\-]+){0,2}"
    r"(?:,|\s+(?:and|&)\s+)[^!?\n]{1,140}?\b(?:19|20)\d{2}[a-z]?\b)"
)
QUOTE_PATTERNS = [
    re.compile(r'"([^"]{12,})"'),
    re.compile(r"\u201c([^\u201d]{12,})\u201d"),
]
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
ET_AL_SENTINEL = "<prd>"
LEADING_REFERENCE_ENTRY_MARKER_PATTERN = re.compile(r"^\s*(?:\[\d+(?:\s*[-,;]\s*\d+)*\]|\d+\.)\s*")
MAX_REFERENCE_RESOLUTION_QUERY_CHARS = 180
MAX_REFERENCE_RESOLUTION_QUERY_TOKENS = 24
MAX_AUTHOR_QUERY_CHARS = 120
MAX_AUTHOR_QUERY_TOKENS = 16
MAX_SUSPICIOUS_REFERENCE_CHARS = 2000
MAX_SUSPICIOUS_REFERENCE_TOKENS = 320
MAX_PDF_BYTES = 100 * 1024 * 1024

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "their",
    "this",
    "to",
    "we",
    "with",
}
NAME_PARTICLE_TOKENS = {
    "al",
    "and",
    "bin",
    "da",
    "de",
    "del",
    "der",
    "di",
    "dos",
    "du",
    "et",
    "la",
    "le",
    "van",
    "von",
}

API_FIELDS = "paperId,title,year,openAccessPdf,citationCount,authors,abstract,venue"
STATE_VERSION = 1
DEFAULT_MAX_REQUEST_RETRIES = 3
DEFAULT_REQUEST_RETRY_BACKOFF_S = 2.0


@dataclass
class ManifestRow:
    source_paper_title: str
    source_paper_url: str
    year: int
    pdf_url: str | None
    pdf_path: str | None
    venue: str | None
    domain: str | None


@dataclass
class CandidateSentence:
    sentence: str
    excerpt: str
    reference_key: str
    citation_marker: str
    citation_count: int
    body_sentence_index: int | None = None


@dataclass
class ResolvedPaper:
    paper_id: str | None
    title: str
    year: int | None
    url: str
    query: str
    score: float


@dataclass
class DiscoverabilityResult:
    query: str
    rank: int


@dataclass
class ParsedCitationGroup:
    citation_marker: str
    reference_keys: list[str]
    display_segments: list[str]
    wrapper: str


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_compare_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return normalize_spaces(lowered)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_source_paper_key(row: ManifestRow) -> str:
    return row.source_paper_url or f"{row.year}::{row.source_paper_title}"


def default_resume_state_path(output_csv_path: Path) -> Path:
    return output_csv_path.with_suffix(output_csv_path.suffix + ".state.json")


def initialize_resume_state(
    starting_id: int,
    split: str,
    citation_mode: str,
    citation_style: str,
    fields_of_study: str | None = "Computer Science",
) -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "starting_id": starting_id,
        "next_id": starting_id,
        "split": split,
        "citation_mode": citation_mode,
        "citation_style": citation_style,
        "fields_of_study": fields_of_study,
        "processed_source_paper_keys": [],
        "processed_source_paper_count": 0,
        "dataset_row_count": 0,
        "audit_row_count": 0,
        "last_processed_source_paper_title": None,
        "last_processed_source_paper_url": None,
        "last_error": None,
    }


def save_resume_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = utc_now_iso()
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")
    temp_path.replace(path)


def load_resume_state(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_resume_state(
    state: dict[str, Any],
    starting_id: int,
    split: str,
    citation_mode: str,
    citation_style: str,
    fields_of_study: str | None = "Computer Science",
) -> None:
    mismatches: list[str] = []
    if state.get("starting_id") != starting_id:
        mismatches.append(f"starting_id={state.get('starting_id')} (expected {starting_id})")
    if state.get("split") != split:
        mismatches.append(f"split={state.get('split')} (expected {split})")
    if state.get("citation_mode") != citation_mode:
        mismatches.append(f"citation_mode={state.get('citation_mode')} (expected {citation_mode})")
    state_citation_style = state.get("citation_style", "numeric")
    if state_citation_style != citation_style:
        mismatches.append(f"citation_style={state_citation_style} (expected {citation_style})")
    state_fields_of_study = state.get("fields_of_study", "Computer Science")
    if state_fields_of_study != fields_of_study:
        mismatches.append(
            f"fields_of_study={state_fields_of_study} (expected {fields_of_study})"
        )
    if mismatches:
        mismatch_text = ", ".join(mismatches)
        raise ValueError(
            f"Resume state at {path_or_placeholder(state)} does not match current arguments: {mismatch_text}"
        )


def path_or_placeholder(state: dict[str, Any]) -> str:
    return state.get("_path", "<unknown>")


def ensure_resume_outputs_are_consistent(
    state: dict[str, Any],
    output_csv_path: Path | None,
    audit_jsonl_path: Path | None,
) -> None:
    if state.get("dataset_row_count", 0) > 0 and output_csv_path and not output_csv_path.exists():
        raise FileNotFoundError(
            f"Resume state expects dataset rows, but output CSV is missing: {output_csv_path}"
        )
    if state.get("audit_row_count", 0) > 0 and audit_jsonl_path and not audit_jsonl_path.exists():
        raise FileNotFoundError(
            f"Resume state expects audit rows, but audit JSONL is missing: {audit_jsonl_path}"
        )


def overwrite_with_header(path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = make_csv_dict_writer(handle, fieldnames)
        writer.writeheader()


def make_csv_dict_writer(handle, fieldnames: list[str]) -> csv.DictWriter:
    return csv.DictWriter(
        handle,
        fieldnames=fieldnames,
        delimiter=",",
        quotechar='"',
        quoting=csv.QUOTE_MINIMAL,
        doublequote=True,
        escapechar="\\",
        lineterminator="\n",
    )


def sanitize_unicode(value: object) -> object:
    """Replace invalid lone UTF-16 surrogates while preserving valid Unicode."""
    if isinstance(value, str):
        return value.encode("utf-8", errors="replace").decode("utf-8")
    if isinstance(value, dict):
        return {key: sanitize_unicode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_unicode(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_unicode(item) for item in value)
    return value


def ensure_output_files_initialized(
    output_csv_path: Path | None,
    audit_jsonl_path: Path | None,
    resume_state_path: Path | None,
    restart: bool,
) -> None:
    managed_paths = [path for path in [output_csv_path, audit_jsonl_path, resume_state_path] if path is not None]
    if restart:
        for path in managed_paths:
            if path.exists():
                path.unlink()
    elif resume_state_path and not resume_state_path.exists():
        existing_outputs = [path for path in [output_csv_path, audit_jsonl_path] if path and path.exists()]
        if existing_outputs:
            output_names = ", ".join(str(path) for path in existing_outputs)
            raise FileExistsError(
                f"Existing output files found without a resume state file: {output_names}. "
                "Use --restart to start fresh or choose new output paths."
            )

    if output_csv_path and not output_csv_path.exists():
        overwrite_with_header(output_csv_path, DATASET_COLUMNS)
    if audit_jsonl_path and not audit_jsonl_path.exists():
        audit_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        audit_jsonl_path.touch()


def append_dataset_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    # Serialize in memory first so a bad row cannot partially append a paper.
    buffer = io.StringIO(newline="")
    writer = make_csv_dict_writer(buffer, DATASET_COLUMNS)
    writer.writerows([sanitize_unicode(row) for row in rows])
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(buffer.getvalue())


def append_audit_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(sanitize_unicode(row), ensure_ascii=True) + "\n")


def run_with_retries(
    action_label: str,
    func: Callable[[], Any],
    max_retries: int,
    retry_backoff_s: float,
) -> Any:
    attempt = 0
    while True:
        attempt += 1
        try:
            return func()
        except (requests.exceptions.RequestException, OSError) as exc:
            response = getattr(exc, "response", None)
            status_code = response.status_code if response is not None else None
            is_retryable_http_error = status_code is None or status_code == 429 or status_code >= 500
            if not is_retryable_http_error:
                raise
            if attempt > max_retries:
                raise
            wait_s = retry_backoff_s * (2 ** (attempt - 1))
            print(
                f"{action_label} failed on attempt {attempt}/{max_retries + 1}: {exc}. "
                f"Retrying in {wait_s:.1f}s"
            )
            time.sleep(wait_s)


def load_manifest(path: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"source_paper_title", "source_paper_url", "year"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
        for row in reader:
            source_paper_url = row["source_paper_url"].strip()
            fallback_pdf_url = source_paper_url if source_paper_url.lower().endswith(".pdf") else None
            rows.append(
                ManifestRow(
                    source_paper_title=row["source_paper_title"].strip(),
                    source_paper_url=source_paper_url,
                    year=int(row["year"]),
                    pdf_url=(row.get("pdf_url") or fallback_pdf_url or "").strip() or None,
                    pdf_path=(row.get("pdf_path") or "").strip() or None,
                    venue=(row.get("venue") or "").strip() or None,
                    domain=(row.get("domain") or "").strip() or None,
                )
            )
    return rows


def safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return slug[:80] or "paper"


def ensure_pdf_available(
    row: ManifestRow,
    cache_dir: Path,
    session: requests.Session,
    max_request_retries: int = DEFAULT_MAX_REQUEST_RETRIES,
    request_retry_backoff_s: float = DEFAULT_REQUEST_RETRY_BACKOFF_S,
) -> Path:
    if row.pdf_path:
        pdf_path = Path(row.pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF path does not exist: {pdf_path}")
        return pdf_path

    if not row.pdf_url:
        raise ValueError(f"No pdf_url or pdf_path available for {row.source_paper_title}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    target_path = cache_dir / f"{safe_slug(row.source_paper_title)}.pdf"
    if target_path.exists():
        if target_path.stat().st_size > MAX_PDF_BYTES:
            raise PdfReadError(
                f"cached PDF exceeds {MAX_PDF_BYTES // (1024 * 1024)} MB: {target_path}"
            )
        if b"%PDF-" in target_path.read_bytes()[:1024]:
            return target_path
        print(f"Replacing invalid cached PDF for {row.source_paper_title}: {target_path}")
        target_path.unlink()

    candidate_urls = [row.pdf_url]
    pmc_match = re.search(r"/articles/(PMC\d+)/", row.pdf_url, flags=re.IGNORECASE)
    if pmc_match:
        candidate_urls.append(
            f"https://europepmc.org/articles/{pmc_match.group(1).upper()}?pdf=render"
        )

    errors: list[str] = []
    for candidate_url in candidate_urls:
        try:
            response = run_with_retries(
                action_label=f"Downloading PDF for {row.source_paper_title}",
                func=lambda candidate_url=candidate_url: session.get(candidate_url, timeout=60),
                max_retries=max_request_retries,
                retry_backoff_s=request_retry_backoff_s,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            errors.append(f"{candidate_url} failed: {exc}")
            continue
        if len(response.content) > MAX_PDF_BYTES:
            raise PdfReadError(
                f"downloaded PDF exceeds {MAX_PDF_BYTES // (1024 * 1024)} MB"
            )
        if b"%PDF-" not in response.content[:1024]:
            content_type = response.headers.get("Content-Type", "unknown")
            errors.append(f"{candidate_url} returned {content_type}, not a PDF")
            continue
        target_path.write_bytes(response.content)
        return target_path

    raise PdfReadError("; ".join(errors) or "download did not return a PDF")


def extract_pdf_text(pdf_path: Path) -> str:
    text_parts: list[str] = []
    reader = PdfReader(str(pdf_path))
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text:
            text_parts.append(page_text)
    return "\n".join(text_parts)


def split_body_and_references(text: str) -> tuple[str, str]:
    normalized = text.replace("\r", "\n")
    matches = list(REFERENCE_HEADING_PATTERN.finditer(normalized))
    if not matches:
        return normalized, ""
    # Use the first plausible back-matter heading. Choosing the last heading
    # can mistake an appendix subsection or extracted table-of-contents entry
    # for the bibliography, leaking real references into the paper body and
    # treating appendix prose as reference entries.
    minimum_offset = min(2000, int(len(normalized) * 0.20))
    match = next((item for item in matches if item.start() >= minimum_offset), matches[-1])
    return normalized[: match.start()], normalized[match.end() :]


def extract_numeric_reference_map(reference_text: str) -> dict[str, str]:
    references: dict[str, str] = {}
    current_key: str | None = None
    current_chunks: list[str] = []

    def flush() -> None:
        nonlocal current_key, current_chunks
        if current_key and current_chunks:
            references[current_key] = normalize_spaces(" ".join(current_chunks))
        current_key = None
        current_chunks = []

    for raw_line in reference_text.splitlines():
        line = normalize_spaces(raw_line)
        if not line:
            continue
        entry_match = None
        for pattern in REFERENCE_ENTRY_PATTERNS:
            entry_match = pattern.match(line)
            if entry_match:
                break
        if entry_match:
            flush()
            current_key = entry_match.group(1)
            remainder = entry_match.group(2).strip()
            if remainder:
                current_chunks.append(remainder)
            continue
        if current_key is not None:
            current_chunks.append(line)
    flush()
    return references


def is_probable_author_year_reference_start(line: str) -> bool:
    if not AUTHOR_YEAR_REFERENCE_START_PATTERN.search(line):
        return False
    year_match = YEAR_TOKEN_PATTERN.search(line)
    if year_match is None:
        return False
    prefix = line[: year_match.start()]
    if any(char.isdigit() for char in prefix):
        return False
    return len(prefix.split()) >= 2


def split_author_year_reference_entries(reference_text: str) -> list[str]:
    entries: list[str] = []
    current_chunks: list[str] = []

    def flush() -> None:
        nonlocal current_chunks
        if current_chunks:
            entries.append(normalize_spaces(" ".join(current_chunks)))
        current_chunks = []

    for raw_line in reference_text.splitlines():
        line = normalize_spaces(raw_line)
        if not line:
            continue
        # PDF extractors frequently collapse an entire bibliography column to
        # one line. Recover high-confidence inline entry boundaries before
        # applying the ordinary line-start detector.
        inline_chunks = AUTHOR_YEAR_INLINE_BOUNDARY_PATTERN.split(line)
        for chunk_index, chunk in enumerate(inline_chunks):
            chunk = normalize_spaces(chunk)
            if not chunk:
                continue
            if current_chunks and (
                chunk_index > 0 or is_probable_author_year_reference_start(chunk)
            ):
                flush()
            current_chunks.append(chunk)
    flush()
    return entries


def extract_author_year_reference_key(text: str) -> str | None:
    year_match = YEAR_TOKEN_PATTERN.search(text)
    if year_match is None:
        return None
    before_year = text[: year_match.start()]
    first_author_fragment = re.split(r"\s*(?:,|;|\band\b|&)\s*", before_year, maxsplit=1)[0]
    tokens = [
        token
        for token in re.findall(r"[A-Za-z][A-Za-z'`-]*", first_author_fragment)
        if token.lower() not in {"et", "al"}
    ]
    if not tokens:
        return None
    surname = tokens[-1].lower()
    year = year_match.group(0).lower()
    return f"{surname}-{year}"


def extract_author_year_reference_map(reference_text: str) -> dict[str, str]:
    references: dict[str, str] = {}
    ambiguous_keys: set[str] = set()

    for entry in split_author_year_reference_entries(reference_text):
        reference_key = extract_author_year_reference_key(entry)
        if not reference_key:
            continue
        if reference_key in ambiguous_keys:
            continue
        if reference_key in references and references[reference_key] != entry:
            references.pop(reference_key, None)
            ambiguous_keys.add(reference_key)
            continue
        references[reference_key] = entry

    return references


def extract_reference_map(reference_text: str, citation_style: str = "numeric") -> dict[str, str]:
    if citation_style == "numeric":
        return extract_numeric_reference_map(reference_text)
    if citation_style == "author_year":
        return extract_author_year_reference_map(reference_text)
    raise ValueError(f"Unsupported citation_style: {citation_style}")


def normalize_body_text(body_text: str) -> str:
    text = body_text.replace("-\n", "")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    protected_text = re.sub(
        r"\bet al\.",
        lambda match: match.group(0).replace(".", ET_AL_SENTINEL),
        text,
        flags=re.IGNORECASE,
    )
    return [
        segment.strip().replace(ET_AL_SENTINEL, ".")
        for segment in SENTENCE_SPLIT_PATTERN.split(protected_text)
        if segment.strip()
    ]


def is_name_like_segment(segment: str) -> bool:
    normalized = normalize_spaces(segment)
    if not normalized:
        return False
    lowered = normalized.lower()
    if lowered.startswith("and "):
        normalized = normalized[4:].strip()
        lowered = normalized.lower()
    if lowered.startswith("& "):
        normalized = normalized[2:].strip()
        lowered = normalized.lower()
    if lowered.startswith("et al"):
        return True

    period_index = normalized.find(".")
    if period_index != -1:
        prefix = normalized[:period_index].strip()
        if prefix:
            normalized = prefix
            lowered = normalized.lower()

    if any(char.isdigit() for char in normalized):
        return False

    tokens = re.findall(r"[^\W\d_][^\W\d_'`-]*", normalized, flags=re.UNICODE)
    if not tokens or len(tokens) > 6:
        return False

    capitalized_token_count = 0
    for token in tokens:
        lowered_token = token.lower()
        if lowered_token in NAME_PARTICLE_TOKENS:
            continue
        if token[0].isupper():
            capitalized_token_count += 1
            continue
        return False

    return capitalized_token_count >= 2


def looks_like_leading_numeric_reference_entry(sentence: str) -> bool:
    marker_match = LEADING_REFERENCE_ENTRY_MARKER_PATTERN.match(sentence)
    if marker_match is None:
        return False

    remainder = sentence[marker_match.end() :].strip()
    if not remainder:
        return False

    segments = [
        normalize_spaces(segment)
        for segment in re.split(r"\s*,\s*|\s+\band\b\s+|\s+&\s+", remainder)
        if normalize_spaces(segment)
    ]
    if len(segments) < 2:
        return False

    leading_name_like_segments = 0
    for segment in segments:
        if not is_name_like_segment(segment):
            break
        leading_name_like_segments += 1

    return leading_name_like_segments >= 2


def looks_like_reference_entry_leakage(sentence: str, reference_text: str) -> bool:
    """Return true when extracted prose is actually the cited bibliography entry."""
    if looks_like_leading_numeric_reference_entry(sentence):
        return True

    def comparison_text(value: str) -> str:
        value = LEADING_REFERENCE_ENTRY_MARKER_PATTERN.sub("", value)
        return " ".join(re.findall(r"[a-z0-9]+", value.lower()))

    normalized_sentence = comparison_text(sentence)
    normalized_reference = comparison_text(reference_text)
    if not normalized_sentence or not normalized_reference:
        return False
    if normalized_sentence == normalized_reference:
        return True

    shorter, longer = sorted(
        (normalized_sentence, normalized_reference), key=len
    )
    return len(shorter) >= 40 and shorter in longer and len(shorter) / len(longer) >= 0.8


def citation_placeholder_for_wrapper(wrapper: str) -> str:
    return "CITATION" if wrapper == "brackets" else "[CITATION]"


def parse_numeric_reference_keys(citation_marker: str) -> list[str]:
    inner_text = citation_marker.strip()
    if inner_text.startswith("[") and inner_text.endswith("]"):
        inner_text = inner_text[1:-1]
    keys: list[str] = []
    for part in re.split(r"\s*[,;]\s*", inner_text):
        if not part:
            continue
        range_match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            if end < start:
                return []
            keys.extend(str(value) for value in range(start, end + 1))
            continue
        if part.isdigit():
            keys.append(part)
            continue
        return []
    return keys


def render_numeric_citation_marker(reference_keys: list[str], highlighted_reference_key: str) -> str:
    rendered_keys = [
        "CITATION" if reference_key == highlighted_reference_key else reference_key
        for reference_key in reference_keys
    ]
    return f"[{', '.join(rendered_keys)}]"


def render_author_year_marker(
    display_segments: list[str],
    reference_keys: list[str],
    highlighted_reference_key: str,
    wrapper: str,
) -> str:
    rendered_segments = [
        citation_placeholder_for_wrapper(wrapper) if reference_key == highlighted_reference_key else display_segment
        for reference_key, display_segment in zip(reference_keys, display_segments)
    ]
    if wrapper == "parens":
        return f"({'; '.join(rendered_segments)})"
    if wrapper == "replace":
        return rendered_segments[0]
    raise ValueError(f"Unsupported author-year wrapper: {wrapper}")


def build_excerpt(sentence: str, citation_group: ParsedCitationGroup, reference_key: str) -> str:
    if citation_group.wrapper == "brackets":
        replacement = render_numeric_citation_marker(citation_group.reference_keys, reference_key)
    else:
        replacement = render_author_year_marker(
            citation_group.display_segments,
            citation_group.reference_keys,
            reference_key,
            citation_group.wrapper,
        )
    return sentence.replace(citation_group.citation_marker, replacement, 1)


def extract_numeric_citation_groups(sentence: str, reference_map: dict[str, str]) -> list[ParsedCitationGroup]:
    citation_matches = list(NUMERIC_CITATION_PATTERN.finditer(sentence))
    if not citation_matches:
        return []
    citation_groups: list[ParsedCitationGroup] = []
    for match in citation_matches:
        citation_marker = match.group(0)
        reference_keys = parse_numeric_reference_keys(citation_marker)
        if not reference_keys:
            return []
        if any(reference_key not in reference_map for reference_key in reference_keys):
            return []
        citation_groups.append(
            ParsedCitationGroup(
                citation_marker=citation_marker,
                reference_keys=reference_keys,
                display_segments=reference_keys,
                wrapper="brackets",
            )
        )
    return citation_groups


def parse_author_year_reference_keys(citation_text: str) -> list[tuple[str, str]]:
    segments = [segment.strip() for segment in citation_text.split(";") if segment.strip()]
    parsed: list[tuple[str, str]] = []
    for segment in segments:
        reference_key = extract_author_year_reference_key(segment)
        if not reference_key:
            return []
        parsed.append((segment, reference_key))
    return parsed


def extract_author_year_citation_groups(sentence: str, reference_map: dict[str, str]) -> list[ParsedCitationGroup]:
    citation_groups: list[ParsedCitationGroup] = []
    occupied_spans: list[tuple[int, int]] = []

    for match in AUTHOR_YEAR_PARENTHETICAL_PATTERN.finditer(sentence):
        parsed_segments = parse_author_year_reference_keys(match.group(1))
        if not parsed_segments:
            continue
        reference_keys = [reference_key for _, reference_key in parsed_segments]
        if any(reference_key not in reference_map for reference_key in reference_keys):
            continue
        citation_groups.append(
            ParsedCitationGroup(
                citation_marker=match.group(0),
                reference_keys=reference_keys,
                display_segments=[segment for segment, _ in parsed_segments],
                wrapper="parens",
            )
        )
        occupied_spans.append(match.span())

    def span_overlaps_existing(start: int, end: int) -> bool:
        return any(start < existing_end and end > existing_start for existing_start, existing_end in occupied_spans)

    for match in AUTHOR_YEAR_NARRATIVE_PATTERN.finditer(sentence):
        start, end = match.span()
        if span_overlaps_existing(start, end):
            continue
        reference_key = extract_author_year_reference_key(match.group(0))
        if not reference_key or reference_key not in reference_map:
            continue
        citation_groups.append(
            ParsedCitationGroup(
                citation_marker=match.group(0),
                reference_keys=[reference_key],
                display_segments=["[CITATION]"],
                wrapper="replace",
            )
        )
        occupied_spans.append((start, end))

    citation_groups.sort(key=lambda citation_group: sentence.find(citation_group.citation_marker))
    return citation_groups


def find_candidate_sentences(
    body_text: str,
    reference_map: dict[str, str],
    min_words: int = 8,
    max_words: int = 80,
    citation_mode: str = "single",
    citation_style: str = "numeric",
) -> list[CandidateSentence]:
    if citation_mode not in {"single", "multi", "any"}:
        raise ValueError(f"Unsupported citation_mode: {citation_mode}")
    if citation_style not in {"numeric", "author_year"}:
        raise ValueError(f"Unsupported citation_style: {citation_style}")

    candidates: list[CandidateSentence] = []
    for body_sentence_index, sentence in enumerate(split_sentences(normalize_body_text(body_text))):
        if citation_style == "numeric" and looks_like_leading_numeric_reference_entry(sentence):
            continue
        if citation_style == "numeric":
            citation_groups = extract_numeric_citation_groups(sentence, reference_map)
        else:
            citation_groups = extract_author_year_citation_groups(sentence, reference_map)
        if not citation_groups:
            continue
        total_reference_count = 0
        for citation_group in citation_groups:
            reference_keys = citation_group.reference_keys
            total_reference_count += len(reference_keys)
        if total_reference_count == 0:
            continue

        if citation_mode == "single" and total_reference_count != 1:
            continue
        if citation_mode == "multi" and total_reference_count <= 1:
            continue

        word_count = len(sentence.split())
        if word_count < min_words or word_count > max_words:
            continue
        for citation_group in citation_groups:
            for reference_key in citation_group.reference_keys:
                excerpt = build_excerpt(sentence, citation_group, reference_key)
                candidates.append(
                    CandidateSentence(
                        sentence=sentence,
                        excerpt=excerpt,
                        reference_key=reference_key,
                        citation_marker=citation_group.citation_marker,
                        citation_count=total_reference_count,
                        body_sentence_index=body_sentence_index,
                    )
                )
    return candidates


def extract_title_guess(reference_text: str) -> str | None:
    for pattern in QUOTE_PATTERNS:
        match = pattern.search(reference_text)
        if match:
            return normalize_spaces(match.group(1))

    segments = [segment.strip() for segment in re.split(r"\.\s+", reference_text) if segment.strip()]
    prioritized_segments = segments[1:] + segments[:1] if len(segments) > 1 else segments
    for segment in prioritized_segments:
        token_count = len(segment.split())
        if 4 <= token_count <= 24 and not re.search(r"\b\d{4}\b", segment):
            return segment
    return None


def compact_query_text(
    text: str,
    max_chars: int = MAX_REFERENCE_RESOLUTION_QUERY_CHARS,
    max_tokens: int = MAX_REFERENCE_RESOLUTION_QUERY_TOKENS,
) -> str | None:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'`:/&+.-]*", normalize_spaces(text))
    if not tokens:
        return None
    compacted_tokens: list[str] = []
    current_length = 0
    for token in tokens[:max_tokens]:
        added_length = len(token) if not compacted_tokens else len(token) + 1
        if compacted_tokens and current_length + added_length > max_chars:
            break
        if not compacted_tokens and len(token) > max_chars:
            compacted_tokens.append(token[:max_chars])
            break
        compacted_tokens.append(token)
        current_length += added_length
    if not compacted_tokens:
        return None
    return " ".join(compacted_tokens)


def extract_author_query(reference_text: str) -> str | None:
    year_match = YEAR_TOKEN_PATTERN.search(reference_text)
    if year_match is None:
        return None
    return compact_query_text(
        reference_text[: year_match.start()],
        max_chars=MAX_AUTHOR_QUERY_CHARS,
        max_tokens=MAX_AUTHOR_QUERY_TOKENS,
    )


def is_suspicious_reference_text(reference_text: str) -> bool:
    if len(reference_text) > MAX_SUSPICIOUS_REFERENCE_CHARS:
        return True
    return len(reference_text.split()) > MAX_SUSPICIOUS_REFERENCE_TOKENS


def build_reference_resolution_queries(reference_text: str) -> list[str]:
    candidate_queries: list[str] = []

    title_guess = extract_title_guess(reference_text)
    if title_guess:
        compact_title_guess = compact_query_text(title_guess)
        if compact_title_guess and compact_title_guess not in candidate_queries:
            candidate_queries.append(compact_title_guess)

    author_query = extract_author_query(reference_text)
    if author_query and author_query not in candidate_queries:
        candidate_queries.append(author_query)

    if not is_suspicious_reference_text(reference_text):
        fallback_query = compact_query_text(reference_text)
        if fallback_query and fallback_query not in candidate_queries:
            candidate_queries.append(fallback_query)

    return candidate_queries


def score_reference_match(reference_text: str, result: dict) -> float:
    title = result.get("title") or ""
    if not title:
        return 0.0
    ref_normalized = normalize_compare_text(reference_text)
    title_normalized = normalize_compare_text(title)
    title_tokens = {token for token in title_normalized.split() if len(token) > 2}
    ref_tokens = set(ref_normalized.split())
    token_overlap = len(title_tokens & ref_tokens) / max(len(title_tokens), 1)
    sequence_score = SequenceMatcher(a=title_normalized, b=ref_normalized).ratio()
    score = 0.7 * token_overlap + 0.3 * sequence_score
    year = result.get("year")
    if year and str(year) in reference_text:
        score += 0.05
    else:
        reference_year = YEAR_TOKEN_PATTERN.search(reference_text)
        if year and reference_year:
            delta = abs(int(year) - int(reference_year.group(0)[:4]))
            if delta > 1:
                score -= min(0.10, 0.02 * (delta - 1))
    return score


def resolve_reference(
    reference_text: str,
    source_year: int,
    api: SemanticScholarAPI,
    request_pause_s: float = 0.0,
    max_request_retries: int = DEFAULT_MAX_REQUEST_RETRIES,
    request_retry_backoff_s: float = DEFAULT_REQUEST_RETRY_BACKOFF_S,
    fields_of_study: str | None = "Computer Science",
    source_paper_title: str | None = None,
) -> ResolvedPaper | None:
    candidate_queries = build_reference_resolution_queries(reference_text)
    if not candidate_queries:
        return None

    best: ResolvedPaper | None = None
    for query in candidate_queries:
        try:
            payload = run_with_retries(
                action_label=f"Resolving reference with Semantic Scholar query: {query[:80]}",
                func=lambda query=query: api.relevance_search(
                    query=query,
                    fields=API_FIELDS,
                    fieldsOfStudy=fields_of_study,
                    year=str(source_year),
                    limit=10,
                ),
                max_retries=max_request_retries,
                retry_backoff_s=request_retry_backoff_s,
            )
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 414:
                print(
                    f"Skipping oversized Semantic Scholar resolution query ({len(query)} chars): "
                    f"{query[:80]}"
                )
                continue
            raise
        if request_pause_s > 0:
            time.sleep(request_pause_s)
        for result in payload.get("data", []):
            if source_paper_title and normalize_compare_text(result.get("title") or "") == normalize_compare_text(source_paper_title):
                continue
            score = score_reference_match(reference_text, result)
            if best is None or score > best.score:
                open_access_pdf = result.get("openAccessPdf") or {}
                best = ResolvedPaper(
                    paper_id=result.get("paperId"),
                    title=result.get("title", ""),
                    year=result.get("year"),
                    url=(open_access_pdf.get("url") or "").strip(),
                    query=query,
                    score=score,
                )

    if best is None or best.score < 0.55:
        return None
    return best


def sentence_window_query(sentence: str, reference_marker: str) -> str | None:
    marker = reference_marker if reference_marker in sentence else (
        f"[{reference_marker}]" if f"[{reference_marker}]" in sentence else reference_marker
    )
    if marker not in sentence:
        return None
    prefix, suffix = sentence.split(marker, 1)
    prefix_tokens = [token for token in tokenize_for_query(prefix) if token not in STOPWORDS]
    suffix_tokens = [token for token in tokenize_for_query(suffix) if token not in STOPWORDS]
    query_tokens = prefix_tokens[-6:] + suffix_tokens[:6]
    if len(query_tokens) < 3:
        return None
    return " ".join(query_tokens)


def tokenize_for_query(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def build_discoverability_queries(sentence: str, reference_marker: str) -> list[str]:
    marker = reference_marker if reference_marker in sentence else (
        f"[{reference_marker}]" if f"[{reference_marker}]" in sentence else reference_marker
    )
    removed_marker = sentence.replace(marker, " ")
    tokens = tokenize_for_query(removed_marker)
    content_tokens = [token for token in tokens if token not in STOPWORDS]

    queries: list[str] = []
    if len(tokens) >= 4:
        queries.append(" ".join(tokens[: min(len(tokens), 14)]))
    if len(content_tokens) >= 3:
        queries.append(" ".join(content_tokens[: min(len(content_tokens), 8)]))
    window_query = sentence_window_query(sentence, marker)
    if window_query:
        queries.append(window_query)

    deduped: list[str] = []
    for query in queries:
        cleaned = normalize_spaces(query)
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)
    return deduped


def maybe_print_source_paper_progress(
    paper_index: int,
    total_papers: int,
    manifest_row: ManifestRow,
    dataset_row_count: int,
    progress_every: int | None,
) -> None:
    if not progress_every or progress_every <= 0:
        return
    if paper_index != 1 and paper_index % progress_every != 0:
        return
    print(
        f"[{paper_index}/{total_papers}] Example source paper: "
        f"{manifest_row.source_paper_title} | kept {dataset_row_count} dataset row(s) so far"
    )


def title_matches(candidate_title: str, oracle_title: str, threshold: float = 0.9) -> bool:
    normalized_candidate = normalize_compare_text(candidate_title)
    normalized_oracle = normalize_compare_text(oracle_title)
    ratio = SequenceMatcher(a=normalized_candidate, b=normalized_oracle).ratio()
    return ratio >= threshold


def check_discoverability(
    oracle_title: str,
    source_year: int,
    queries: Iterable[str],
    api: SemanticScholarAPI,
    limit: int = 20,
    request_pause_s: float = 0.0,
    max_request_retries: int = DEFAULT_MAX_REQUEST_RETRIES,
    request_retry_backoff_s: float = DEFAULT_REQUEST_RETRY_BACKOFF_S,
    fields_of_study: str | None = "Computer Science",
) -> DiscoverabilityResult | None:
    for query in queries:
        payload = run_with_retries(
            action_label=f"Checking discoverability with Semantic Scholar query: {query[:80]}",
            func=lambda query=query: api.relevance_search(
                query=query,
                fields=API_FIELDS,
                fieldsOfStudy=fields_of_study,
                year=str(source_year),
                limit=limit,
            ),
            max_retries=max_request_retries,
            retry_backoff_s=request_retry_backoff_s,
        )
        if request_pause_s > 0:
            time.sleep(request_pause_s)
        for rank, result in enumerate(payload.get("data", []), start=1):
            if title_matches(result.get("title", ""), oracle_title):
                return DiscoverabilityResult(query=query, rank=rank)
    return None


def iter_rows(
    manifest_rows: Iterable[ManifestRow],
    api: SemanticScholarAPI,
    cache_dir: Path,
    split: str,
    starting_id: int,
    request_pause_s: float,
    session: requests.Session,
    citation_mode: str = "single",
    citation_style: str = "numeric",
    progress_every: int | None = None,
    output_csv_path: Path | None = None,
    audit_jsonl_path: Path | None = None,
    resume_state_path: Path | None = None,
    restart: bool = False,
    max_request_retries: int = DEFAULT_MAX_REQUEST_RETRIES,
    request_retry_backoff_s: float = DEFAULT_REQUEST_RETRY_BACKOFF_S,
    fields_of_study: str | None = "Computer Science",
    max_dataset_rows: int | None = None,
) -> tuple[list[dict[str, str | int]], list[dict[str, object]]]:
    manifest_rows = list(manifest_rows)
    dataset_rows: list[dict[str, str | int]] = []
    audit_rows: list[dict[str, object]] = []
    total_papers = len(manifest_rows)
    next_id = starting_id
    processed_source_paper_keys: set[str] = set()
    state: dict[str, Any] | None = None
    reference_resolution_cache: dict[tuple[int, str], ResolvedPaper | None] = {}
    discoverability_cache: dict[tuple[str, int, tuple[str, ...]], DiscoverabilityResult | None] = {}

    if output_csv_path or audit_jsonl_path or resume_state_path:
        ensure_output_files_initialized(
            output_csv_path=output_csv_path,
            audit_jsonl_path=audit_jsonl_path,
            resume_state_path=resume_state_path,
            restart=restart,
        )

    if resume_state_path:
        if resume_state_path.exists():
            state = load_resume_state(resume_state_path)
            state["_path"] = str(resume_state_path)
            validate_resume_state(
                state,
                starting_id=starting_id,
                split=split,
                citation_mode=citation_mode,
                citation_style=citation_style,
                fields_of_study=fields_of_study,
            )
            ensure_resume_outputs_are_consistent(
                state,
                output_csv_path=output_csv_path,
                audit_jsonl_path=audit_jsonl_path,
            )
            processed_source_paper_keys = set(state.get("processed_source_paper_keys", []))
            next_id = int(state.get("next_id", starting_id))
            print(
                f"Loaded resume state from {resume_state_path}: "
                f"{state.get('processed_source_paper_count', len(processed_source_paper_keys))} source paper(s) already processed, "
                f"{state.get('dataset_row_count', 0)} dataset row(s) already saved"
            )
            if state.get("last_error"):
                last_error = state["last_error"]
                print(
                    f"Last recorded error was on {last_error.get('source_paper_title')}: "
                    f"{last_error.get('message')}"
                )
        else:
            state = initialize_resume_state(
                starting_id=starting_id,
                split=split,
                citation_mode=citation_mode,
                citation_style=citation_style,
                fields_of_study=fields_of_study,
            )
            state["_path"] = str(resume_state_path)
            save_resume_state(resume_state_path, state)

    remaining_papers = total_papers - len(processed_source_paper_keys)
    if processed_source_paper_keys:
        print(
            f"Processing {remaining_papers} remaining source paper(s) for citation extraction "
            f"({len(processed_source_paper_keys)} already completed)"
        )
    else:
        print(f"Processing {total_papers} source paper(s) for citation extraction")

    for paper_index, manifest_row in enumerate(manifest_rows, start=1):
        source_paper_key = build_source_paper_key(manifest_row)
        if source_paper_key in processed_source_paper_keys:
            continue
        maybe_print_source_paper_progress(
            paper_index=paper_index,
            total_papers=total_papers,
            manifest_row=manifest_row,
            dataset_row_count=(
                (state.get("dataset_row_count", 0) if state else 0) + len(dataset_rows)
            ),
            progress_every=progress_every,
        )
        current_dataset_rows: list[dict[str, str | int]] = []
        current_audit_rows: list[dict[str, object]] = []

        try:
            pdf_path = ensure_pdf_available(
                manifest_row,
                cache_dir,
                session,
                max_request_retries=max_request_retries,
                request_retry_backoff_s=request_retry_backoff_s,
            )
            pdf_text = extract_pdf_text(pdf_path)
            body_text, references_text = split_body_and_references(pdf_text)
            reference_map = extract_reference_map(references_text, citation_style=citation_style)
            candidates = find_candidate_sentences(
                body_text,
                reference_map,
                citation_mode=citation_mode,
                citation_style=citation_style,
            )

            for candidate in candidates:
                reference_text = reference_map[candidate.reference_key]
                if looks_like_reference_entry_leakage(candidate.sentence, reference_text):
                    continue
                reference_cache_key = (
                    manifest_row.year,
                    manifest_row.source_paper_title,
                    reference_text,
                )
                if reference_cache_key not in reference_resolution_cache:
                    reference_resolution_cache[reference_cache_key] = resolve_reference(
                        reference_text,
                        manifest_row.year,
                        api,
                        request_pause_s=request_pause_s,
                        max_request_retries=max_request_retries,
                        request_retry_backoff_s=request_retry_backoff_s,
                        fields_of_study=fields_of_study,
                        source_paper_title=manifest_row.source_paper_title,
                    )
                resolved = reference_resolution_cache[reference_cache_key]
                if resolved is None:
                    continue
                discoverability_queries = tuple(
                    build_discoverability_queries(candidate.sentence, candidate.citation_marker)
                )
                discoverability_cache_key = (
                    resolved.title,
                    manifest_row.year,
                    discoverability_queries,
                )
                if discoverability_cache_key not in discoverability_cache:
                    discoverability_cache[discoverability_cache_key] = check_discoverability(
                        resolved.title,
                        manifest_row.year,
                        discoverability_queries,
                        api,
                        limit=20,
                        request_pause_s=request_pause_s,
                        max_request_retries=max_request_retries,
                        request_retry_backoff_s=request_retry_backoff_s,
                        fields_of_study=fields_of_study,
                    )
                discoverability = discoverability_cache[discoverability_cache_key]
                if discoverability is None:
                    continue

                current_dataset_rows.append(
                    {
                        "id": next_id,
                        "excerpt": candidate.excerpt,
                        "target_paper_title": resolved.title,
                        "target_paper_url": resolved.url,
                        "source_paper_title": manifest_row.source_paper_title,
                        "source_paper_url": manifest_row.source_paper_url,
                        "year": manifest_row.year,
                        "split": split,
                    }
                )
                current_audit_rows.append(
                    {
                        "id": next_id,
                        "source_paper_title": manifest_row.source_paper_title,
                        "source_paper_url": manifest_row.source_paper_url,
                        "venue": manifest_row.venue,
                        "domain": manifest_row.domain,
                        "pdf_path": str(pdf_path),
                        "citation_marker": candidate.citation_marker,
                        "sentence_citation_count": candidate.citation_count,
                        "reference_key": candidate.reference_key,
                        "raw_sentence": candidate.sentence,
                        "body_sentence_index": candidate.body_sentence_index,
                        "passage_sha256": hashlib.sha256(
                            normalize_spaces(candidate.sentence).encode("utf-8")
                        ).hexdigest(),
                        "raw_reference": reference_text,
                        "resolved_paper_id": resolved.paper_id,
                        "resolved_title": resolved.title,
                        "resolved_year": resolved.year,
                        "resolution_query": resolved.query,
                        "resolution_score": resolved.score,
                        "discoverability_query": discoverability.query,
                        "discoverability_rank": discoverability.rank,
                    }
                )
                next_id += 1
                if max_dataset_rows is not None and (
                    len(dataset_rows) + len(current_dataset_rows) >= max_dataset_rows
                ):
                    break
        except PdfReadError as exc:
            print(
                f"Skipping unreadable PDF for {manifest_row.source_paper_title}: "
                f"{exc or exc.__class__.__name__}"
            )
        except Exception as exc:
            if state is not None and resume_state_path is not None:
                state["next_id"] = next_id
                state["last_error"] = {
                    "source_paper_title": manifest_row.source_paper_title,
                    "source_paper_url": manifest_row.source_paper_url,
                    "message": str(exc),
                    "recorded_at": utc_now_iso(),
                }
                save_resume_state(resume_state_path, state)
            raise

        dataset_rows.extend(current_dataset_rows)
        audit_rows.extend(current_audit_rows)

        if output_csv_path is not None:
            append_dataset_rows(output_csv_path, current_dataset_rows)
        if audit_jsonl_path is not None:
            append_audit_rows(audit_jsonl_path, current_audit_rows)

        if state is not None and resume_state_path is not None:
            processed_source_paper_keys.add(source_paper_key)
            state["processed_source_paper_keys"] = sorted(processed_source_paper_keys)
            state["processed_source_paper_count"] = len(processed_source_paper_keys)
            state["dataset_row_count"] = int(state.get("dataset_row_count", 0)) + len(current_dataset_rows)
            state["audit_row_count"] = int(state.get("audit_row_count", 0)) + len(current_audit_rows)
            state["next_id"] = next_id
            state["last_processed_source_paper_title"] = manifest_row.source_paper_title
            state["last_processed_source_paper_url"] = manifest_row.source_paper_url
            state["last_error"] = None
            save_resume_state(resume_state_path, state)

        if max_dataset_rows is not None and len(dataset_rows) >= max_dataset_rows:
            print(f"Reached --max_dataset_rows={max_dataset_rows}; stopping collection")
            break

    return dataset_rows, audit_rows


def write_dataset_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = make_csv_dict_writer(handle, DATASET_COLUMNS)
        writer.writeheader()
        writer.writerows([sanitize_unicode(row) for row in rows])


def write_audit_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(sanitize_unicode(row), ensure_ascii=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect DATASET.csv-style training rows from CS papers with numeric citations."
    )
    parser.add_argument("--manifest", required=True, help="CSV with source paper metadata.")
    parser.add_argument("--output_csv", required=True, help="Output CSV path.")
    parser.add_argument("--audit_jsonl", required=True, help="Output JSONL audit path.")
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
        default="single",
        help="Keep only sentences with a single citation, multiple citations, or any citation count.",
    )
    parser.add_argument(
        "--citation_style",
        choices=["numeric", "author_year"],
        default="numeric",
        help="Citation style to extract from source papers.",
    )
    parser.add_argument(
        "--fields_of_study",
        default="Computer Science",
        help=(
            "Semantic Scholar field filter used for reference resolution and discoverability "
            '(for example, "Medicine" or "Biology"). Use an empty string to disable the filter.'
        ),
    )
    parser.add_argument(
        "--request_pause_s",
        type=float,
        default=0.0,
        help="Sleep duration between Semantic Scholar API calls.",
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
        help="Number of retries for transient PDF and Semantic Scholar request failures.",
    )
    parser.add_argument(
        "--request_retry_backoff_s",
        type=float,
        default=DEFAULT_REQUEST_RETRY_BACKOFF_S,
        help="Base backoff in seconds for request retries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session = requests.Session()
    session.headers.update({"User-Agent": "CiteGuard training-data collector"})
    api = SemanticScholarAPI(session=session)
    output_csv_path = Path(args.output_csv)
    audit_jsonl_path = Path(args.audit_jsonl)
    resume_state_path = (
        Path(args.resume_state_path)
        if args.resume_state_path
        else default_resume_state_path(output_csv_path)
    )

    manifest_rows = load_manifest(Path(args.manifest))
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
    print(f"Wrote {final_state.get('dataset_row_count', 0)} dataset rows to {args.output_csv}")
    print(f"Wrote {final_state.get('audit_row_count', 0)} audit rows to {args.audit_jsonl}")
    print(f"Checkpoint state saved at {resume_state_path}")


if __name__ == "__main__":
    main()
