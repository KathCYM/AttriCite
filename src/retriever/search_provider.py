from src.utils.semantic_scholar import SemanticScholarAPI, SemanticScholarWebSearch
from src.utils.str_matcher import find_match_psr
from src.utils import PaperSearchResult
from typing import List

TITLE_SEPERATOR = "[TITLE_SEPARATOR]"

class SearchProvider:
    def __call__(self, search_term: str, year: str) -> List[PaperSearchResult]:
        raise NotImplementedError("Subclasses must implement this method")


class SemanticScholarSearchProvider(SearchProvider):
    def __init__(
        self,
        fieldsOfStudy: str = "Computer Science",
        limit: int = 10,
        # sort: str = "citationCount:desc",
        only_open_access: bool = False,
        console=None,
    ):
        # These fields are required for the PaperSearchResult model
        self.fields = (
            "paperId,title,authors,venue,year,citationCount,abstract,openAccessPdf"
        )
        self.fieldsOfStudy = fieldsOfStudy
        self.limit = limit
        self.only_open_access = only_open_access
        # self.sort = sort
        self.s2api = SemanticScholarAPI()
        self.console = console

    def citation_count_search(
        self, query: str, year: str | None, max_search_limit: int = 100, skip=[]
        ) -> List[PaperSearchResult]:
        papers: List[PaperSearchResult] = []
        for offset in range(0, max_search_limit, 100):
            tmp_papers = self.s2api.relevance_search(
                query,
                self.fields,
                self.fieldsOfStudy,
                year=year,
                limit=min(100, max_search_limit - offset),
                offset=offset,
                only_open_access=self.only_open_access,
            )
            if "data" not in tmp_papers:
                break
            self.console.log(f"Fetched {len(tmp_papers.get('data', []))} papers for offset {offset}")
            for paper in tmp_papers.get("data", []):
                skip_flag = False
                for skip_title in skip:
                    if find_match_psr(skip_title.split(TITLE_SEPERATOR), paper.get('title'), 0.9) is not None:
                        self.console.log(f"Skipped paper title: {paper.get('title')}")
                        skip_flag = True
                        break
                if skip_flag:
                    continue
                papers.append(PaperSearchResult(**paper))

        papers = sorted(papers, key=lambda x: x.citationCount, reverse=True)
        return papers[: self.limit]

    def snippet_search(
        self, query: str, year: str, src_paper_title: str, skip=[]
    ) -> str:
        snippets = self.s2api.snippet_search(
            query,
            year,
            self.fieldsOfStudy,
            limit=self.limit
        )
        snippets_str = ""
        self.console.log(f"Fetched {len(snippets.get('data', []))} snippets")
        for item in snippets.get("data", []):
            snippet = item.get("snippet", {})
            paper = item.get('paper', {})
            # Skip if the paper title matches any of the skip titles
            skip_flag = False
            for skip_title in skip:
                if find_match_psr(skip_title.split(TITLE_SEPERATOR), paper.get('title'), 0.9) is not None:
                    self.console.log(f"Skipped snippet from paper title: {paper.get('title')}")
                    skip_flag = True
                    break
            if skip_flag:
                continue
            snippets_str += (
                f"- Paper ID: {paper.get('corpusId')}\n"
                f"  Title: {paper.get('title')}\n"
                f"  Section: {snippet.get('section')}\n"
                f"  Snippet: {snippet.get('text')}\n"
                "\n"
            )
        self.console.log(snippets_str)
        if not snippets.get("data"):
            snippets_str = "No papers were found for the given search query. Please use a different query."
        return snippets_str

    def __call__(self, query: str, year: str | None = None, skip=[]) -> List[PaperSearchResult]:
        # papers = self.s2api.bulk_search(query, self.fields, self.fieldsOfStudy, self.sort)
        self.console.log(f"Searching for query: {query} with year: {year}")
        papers = self.s2api.relevance_search(
            query,
            self.fields,
            self.fieldsOfStudy,
            year,
            self.limit,
            only_open_access=self.only_open_access,
        )
        if "data" not in papers:
            return []
        self.console.log(f"Fetched {len(papers.get('data', []))} papers")
        final_papers = []
        for paper in papers.get("data", []):
            skip_flag = False
            for skip_title in skip:
                if find_match_psr(skip_title.split(TITLE_SEPERATOR), paper.get('title'), 0.9) is not None:
                    self.console.log(f"Skipped paper title: {paper.get('title')}")
                    skip_flag = True
                    break
            if skip_flag:
                continue
            final_papers.append(PaperSearchResult(**paper))
        return final_papers


class SemanticScholarWebSearchProvider(SearchProvider):
    def __init__(
        self,
        fieldsOfStudy: str = "Computer Science",
        limit: int = 10,
        # sort: str = "citationCount:desc",
        only_open_access: bool = False,
    ):
        # These fields are required for the PaperSearchResult model
        self.fields = (
            "paperId,title,authors,venue,year,citationCount,abstract,openAccessPdf"
        )
        self.fieldsOfStudy = fieldsOfStudy
        self.limit = limit
        self.only_open_access = only_open_access
        # self.sort = sort
        self.s2api = SemanticScholarWebSearch()

    def citation_count_search(
        self, query: str, year: str | None
    ) -> List[PaperSearchResult]:
        papers = self.s2api.web_search(
            query,
            fields=self.fields,
            fields_of_study=(
                self.fieldsOfStudy.replace(" ", "-").lower()
                if self.fieldsOfStudy
                else None
            ),
            sort="total-citations",
            cutoff_year=year.replace("-", "") if year else None,
            limit=self.limit,
            only_open_access=self.only_open_access,
        )
        papers = [PaperSearchResult(**paper) for paper in papers]
        return papers[: self.limit]

    def __call__(self, query: str, year: str | None = None) -> List[PaperSearchResult]:
        # papers = self.s2api.bulk_search(query, self.fields, self.fieldsOfStudy, self.sort)
        papers = self.s2api.relevance_search(
            query,
            self.fields,
            self.fieldsOfStudy,
            year,
            self.limit,
            only_open_access=self.only_open_access,
        )
        papers = [PaperSearchResult(**paper) for paper in papers]
        return papers
