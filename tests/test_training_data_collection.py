import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import requests
from PyPDF2.errors import PdfReadError

from training_data_collection.collect_cs_conference_candidates import (
    build_manifest_rows_from_proceedings,
    collect_proceedings_papers,
    extract_listing_page_urls,
    derive_pdf_url_from_cvf_abstract_url,
    extract_proceedings_papers_from_html,
    write_manifest_csv,
)
from training_data_collection.collect_numeric_cs_candidates import (
    CandidateSentence,
    DATASET_COLUMNS,
    DiscoverabilityResult,
    ManifestRow,
    ResolvedPaper,
    append_dataset_rows,
    build_discoverability_queries,
    extract_reference_map,
    split_author_year_reference_entries,
    extract_author_year_reference_key,
    ensure_pdf_available,
    find_candidate_sentences,
    iter_rows,
    load_resume_state,
    looks_like_reference_entry_leakage,
    maybe_print_source_paper_progress,
    overwrite_with_header,
    resolve_reference,
    sanitize_unicode,
    split_body_and_references,
)
from training_data_collection.collect_pubmed_candidates import (
    build_europe_pmc_query,
    build_pmc_query,
    manifest_rows_from_europe_pmc_summaries,
    manifest_rows_from_pmc_summaries,
)


class TrainingDataCollectionTests(unittest.TestCase):
    def test_body_split_uses_first_plausible_references_heading(self):
        text = (
            "Table of Contents\nReferences\n" + "Body text. " * 500
            + "\nReferences\nSmith. 2020. Target paper.\n"
            + "\nAppendix A\nAdditional analysis.\nReferences\nNot a bibliography heading."
        )
        body, references = split_body_and_references(text)
        self.assertIn("Body text", body)
        self.assertNotIn("Smith. 2020", body)
        self.assertTrue(references.startswith("Smith. 2020"))

    def test_ensure_pdf_available_replaces_html_cache_and_uses_europe_pmc_fallback(self):
        class FakeResponse:
            def __init__(self, content, content_type):
                self.content = content
                self.headers = {"Content-Type": content_type}

            def raise_for_status(self):
                return None

        class FakeSession:
            def __init__(self):
                self.urls = []

            def get(self, url, timeout=60):
                self.urls.append(url)
                if "ncbi.nlm.nih.gov" in url:
                    return FakeResponse(b"<!doctype html>recaptcha", "text/html")
                return FakeResponse(b"%PDF-1.7\nvalid fallback", "application/pdf")

        row = ManifestRow(
            source_paper_title="PMC Test Paper",
            source_paper_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC123456/",
            year=2025,
            pdf_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC123456/pdf/",
            pdf_path=None,
            venue="Test Journal",
            domain="Biomedical Research",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            cached_path = cache_dir / "pmc-test-paper.pdf"
            cached_path.write_bytes(b"<!doctype html>old recaptcha")
            session = FakeSession()

            result = ensure_pdf_available(row, cache_dir, session)

            self.assertEqual(result, cached_path)
            self.assertTrue(result.read_bytes().startswith(b"%PDF-"))
            self.assertEqual(
                session.urls[-1],
                "https://europepmc.org/articles/PMC123456?pdf=render",
            )

    def test_ensure_pdf_available_turns_fallback_http_error_into_skippable_pdf_error(self):
        class FakeResponse:
            def __init__(self, content=b"", status_code=200):
                self.content = content
                self.status_code = status_code
                self.headers = {"Content-Type": "text/html"}

            def raise_for_status(self):
                if self.status_code >= 400:
                    response = requests.Response()
                    response.status_code = self.status_code
                    raise requests.exceptions.HTTPError(
                        f"{self.status_code} Server Error", response=response
                    )

        class FakeSession:
            def get(self, url, timeout=60):
                if "ncbi.nlm.nih.gov" in url:
                    return FakeResponse(b"<!doctype html>recaptcha")
                return FakeResponse(status_code=500)

        row = ManifestRow(
            source_paper_title="Unavailable PMC Paper",
            source_paper_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC999999/",
            year=2025,
            pdf_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC999999/pdf/",
            pdf_path=None,
            venue="Test Journal",
            domain="Biomedical Research",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(PdfReadError) as context:
                ensure_pdf_available(
                    row,
                    Path(temp_dir),
                    FakeSession(),
                    max_request_retries=0,
                )

        self.assertIn("not a PDF", str(context.exception))
        self.assertIn("500 Server Error", str(context.exception))

    def test_build_pmc_query_always_requires_open_access_and_year(self):
        self.assertEqual(build_pmc_query(2025), "2025[PDAT] AND open access[filter]")
        self.assertEqual(
            build_pmc_query(2025, "cancer AND imaging"),
            "(cancer AND imaging) AND 2025[PDAT] AND open access[filter]",
        )

    def test_build_europe_pmc_query_requires_an_indexed_pdf(self):
        self.assertEqual(
            build_europe_pmc_query(2025, '"medical imaging"'),
            '("medical imaging") AND PUB_YEAR:2025 AND OPEN_ACCESS:Y AND HAS_PDF:Y',
        )

    def test_manifest_rows_from_europe_pmc_uses_indexed_pdf_link(self):
        rows = manifest_rows_from_europe_pmc_summaries(
            [
                {
                    "pmcid": "PMC12345678",
                    "title": "An Imaging Study",
                    "journalInfo": {"journal": {"title": "Imaging Journal"}},
                    "fullTextUrlList": {
                        "fullTextUrl": [
                            {
                                "availabilityCode": "OA",
                                "documentStyle": "pdf",
                                "url": "https://europepmc.org/articles/PMC12345678?pdf=render",
                            }
                        ]
                    },
                }
            ],
            year=2025,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].venue, "Imaging Journal")
        self.assertEqual(
            rows[0].pdf_url,
            "https://europepmc.org/articles/PMC12345678?pdf=render",
        )

    def test_manifest_rows_from_pmc_summaries_builds_open_pdf_urls(self):
        rows = manifest_rows_from_pmc_summaries(
            [
                {
                    "uid": "12345678",
                    "title": "  A Biomedical   Study ",
                    "fulljournalname": "Example Medical Journal",
                    "articleids": [{"idtype": "pmcid", "value": "PMC12345678"}],
                }
            ],
            year=2025,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].source_paper_title, "A Biomedical Study")
        self.assertEqual(rows[0].venue, "Example Medical Journal")
        self.assertEqual(
            rows[0].pdf_url,
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC12345678/pdf/",
        )

    def test_sanitize_unicode_replaces_lone_surrogate(self):
        self.assertEqual(sanitize_unicode("valid café \ud835 text"), "valid café ? text")

    def test_derive_pdf_url_from_cvf_abstract_url(self):
        abstract_url = (
            "https://openaccess.thecvf.com/content/CVPR2025/html/"
            "Yuan_From_Poses_to_Identity_CVPR_2025_paper.html"
        )
        self.assertEqual(
            derive_pdf_url_from_cvf_abstract_url(abstract_url),
            "https://openaccess.thecvf.com/content/CVPR2025/papers/"
            "Yuan_From_Poses_to_Identity_CVPR_2025_paper.pdf",
        )

    def test_reference_entry_leakage_matches_author_list(self):
        self.assertTrue(
            looks_like_reference_entry_leakage(
                "[108] Mostafa Kalhor, Joel Lapin, Mario Picciani, and Mathias Wilhelm.",
                "Mostafa Kalhor, Joel Lapin, Mario Picciani, and Mathias Wilhelm.",
            )
        )

    def test_reference_entry_leakage_matches_unicode_author_list(self):
        self.assertTrue(
            looks_like_reference_entry_leakage(
                "[22] Konstantin Weißenow, Michael Heinzinger, and Burkhard Rost.",
                "Konstantin Weißenow, Michael Heinzinger, and Burkhard Rost. "
                "Protein language-model embeddings. Structure, 2022.",
            )
        )

    def test_reference_entry_leakage_keeps_normal_citing_sentence(self):
        self.assertFalse(
            looks_like_reference_entry_leakage(
                "Our method extends the robust objective introduced in [12].",
                "A. Smith and B. Jones. Robust objectives for learning. 2020.",
            )
        )

    def test_extract_reference_map_supports_bracketed_numeric_entries(self):
        reference_text = """
        [1] Alice Smith and Bob Jones. Great Retrieval Paper. ICML 2020.
        [2] Carol Doe. Another Paper. NeurIPS 2021.
        """

        reference_map = extract_reference_map(reference_text)

        self.assertEqual(
            reference_map["1"],
            "Alice Smith and Bob Jones. Great Retrieval Paper. ICML 2020.",
        )
        self.assertEqual(
            reference_map["2"],
            "Carol Doe. Another Paper. NeurIPS 2021.",
        )

    def test_extract_reference_map_supports_author_year_entries(self):
        reference_text = """
        Zhengxin Zhang, Dan Zhao, and Xupeng Miao. 2024. Quantized Side Tuning.
        Jacob Devasier, Yogesh Gurjar, and Chengkai Li. 2024. Robust Frame-Semantic Models.
        """

        reference_map = extract_reference_map(reference_text, citation_style="author_year")

        self.assertEqual(
            reference_map["zhang-2024"],
            "Zhengxin Zhang, Dan Zhao, and Xupeng Miao. 2024. Quantized Side Tuning.",
        )
        self.assertEqual(
            reference_map["devasier-2024"],
            "Jacob Devasier, Yogesh Gurjar, and Chengkai Li. 2024. Robust Frame-Semantic Models.",
        )

    def test_find_candidate_sentences_keeps_single_numeric_citation_only(self):
        body_text = (
            "We improve retrieval with sparse attention [1]. "
            "This sentence cites two papers [2,3]. "
            "Ablations are reported in Table 2."
        )
        reference_map = {
            "1": "Alice Smith and Bob Jones. Great Retrieval Paper. ICML 2020.",
            "2": "Paper Two.",
            "3": "Paper Three.",
        }

        candidates = find_candidate_sentences(body_text, reference_map, min_words=4, max_words=20)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].reference_key, "1")
        self.assertEqual(
            candidates[0].excerpt,
            "We improve retrieval with sparse attention [CITATION].",
        )

    def test_find_candidate_sentences_supports_author_year_parenthetical_citation(self):
        body_text = "We build on quantized tuning methods (Zhang et al., 2024) for efficient adaptation."
        reference_map = {
            "zhang-2024": "Zhengxin Zhang, Dan Zhao, and Xupeng Miao. 2024. Quantized Side Tuning.",
        }

        candidates = find_candidate_sentences(
            body_text,
            reference_map,
            min_words=4,
            max_words=20,
            citation_mode="single",
            citation_style="author_year",
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].reference_key, "zhang-2024")
        self.assertEqual(
            candidates[0].excerpt,
            "We build on quantized tuning methods ([CITATION]) for efficient adaptation.",
        )

    def test_find_candidate_sentences_supports_author_year_narrative_citation(self):
        body_text = "Zhang et al. (2024) propose an efficient adaptation recipe for quantized models."
        reference_map = {
            "zhang-2024": "Zhengxin Zhang, Dan Zhao, and Xupeng Miao. 2024. Quantized Side Tuning.",
        }

        candidates = find_candidate_sentences(
            body_text,
            reference_map,
            min_words=4,
            max_words=20,
            citation_mode="single",
            citation_style="author_year",
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].reference_key, "zhang-2024")
        self.assertEqual(
            candidates[0].excerpt,
            "[CITATION] propose an efficient adaptation recipe for quantized models.",
        )

    def test_find_candidate_sentences_skips_numeric_reference_author_list(self):
        body_text = "[108] Mostafa Kalhor, Joel Lapin, Mario Picciani, and Mathias Wilhelm."
        reference_map = {
            "108": (
                "Mostafa Kalhor, Joel Lapin, Mario Picciani, and Mathias Wilhelm. "
                "Rescoring peptide spectrum matches."
            ),
        }

        candidates = find_candidate_sentences(
            body_text,
            reference_map,
            min_words=4,
            max_words=40,
            citation_mode="single",
            citation_style="numeric",
        )

        self.assertEqual(candidates, [])

    def test_find_candidate_sentences_skips_numeric_reference_entry_with_title(self):
        body_text = (
            "[67] Mario Picciani, Wassim Gabriel, Victor-George Giurcoiu, Omar Shouman, "
            "Firas Hamood, Ludwig Lautenbacher, Cecilia Bang Jensen, Julian Muller, "
            "Mostafa Kalhor, Armin Soleymaniniya, et al. Oktoberfest: Open-source "
            "spectral library generation and rescoring pipeline based on prosit."
        )
        reference_map = {
            "67": (
                "Mario Picciani, Wassim Gabriel, Victor-George Giurcoiu, Omar Shouman, "
                "Firas Hamood, Ludwig Lautenbacher, Cecilia Bang Jensen, Julian Muller, "
                "Mostafa Kalhor, Armin Soleymaniniya, et al. Oktoberfest."
            ),
        }

        candidates = find_candidate_sentences(
            body_text,
            reference_map,
            min_words=4,
            max_words=80,
            citation_mode="single",
            citation_style="numeric",
        )

        self.assertEqual(candidates, [])

    def test_find_candidate_sentences_keeps_sentence_starting_with_numeric_citation(self):
        body_text = "[12] We show that sparse retrieval improves recall in long-context settings."
        reference_map = {"12": "Paper Twelve."}

        candidates = find_candidate_sentences(
            body_text,
            reference_map,
            min_words=4,
            max_words=20,
            citation_mode="single",
            citation_style="numeric",
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].reference_key, "12")
        self.assertEqual(
            candidates[0].excerpt,
            "[CITATION] We show that sparse retrieval improves recall in long-context settings.",
        )

    def test_build_discoverability_queries_returns_multiple_deduped_queries(self):
        sentence = "We improve retrieval with sparse attention [12] for long-context citation matching."

        queries = build_discoverability_queries(sentence, "12")

        self.assertGreaterEqual(len(queries), 2)
        self.assertEqual(len(queries), len(set(queries)))
        self.assertTrue(all("[12]" not in query for query in queries))

    def test_find_candidate_sentences_supports_multi_citation_sentences(self):
        body_text = "We improve retrieval with sparse attention [2,3] and stronger negatives [4-5]."
        reference_map = {
            "2": "Paper Two.",
            "3": "Paper Three.",
            "4": "Paper Four.",
            "5": "Paper Five.",
        }

        candidates = find_candidate_sentences(
            body_text,
            reference_map,
            min_words=4,
            max_words=20,
            citation_mode="any",
        )

        self.assertEqual([candidate.reference_key for candidate in candidates], ["2", "3", "4", "5"])
        self.assertEqual(candidates[0].excerpt, "We improve retrieval with sparse attention [CITATION, 3] and stronger negatives [4-5].")
        self.assertEqual(candidates[1].excerpt, "We improve retrieval with sparse attention [2, CITATION] and stronger negatives [4-5].")
        self.assertEqual(candidates[2].excerpt, "We improve retrieval with sparse attention [2,3] and stronger negatives [CITATION, 5].")
        self.assertEqual(candidates[3].excerpt, "We improve retrieval with sparse attention [2,3] and stronger negatives [4, CITATION].")
        self.assertTrue(all(candidate.citation_count == 4 for candidate in candidates))

    def test_build_discoverability_queries_accepts_multi_citation_marker(self):
        sentence = "We improve retrieval with sparse attention [12, 14] for long-context citation matching."

        queries = build_discoverability_queries(sentence, "[12, 14]")

        self.assertGreaterEqual(len(queries), 2)
        self.assertEqual(len(queries), len(set(queries)))
        self.assertTrue(all("[12, 14]" not in query for query in queries))

    def test_build_discoverability_queries_accepts_author_year_marker(self):
        sentence = "We build on quantized tuning methods (Zhang et al., 2024) for efficient adaptation."

        queries = build_discoverability_queries(sentence, "(Zhang et al., 2024)")

        self.assertGreaterEqual(len(queries), 2)
        self.assertEqual(len(queries), len(set(queries)))
        self.assertTrue(all("(Zhang et al., 2024)" not in query for query in queries))
        self.assertTrue(any("quantized" in query for query in queries))
        self.assertTrue(any("adaptation" in query for query in queries))

    def test_extract_proceedings_papers_from_html_pairs_titles_with_pdf_links(self):
        html = """
        <html>
          <body>
            <dt class="ptitle">
              <a href="/content/CVPR2024/html/Author_Paper_Title_CVPR_2024_paper.html">
                Paper Title
              </a>
            </dt>
            <dd>
              <a href="#">Wilfried Philips</a>
              <a href="/content/CVPR2024/papers/Author_Paper_Title_CVPR_2024_paper.pdf">pdf</a>
              <a href="/content/CVPR2024/supp/Author_Paper_Title_CVPR_2024_supp.pdf">supp</a>
            </dd>
          </body>
        </html>
        """

        papers = extract_proceedings_papers_from_html(html, "https://openaccess.thecvf.com/CVPR2024")

        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0].title, "Paper Title")
        self.assertEqual(
            papers[0].source_paper_url,
            "https://openaccess.thecvf.com/content/CVPR2024/html/Author_Paper_Title_CVPR_2024_paper.html",
        )
        self.assertEqual(
            papers[0].pdf_url,
            "https://openaccess.thecvf.com/content/CVPR2024/papers/Author_Paper_Title_CVPR_2024_paper.pdf",
        )

    def test_extract_proceedings_papers_from_acl_style_html(self):
        html = """
        <html>
          <body>
            <p>
              <a href=/2024.acl-long.1.pdf>pdf</a>
              <a href=/2024.acl-long.1.bib>bib</a>
              <a href=/2024.acl-long.1/>abs</a>
              <strong><a href=/2024.acl-long.1/>Paper Title</a></strong>
            </p>
          </body>
        </html>
        """

        papers = extract_proceedings_papers_from_html(html, "https://aclanthology.org/volumes/2024.acl-long/")

        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0].title, "Paper Title")
        self.assertEqual(papers[0].source_paper_url, "https://aclanthology.org/2024.acl-long.1/")
        self.assertEqual(papers[0].pdf_url, "https://aclanthology.org/2024.acl-long.1.pdf")

    def test_extract_proceedings_papers_prefers_matching_title_href_over_nearby_author(self):
        html = """
        <html>
          <body>
            <p>
              <a href=/people/vivek-srikumar/>Vivek Srikumar</a>
              <a href=/2024.acl-long.1.pdf>pdf</a>
              <a href=/2024.acl-long.1.bib>bib</a>
              <a href=/2024.acl-long.1/>Quantized Side Tuning</a>
            </p>
          </body>
        </html>
        """

        papers = extract_proceedings_papers_from_html(html, "https://aclanthology.org/volumes/2024.acl-long/")

        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0].title, "Quantized Side Tuning")
        self.assertEqual(papers[0].source_paper_url, "https://aclanthology.org/2024.acl-long.1/")

    def test_extract_proceedings_papers_skips_volume_metadata_entries(self):
        html = """
        <html>
          <body>
            <p>
              <a href=/2024.acl-long.pdf>PDF (full)</a>
              <a href=/2024.acl-long/>https://aclanthology.org/2024.acl-long/</a>
            </p>
            <p>
              <a href=/2024.acl-long.0.pdf>pdf</a>
              <a href=/2024.acl-long.0/>Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics</a>
            </p>
            <p>
              <a href=/2024.acl-long.1.pdf>pdf</a>
              <a href=/2024.acl-long.1/>Quantized Side Tuning</a>
            </p>
          </body>
        </html>
        """

        papers = extract_proceedings_papers_from_html(html, "https://aclanthology.org/volumes/2024.acl-long/")

        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0].title, "Quantized Side Tuning")

    def test_extract_proceedings_papers_from_neurips_title_only_listing(self):
        html = """
        <html>
          <body>
            <ul>
              <li>
                <a href="/paper_files/paper/2024/hash/000f947dcaff8fbffcc3f53a1314f358-Abstract-Conference.html">
                  Test-Time Scaling with Reflective Generative Model
                </a>
              </li>
            </ul>
          </body>
        </html>
        """

        papers = extract_proceedings_papers_from_html(html, "https://papers.nips.cc/paper_files/paper/2024")

        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0].title, "Test-Time Scaling with Reflective Generative Model")
        self.assertEqual(
            papers[0].source_paper_url,
            "https://papers.nips.cc/paper_files/paper/2024/hash/000f947dcaff8fbffcc3f53a1314f358-Abstract-Conference.html",
        )
        self.assertEqual(
            papers[0].pdf_url,
            "https://papers.nips.cc/paper_files/paper/2024/file/000f947dcaff8fbffcc3f53a1314f358-Paper-Conference.pdf",
        )

    def test_extract_proceedings_papers_from_neurips_dataset_track_listing(self):
        html = """
        <html>
          <body>
            <ul>
              <li>
                <a href="/paper_files/paper/2024/hash/013cf29a9e68e4411d0593040a8a1eb3-Abstract-Datasets_and_Benchmarks_Track.html">
                  Open-FLAN: Effective Open-Source Instruction Tuning
                </a>
              </li>
            </ul>
          </body>
        </html>
        """

        papers = extract_proceedings_papers_from_html(html, "https://papers.nips.cc/paper_files/paper/2024")

        self.assertEqual(len(papers), 1)
        self.assertEqual(
            papers[0].pdf_url,
            "https://papers.nips.cc/paper_files/paper/2024/file/013cf29a9e68e4411d0593040a8a1eb3-Paper-Datasets_and_Benchmarks_Track.pdf",
        )

    def test_extract_proceedings_papers_from_pmlr_icml_listing(self):
        html = """
        <html>
          <body>
            <div class="paper">
              <p class="title">Revisiting Character-level Adversarial Attacks for Language Models</p>
              <p class="details">
                <span class="authors">Elias Abad Rocamora,&nbsp;Yongtao Wu</span>
              </p>
              <p class="links">
                [<a href="https://proceedings.mlr.press/v235/abad-rocamora24a.html">abs</a>]
                [<a href="https://raw.githubusercontent.com/mlresearch/v235/main/assets/abad-rocamora24a/abad-rocamora24a.pdf">Download PDF</a>]
              </p>
            </div>
          </body>
        </html>
        """

        papers = extract_proceedings_papers_from_html(html, "https://proceedings.mlr.press/v235/")

        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0].title, "Revisiting Character-level Adversarial Attacks for Language Models")
        self.assertEqual(papers[0].source_paper_url, "https://proceedings.mlr.press/v235/abad-rocamora24a.html")
        self.assertEqual(
            papers[0].pdf_url,
            "https://raw.githubusercontent.com/mlresearch/v235/main/assets/abad-rocamora24a/abad-rocamora24a.pdf",
        )

    def test_extract_listing_page_urls_prefers_all_papers_link(self):
        html = """
        <html>
          <body>
            <a href="/CVPR2024?day=2024-06-19">Day 1: 2024-06-19</a>
            <a href="/CVPR2024?day=2024-06-20">Day 2: 2024-06-20</a>
            <a href="/CVPR2024?day=all">All Papers</a>
          </body>
        </html>
        """

        urls = extract_listing_page_urls(html, "https://openaccess.thecvf.com/CVPR2024")

        self.assertEqual(urls, ["https://openaccess.thecvf.com/CVPR2024?day=all"])

    def test_build_manifest_rows_from_proceedings_follows_all_papers_link(self):
        landing_html = """
        <html>
          <body>
            <a href="/CVPR2024?day=all">All Papers</a>
          </body>
        </html>
        """
        all_papers_html = """
        <html>
          <body>
            <dt class="ptitle">
              <a href="/content/CVPR2024/html/Author_Paper_Title_CVPR_2024_paper.html">
                Paper Title
              </a>
            </dt>
            <dd>
              <a href="/content/CVPR2024/papers/Author_Paper_Title_CVPR_2024_paper.pdf">pdf</a>
            </dd>
          </body>
        </html>
        """

        class FakeResponse:
            def __init__(self, text):
                self.text = text

            def raise_for_status(self):
                return None

        class FakeSession:
            def __init__(self, html_by_url):
                self.html_by_url = html_by_url

            def get(self, url, timeout=60):
                return FakeResponse(self.html_by_url[url])

        session = FakeSession(
            {
                "https://openaccess.thecvf.com/CVPR2024": landing_html,
                "https://openaccess.thecvf.com/CVPR2024?day=all": all_papers_html,
            }
        )

        rows = build_manifest_rows_from_proceedings(
            proceedings_url="https://openaccess.thecvf.com/CVPR2024",
            year=2024,
            venue="CVPR",
            domain="Computer Science",
            session=session,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].source_paper_title, "Paper Title")
        self.assertEqual(
            rows[0].source_paper_url,
            "https://openaccess.thecvf.com/content/CVPR2024/html/Author_Paper_Title_CVPR_2024_paper.html",
        )
        self.assertEqual(
            rows[0].pdf_url,
            "https://openaccess.thecvf.com/content/CVPR2024/papers/Author_Paper_Title_CVPR_2024_paper.pdf",
        )

    def test_collect_proceedings_papers_reports_fetched_listing_pages(self):
        landing_html = """
        <html>
          <body>
            <a href="/CVPR2024?day=all">All Papers</a>
          </body>
        </html>
        """
        all_papers_html = """
        <html>
          <body>
            <dt class="ptitle">
              <a href="/content/CVPR2024/html/Author_Paper_Title_CVPR_2024_paper.html">
                Paper Title
              </a>
            </dt>
            <dd>
              <a href="/content/CVPR2024/papers/Author_Paper_Title_CVPR_2024_paper.pdf">pdf</a>
            </dd>
          </body>
        </html>
        """

        class FakeResponse:
            def __init__(self, text):
                self.text = text

            def raise_for_status(self):
                return None

        class FakeSession:
            def __init__(self, html_by_url):
                self.html_by_url = html_by_url

            def get(self, url, timeout=60):
                return FakeResponse(self.html_by_url[url])

        session = FakeSession(
            {
                "https://openaccess.thecvf.com/CVPR2024": landing_html,
                "https://openaccess.thecvf.com/CVPR2024?day=all": all_papers_html,
            }
        )

        collection = collect_proceedings_papers("https://openaccess.thecvf.com/CVPR2024", session)

        self.assertEqual(len(collection.papers), 1)
        self.assertEqual(
            collection.fetched_page_urls,
            [
                "https://openaccess.thecvf.com/CVPR2024",
                "https://openaccess.thecvf.com/CVPR2024?day=all",
            ],
        )
        self.assertEqual(collection.listing_page_urls, ["https://openaccess.thecvf.com/CVPR2024?day=all"])

    def test_maybe_print_source_paper_progress_prints_first_and_hundredth_examples(self):
        manifest_row = ManifestRow(
            source_paper_title="Example Paper",
            source_paper_url="https://example.org/paper",
            year=2024,
            pdf_url="https://example.org/paper.pdf",
            pdf_path=None,
            venue="CVPR",
            domain="Computer Science",
        )

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            maybe_print_source_paper_progress(
                paper_index=1,
                total_papers=250,
                manifest_row=manifest_row,
                dataset_row_count=3,
                progress_every=100,
            )
            maybe_print_source_paper_progress(
                paper_index=99,
                total_papers=250,
                manifest_row=manifest_row,
                dataset_row_count=5,
                progress_every=100,
            )
            maybe_print_source_paper_progress(
                paper_index=100,
                total_papers=250,
                manifest_row=manifest_row,
                dataset_row_count=7,
                progress_every=100,
            )

        lines = [line for line in buffer.getvalue().splitlines() if line.strip()]
        self.assertEqual(len(lines), 2)
        self.assertIn("[1/250] Example source paper: Example Paper | kept 3 dataset row(s) so far", lines[0])
        self.assertIn("[100/250] Example source paper: Example Paper | kept 7 dataset row(s) so far", lines[1])

    def test_iter_rows_resumes_after_partial_failure(self):
        manifest_rows = [
            ManifestRow(
                source_paper_title="Paper One",
                source_paper_url="https://example.org/paper1",
                year=2024,
                pdf_url="https://example.org/paper1.pdf",
                pdf_path=None,
                venue="CVPR",
                domain="Computer Science",
            ),
            ManifestRow(
                source_paper_title="Paper Two",
                source_paper_url="https://example.org/paper2",
                year=2024,
                pdf_url="https://example.org/paper2.pdf",
                pdf_path=None,
                venue="CVPR",
                domain="Computer Science",
            ),
        ]
        candidate = CandidateSentence(
            sentence="A useful sentence [1].",
            excerpt="A useful sentence [CITATION].",
            reference_key="1",
            citation_marker="[1]",
            citation_count=1,
        )
        resolved = ResolvedPaper(
            paper_id="paper-id",
            title="Resolved Paper",
            year=2020,
            url="https://example.org/resolved.pdf",
            query="reference query",
            score=0.9,
        )
        discoverability = DiscoverabilityResult(query="discoverability query", rank=1)

        def fake_pdf_path(row, *_args, **_kwargs):
            return Path(f"{row.source_paper_title}.pdf")

        def failing_extract_pdf_text(path):
            if "Paper Two" in str(path):
                raise OSError("Remote end closed connection without response")
            return "dummy pdf text"

        with tempfile.TemporaryDirectory() as temp_dir:
            output_csv_path = Path(temp_dir) / "output.csv"
            audit_jsonl_path = Path(temp_dir) / "output.audit.jsonl"
            resume_state_path = Path(temp_dir) / "output.csv.state.json"

            with patch("training_data_collection.collect_numeric_cs_candidates.ensure_pdf_available", side_effect=fake_pdf_path), patch(
                "training_data_collection.collect_numeric_cs_candidates.extract_pdf_text",
                side_effect=failing_extract_pdf_text,
            ), patch(
                "training_data_collection.collect_numeric_cs_candidates.split_body_and_references",
                return_value=("body", "references"),
            ), patch(
                "training_data_collection.collect_numeric_cs_candidates.extract_reference_map",
                return_value={"1": "Reference One"},
            ), patch(
                "training_data_collection.collect_numeric_cs_candidates.find_candidate_sentences",
                return_value=[candidate],
            ), patch(
                "training_data_collection.collect_numeric_cs_candidates.resolve_reference",
                return_value=resolved,
            ), patch(
                "training_data_collection.collect_numeric_cs_candidates.check_discoverability",
                return_value=discoverability,
            ):
                with self.assertRaises(OSError):
                    iter_rows(
                        manifest_rows=manifest_rows,
                        api=object(),
                        cache_dir=Path(temp_dir),
                        split="train",
                        starting_id=100000,
                        request_pause_s=0.0,
                        session=object(),
                        citation_mode="single",
                        progress_every=None,
                        output_csv_path=output_csv_path,
                        audit_jsonl_path=audit_jsonl_path,
                        resume_state_path=resume_state_path,
                    )

            partial_state = load_resume_state(resume_state_path)
            self.assertEqual(partial_state["processed_source_paper_count"], 1)
            self.assertEqual(partial_state["dataset_row_count"], 1)
            self.assertEqual(partial_state["last_error"]["source_paper_title"], "Paper Two")

            with patch("training_data_collection.collect_numeric_cs_candidates.ensure_pdf_available", side_effect=fake_pdf_path), patch(
                "training_data_collection.collect_numeric_cs_candidates.extract_pdf_text",
                return_value="dummy pdf text",
            ), patch(
                "training_data_collection.collect_numeric_cs_candidates.split_body_and_references",
                return_value=("body", "references"),
            ), patch(
                "training_data_collection.collect_numeric_cs_candidates.extract_reference_map",
                return_value={"1": "Reference One"},
            ), patch(
                "training_data_collection.collect_numeric_cs_candidates.find_candidate_sentences",
                return_value=[candidate],
            ), patch(
                "training_data_collection.collect_numeric_cs_candidates.resolve_reference",
                return_value=resolved,
            ), patch(
                "training_data_collection.collect_numeric_cs_candidates.check_discoverability",
                return_value=discoverability,
            ):
                iter_rows(
                    manifest_rows=manifest_rows,
                    api=object(),
                    cache_dir=Path(temp_dir),
                    split="train",
                    starting_id=100000,
                    request_pause_s=0.0,
                    session=object(),
                    citation_mode="single",
                    progress_every=None,
                    output_csv_path=output_csv_path,
                    audit_jsonl_path=audit_jsonl_path,
                    resume_state_path=resume_state_path,
                )

            final_state = load_resume_state(resume_state_path)
            self.assertEqual(final_state["processed_source_paper_count"], 2)
            self.assertEqual(final_state["dataset_row_count"], 2)
            self.assertIsNone(final_state["last_error"])

            with output_csv_path.open("r", encoding="utf-8", newline="") as handle:
                dataset_rows = list(csv.DictReader(handle))
            self.assertEqual(len(dataset_rows), 2)
            self.assertEqual(
                [row["source_paper_title"] for row in dataset_rows],
                ["Paper One", "Paper Two"],
            )

            with audit_jsonl_path.open("r", encoding="utf-8") as handle:
                audit_rows = [line for line in handle.readlines() if line.strip()]
            self.assertEqual(len(audit_rows), 2)

    def test_iter_rows_caches_reference_resolution_for_repeated_reference(self):
        manifest_rows = [
            ManifestRow(
                source_paper_title="Paper One",
                source_paper_url="https://example.org/paper1",
                year=2024,
                pdf_url="https://example.org/paper1.pdf",
                pdf_path=None,
                venue="CVPR",
                domain="Computer Science",
            ),
        ]
        candidates = [
            CandidateSentence(
                sentence="Sentence one [1].",
                excerpt="Sentence one [CITATION].",
                reference_key="1",
                citation_marker="[1]",
                citation_count=1,
            ),
            CandidateSentence(
                sentence="Sentence two [1].",
                excerpt="Sentence two [CITATION].",
                reference_key="1",
                citation_marker="[1]",
                citation_count=1,
            ),
        ]
        resolved = ResolvedPaper(
            paper_id="paper-id",
            title="Resolved Paper",
            year=2020,
            url="https://example.org/resolved.pdf",
            query="reference query",
            score=0.9,
        )
        discoverability = DiscoverabilityResult(query="discoverability query", rank=1)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_csv_path = Path(temp_dir) / "output.csv"
            audit_jsonl_path = Path(temp_dir) / "output.audit.jsonl"
            resume_state_path = Path(temp_dir) / "output.csv.state.json"

            with patch(
                "training_data_collection.collect_numeric_cs_candidates.ensure_pdf_available",
                return_value=Path("Paper One.pdf"),
            ), patch(
                "training_data_collection.collect_numeric_cs_candidates.extract_pdf_text",
                return_value="dummy pdf text",
            ), patch(
                "training_data_collection.collect_numeric_cs_candidates.split_body_and_references",
                return_value=("body", "references"),
            ), patch(
                "training_data_collection.collect_numeric_cs_candidates.extract_reference_map",
                return_value={"1": "Reference One"},
            ), patch(
                "training_data_collection.collect_numeric_cs_candidates.find_candidate_sentences",
                return_value=candidates,
            ), patch(
                "training_data_collection.collect_numeric_cs_candidates.resolve_reference",
                return_value=resolved,
            ) as resolve_mock, patch(
                "training_data_collection.collect_numeric_cs_candidates.check_discoverability",
                return_value=discoverability,
            ):
                iter_rows(
                    manifest_rows=manifest_rows,
                    api=object(),
                    cache_dir=Path(temp_dir),
                    split="train",
                    starting_id=100000,
                    request_pause_s=0.0,
                    session=object(),
                    citation_mode="single",
                    progress_every=None,
                    output_csv_path=output_csv_path,
                    audit_jsonl_path=audit_jsonl_path,
                    resume_state_path=resume_state_path,
                )

            self.assertEqual(resolve_mock.call_count, 1)

    def test_iter_rows_skips_unreadable_pdf_and_continues(self):
        manifest_rows = [
            ManifestRow(
                source_paper_title="Bad PDF Paper",
                source_paper_url="https://example.org/bad-paper",
                year=2024,
                pdf_url="https://example.org/bad-paper.pdf",
                pdf_path=None,
                venue="ICML",
                domain="Computer Science",
            ),
            ManifestRow(
                source_paper_title="Good PDF Paper",
                source_paper_url="https://example.org/good-paper",
                year=2024,
                pdf_url="https://example.org/good-paper.pdf",
                pdf_path=None,
                venue="ICML",
                domain="Computer Science",
            ),
        ]
        candidate = CandidateSentence(
            sentence="A useful sentence (Smith et al., 2023).",
            excerpt="A useful sentence ([CITATION]).",
            reference_key="smith-2023",
            citation_marker="(Smith et al., 2023)",
            citation_count=1,
        )
        resolved = ResolvedPaper(
            paper_id="paper-id",
            title="Resolved Paper",
            year=2023,
            url="https://example.org/resolved.pdf",
            query="reference query",
            score=0.9,
        )
        discoverability = DiscoverabilityResult(query="discoverability query", rank=1)

        def fake_pdf_path(row, *_args, **_kwargs):
            return Path(f"{row.source_paper_title}.pdf")

        def maybe_bad_extract_pdf_text(path):
            if "Bad PDF Paper" in str(path):
                raise PdfReadError("malformed PDF stream")
            return "dummy pdf text"

        with tempfile.TemporaryDirectory() as temp_dir:
            output_csv_path = Path(temp_dir) / "output.csv"
            audit_jsonl_path = Path(temp_dir) / "output.audit.jsonl"
            resume_state_path = Path(temp_dir) / "output.csv.state.json"

            buffer = io.StringIO()
            with patch(
                "training_data_collection.collect_numeric_cs_candidates.ensure_pdf_available",
                side_effect=fake_pdf_path,
            ), patch(
                "training_data_collection.collect_numeric_cs_candidates.extract_pdf_text",
                side_effect=maybe_bad_extract_pdf_text,
            ), patch(
                "training_data_collection.collect_numeric_cs_candidates.split_body_and_references",
                return_value=("body", "references"),
            ), patch(
                "training_data_collection.collect_numeric_cs_candidates.extract_reference_map",
                return_value={"smith-2023": "Reference One"},
            ), patch(
                "training_data_collection.collect_numeric_cs_candidates.find_candidate_sentences",
                return_value=[candidate],
            ), patch(
                "training_data_collection.collect_numeric_cs_candidates.resolve_reference",
                return_value=resolved,
            ), patch(
                "training_data_collection.collect_numeric_cs_candidates.check_discoverability",
                return_value=discoverability,
            ), redirect_stdout(buffer):
                iter_rows(
                    manifest_rows=manifest_rows,
                    api=object(),
                    cache_dir=Path(temp_dir),
                    split="train",
                    starting_id=100000,
                    request_pause_s=0.0,
                    session=object(),
                    citation_mode="single",
                    citation_style="author_year",
                    progress_every=None,
                    output_csv_path=output_csv_path,
                    audit_jsonl_path=audit_jsonl_path,
                    resume_state_path=resume_state_path,
                )

            self.assertIn("Skipping unreadable PDF for Bad PDF Paper", buffer.getvalue())

            final_state = load_resume_state(resume_state_path)
            self.assertEqual(final_state["processed_source_paper_count"], 2)
            self.assertEqual(final_state["dataset_row_count"], 1)
            self.assertIsNone(final_state["last_error"])

            with output_csv_path.open("r", encoding="utf-8", newline="") as handle:
                dataset_rows = list(csv.DictReader(handle))
            self.assertEqual(len(dataset_rows), 1)
            self.assertEqual(dataset_rows[0]["source_paper_title"], "Good PDF Paper")

    def test_extract_author_year_reference_key_handles_acl_style_names(self):
        self.assertEqual(
            extract_author_year_reference_key("Zhengxin Zhang, Dan Zhao, and Xupeng Miao. 2024. Quantized Side Tuning."),
            "zhang-2024",
        )

    def test_author_year_entries_split_when_pdf_collapses_bibliography_line(self):
        text = (
            "Alice Smith and Bob Jones. 2020. First Paper. ICML. "
            "Carol Doe and Dan Roe. 2021. Second Paper. NeurIPS."
        )
        self.assertEqual(
            split_author_year_reference_entries(text),
            [
                "Alice Smith and Bob Jones. 2020. First Paper. ICML.",
                "Carol Doe and Dan Roe. 2021. Second Paper. NeurIPS.",
            ],
        )

    def test_resolve_reference_never_returns_source_paper(self):
        class FakeAPI:
            def relevance_search(self, query, **kwargs):
                return {
                    "data": [
                        {"paperId": "source", "title": "The Source Paper", "year": 2024},
                        {"paperId": "target", "title": "The Intended Method", "year": 2020},
                    ]
                }

        resolved = resolve_reference(
            "A. Author. 2020. The Intended Method.",
            2024,
            FakeAPI(),
            source_paper_title="The Source Paper",
        )
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.paper_id, "target")
        self.assertEqual(
            extract_author_year_reference_key("Smith, John and Doe, Jane. 2020a. Another Paper."),
            "smith-2020a",
        )

    def test_resolve_reference_compacts_queries_for_semantic_scholar(self):
        class FakeAPI:
            def __init__(self):
                self.queries = []
                self.fields_of_study = []

            def relevance_search(self, query, **kwargs):
                self.queries.append(query)
                self.fields_of_study.append(kwargs.get("fieldsOfStudy"))
                return {"data": []}

        reference_text = (
            "Alice Smith, Bob Jones, and Carol Doe. 2024. A Useful Retrieval Benchmark for Agentic Citation Search. "
            + "This malformed extraction keeps going with appendix text and tool descriptions. " * 80
        )
        api = FakeAPI()

        resolved = resolve_reference(reference_text, 2024, api, fields_of_study="Medicine")

        self.assertIsNone(resolved)
        self.assertGreaterEqual(len(api.queries), 2)
        self.assertTrue(all(len(query) <= 180 for query in api.queries))
        self.assertIn("A Useful Retrieval Benchmark for Agentic Citation Search", api.queries[0])
        self.assertTrue(all(field == "Medicine" for field in api.fields_of_study))

    def test_run_with_retries_does_not_retry_non_retryable_http_errors(self):
        response = requests.Response()
        response.status_code = 414
        error = requests.exceptions.HTTPError("414 Client Error", response=response)
        attempts = {"count": 0}

        def always_fail():
            attempts["count"] += 1
            raise error

        with self.assertRaises(requests.exceptions.HTTPError):
            from training_data_collection.collect_numeric_cs_candidates import run_with_retries

            run_with_retries(
                action_label="Testing non-retryable HTTP error",
                func=always_fail,
                max_retries=3,
                retry_backoff_s=0.0,
            )

        self.assertEqual(attempts["count"], 1)

    def test_dataset_csv_writers_ignore_mutated_global_excel_quoting(self):
        original_quoting = csv.excel.quoting
        original_doublequote = csv.excel.doublequote
        original_escapechar = csv.excel.escapechar
        csv.excel.quoting = csv.QUOTE_NONE
        csv.excel.doublequote = False
        csv.excel.escapechar = None
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_csv_path = Path(temp_dir) / "output.csv"
                overwrite_with_header(output_csv_path, DATASET_COLUMNS)
                append_dataset_rows(
                    output_csv_path,
                    [
                        {
                            "id": 1,
                            "excerpt": 'Sentence with comma, quote "inside", and newline\nsecond line.',
                            "target_paper_title": "A, \"Quoted\" Title",
                            "target_paper_url": "https://example.org/target",
                            "source_paper_title": "Source Paper",
                            "source_paper_url": "https://example.org/source",
                            "year": 2024,
                            "split": "train",
                        }
                    ],
                )

                with output_csv_path.open("r", encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["excerpt"], 'Sentence with comma, quote "inside", and newline\nsecond line.')
            self.assertEqual(rows[0]["target_paper_title"], 'A, "Quoted" Title')
        finally:
            csv.excel.quoting = original_quoting
            csv.excel.doublequote = original_doublequote
            csv.excel.escapechar = original_escapechar

    def test_manifest_csv_writer_ignores_mutated_global_excel_quoting(self):
        original_quoting = csv.excel.quoting
        original_doublequote = csv.excel.doublequote
        original_escapechar = csv.excel.escapechar
        csv.excel.quoting = csv.QUOTE_NONE
        csv.excel.doublequote = False
        csv.excel.escapechar = None
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                manifest_path = Path(temp_dir) / "manifest.csv"
                write_manifest_csv(
                    manifest_path,
                    [
                        ManifestRow(
                            source_paper_title='Paper with comma, "quote", and newline\nmarker',
                            source_paper_url="https://example.org/source",
                            year=2024,
                            pdf_url="https://example.org/paper.pdf?download=1&name=a,b",
                            pdf_path=None,
                            venue="ACL, Findings",
                            domain='Language "Processing"',
                        )
                    ],
                )

                with manifest_path.open("r", encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["source_paper_title"], 'Paper with comma, "quote", and newline\nmarker')
            self.assertEqual(rows[0]["venue"], "ACL, Findings")
            self.assertEqual(rows[0]["domain"], 'Language "Processing"')
        finally:
            csv.excel.quoting = original_quoting
            csv.excel.doublequote = original_doublequote
            csv.excel.escapechar = original_escapechar


if __name__ == "__main__":
    unittest.main()
