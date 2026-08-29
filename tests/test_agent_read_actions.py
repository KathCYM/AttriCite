import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage

from src.retriever.agent import LLMSelfAskAgentPydantic, PaperNotFoundError
from src.utils.data_model import OpenAccessPdf, PaperAuthor, PaperSearchResult


def make_paper(
    paper_id: str = "paper-1",
    title: str = "Example Paper",
    pdf_url: str | None = "https://example.com/paper.pdf",
):
    return PaperSearchResult(
        paperId=paper_id,
        title=title,
        authors=[PaperAuthor(authorId="author-1", name="Author One")],
        abstract="Example abstract.",
        venue="TestConf",
        year=2025,
        citationCount=42,
        openAccessPdf=OpenAccessPdf(url=pdf_url, status="GREEN" if pdf_url else None),
    )


class AgentReadAndSelectTests(unittest.TestCase):
    def make_agent(self):
        agent = LLMSelfAskAgentPydantic.__new__(LLMSelfAskAgentPydantic)
        agent.console = MagicMock()
        agent.paper_buffer = []
        return agent

    def test_select_returns_matching_paper(self):
        agent = self.make_agent()
        paper = make_paper()
        agent.paper_buffer = [[paper]]

        result = agent._select("paper-1")

        self.assertEqual(result.title, "Example Paper")

    def test_select_raises_when_paper_missing(self):
        agent = self.make_agent()
        agent.paper_buffer = [[make_paper()]]

        with self.assertRaises(PaperNotFoundError):
            agent._select("missing-paper")

    def test_find_in_text_returns_matching_sentences(self):
        agent = self.make_agent()
        agent._read = MagicMock(
            return_value=HumanMessage(
                content=(
                    "The first sentence is unrelated. "
                    "The benchmark setup appears here. "
                    "Final sentence."
                )
            )
        )

        message = agent._read_and_find_in_text("paper-1", "benchmark setup")

        self.assertIn("The benchmark setup appears here.", message.content)

    def test_find_in_text_returns_no_match_message(self):
        agent = self.make_agent()
        agent._read = MagicMock(return_value=HumanMessage(content="Nothing useful is here."))

        message = agent._read_and_find_in_text("paper-1", "benchmark setup")

        self.assertEqual(message.content, "No sentence found containing 'benchmark setup'.")

    def test_find_in_text_returns_read_error_unchanged(self):
        agent = self.make_agent()
        agent._read = MagicMock(
            return_value=HumanMessage(content="This paper does not have an open access PDF.")
        )

        message = agent._read_and_find_in_text("paper-1", "benchmark setup")

        self.assertEqual(message.content, "This paper does not have an open access PDF.")

    def test_read_returns_message_when_no_open_access_pdf(self):
        agent = self.make_agent()
        agent.paper_buffer = [[make_paper(pdf_url=None)]]

        message = agent._read("paper-1")

        self.assertEqual(message.content, "This paper does not have an open access PDF.")

    def test_read_extracts_pdf_text_when_available(self):
        agent = self.make_agent()
        agent.paper_buffer = [[make_paper()]]
        fake_reader = SimpleNamespace(
            pages=[
                SimpleNamespace(extract_text=lambda: "Page one. "),
                SimpleNamespace(extract_text=lambda: "Page two."),
            ]
        )

        with patch("src.retriever.agent.requests.get", return_value=SimpleNamespace(content=b"%PDF-1.4")):
            with patch("src.retriever.agent.PdfReader", return_value=fake_reader):
                message = agent._read("paper-1")

        self.assertEqual(message.content, "Page one. Page two.")

    def test_read_uses_handle_aaai_for_aaai_urls(self):
        agent = self.make_agent()
        aaai_paper = make_paper(pdf_url="https://ojs.aaai.org/index.php/AAAI/article/download/12345/12000")
        agent.paper_buffer = [[aaai_paper]]
        agent.handle_aaai = MagicMock(return_value=b"%PDF-1.4")
        fake_reader = SimpleNamespace(pages=[SimpleNamespace(extract_text=lambda: "AAAI paper text.")])

        with patch("src.retriever.agent.requests.get") as mock_get:
            with patch("src.retriever.agent.PdfReader", return_value=fake_reader):
                message = agent._read("paper-1")

        self.assertEqual(message.content, "AAAI paper text.")
        agent.handle_aaai.assert_called_once()
        mock_get.assert_not_called()

    def test_read_returns_error_message_on_exception(self):
        agent = self.make_agent()
        agent.paper_buffer = [[make_paper()]]

        with patch("src.retriever.agent.requests.get", side_effect=RuntimeError("network down")):
            message = agent._read("paper-1")

        self.assertIn("There was an error reading the PDF.", message.content)
        self.assertIn("network down", message.content)


if __name__ == "__main__":
    unittest.main()
