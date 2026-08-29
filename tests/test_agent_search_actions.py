import unittest
from unittest.mock import MagicMock

from langchain_core.messages import HumanMessage

from src.retriever.agent import LLMSelfAskAgentPydantic
from src.utils.data_model import OpenAccessPdf, PaperAuthor, PaperSearchResult


def make_paper(
    paper_id: str = "paper-1",
    title: str = "Example Paper",
    abstract: str = "Example abstract.",
    citation_count: int = 42,
):
    return PaperSearchResult(
        paperId=paper_id,
        title=title,
        authors=[PaperAuthor(authorId="author-1", name="Author One")],
        abstract=abstract,
        venue="TestConf",
        year=2025,
        citationCount=citation_count,
        openAccessPdf=OpenAccessPdf(url="https://example.com/paper.pdf", status="GREEN"),
    )


class AgentSearchActionTests(unittest.TestCase):
    def make_agent(self):
        agent = LLMSelfAskAgentPydantic.__new__(LLMSelfAskAgentPydantic)
        agent.console = MagicMock()
        agent.search_provider = MagicMock()
        agent.source_papers_title = []
        agent.paper_buffer = []
        return agent

    def test_search_relevance_formats_results_and_updates_buffer(self):
        agent = self.make_agent()
        paper = make_paper()
        agent.search_provider.return_value = [paper]

        message = agent._search_relevance("benchmark setup", "2025")

        self.assertIsInstance(message, HumanMessage)
        self.assertIn("Paper ID: paper-1", message.content)
        self.assertIn("Title: Example Paper", message.content)
        self.assertIn("Abstract: Example abstract.", message.content)
        self.assertEqual(agent.paper_buffer, [[paper]])
        agent.search_provider.assert_called_once_with("benchmark setup", "2025", skip=[])

    def test_search_citation_count_uses_citation_count_search(self):
        agent = self.make_agent()
        paper = make_paper(paper_id="paper-2", title="Citation Ranked Paper")
        agent.search_provider.citation_count_search.return_value = [paper]

        message = agent._search_citation_count("imagenet", "2014")

        self.assertIsInstance(message, HumanMessage)
        self.assertIn("Citation Ranked Paper", message.content)
        agent.search_provider.citation_count_search.assert_called_once_with(
            "imagenet", "2014", skip=[]
        )

    def test_search_text_snippet_returns_provider_response(self):
        agent = self.make_agent()
        agent.search_provider.snippet_search.return_value = "Title: Example\nSnippet: More context"

        message = agent._search_snippet(
            "imagenet benchmark",
            "2014",
            "Example Source Paper",
            skip=["Paper A"],
        )

        self.assertIsInstance(message, HumanMessage)
        self.assertEqual(message.content, "Title: Example\nSnippet: More context")
        agent.search_provider.snippet_search.assert_called_once_with(
            "imagenet benchmark",
            "2014",
            "Example Source Paper",
            skip=["Paper A"],
        )

    def test_process_search_filters_source_paper_title(self):
        agent = self.make_agent()
        agent.source_papers_title = ["Example Source Paper"]
        source_paper = make_paper(paper_id="paper-source", title="Example Source Paper")
        cited_paper = make_paper(paper_id="paper-cited", title="Different Cited Paper")

        message = agent._LLMSelfAskAgentPydantic__process_search([source_paper, cited_paper])

        self.assertIn("Different Cited Paper", message.content)
        self.assertNotIn("Example Source Paper", message.content)
        self.assertEqual(agent.paper_buffer, [[cited_paper]])

    def test_process_search_returns_no_papers_message_when_empty(self):
        agent = self.make_agent()

        message = agent._LLMSelfAskAgentPydantic__process_search([])

        self.assertEqual(
            message.content,
            "No papers were found for the given search query. Please use a different query.",
        )
        self.assertEqual(agent.paper_buffer, [[]])


if __name__ == "__main__":
    unittest.main()
