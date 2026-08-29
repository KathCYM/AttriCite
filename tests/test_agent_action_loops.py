import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain_core.messages import HumanMessage

from src.retriever.agent import LLMSelfAskAgentPydantic
from src.utils.data_model import (
    CitationCountSearchAction,
    FindInTextAction,
    Output,
    ReadAction,
    RelevanceSearchAction,
    SearchTextSnippetAction,
    SelectAction,
)


class AgentActionLoopTests(unittest.TestCase):
    def make_agent(self):
        agent = LLMSelfAskAgentPydantic.__new__(LLMSelfAskAgentPydantic)
        agent.console = MagicMock()
        agent.human_intro = "You are now given an excerpt. Find me the paper cited in the excerpt."
        agent.source_papers_title = ["Example Source Paper"]
        agent.paper_buffer = []
        return agent

    def test_search_relevance_then_select_loop(self):
        agent = self.make_agent()
        selected_paper = SimpleNamespace(title="Selected Paper")
        responses = iter(
            [
                Output(
                    reason="Search first.",
                    action=RelevanceSearchAction(name="search_relevance", query="benchmark setup"),
                ),
                Output(
                    reason="Select after search.",
                    action=SelectAction(name="select", paper_id="paper-1"),
                ),
            ]
        )
        agent._ask_llm = MagicMock(side_effect=lambda message, last_action=False: next(responses))
        agent._search_relevance = MagicMock(return_value=HumanMessage(content="search results"))
        agent._select = MagicMock(return_value=selected_paper)

        result = agent(
            excerpt="Example excerpt with [CITATION].",
            year="2025",
            src_paper_title="Example Source Paper",
            max_actions=3,
            skip=[],
        )

        self.assertEqual(result.title, "Selected Paper")
        agent._search_relevance.assert_called_once_with("benchmark setup", "2025", skip=[])
        agent._select.assert_called_once_with("paper-1")

    def test_search_citation_count_read_then_select_loop(self):
        agent = self.make_agent()
        selected_paper = SimpleNamespace(title="Selected Paper")
        responses = iter(
            [
                Output(
                    reason="Search by citation count.",
                    action=CitationCountSearchAction(
                        name="search_citation_count",
                        query="imagenet challenge",
                    ),
                ),
                Output(
                    reason="Read the paper.",
                    action=ReadAction(name="read", paper_id="paper-2"),
                ),
                Output(
                    reason="Now select.",
                    action=SelectAction(name="select", paper_id="paper-2"),
                ),
            ]
        )
        agent._ask_llm = MagicMock(side_effect=lambda message, last_action=False: next(responses))
        agent._search_citation_count = MagicMock(return_value=HumanMessage(content="ranked search results"))
        agent._read = MagicMock(return_value=HumanMessage(content="paper text"))
        agent._select = MagicMock(return_value=selected_paper)

        result = agent(
            excerpt="Example excerpt with [CITATION].",
            year="2025",
            src_paper_title="Example Source Paper",
            max_actions=4,
            skip=[],
        )

        self.assertEqual(result.title, "Selected Paper")
        agent._search_citation_count.assert_called_once_with("imagenet challenge", "2025", skip=[])
        agent._read.assert_called_once_with("paper-2")
        agent._select.assert_called_once_with("paper-2")

    def test_search_snippet_find_in_text_then_select_loop(self):
        agent = self.make_agent()
        selected_paper = SimpleNamespace(title="Selected Paper")
        responses = iter(
            [
                Output(
                    reason="Search snippets first.",
                    action=SearchTextSnippetAction(
                        name="search_text_snippet",
                        query="ILSVRC 2014",
                    ),
                ),
                Output(
                    reason="Find the exact phrase.",
                    action=FindInTextAction(
                        name="find_in_text",
                        paper_id="paper-3",
                        query="ImageNet",
                    ),
                ),
                Output(
                    reason="Now select the paper.",
                    action=SelectAction(name="select", paper_id="paper-3"),
                ),
            ]
        )
        agent._ask_llm = MagicMock(side_effect=lambda message, last_action=False: next(responses))
        agent._search_snippet = MagicMock(return_value=HumanMessage(content="snippet results"))
        agent._read_and_find_in_text = MagicMock(return_value=HumanMessage(content="ImageNet sentence"))
        agent._select = MagicMock(return_value=selected_paper)

        result = agent(
            excerpt="Example excerpt with [CITATION].",
            year="2025",
            src_paper_title="Example Source Paper",
            max_actions=4,
            skip=[],
        )

        self.assertEqual(result.title, "Selected Paper")
        agent._search_snippet.assert_called_once_with(
            "ILSVRC 2014",
            "2025",
            "Example Source Paper",
            skip=[],
        )
        agent._read_and_find_in_text.assert_called_once_with("paper-3", "ImageNet")
        agent._select.assert_called_once_with("paper-3")


if __name__ == "__main__":
    unittest.main()
