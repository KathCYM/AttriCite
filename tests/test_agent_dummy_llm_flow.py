import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.retriever.agent import LLMSelfAskAgentPydantic, StaticContextProvider
from src.utils.data_model import AskForMoreContextAction, Output, SelectAction


class AgentDummyLlmFlowTests(unittest.TestCase):
    def test_additional_context_is_used_in_follow_up_llm_turn(self):
        agent = LLMSelfAskAgentPydantic.__new__(LLMSelfAskAgentPydantic)
        agent.console = MagicMock()
        agent.context_provider = StaticContextProvider(
            context="The previous sentence explains the benchmark setup in detail.",
            console=MagicMock(),
        )
        agent.human_intro = "You are now given an excerpt. Find me the paper cited in the excerpt."
        agent.source_papers_title = ["Example Source Paper"]
        agent.paper_buffer = []

        selected_paper = SimpleNamespace(
            paperId="paper-1",
            title="Selected Dummy Paper",
            model_dump=lambda: {
                "paperId": "paper-1",
                "title": "Selected Dummy Paper",
            },
        )

        calls = {"count": 0}

        def fake_ask_llm(message, last_action=False):
            calls["count"] += 1
            if calls["count"] == 1:
                self.assertIn("Example excerpt with [CITATION].", message.content)
                return Output(
                    reason="Need more context.",
                    action=AskForMoreContextAction(
                        name="ask_for_more_context",
                        query="benchmark setup",
                        paper_title="Example Source Paper",
                    ),
                )

            self.assertIn("Additional context from the user", message.content)
            self.assertIn(
                "The previous sentence explains the benchmark setup in detail.",
                message.content,
            )
            return Output(
                reason="Now I can select the paper.",
                action=SelectAction(
                    name="select",
                    paper_id="paper-1",
                ),
            )

        agent._ask_llm = fake_ask_llm
        agent._select = MagicMock(return_value=selected_paper)

        result = agent(
            excerpt="Example excerpt with [CITATION].",
            year="2025",
            src_paper_title="Example Source Paper",
            max_actions=3,
            skip=[],
        )

        self.assertEqual(result.title, "Selected Dummy Paper")
        agent._select.assert_called_once_with("paper-1")


if __name__ == "__main__":
    unittest.main()
