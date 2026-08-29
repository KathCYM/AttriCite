import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage

from src.retriever.agent import (
    InteractiveConsoleContextProvider,
    LLMSelfAskAgentPydantic,
    StaticContextProvider,
)


class InteractiveConsoleContextProviderTests(unittest.TestCase):
    def test_collects_multiline_context_until_blank_line(self):
        provider = InteractiveConsoleContextProvider(console=MagicMock())

        with patch("builtins.input", side_effect=["first line", "second line", ""]):
            context = provider.request_context(
                query="transformer retrieval",
                paper_title="Example Paper",
            )

        self.assertEqual(context, "first line\nsecond line")

    def test_skip_returns_empty_string(self):
        provider = InteractiveConsoleContextProvider(console=MagicMock())

        with patch("builtins.input", side_effect=["SKIP"]):
            context = provider.request_context(
                query="transformer retrieval",
                paper_title="Example Paper",
            )

        self.assertEqual(context, "")

    def test_keyboard_interrupt_returns_empty_string(self):
        provider = InteractiveConsoleContextProvider(console=MagicMock())

        with patch("builtins.input", side_effect=KeyboardInterrupt()):
            context = provider.request_context(
                query="transformer retrieval",
                paper_title="Example Paper",
            )

        self.assertEqual(context, "")


class AskForMoreContextAgentTests(unittest.TestCase):
    def make_agent(self, context_provider):
        agent = LLMSelfAskAgentPydantic.__new__(LLMSelfAskAgentPydantic)
        agent.console = MagicMock()
        agent.context_provider = context_provider
        return agent

    def test_agent_wraps_user_supplied_context_in_human_message(self):
        agent = self.make_agent(
            StaticContextProvider(
                context="The previous sentence discusses the benchmark setup.",
                console=MagicMock(),
            )
        )

        message = agent._ask_for_more_context(
            query="benchmark setup",
            paper_title="Example Paper",
        )

        self.assertIsInstance(message, HumanMessage)
        self.assertIn("Additional context from the user", message.content)
        self.assertIn("benchmark setup", message.content)
        self.assertIn("Example Paper", message.content)
        self.assertIn(
            "The previous sentence discusses the benchmark setup.",
            message.content,
        )

    def test_agent_returns_fallback_message_when_no_context_is_available(self):
        agent = self.make_agent(StaticContextProvider(context=None, console=MagicMock()))

        message = agent._ask_for_more_context(
            query="benchmark setup",
            paper_title="Example Paper",
        )

        self.assertIsInstance(message, HumanMessage)
        self.assertIn("No additional context was provided by the user.", message.content)
        self.assertIn("benchmark setup", message.content)
        self.assertIn("Example Paper", message.content)


if __name__ == "__main__":
    unittest.main()
