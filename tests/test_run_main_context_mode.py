import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import src.run_main as run_main_module
from src.retriever.agent import InteractiveConsoleContextProvider, StaticContextProvider
from src.run_main import run


class DummySelection:
    title = "Dummy Paper"

    def model_dump(self):
        return {"title": self.title}


class CapturingAgent:
    last_context_provider = None
    last_max_actions = None

    def __init__(self, *args, **kwargs):
        CapturingAgent.last_context_provider = kwargs["context_provider"]
        self.paper_buffer = []

    def reset(self, source_papers_title=None, skip=None):
        return None

    def __call__(self, excerpt, year, src_paper_title=None, max_actions=5, skip=None):
        CapturingAgent.last_max_actions = max_actions
        return DummySelection()

    def get_paper_buffer(self):
        return []

    def get_history(self, ignore_system_messages=False):
        return []


class RunMainContextModeTests(unittest.TestCase):
    def make_args(self, **overrides):
        base = {
            "dataset": None,
            "result_path": None,
            "model_name": "dummy-model",
            "local_model": False,
            "id": "manual",
            "source_paper_title": "Example Source Paper",
            "target_paper_title": None,
            "excerpt": "Example excerpt with [CITATION].",
            "year": 2025,
            "skip_citations": "",
            "temperature": 0.95,
            "max_actions": 15,
            "additional_context": None,
            "no_interactive_context": False,
        }
        base.update(overrides)
        return Namespace(**base)

    def run_with_args(self, args, is_tty):
        console = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            args.result_path = f"{tmpdir}/results.json"
            if args.dataset:
                dataset_path = tmpdir_path / "dataset.csv"
                dataset_path.write_text(
                    "id,excerpt,year,source_paper_title,target_paper_title\n"
                    "1,Dataset excerpt with [CITATION].,2025,Example Source Paper,Dummy Gold Paper\n",
                    encoding="utf-8",
                )
                args.dataset = str(dataset_path)
            fake_stdin = SimpleNamespace(isatty=lambda: is_tty)
            with patch("src.run_main.LLMSelfAskAgentPydantic", CapturingAgent):
                with patch.object(run_main_module.sys, "stdin", fake_stdin):
                    run(args, console)
        return CapturingAgent.last_context_provider

    def test_single_excerpt_mode_uses_interactive_provider_when_tty(self):
        provider = self.run_with_args(self.make_args(dataset=None), is_tty=True)
        self.assertIsInstance(provider, InteractiveConsoleContextProvider)

    def test_dataset_mode_uses_static_provider_even_when_tty(self):
        provider = self.run_with_args(
            self.make_args(dataset="dummy.csv", excerpt=None),
            is_tty=True,
        )
        self.assertIsInstance(provider, StaticContextProvider)

    def test_explicit_noninteractive_mode_uses_static_provider(self):
        provider = self.run_with_args(
            self.make_args(dataset=None, no_interactive_context=True),
            is_tty=True,
        )
        self.assertIsInstance(provider, StaticContextProvider)

    def test_web_style_interactive_flag_false_uses_static_provider(self):
        provider = self.run_with_args(
            self.make_args(dataset=None, interactive_context=False),
            is_tty=True,
        )
        self.assertIsInstance(provider, StaticContextProvider)

    def test_max_actions_is_forwarded_to_agent(self):
        self.run_with_args(self.make_args(max_actions=5), is_tty=False)
        self.assertEqual(CapturingAgent.last_max_actions, 5)


if __name__ == "__main__":
    unittest.main()
