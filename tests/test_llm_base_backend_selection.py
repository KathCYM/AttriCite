import unittest
from unittest.mock import patch

from src.retriever.llm_base import get_model_by_name


class LlmBaseBackendSelectionTests(unittest.TestCase):
    @patch.dict("os.environ", {}, clear=True)
    @patch("src.retriever.llm_base.ChatOpenAI")
    def test_use_vllm_routes_qwen3_to_openai_compatible_client_with_thinking_disabled(self, mock_chat_openai):
        get_model_by_name(
            "Qwen/Qwen3-8B",
            temperature=0.2,
            use_vllm=True,
            vllm_base_url="http://localhost:8000/v1",
        )

        mock_chat_openai.assert_called_once_with(
            model="Qwen/Qwen3-8B",
            temperature=0.2,
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

    @patch.dict("os.environ", {}, clear=True)
    @patch("src.retriever.llm_base.ChatOllama")
    def test_local_qwen3_keeps_ollama_but_disables_reasoning(self, mock_chat_ollama):
        get_model_by_name(
            "qwen3:8b",
            temperature=0.3,
            local_model=True,
        )

        mock_chat_ollama.assert_called_once_with(
            model="qwen3:8b",
            validate_model_on_init=True,
            temperature=0.3,
            reasoning=False,
        )


if __name__ == "__main__":
    unittest.main()
