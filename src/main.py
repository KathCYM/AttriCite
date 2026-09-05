import sys
from pathlib import Path
from rich.console import Console
import argparse

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retriever.llm_base import DEFAULT_TEMPERATURE
from src.run_main import run

def main():
    parser = argparse.ArgumentParser(description="Run citation retrieval agent.")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to the dataset CSV file. If omitted, runs in single input mode.")
    parser.add_argument("--result_path", type=str, default="results.json",
                        help="Path to result JSON file (default: results.json).")
    parser.add_argument("--model_name", type=str, help="model name", required=True)
    parser.add_argument("--local_model", action="store_true", help="Use local model if set.")
    parser.add_argument("--use_vllm", action="store_true", help="Use a vLLM OpenAI-compatible server instead of a hosted provider.")
    parser.add_argument("--use_together", action="store_true", help="Use Together AI for the specified model name.")
    parser.add_argument("--vllm_base_url", type=str, default=None, help="Base URL for the vLLM OpenAI-compatible server (default: http://localhost:8000/v1 or VLLM_BASE_URL).")
    parser.add_argument(
        "--fields_of_study",
        type=str,
        default="Computer Science",
        help='Semantic Scholar field-of-study filter (for example, "Medicine"); use an empty string to disable it.',
    )

    # single data point inputs
    parser.add_argument("--id", type=str, default="manual",
                        help="Unique ID for single input mode (default: manual).")
    parser.add_argument("--source_paper_title", type=str, default=None,
                        help="Optional source paper title.")
    parser.add_argument("--target_paper_title", type=str, default=None, help="Target paper title if running evaluation, separated by [TITLE_SEPARATOR] if needed.")
    parser.add_argument("--excerpt", type=str, help="Citation excerpt text.")
    parser.add_argument("--year", type=int, default=2025, help="Publication year of source paper which contains the excerpt (default: 2025).")
    parser.add_argument("--skip_citations", type=str, default="",
                        help="Comma-separated list of found citations for the excerpt (used in single input mode).")
    parser.add_argument("--additional_context", type=str, default=None,
                        help="Optional extra context to use if the agent asks for more context.")
    parser.add_argument("--no_interactive_context", action="store_true",
                        help="Disable terminal prompts when the agent asks for more context.")

    # parameters
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help="Temperature to use for the LLM (default: 0.95).")
    parser.add_argument("--max_actions", type=int, default=5,
                        help="Maximum number of agent actions per example (default: 5).")
    
    args = parser.parse_args()
    # Excerpts, paper metadata, and model/error text may legitimately contain
    # bracketed tokens such as ``[/INST]``.  Treat all CLI log strings as
    # literal text so Rich does not parse untrusted content as markup and abort
    # a dataset evaluation with MarkupError.
    console = Console(markup=False)
    results = run(args, console)
    return results

if __name__ == "__main__":
    main()
