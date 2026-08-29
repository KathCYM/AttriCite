# AttriCite

AttriCite is an open 4B-parameter model trained for tool-using citation
recovery. Given a scientific passage with one citation replaced by
`[CITATION]`, the model searches Semantic Scholar, inspects candidate papers,
and selects the paper cited by the original author. The repository includes the
retrieval agent, collection and audit pipeline, metadata-only dataset release,
passage reconstruction utility, exact prompts, full-parameter GRPO recipe, and
evaluation code.

This anonymous review snapshot intentionally omits permanent paper, repository,
dataset, and checkpoint URLs. Start with [RELEASE.md](RELEASE.md) for the
paper-to-artifact map and release checklist.

## Setup

Run from the repository root.

```bash
pip install -r requirements.txt
```

Required:

- `S2_API_KEY`

Set the model provider key that matches the model you want to use:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `TOGETHER_API_KEY`
- `DEEPSEEK_API_KEY`
- `GOOGLE_API_KEY`
- `VLLM_API_KEY` (optional, defaults to `EMPTY`)

If you use `--local_model`, CiteGuard uses Ollama and does not need a cloud model API key.

If you use `--use_vllm`, CiteGuard sends requests to a vLLM OpenAI-compatible server. Set `VLLM_BASE_URL` or pass `--vllm_base_url`. If you do not set either one, the default is `http://localhost:8000/v1`.

## CLI

Single excerpt:

```bash
python -m src.main --model_name gpt-4o --excerpt "Your excerpt with [CITATION]"
```

Dataset:

```bash
python -m src.main --model_name gpt-4o --dataset DATASET.csv --result_path results/run.json
```

Useful options:

- `--result_path`: JSON output file. Existing results are loaded first, and already-processed IDs are skipped.
- `--source_paper_title`: source paper title for the excerpt.
- `--target_paper_title`: gold title for evaluation. Use `[TITLE_SEPARATOR]` for multiple acceptable titles.
- `--skip_citations`: comma-separated titles to exclude.
- `--additional_context`: extra surrounding text provided up front.
- `--no_interactive_context`: disable terminal prompts for more context.
- `--year`: source paper year. Default: `2025`.
- `--temperature`: model temperature. Default: `0.95`.
- `--local_model`: use Ollama instead of a hosted API model.
- `--use_vllm`: use a vLLM OpenAI-compatible server instead of a hosted API model.
- `--vllm_base_url`: override the vLLM base URL. Default: `http://localhost:8000/v1`.

Example:

```bash
python -m src.main \
  --model_name gpt-4o \
  --source_paper_title "Example Source Paper" \
  --excerpt "Transformer-based retrieval improves grounded generation [CITATION]." \
  --skip_citations "Attention Is All You Need,Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
```

vLLM example:

```bash
python -m src.main \
  --model_name Qwen/Qwen3-8B \
  --use_vllm \
  --vllm_base_url http://localhost:8000/v1 \
  --dataset DATASET.csv \
  --result_path results/qwen3-vllm.json
```

For Qwen3 models served through vLLM, CiteGuard automatically disables thinking mode by sending `chat_template_kwargs.enable_thinking=false`.

In single-excerpt CLI mode, the agent can ask for more context. Paste one or more lines and press Enter on an empty line to submit, or type `SKIP` to continue without extra context.

## Dataset Format

Expected CSV columns:

- `id`
- `excerpt`
- `year`
- `source_paper_title` (optional)
- `target_paper_title` (optional)

The CSV files used internally for training contain reconstructed third-party
passages and are not the public distribution format. The public CiteAlign
artifact contains metadata, source URLs, sentence locators, and SHA-256
fingerprints. Use `training_data_collection/reconstruct_passages.py` to recover
passages locally from their original hosts. See `RELEASE.md` for exact commands.

## AttriCite-4B training

The reproducible training implementation is under `training/verl/`. It contains
the complete prompt, seven-action schema, bounded observations, agent loop,
binary title-match reward, parquet conversion, full-parameter GRPO launcher,
and evaluation launchers. See `training/verl/README.md` before running the
four-GPU recipe.

## Web UI

Start the local server:

```bash
python app.py
```

Then open [http://127.0.0.1:5000/](http://127.0.0.1:5000/).

The web UI supports:

- single-excerpt runs
- dataset runs
- browsing results by ID after a run
- loading an existing `result_path` without re-running the agent via `Load Result Path`

If the `result_path` already exists, the web app loads it first, skips IDs that are already done, and writes the updated results back to that path after the run.

The web UI is not mid-run conversational. If you want to provide extra context, use the `Additional Context` field.

## Output

Each result JSON file contains:

- `metadata`: run configuration
- `results`: one entry per excerpt

Each result entry may include:

- `id`
- `excerpt`
- `selected`
- `status`
- `error`
- `papers`
- `history`
- `duration`
- `is_correct`
- `is_in_search`

## Key Files

- `src/main.py`: CLI entrypoint
- `src/run_main.py`: main execution flow
- `src/retriever/agent.py`: agent loop and tool actions
- `app.py`: Flask backend
- `templates/index.html`: web frontend

## License

- Code: MIT. See `LICENSE`.
- Dataset: CC-BY-4.0. See `LICENSE_DATASET`.

## Acknowledgements

AttriCite builds on the retrieval environment and agent interface introduced by
[CiteGuard](https://github.com/KathCYM/CiteGuard). Scholarly search and paper
metadata are provided by the [Semantic Scholar API](https://www.semanticscholar.org/product/api).
We thank both projects for making this work possible. Semantic Scholar is an
evolving external service; returned records and rankings can change over time.
