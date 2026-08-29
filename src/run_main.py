import os
import json
import sys
import pandas as pd
from rich.console import Console
from src.retriever.agent import (
    InteractiveConsoleContextProvider,
    LLMSelfAskAgentPydantic,
    LLMNoSearch,
    OutputNoRead,
    OutputSearchOnly,
    Output,
    StaticContextProvider,
)
from src.utils.str_matcher import find_match_psr, find_multi_match_psr
from src.utils.tokens import num_tokens_from_string
from datetime import datetime
from time import time
from rich.progress import track
from src.retriever.llm_base import DEFAULT_TEMPERATURE
import argparse

def run(args, console):
    TITLE_SEPERATOR = "[TITLE_SEPARATOR]"
    MODEL_NAME = args.model_name
    RESULT_FILE_NAME = args.result_path
    TEMPERATURE = args.temperature

    metadata = {
        "model": MODEL_NAME,
        "temperature": TEMPERATURE,
        "search_provider": "SemanticScholarSearchProvider",
        "search_limit": 10,
        "threshold": 0.8,
        "execution_date": datetime.now().isoformat(),
        "only_open_access": False,
        "use_web_search": False,
        "max_actions": int(getattr(args, "max_actions", 5)),
    }

    prompt_name = "few_shot_tool"

    ## Load the dataset
    if args.dataset is not None:
        if not os.path.exists(args.dataset):
            raise FileNotFoundError(f"Dataset not found: {args.dataset}")
        c = pd.read_csv(args.dataset, sep=",")
        c.set_index("id", inplace=True)
        to_iterate = list(c.iterrows())
    else:
        if not args.excerpt:
            raise ValueError(
                "For single input mode, you must provide --excerpt."
            )
        single = {
            "id": args.id,
            "source_paper_title": args.source_paper_title or None,
            "target_paper_title": args.target_paper_title or None,
            "excerpt": args.excerpt,
            "year": args.year,
            "skip": args.skip_citations.split(",") if args.skip_citations else [],
        }
        to_iterate = [(args.id, single)]

    ## Select executor
    executor = "LLMSelfAskAgentPydantic"
    actions = "search_relevance,search_citation_count,read,select,find_in_text,ask_for_more_context,search_text_snippet"
    pdo = Output
    interactive_context = getattr(
        args,
        "interactive_context",
        args.dataset is None and sys.stdin.isatty(),
    )
    if getattr(args, "no_interactive_context", False):
        interactive_context = False

    additional_context = getattr(args, "additional_context", None)
    if interactive_context:
        context_provider = InteractiveConsoleContextProvider(console=console)
    else:
        context_provider = StaticContextProvider(
            context=additional_context,
            console=console,
        )

    console.log("using Ouput pdo")
    console.log(metadata)
    agent = LLMSelfAskAgentPydantic(
        only_open_access=metadata["only_open_access"],
        search_limit=metadata["search_limit"],
        model_name=metadata["model"],
        temperature=metadata["temperature"],
        use_web_search=metadata["use_web_search"],
        prompt_name=prompt_name,
        pydantic_object=pdo,
        console = console,
        local_model=args.local_model,
        use_vllm=getattr(args, "use_vllm", False),
        use_together=getattr(args, "use_together", False),
        vllm_base_url=getattr(args, "vllm_base_url", None),
        context_provider=context_provider,
    )

    results = []
    paper_buffer = None
    if os.path.exists(RESULT_FILE_NAME):
        with open(RESULT_FILE_NAME, "r") as f:
            existing_data = json.load(f)
            old_results = existing_data.get("results", [])
            results += old_results

    processed_ids = [item["id"] for item in results]
    console.log(f"Already processed: {processed_ids}")

    for cid, citation in track(
        to_iterate, description="Processing citations", total=len(to_iterate), console=console
    ):
        if cid in processed_ids:
            continue
        skip = citation.get("skip", [])
        if citation["source_paper_title"]:
            agent.reset([citation["source_paper_title"]], skip=skip)
        else:
            agent.reset([], skip=skip)
        
        start_time = time()
        citation_text = citation["excerpt"]
        src_paper_title = citation["source_paper_title"]
        target_titles = None
        if citation["target_paper_title"] is not None:
            target_titles = citation["target_paper_title"].split(TITLE_SEPERATOR)
        year = int(citation["year"] if not pd.isna(citation["year"]) else 2025)
        result_data = {
            "id": cid,
            "cited_paper_titles": target_titles,
            "excerpt": citation_text,
            "skip": skip,
        }
        try:
            selection = agent(citation_text, f"{year}", src_paper_title=src_paper_title, max_actions=metadata["max_actions"], skip=skip)
            result_data["selected"] = selection.model_dump()
            result_data["is_correct"] = None
            result_data["is_in_search"] = None
            paper_buffer = agent.paper_buffer 
            if target_titles:
                is_in_search = find_multi_match_psr(
                    sum(agent.paper_buffer, []), target_titles, threshold=metadata["threshold"]
                )
                result_data["is_in_search"] = is_in_search[1] if is_in_search else None
            
                correct = (
                    find_match_psr(target_titles, selection.title, metadata["threshold"])
                    is not None
                )
                result_data["is_correct"] = correct
            result_data["status"] = "success"
            result_data["error"] = None
            if target_titles:
                console.log(
                    f"{cid:3d}: {target_titles[0]}\n"
                    + f"     [{'green' if correct else 'red'}]{selection.title}\n"
                    + f"     [black]search: {'✅' if is_in_search is not None else '❌'}, selection: {'✅' if correct else '❌'}"
                )
        except Exception as e:
            result_data["selected"] = None
            result_data["is_in_search"] = None
            result_data["is_correct"] = None
            result_data["status"] = "error"
            result_data["error"] = str(e)
            if target_titles:
                console.log(f"[red]{cid:3d}: {target_titles[0]}\n     [bold]{e}")

        result_data["duration"] = time() - start_time
        result_data["papers"] = agent.get_paper_buffer()
        result_data["history"] = agent.get_history(ignore_system_messages=False)
        if metadata["model"].startswith("gpt"):
            result_data["tokens"] = {
                "input": sum(
                    [
                        num_tokens_from_string(m["content"], metadata["model"])
                        for m in result_data["history"]
                        if m["role"] != "assistant"
                    ]
                ),
                "output": sum(
                    [
                        num_tokens_from_string(m["content"], metadata["model"])
                        for m in result_data["history"]
                        if m["role"] == "assistant"
                    ]
                ),
            }

        results.append(result_data)


        with open(RESULT_FILE_NAME, "w") as f:
            json.dump({"metadata": metadata, "results": results}, f, indent=4)
    return results, paper_buffer
