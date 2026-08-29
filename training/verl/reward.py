"""Dataset-level reward hook for current veRL reward-loop workers."""

from __future__ import annotations

from typing import Any


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **_kwargs: Any,
) -> dict[str, float]:
    """Return the binary selection reward produced by the CiteGuard tool.

    Current veRL still requires a dataset-level reward function even when a
    stateful tool emits per-call rewards. ToolAgentLoop exposes those values as
    ``extra_info['tool_rewards']``. Only a correct select call can emit 1.0.
    """
    if str(data_source) != "citeguard":
        raise ValueError(f"Unexpected data source: {data_source}")
    info = extra_info or {}
    state = info.get("citeguard_state") or {}
    selected = float(bool(state.get("selected", False)))
    if "correct_selection" in state:
        score = float(bool(state["correct_selection"]))
    else:
        raw_rewards = info.get("tool_rewards")
        rewards = [float(value) for value in (raw_rewards if raw_rewards is not None else []) if value is not None]
        score = max(rewards, default=0.0)
    return {
        "score": score,
        "acc": score,
        "selected": selected,
        "correct_selection": score,
    }
