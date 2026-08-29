"""Shared observation bounds for evaluation and RL environments."""

MAX_ABSTRACT_CHARS = 1000
MAX_OBSERVATION_CHARS = 16000


def bound_observation(text: str, limit: int = MAX_OBSERVATION_CHARS) -> str:
    """Deterministically retain the beginning of an oversized tool observation."""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n[Observation truncated to fit the model context.]"
