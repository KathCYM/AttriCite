import tiktoken


def get_encoding_for_model(model_name: str):
    """Resolve a tokenizer while tolerating newer dated model aliases.

    Older tiktoken releases may not yet contain mappings for snapshots such as
    ``gpt-5.4-mini-2026-03-17`` even though they use the GPT-5 tokenizer family.
    """
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        normalized = model_name.lower()
        if normalized.startswith(("gpt-5", "gpt-4o", "o1", "o3", "o4")):
            return tiktoken.get_encoding("o200k_base")
        return tiktoken.get_encoding("cl100k_base")


def num_tokens_from_string(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = get_encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
