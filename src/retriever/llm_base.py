import os
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_anthropic import ChatAnthropic
from langchain_together import ChatTogether
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

DEFAULT_TEMPERATURE = 0.95

def _build_vllm_model(model_name: str, temperature: float = DEFAULT_TEMPERATURE, base_url: str | None = None):
    resolved_base_url = (base_url or os.environ.get("VLLM_BASE_URL") or "http://localhost:8000/v1").rstrip("/")
    extra_body = None
    # Qwen3 uses thinking mode by default unless explicitly disabled.
    if model_name.lower().startswith("qwen3") or "qwen/qwen3" in model_name.lower():
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        base_url=resolved_base_url,
        api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
        extra_body=extra_body,
    )

def get_model_by_name(
    model_name: str,
    temperature: float = DEFAULT_TEMPERATURE,
    local_model: bool = False,
    use_vllm: bool = False,
    use_together: bool = False,
    vllm_base_url: str | None = None,
):
    if use_vllm:
        return _build_vllm_model(
            model_name,
            temperature=temperature,
            base_url=vllm_base_url,
        )
    if use_together:
        return ChatTogether(
            model=model_name,
            temperature=temperature,
        )
    if local_model:
        ollama_kwargs = {
            "model": model_name,
            "validate_model_on_init": True,
            "temperature": temperature,
        }
        # Qwen3 defaults to thinking mode unless we explicitly disable it.
        if model_name.lower().startswith("qwen3"):
            ollama_kwargs["reasoning"] = False
        return ChatOllama(**ollama_kwargs)
    if model_name.startswith("gpt-") or model_name.startswith("o1-"):
        return ChatOpenAI(model=model_name, temperature=temperature)
    if model_name.startswith("claude-"):
        return ChatAnthropic(
            temperature=temperature,
            model_name=model_name,
            timeout=60*10,
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        )
    if "llama" in model_name.lower() or "phi" in model_name.lower() or "mistral" in model_name.lower():
        return ChatTogether(
            # together_api_key="YOUR_API_KEY",
            temperature=temperature,
            model=model_name,
        )
    if "deepseek" in model_name.lower():
        return ChatDeepSeek(
            model=model_name,
            temperature=temperature,
        )
    if "gemini" in model_name.lower():
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
        )
    if "qwen" in model_name.lower() or "kimi" in model_name.lower():
        return ChatTogether(
            # together_api_key="YOUR_API_KEY",
            temperature=temperature,
            model=model_name,
        )
    raise ValueError(f"Model {model_name} not found")
