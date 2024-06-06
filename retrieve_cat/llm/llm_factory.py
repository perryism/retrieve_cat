import logging
from langchain_openai import OpenAI

logger = logging.getLogger(__name__)

def create(name, **kwargs):
    if name == "local":
        params = {"api_key": "not-needed", "temperature": 0.1, "max_tokens": 2048}
        params.update(kwargs)
        if "base_url" not in params:
            raise ValueError("base_url is required for local llm")
        return OpenAI(**params)
    else:
        raise ValueError(f"Unknown name {name}")
