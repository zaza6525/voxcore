"""LLM — Local LLM via OpenAI-compatible API (llama.cpp, vLLM, LM Studio)."""
from openai import OpenAI
from typing import Optional

class LLMEngine:
    """Client OpenAI-compatible pour LLM local."""
    
    def __init__(self, base_url: str = "http://127.0.0.1:1234/v1", model: str = "local", api_key: str = "sk-local"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
    
    def chat(self, messages: list, max_tokens: int = 4096, temperature: float = 0.7) -> str:
        """Chat complet avec le LLM local."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    
    def complete(self, prompt: str, system: str = "You are a helpful assistant.", max_tokens: int = 4096) -> str:
        """Complétion simple."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        return self.chat(messages, max_tokens=max_tokens)
