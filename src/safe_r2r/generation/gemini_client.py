import os
from dataclasses import dataclass
from typing import Optional
from google import genai
from dotenv import load_dotenv

load_dotenv()

@dataclass
class GeminiConfig:
    model: str = "gemini-2.5-flash"
    temperature: float = 0.2
    max_output_tokens: int = 256

class GeminiClient:
    def __init__(self, cfg: GeminiConfig, api_key: Optional[str] = None):
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY.")
        self.client = genai.Client(api_key=api_key)
        self.cfg = cfg

    def generate_text(self, prompt: str) -> str:
        resp = self.client.models.generate_content(
            model=self.cfg.model,
            contents=prompt,
            config={
                "temperature": self.cfg.temperature,
                "max_output_tokens": self.cfg.max_output_tokens,
            },
        )
        return (resp.text or "").strip()