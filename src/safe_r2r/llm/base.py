from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class LLMConfig:
    backend: str
    model_name: str
    temperature: float = 0.2
    max_new_tokens: int = 128
    seed: Optional[int] = 42

class BaseLLM(ABC):
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

    @abstractmethod
    def generate(self, prompt: str) -> Dict[str, Any]:
        """
        Returns dict with at least:
          - text: str
          - meta: dict (latency, tokens if available)
        """
        raise NotImplementedError