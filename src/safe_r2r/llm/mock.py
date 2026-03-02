import time
from typing import Dict, Any
from .base import BaseLLM

class MockLLM(BaseLLM):
    def generate(self, prompt: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        # deterministic fake answer: last question line if present
        text = "mock_answer"
        t1 = time.perf_counter()
        return {
            "text": text,
            "meta": {"latency_ms": (t1 - t0) * 1000.0}
        }