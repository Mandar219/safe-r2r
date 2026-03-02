from .base import LLMConfig
from .mock import MockLLM
from .hf_local import HfLocalLLM

def make_llm(cfg: LLMConfig):
    if cfg.backend == "mock":
        return MockLLM(cfg)
    if cfg.backend == "hf_local":
        return HfLocalLLM(cfg)
    raise ValueError(f"Unknown LLM backend: {cfg.backend}")