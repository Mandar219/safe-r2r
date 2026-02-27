import time
from typing import Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from .base import BaseLLM

class HfLocalLLM(BaseLLM):
    def __init__(self, cfg):
        super().__init__(cfg)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
        )
        self.model.eval()

    def generate(self, prompt: str) -> Dict[str, Any]:
        t0 = time.perf_counter()

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=self.cfg.temperature > 0,
                temperature=self.cfg.temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        text = self.tokenizer.decode(out[0], skip_special_tokens=True)

        # Heuristic: return only generated continuation
        prompt_text = self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        if text.startswith(prompt_text):
            text = text[len(prompt_text):].strip()

        t1 = time.perf_counter()
        return {
            "text": text,
            "meta": {"latency_ms": (t1 - t0) * 1000.0}
        }