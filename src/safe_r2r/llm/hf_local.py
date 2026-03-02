import time
from typing import Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from .base import BaseLLM


class HfLocalLLM(BaseLLM):
    """
    Local HuggingFace LLM backend optimized for Colab T4 using 4-bit quantization.
    Uses chat template for instruction models (Qwen, Llama, etc).
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

        # Some tokenizers/models don't have pad token configured
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
        )
        self.model.eval()

        # Optional: disable gradients globally
        torch.set_grad_enabled(False)

    def _build_chat_prompt(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Builds a chat-formatted prompt using tokenizer's chat template.
        """
        system_prompt = system_prompt or (
            "You are a question answering system. "
            "Follow the user's rules strictly. "
            "Output only the final answer."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # add_generation_prompt=True adds the assistant turn header
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(self, prompt: str) -> Dict[str, Any]:
        """
        Returns:
          {"text": <assistant completion>, "meta": {"latency_ms": ...}}
        """
        t0 = time.perf_counter()

        chat_prompt = self._build_chat_prompt(prompt)

        inputs = self.tokenizer(
            chat_prompt,
            return_tensors="pt",
            truncation=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Deterministic vs sampling
        temperature = float(self.cfg.temperature)
        do_sample = temperature > 0.0

        out = self.model.generate(
            **inputs,
            max_new_tokens=int(self.cfg.max_new_tokens),
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Decode ONLY the new tokens (completion), not the full prompt
        # out shape: [1, prompt_len + gen_len]
        gen_ids = out[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        t1 = time.perf_counter()
        return {"text": text, "meta": {"latency_ms": (t1 - t0) * 1000.0}}