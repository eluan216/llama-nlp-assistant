"""LLM generation pipeline using Hugging Face Transformers.

Optimized for Streamlit Cloud free tier (CPU + limited RAM).
"""

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMPipeline:
    """Lightweight wrapper for instruction-tuned causal LMs."""

    DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"

    _TRUST_REMOTE_CODE_MODELS = {
        "microsoft/Phi-3-mini-4k-instruct",
    }

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        self.model_name = model_name

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        trust_remote = self._requires_trust_remote_code(model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote,
        )

        # Ensure pad token exists (needed for batch-safe generate on many instruct models)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "trust_remote_code": trust_remote,
            "low_cpu_mem_usage": True,
        }

        # Prefer `dtype` (new API); fall back to torch_dtype for older transformers
        resolved_dtype = torch_dtype
        if resolved_dtype is None:
            resolved_dtype = torch.float16 if device == "cuda" else torch.float32

        try:
            model_kwargs["dtype"] = resolved_dtype
        except Exception:
            model_kwargs["torch_dtype"] = resolved_dtype

        if device == "cuda":
            model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, **model_kwargs
        )

        if device == "cpu":
            self.model = self.model.to("cpu")

        self.model.eval()

    @classmethod
    def _requires_trust_remote_code(cls, model_name: str) -> bool:
        name = model_name.strip()
        if name in cls._TRUST_REMOTE_CODE_MODELS:
            return True
        if "phi-3" in name.lower():
            return True
        return False

    def _build_prompt(self, system: str, user: str) -> str:
        name = self.model_name.lower()

        if "phi-3" in name:
            return (
                f"<|system|>\n{system}<|end|>\n"
                f"<|user|>\n{user}<|end|>\n"
                f"<|assistant|>\n"
            )

        if "smollm" in name or "chatml" in name:
            return (
                f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{user}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        # TinyLlama / generic chat-style fallback
        if "tinyllama" in name:
            return (
                f"<|system|>\n{system}</s>\n"
                f"<|user|>\n{user}</s>\n"
                f"<|assistant|>\n"
            )

        return f"System: {system}\n\nUser: {user}\n\nAssistant:"

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.3,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """Generate completion using model.generate (avoids pipeline config clashes)."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }

        if do_sample and temperature is not None and temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = float(top_p)
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        # Decode only newly generated tokens
        prompt_len = inputs["input_ids"].shape[-1]
        new_tokens = output_ids[0][prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        if not text:
            # Fallback: decode full sequence minus prompt text (some tokenizers differ)
            full = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            if full.startswith(prompt):
                text = full[len(prompt):].strip()
            else:
                text = full.strip()

        return text or "(No text was generated. Try a shorter question or lower max tokens.)"

    def answer(
        self,
        question: str,
        context: str,
        max_new_tokens: int = 128,
        temperature: float = 0.2,
    ) -> str:
        system = (
            "You are a helpful assistant that answers questions based only on the "
            "provided context. If the context does not contain enough information, "
            "say so clearly. Be concise and accurate."
        )
        if len(context) > 2000:
            context = context[:2000] + "\n\n[Context truncated...]"

        user = f"Context:\n{context}\n\nQuestion: {question}"
        prompt = self._build_prompt(system, user)
        return self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature is not None and temperature > 0,
        )

    def summarize(
        self,
        text: str,
        max_new_tokens: int = 128,
        temperature: float = 0.3,
    ) -> str:
        system = (
            "You are a helpful assistant that writes clear, concise summaries. "
            "Capture the main points without adding external information."
        )
        max_chars = 2500
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[Text truncated...]"

        user = f"Please summarize the following document:\n\n{text}"
        prompt = self._build_prompt(system, user)
        return self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature is not None and temperature > 0,
        )
