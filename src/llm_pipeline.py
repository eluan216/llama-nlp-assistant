"""LLM generation pipeline using Hugging Face Transformers.

Optimized for Streamlit Cloud free tier (CPU + limited RAM).
"""

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class LLMPipeline:
    """Lightweight wrapper for instruction-tuned causal LMs."""

    DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"

    # Models that ship custom modeling code and currently require trust_remote_code.
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

        model_kwargs = {
            "trust_remote_code": trust_remote,
            "low_cpu_mem_usage": True,
        }

        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        elif device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, **model_kwargs
        )

        if device == "cpu":
            self.model = self.model.to("cpu")

        self.model.eval()

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1 if device == "cpu" else 0,
        )

    @classmethod
    def _requires_trust_remote_code(cls, model_name: str) -> bool:
        """Return True only for models known to need custom remote code."""
        name = model_name.strip()
        if name in cls._TRUST_REMOTE_CODE_MODELS:
            return True
        lower = name.lower()
        if "phi-3" in lower:
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

        return f"System: {system}\n\nUser: {user}\n\nAssistant:"

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.3,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        outputs = self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            do_sample=do_sample,
            return_full_text=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return outputs[0]["generated_text"].strip()

    def answer(
        self,
        question: str,
        context: str,
        max_new_tokens: int = 200,
        temperature: float = 0.2,
    ) -> str:
        system = (
            "You are a helpful assistant that answers questions based only on the "
            "provided context. If the context does not contain enough information, "
            "say so clearly. Be concise and accurate."
        )
        if len(context) > 2500:
            context = context[:2500] + "\n\n[Context truncated...]"

        user = f"Context:\n{context}\n\nQuestion: {question}"
        prompt = self._build_prompt(system, user)
        return self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

    def summarize(
        self,
        text: str,
        max_new_tokens: int = 180,
        temperature: float = 0.3,
    ) -> str:
        system = (
            "You are a helpful assistant that writes clear, concise summaries. "
            "Capture the main points without adding external information."
        )
        max_chars = 3000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[Text truncated...]"

        user = f"Please summarize the following document:\n\n{text}"
        prompt = self._build_prompt(system, user)
        return self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
