"""LLM generation pipeline using Hugging Face Transformers.

Optimized for Streamlit Cloud free tier (CPU + limited RAM).
"""

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class LLMPipeline:
    """Lightweight wrapper for instruction-tuned causal LMs."""

    # Small model that fits Streamlit Cloud free tier (~360M parameters)
    DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        """
        Args:
            model_name: Hugging Face model id.
            device: "cuda", "cpu", or None (auto).
            torch_dtype: Optional dtype override.
        """
        self.model_name = model_name

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # Force low memory usage on CPU (Streamlit Cloud)
        model_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        elif device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"
        else:
            # CPU: use float32 and keep it simple
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

    def _build_prompt(self, system: str, user: str) -> str:
        """Build a chat-style prompt compatible with most instruct models."""
        name = self.model_name.lower()

        # Phi-3 style
        if "phi-3" in name:
            return (
                f"<|system|>\n{system}<|end|>\n"
                f"<|user|>\n{user}<|end|>\n"
                f"<|assistant|>\n"
            )

        # SmolLM / ChatML style
        if "smollm" in name or "chatml" in name:
            return (
                f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{user}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        # Generic fallback
        return f"System: {system}\n\nUser: {user}\n\nAssistant:"

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.3,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """Generate a completion for a raw prompt."""
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
        """Answer a question given retrieved context."""
        system = (
            "You are a helpful assistant that answers questions based only on the "
            "provided context. If the context does not contain enough information, "
            "say so clearly. Be concise and accurate."
        )
        # Keep context short for small models / free tier
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
        """Produce a concise summary of the given text."""
        system = (
            "You are a helpful assistant that writes clear, concise summaries. "
            "Capture the main points without adding external information."
        )
        # Aggressive truncation for free-tier memory limits
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
