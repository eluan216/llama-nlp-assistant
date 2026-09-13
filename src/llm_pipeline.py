"""LLM generation pipeline using Hugging Face Transformers."""

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class LLMPipeline:
    """Lightweight wrapper for instruction-tuned causal LMs."""

    # Default: small, capable, and relatively fast even on CPU
    DEFAULT_MODEL = "microsoft/Phi-3-mini-4k-instruct"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        load_in_4bit: bool = False,
    ):
        """
        Args:
            model_name: Hugging Face model id.
            device: "cuda", "cpu", or None (auto).
            torch_dtype: Optional dtype override.
            load_in_4bit: Use bitsandbytes 4-bit quantization (requires GPU + bitsandbytes).
        """
        self.model_name = model_name

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto" if device == "cuda" else None,
        }

        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        elif device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16

        if load_in_4bit and device == "cuda":
            try:
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            except ImportError:
                pass  # fall back to normal loading

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, **model_kwargs
        )

        if device == "cpu":
            self.model = self.model.to("cpu")

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device == "cuda" else -1,
        )

    def _build_prompt(self, system: str, user: str) -> str:
        """Build a chat-style prompt compatible with Phi-3 / most instruct models."""
        # Phi-3 style
        if "phi-3" in self.model_name.lower():
            return (
                f"<|system|>\n{system}<|end|>\n"
                f"<|user|>\n{user}<|end|>\n"
                f"<|assistant|>\n"
            )

        # Generic fallback
        return f"System: {system}\n\nUser: {user}\n\nAssistant:"

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
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
        max_new_tokens: int = 400,
        temperature: float = 0.2,
    ) -> str:
        """Answer a question given retrieved context."""
        system = (
            "You are a helpful assistant that answers questions based only on the "
            "provided context. If the context does not contain enough information, "
            "say so clearly. Be concise and accurate."
        )
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
        max_new_tokens: int = 300,
        temperature: float = 0.3,
    ) -> str:
        """Produce a concise summary of the given text."""
        system = (
            "You are a helpful assistant that writes clear, concise summaries. "
            "Capture the main points without adding external information."
        )
        # Truncate very long inputs to stay within context limits
        max_chars = 6000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[Text truncated...]"

        user = f"Please summarize the following document:\n\n{text}"
        prompt = self._build_prompt(system, user)
        return self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
