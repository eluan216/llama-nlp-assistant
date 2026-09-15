"""LLM generation for RAG answers.

Default path: Hugging Face Inference API (no local model weights in RAM).
This is required for Streamlit Cloud free tier, which OOMs when loading
PyTorch + embeddings + a causal LM together.

Optional local path: set environment variable LOCAL_LLM=1 to load weights
in-process (needs much more RAM; fine on a laptop/Docker with memory).
"""

from __future__ import annotations

import os
from typing import Optional


class LLMPipeline:
    """Answer / summarize via HF Inference API (default) or local transformers."""

    DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        torch_dtype=None,
    ):
        self.model_name = model_name
        self.device = device or "cpu"
        self._local = None
        self._client = None

        use_local = os.environ.get("LOCAL_LLM", "").strip() in {"1", "true", "True", "yes"}

        if use_local:
            self._init_local(torch_dtype)
        else:
            self._init_inference_client()

    def _init_inference_client(self) -> None:
        from huggingface_hub import InferenceClient

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        # token may be None — public models still work with stricter rate limits
        self._client = InferenceClient(model=self.model_name, token=token)

    def _init_local(self, torch_dtype=None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        trust_remote = "phi-3" in self.model_name.lower()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=trust_remote
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch_dtype or (torch.float16 if torch.cuda.is_available() else torch.float32)
        kwargs = {
            "trust_remote_code": trust_remote,
            "low_cpu_mem_usage": True,
        }
        try:
            kwargs["dtype"] = dtype
        except Exception:
            kwargs["torch_dtype"] = dtype

        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
            self.device = "cuda"

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        if self.device == "cpu":
            self.model = self.model.to("cpu")
        self.model.eval()
        self._local = True

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
        max_new_tokens: int = 96,
        temperature: float = 0.3,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        if self._local:
            return self._generate_local(
                prompt, max_new_tokens, temperature, top_p, do_sample
            )
        return self._generate_remote(
            prompt, max_new_tokens, temperature, top_p, do_sample
        )

    def _generate_remote(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
    ) -> str:
        assert self._client is not None
        try:
            # huggingface_hub InferenceClient text_generation
            text = self._client.text_generation(
                prompt,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature) if do_sample and temperature else 0.01,
                top_p=float(top_p),
                do_sample=bool(do_sample and temperature and temperature > 0),
                return_full_text=False,
            )
            if isinstance(text, str):
                return text.strip() or "(Empty model response.)"
            return str(text).strip()
        except Exception as e:
            msg = str(e)
            hint = ""
            if "401" in msg or "403" in msg or "token" in msg.lower():
                hint = (
                    " Set a free HF_TOKEN in Streamlit secrets for higher limits "
                    "(https://huggingface.co/settings/tokens)."
                )
            if "429" in msg or "rate" in msg.lower():
                hint = " Rate limited — wait a minute or add HF_TOKEN in secrets."
            if "loading" in msg.lower() or "503" in msg:
                hint = " Model is cold-starting on HF — wait 30s and try again."
            raise RuntimeError(f"Inference API error: {msg}.{hint}") from e

    def _generate_local(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
    ) -> str:
        import torch

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        gen_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        if do_sample and temperature and temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = float(top_p)
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[-1]
        new_tokens = output_ids[0][prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return text or "(No text was generated.)"

    def answer(
        self,
        question: str,
        context: str,
        max_new_tokens: int = 96,
        temperature: float = 0.2,
    ) -> str:
        system = (
            "You are a helpful assistant that answers questions based only on the "
            "provided context. If the context does not contain enough information, "
            "say so clearly. Be concise and accurate."
        )
        if len(context) > 1800:
            context = context[:1800] + "\n\n[Context truncated...]"

        user = f"Context:\n{context}\n\nQuestion: {question}"
        prompt = self._build_prompt(system, user)
        return self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=bool(temperature and temperature > 0),
        )

    def summarize(
        self,
        text: str,
        max_new_tokens: int = 96,
        temperature: float = 0.3,
    ) -> str:
        system = (
            "You are a helpful assistant that writes clear, concise summaries. "
            "Capture the main points without adding external information."
        )
        if len(text) > 2000:
            text = text[:2000] + "\n\n[Text truncated...]"

        user = f"Please summarize the following document:\n\n{text}"
        prompt = self._build_prompt(system, user)
        return self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=bool(temperature and temperature > 0),
        )
