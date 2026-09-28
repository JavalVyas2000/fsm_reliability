from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_hf_model_and_tokenizer(
    model_name: str,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    revision: Optional[str] = None,
    attn_implementation: Optional[str] = None,
    local_files_only: bool = False,
    quantization: Optional[str] = None,
):
    """
    Load a Hugging Face causal LM and tokenizer.

    Args:
        model_name: e.g. "Qwen/Qwen2.5-7B-Instruct" or "mistralai/Mistral-7B-Instruct-v0.3"
        device_map: usually "auto"
        torch_dtype: "auto", "float16", "bfloat16", or "float32"
        revision: optional commit hash / branch to pin
        attn_implementation: optional, e.g. "sdpa" or "eager"
        local_files_only: do not contact the Hub
        quantization: None or "4bit" (bitsandbytes NF4, bf16 compute, double quantisation)
    """
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if torch_dtype not in dtype_map:
        raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")
    chosen_dtype = dtype_map[torch_dtype]

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, revision=revision, local_files_only=local_files_only
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = dict(
        device_map=device_map,
        dtype=chosen_dtype,
        revision=revision,
        local_files_only=local_files_only,
    )
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    if quantization == "4bit":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif quantization is not None:
        raise ValueError(f"Unsupported quantization: {quantization}")

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()

    return model, tokenizer


def resolved_revision(model) -> Optional[str]:
    """Commit hash of the loaded checkpoint, when the Hub cache records it."""
    return getattr(model.config, "_commit_hash", None)
