from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_hf_model_and_tokenizer(
    model_name: str,
    device_map: str = "auto",
    torch_dtype: str = "auto",
):
    """
    Load a Hugging Face causal LM and tokenizer.

    Args:
        model_name: e.g. "Qwen/Qwen2.5-7B-Instruct" or "mistralai/Mistral-7B-Instruct-v0.3"
        device_map: usually "auto"
        torch_dtype: "auto", "float16", "bfloat16", or "float32"
    """
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    chosen_dtype = dtype_map.get(torch_dtype, "auto")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=chosen_dtype,
    )
    model.eval()

    return model, tokenizer