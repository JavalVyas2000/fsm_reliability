from __future__ import annotations

from typing import Any, Dict

import torch


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 32,
    do_sample: bool = False,
    output_scores: bool = True,
    return_attentions: bool = False,
) -> Dict[str, Any]:
    """
    Run generation and optionally collect attentions on the full prompt+generation sequence.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    generation_output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        return_dict_in_generate=True,
        output_scores=output_scores,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    prompt_len = inputs["input_ids"].shape[1]
    full_sequence = generation_output.sequences[0]
    generated_ids = full_sequence[prompt_len:]

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if generated_text:
        generated_text = generated_text.splitlines()[0].strip()

    attentions = None

    if return_attentions:
        full_inputs = {
            "input_ids": generation_output.sequences,
            "attention_mask": torch.ones_like(generation_output.sequences, device=model.device),
        }

        forward_out = model(
            **full_inputs,
            output_attentions=True,
            use_cache=False,
        )
        attentions = forward_out.attentions

    return {
        "prompt": prompt,
        "prompt_len": prompt_len,
        "generated_text": generated_text,
        "generated_ids": generated_ids.detach().cpu().tolist(),
        "full_input_ids": full_sequence.detach().cpu().tolist(),
        "full_sequences": generation_output.sequences.detach().cpu(),
        "scores": generation_output.scores if hasattr(generation_output, "scores") else None,
        "attentions": attentions,
    }