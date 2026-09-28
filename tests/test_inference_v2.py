"""
Model-backed checks for answer-aligned internals. Uses a locally cached small
instruct model in float32, where the extraction must match a full eager pass.
Skipped when no GPU or no cached model is available.
"""
import pytest

torch = pytest.importorskip("torch")

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache  # noqa: E402

from src.data.fsm_dataset_v2 import generate_partitions, load_graph  # noqa: E402
from src.evaluation.fsm_labels_v2 import find_first_json_object  # noqa: E402
from src.models.inference_v2 import InternalsConfig, decision_pass, run_candidate  # noqa: E402
from src.prompts.fsm_prompts_v2 import build_messages, region_char_spans  # noqa: E402

TEST_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


@pytest.fixture(scope="module")
def model_tok():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        tok = AutoTokenizer.from_pretrained(TEST_MODEL, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            TEST_MODEL, dtype=torch.float32, device_map="cuda", local_files_only=True, attn_implementation="sdpa"
        ).eval()
    except OSError:
        pytest.skip(f"{TEST_MODEL} not cached")
    return model, tok


@pytest.fixture(scope="module")
def instances():
    parts, _ = generate_partitions(99, {"train": 3}, num_nodes_list=(5, 10, 15))
    return parts["train"]


CFG = InternalsConfig(max_new_tokens=64)


def _run(model, tok, x):
    msgs = build_messages(x.graph_text, x.start, x.goal)
    return run_candidate(
        model, tok, msgs, lambda r: region_char_spans(r, x.graph_text, x.start, x.goal), CFG
    )


def test_answer_span_alignment(model_tok, instances):
    model, tok = model_tok
    for x in instances:
        rec = _run(model, tok, x)
        if rec["stop_reason"] != "json_object":
            continue
        ans = rec["answer_text"]
        s, e = find_first_json_object(ans)
        # the object ends inside the last answer token, not earlier
        shorter = tok.decode(rec["generated_ids"][: rec["answer_token_count"] - 1], skip_special_tokens=True)
        assert find_first_json_object(shorter) is None
        assert e <= len(ans)
        assert rec["greedy_consistent"]


def test_region_masses_sum_to_one(model_tok, instances):
    model, tok = model_tok
    rec = _run(model, tok, instances[0])
    f = rec["features"]
    tag = "L050"
    total = sum(v for k, v in f.items() if k.startswith(f"att_{tag}_") and k.endswith("_mean"))
    assert abs(total - 1.0) < 1e-3


def test_decision_pass_matches_full_eager_forward(model_tok, instances):
    model, tok = model_tok
    x = instances[1]
    rendered = tok.apply_chat_template(build_messages(x.graph_text, x.start, x.goal), tokenize=False, add_generation_prompt=True)
    ids = tok(rendered, add_special_tokens=False, return_tensors="pt")["input_ids"].cuda()
    P = ids.shape[1]
    answer = tok('{"path": [1, 2, 3]}', add_special_tokens=False, return_tensors="pt")["input_ids"][0].cuda()
    k = answer.shape[0]

    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        model(ids, past_key_values=cache, use_cache=True)
    n_layers = model.config.num_hidden_layers
    blocks = [1, n_layers // 2 + 1, n_layers]
    fo, attn = decision_pass(model, cache, ids, answer, CFG, attn_blocks=blocks)

    model.set_attn_implementation("eager")
    full_ids = torch.cat([ids, answer[None, :-1]], dim=1)
    with torch.no_grad():
        full = model(full_ids, output_attentions=True, output_hidden_states=True)
    model.set_attn_implementation("sdpa")

    rows = slice(P - 1, P - 1 + k)
    for b in blocks:
        l = b - 1
        diff = (full.attentions[l][0, :, rows, :] - attn[b][0]).abs().max().item()
        assert diff <= 1e-3, f"layer {l} attention diff {diff}"
    hdiff = (full.hidden_states[-1][0, rows] - fo.hidden_states[-1][0]).abs().max().item()
    assert hdiff <= 1e-2
    ldiff = (full.logits[0, rows] - fo.logits[0]).abs().max().item()
    assert ldiff <= 1e-2


def test_greedy_generation_is_deterministic(model_tok, instances):
    model, tok = model_tok
    a = _run(model, tok, instances[2])
    b = _run(model, tok, instances[2])
    assert a["generated_ids"] == b["generated_ids"]
    for key, v in a["features"].items():
        assert abs(v - b["features"][key]) < 1e-5 or (v != v and b["features"][key] != b["features"][key])


def test_chunked_prefill_matches_plain_generation(model_tok, instances):
    from dataclasses import replace

    model, tok = model_tok
    x = instances[0]
    msgs = build_messages(x.graph_text, x.start, x.goal)
    spans = lambda r: region_char_spans(r, x.graph_text, x.start, x.goal)
    plain = run_candidate(model, tok, msgs, spans, CFG)
    chunked = run_candidate(model, tok, msgs, spans, replace(CFG, prefill_chunk=64))
    assert chunked["chunked_prefill"] and not plain["chunked_prefill"]
    assert chunked["generated_ids"] == plain["generated_ids"]
    for k, v in plain["features"].items():
        assert abs(v - chunked["features"][k]) < 1e-3, k


def test_prefix_cache_matches_plain_generation(model_tok, instances):
    from dataclasses import replace

    from src.models.inference_v2 import PrefixCache

    model, tok = model_tok
    x = instances[1]
    msgs = build_messages(x.graph_text, x.start, x.goal)
    spans = lambda r: region_char_spans(r, x.graph_text, x.start, x.goal)
    rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **CFG.chat_template_kwargs)
    ids = tok(rendered, add_special_tokens=False)["input_ids"]
    pc = PrefixCache(model, ids[:60], chunk=32)
    plain = run_candidate(model, tok, msgs, spans, CFG)
    cached = run_candidate(model, tok, msgs, spans, replace(CFG, prefix_cache=pc))
    assert cached["prefix_cache_hit"] and not plain["prefix_cache_hit"]
    assert cached["generated_ids"] == plain["generated_ids"]
    for k, v in plain["features"].items():
        assert abs(v - cached["features"][k]) < 1e-3, k
    # the prefix cache itself must not be modified by use
    assert pc.cache.get_seq_length() == 60
