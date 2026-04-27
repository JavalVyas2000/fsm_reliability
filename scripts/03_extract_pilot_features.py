from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.evaluation.correctness import score_prediction
from src.features.attention_features import extract_attention_features
from src.features.attention_region_features import (
    build_prompt_regions,
    extract_attention_region_features,
)
from src.features.token_confidence_features import extract_token_confidence_features
from src.models.load_model import load_hf_model_and_tokenizer
from src.models.output_parser import parse_path_from_text
from src.models.run_inference import generate_text
from src.prompts.fsm_prompts import build_path_prompt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract internal features for FSM traversal."
    )
    parser.add_argument("--data_path", type=str, default="data/raw_multi_node_large/test.csv")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--output_path", type=str, default="data/processed/pilot_features.csv")
    return parser.parse_args()


def detect_output_format(text: str) -> str:
    if text is None or not str(text).strip():
        return "empty"

    s = str(text).strip()
    if s.startswith("{"):
        return "json_like"
    if s.startswith("["):
        return "list_like"
    return "other"


def parse_json_list_field(value):
    if isinstance(value, list):
        return [int(x) for x in value]

    if isinstance(value, str):
        parsed = json.loads(value)
        return [int(x) for x in parsed]

    raise ValueError(f"Unsupported list field type: {type(value)}")


def normalize_graph_keys_and_values(graph):
    return {int(k): [int(v) for v in vals] for k, vals in graph.items()}


def choose_max_new_tokens(row, user_value):
    if user_value is not None:
        return user_value

    num_nodes = int(row["num_nodes"])
    shortest_length = None

    if "shortest_path_length" in row and pd.notna(row["shortest_path_length"]):
        shortest_length = int(row["shortest_path_length"])

    if shortest_length is not None:
        return max(100, shortest_length * 3)

    return max(100, num_nodes * 2)


def main():
    args = parse_args()

    df = pd.read_csv(args.data_path).head(args.num_samples).copy()

    model, tokenizer = load_hf_model_and_tokenizer(args.model_name)

    rows = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Extracting features",
    ):
        graph = json.loads(row["graph_json"])
        graph = normalize_graph_keys_and_values(graph)

        start = int(row["start"])
        goal = int(row["goal"])
        shortest_path = parse_json_list_field(row["shortest_path"])

        prompt = build_path_prompt(
            graph_text=row["graph_text"],
            start=start,
            goal=goal,
        )

        effective_max_new_tokens = choose_max_new_tokens(row, args.max_new_tokens)

        result = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=effective_max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_attentions=True,
        )

        parsed_result = parse_path_from_text(result["generated_text"])
        parsed_path = parsed_result.path

        scored = score_prediction(
            graph=graph,
            predicted_path=parsed_path,
            start=start,
            goal=goal,
            shortest_path=shortest_path,
        )

        token_features = extract_token_confidence_features(
            generation_scores=result["scores"],
            generated_ids=result["generated_ids"],
        )

        pooled_attention_features = extract_attention_features(
            attentions=result["attentions"],
        )

        regions = build_prompt_regions(
            tokenizer=tokenizer,
            prompt_text=prompt,
            full_input_ids=result["full_input_ids"],
            prompt_len=result["prompt_len"],
            start=start,
            goal=goal,
        )

        region_attention_features = extract_attention_region_features(
            attentions=result["attentions"],
            regions=regions,
        )

        out_row = {
            "instance_id": row["instance_id"],
            "split": row["split"] if "split" in row else None,
            "start": start,
            "goal": goal,
            "num_nodes": int(row["num_nodes"]),
            "edge_prob": float(row["edge_prob"]) if "edge_prob" in row and pd.notna(row["edge_prob"]) else None,
            "num_edges": int(row["num_edges"]),
            "ground_truth_shortest_path": json.dumps(shortest_path),
            "generated_text": result["generated_text"],
            "parsed_path": json.dumps(scored["parsed_path"]) if scored["parsed_path"] is not None else None,
            "output_format": detect_output_format(result["generated_text"]),
            "parse_success": parsed_result.parse_success,
            "parse_mode": parsed_result.parse_mode,
            "strict_json_success": int(parsed_result.parse_mode == "json"),
            "valid_path": scored["valid_path"],
            "optimal_path": scored["optimal_path"],
            "path_length": scored["path_length"],
            "shortest_length": scored["shortest_length"],
            "max_new_tokens_used": effective_max_new_tokens,
        }

        if "length_gap" in scored:
            out_row["length_gap"] = scored["length_gap"]

        out_row.update(token_features)
        out_row.update(pooled_attention_features)
        out_row.update(region_attention_features)
        rows.append(out_row)

    out_df = pd.DataFrame(rows)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    print("\nSaved feature table to:", output_path.resolve())
    print("Rows:", len(out_df))
    print("Columns:", len(out_df.columns))

    if "parse_mode" in out_df.columns:
        print("\nParse mode counts:")
        print(out_df["parse_mode"].value_counts(dropna=False))

    if "num_nodes" in out_df.columns:
        print("\nNode-count breakdown:")
        print(out_df["num_nodes"].value_counts().sort_index())


if __name__ == "__main__":
    main()