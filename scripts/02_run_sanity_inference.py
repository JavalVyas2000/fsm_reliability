from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.evaluation.correctness import score_prediction
from src.models.load_model import load_hf_model_and_tokenizer
from src.models.output_parser import parse_path_from_text
from src.models.run_inference import generate_text
from src.prompts.fsm_prompts import build_path_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Run sanity-check inference on FSM dataset.")
    parser.add_argument("--data_path", type=str, default="data/raw/test.csv")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=24)
    parser.add_argument("--output_path", type=str, default="data/runs/sanity_inference.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.data_path)
    df = df.head(args.num_samples).copy()

    model, tokenizer = load_hf_model_and_tokenizer(args.model_name)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_valid = 0
    num_optimal = 0
    num_parsed = 0
    num_json = 0
    num_list = 0

    with output_path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            graph = json.loads(row["graph_json"])
            graph = {int(k): v for k, v in graph.items()}

            start = int(row["start"])
            goal = int(row["goal"])

            shortest_path = row["shortest_path"]
            if isinstance(shortest_path, str):
                shortest_path = json.loads(shortest_path)

            prompt = build_path_prompt(
                graph_text=row["graph_text"],
                start=start,
                goal=goal,
            )

            result = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
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

            record = {
                "instance_id": row["instance_id"],
                "start": start,
                "goal": goal,
                "shortest_path": shortest_path,
                "prompt": prompt,
                "generated_text": result["generated_text"],
                "parsed_path": scored["parsed_path"],
                "parse_success": parsed_result.parse_success,
                "parse_mode": parsed_result.parse_mode,
                "strict_json_success": int(parsed_result.parse_mode == "json"),
                "valid_path": scored["valid_path"],
                "optimal_path": scored["optimal_path"],
                "path_length": scored["path_length"],
                "shortest_length": scored["shortest_length"],
            }

            f.write(json.dumps(record) + "\n")

            num_parsed += parsed_result.parse_success
            num_json += int(parsed_result.parse_mode == "json")
            num_list += int(parsed_result.parse_mode == "list")
            num_valid += scored["valid_path"]
            num_optimal += scored["optimal_path"]

            print("-" * 80)
            print(f"Instance: {row['instance_id']}")
            print(f"Start -> Goal: {start} -> {goal}")
            print(f"Ground truth shortest path: {shortest_path}")
            print(f"Model output: {result['generated_text']}")
            print(f"Parsed path: {scored['parsed_path']}")
            print(f"Parse mode: {parsed_result.parse_mode}")
            print(f"Valid: {scored['valid_path']} | Optimal: {scored['optimal_path']}")

    total = len(df)
    print("\nSummary")
    print(f"Samples      : {total}")
    print(f"Parsed       : {num_parsed}/{total} = {num_parsed / total:.3f}")
    print(f"JSON parsed  : {num_json}/{total} = {num_json / total:.3f}")
    print(f"List parsed  : {num_list}/{total} = {num_list / total:.3f}")
    print(f"Valid        : {num_valid}/{total} = {num_valid / total:.3f}")
    print(f"Optimal      : {num_optimal}/{total} = {num_optimal / total:.3f}")
    print(f"Saved results: {output_path.resolve()}")


if __name__ == "__main__":
    main()