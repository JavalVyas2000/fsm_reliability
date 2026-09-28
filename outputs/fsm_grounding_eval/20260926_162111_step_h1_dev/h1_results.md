# H1: step-level grounding (share of graph attention on the source node's adjacency line)

Partitions: train, dev_cal, dev_thr. Score = −share (higher = more likely invalid). AUROC with 95% bootstrap CI grouped by instance.

| Model | Partition set | Steps (invalid) | Mean share valid / invalid | AUROC −share (layer mean) [CI] | Best single layer | AUROC −share_hmax | top-1 rate valid / invalid |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-3B-Instruct | train+dev_cal+dev_thr | 9797 (2243) | 0.114 / 0.076 | 0.644 [0.631, 0.656] | L050 0.659 | 0.614 | 0.15 / 0.07 |
| meta-llama/Llama-3.2-3B-Instruct | train+dev_cal+dev_thr | 9299 (2354) | 0.223 / 0.125 | 0.813 [0.804, 0.823] | L075 0.844 | 0.769 | 0.51 / 0.20 |
| Qwen/Qwen2.5-1.5B-Instruct | train+dev_cal+dev_thr | 9515 (3378) | 0.188 / 0.102 | 0.757 [0.747, 0.767] | L075 0.762 | 0.712 | 0.32 / 0.13 |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | train+dev_cal+dev_thr | 10192 (5505) | 0.138 / 0.077 | 0.746 [0.736, 0.756] | L075 0.782 | 0.660 | 0.21 / 0.07 |