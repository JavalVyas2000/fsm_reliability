# H1: step-level grounding (share of graph attention on the source node's adjacency line)

Partitions: train, dev_cal, dev_thr, test_iid. Score = −share (higher = more likely invalid). AUROC with 95% bootstrap CI grouped by instance.

| Model | Partition set | Steps (invalid) | Mean share valid / invalid | AUROC −share (layer mean) [CI] | Best single layer | AUROC −share_hmax | top-1 rate valid / invalid |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-3B-Instruct | train+dev_cal+dev_thr+test_iid | 11748 (2659) | 0.114 / 0.076 | 0.648 [0.637, 0.660] | L050 0.663 | 0.617 | 0.16 / 0.07 |
| meta-llama/Llama-3.2-3B-Instruct | train+dev_cal+dev_thr+test_iid | 11127 (2842) | 0.223 / 0.124 | 0.813 [0.804, 0.822] | L075 0.844 | 0.769 | 0.51 / 0.20 |
| Qwen/Qwen2.5-1.5B-Instruct | train+dev_cal+dev_thr+test_iid | 11433 (4063) | 0.188 / 0.102 | 0.758 [0.748, 0.767] | L075 0.763 | 0.714 | 0.32 / 0.13 |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | train+dev_cal+dev_thr+test_iid | 12340 (6704) | 0.138 / 0.077 | 0.745 [0.736, 0.754] | L075 0.783 | 0.658 | 0.21 / 0.07 |