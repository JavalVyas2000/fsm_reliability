# CSTR: validator calls skipped per signal (test_iid)

Cell: share of validator calls skipped (accepted: failures/accepted; cert = certified failure bound <= X; rej-good = good proposals rejected unchecked; Δ = change vs observables, points, paired bootstrap 95% CI).

## X = 10%

| signal | Qwen2.5-1.5B | Qwen2.5-3B* | Qwen2.5-7B (4-bit) | Llama-3.2-3B | SmolLM2-1.7B |
|---|---|---|---|---|---|
| Observables | 63.8% (acc 9/108; rej-good 5) | 43.6% (acc 0; rej-good 5) | 88.4% (acc 0/58, cert; rej-good 18) | 75.9% (acc 0; rej-good 22) | 61.5% (acc 9/111; rej-good 4) |
| Plant readings only | 63.2% (acc 10/106; rej-good 5; Δ -0.6 [-2.8, +1.6]) | 1.4% (acc 0; rej-good 0) | 82.4% (acc 2/47; rej-good 20; Δ -6.0 [-9.2, -2.8]) | 5.1% (acc 0; rej-good 2) | 58.3% (acc 10/107; rej-good 3; Δ -3.2 [-5.6, -0.8]) |
| Obs. + grounding | 64.6% (acc 7/99; rej-good 7; Δ +0.8 [-1.8, +3.4]) | 40.2% (acc 0; rej-good 4) | 88.8% (acc 0/63, cert; rej-good 15; Δ +0.4 [-0.8, +1.8]) | 77.8% (acc 0; rej-good 18) | 67.1% (acc 13/118; rej-good 7; Δ +5.6 [+3.4, +8.0]) |
| Obs. + all internals | 62.1% (acc 7/101; rej-good 6; Δ -1.6 [-4.7, +1.2]) | 54.4% (acc 4/44; rej-good 9) | 86.2% (acc 0/50; rej-good 19; Δ -2.2 [-4.8, +0.2]) | 72.9% (acc 0; rej-good 8) | 66.7% (acc 8/98; rej-good 11; Δ +5.2 [+1.8, +8.6]) |
| Obs. + internals + grounding | 59.9% (acc 9/102; rej-good 6; Δ -3.8 [-7.1, -0.6]) | 51.9% (acc 4/49; rej-good 6) | 83.6% (acc 0/37; rej-good 20; Δ -4.8 [-7.8, -1.8]) | 75.1% (acc 0; rej-good 8) | 65.9% (acc 8/99; rej-good 9; Δ +4.4 [+1.2, +7.8]) |
| Region grounding | 27.9% (acc 0; rej-good 6; Δ -35.8 [-40.7, -31.2]) | 11.4% (acc 0; rej-good 6) | 47.5% (acc 0; rej-good 11; Δ -40.9 [-45.5, -36.5]) | 40.4% (acc 0; rej-good 7) | 26.5% (acc 0; rej-good 5; Δ -35.1 [-39.5, -30.7]) |
| Attention (regions) | 28.9% (acc 0; rej-good 6; Δ -34.8 [-39.5, -30.0]) | 14.6% (acc 0; rej-good 7) | 49.1% (acc 0; rej-good 12; Δ -39.3 [-43.7, -34.9]) | 53.1% (acc 0; rej-good 6) | 32.9% (acc 0; rej-good 12; Δ -28.7 [-33.9, -23.6]) |
| Hidden states | 32.2% (acc 0; rej-good 12; Δ -31.6 [-36.2, -26.7]) | 14.2% (acc 0; rej-good 3) | 50.1% (acc 0; rej-good 12; Δ -38.3 [-42.9, -33.9]) | 53.7% (acc 0; rej-good 4) | 46.7% (acc 2/68; rej-good 5; Δ -14.8 [-18.6, -11.2]) |
| All internals | 32.0% (acc 0; rej-good 11; Δ -31.8 [-36.8, -26.9]) | 14.0% (acc 0; rej-good 1) | 48.9% (acc 0; rej-good 12; Δ -39.5 [-44.1, -35.1]) | 56.7% (acc 0; rej-good 5) | 41.7% (acc 2/41; rej-good 3; Δ -19.8 [-23.8, -15.8]) |
| All internals + grounding | 30.6% (acc 0; rej-good 11; Δ -33.2 [-38.1, -28.1]) | 12.2% (acc 0; rej-good 0) | 49.5% (acc 0; rej-good 11; Δ -38.9 [-43.5, -34.5]) | 56.7% (acc 0; rej-good 5) | 43.1% (acc 2/36; rej-good 6; Δ -18.4 [-22.4, -14.0]) |
| Token confidence | 3.2% (acc 0; rej-good 1; Δ -60.5 [-64.8, -56.1]) | 0.0% (acc 0; rej-good 0) | 19.2% (acc 0; rej-good 4; Δ -69.1 [-73.3, -64.7]) | 36.6% (acc 0; rej-good 9) | 23.4% (acc 0; rej-good 7; Δ -38.1 [-42.9, -33.3]) |

## X = 5%

| signal | Qwen2.5-1.5B | Qwen2.5-3B* | Qwen2.5-7B (4-bit) | Llama-3.2-3B | SmolLM2-1.7B |
|---|---|---|---|---|---|
| Observables | 58.3% (acc 1/81; rej-good 5) | 43.6% (acc 0; rej-good 5) | 86.2% (acc 0/47; rej-good 18) | 75.9% (acc 0; rej-good 22) | 54.5% (acc 0/76; rej-good 4) |
| Plant readings only | 55.3% (acc 0/67; rej-good 5; Δ -3.0 [-5.5, -0.4]) | 1.4% (acc 0; rej-good 0) | 72.9% (acc 0; rej-good 20; Δ -13.2 [-17.2, -9.4]) | 5.1% (acc 0; rej-good 2) | 50.7% (acc 0/69; rej-good 3; Δ -3.8 [-6.2, -1.6]) |
| Obs. + grounding | 58.9% (acc 1/71; rej-good 7; Δ +0.6 [-1.8, +3.0]) | 40.2% (acc 0; rej-good 4) | 81.6% (acc 0/27; rej-good 15; Δ -4.6 [-6.6, -2.6]) | 77.8% (acc 0; rej-good 18) | 59.1% (acc 1/78; rej-good 7; Δ +4.6 [+2.6, +6.8]) |
| Obs. + all internals | 55.3% (acc 2/67; rej-good 6; Δ -3.0 [-6.1, +0.0]) | 45.4% (acc 0; rej-good 9) | 84.4% (acc 0/41; rej-good 19; Δ -1.8 [-4.8, +1.0]) | 72.9% (acc 0; rej-good 8) | 61.1% (acc 1/70; rej-good 11; Δ +6.6 [+3.6, +9.6]) |
| Obs. + internals + grounding | 46.6% (acc 0/36; rej-good 6; Δ -11.7 [-15.2, -8.5]) | 42.0% (acc 0; rej-good 6) | 82.8% (acc 0/33; rej-good 20; Δ -3.4 [-6.4, -0.4]) | 75.1% (acc 0; rej-good 8) | 58.9% (acc 1/64; rej-good 9; Δ +4.4 [+1.2, +7.6]) |
| Region grounding | 27.9% (acc 0; rej-good 6; Δ -30.4 [-35.0, -25.7]) | 11.4% (acc 0; rej-good 6) | 47.5% (acc 0; rej-good 11; Δ -38.7 [-43.1, -34.3]) | 40.4% (acc 0; rej-good 7) | 26.5% (acc 0; rej-good 5; Δ -28.1 [-32.5, -23.8]) |
| Attention (regions) | 28.9% (acc 0; rej-good 6; Δ -29.4 [-34.0, -24.5]) | 14.6% (acc 0; rej-good 7) | 49.1% (acc 0; rej-good 12; Δ -37.1 [-41.5, -32.5]) | 53.1% (acc 0; rej-good 6) | 32.9% (acc 0; rej-good 12; Δ -21.6 [-26.1, -17.2]) |
| Hidden states | 32.2% (acc 0; rej-good 12; Δ -26.1 [-30.8, -21.5]) | 14.2% (acc 0; rej-good 3) | 50.1% (acc 0; rej-good 12; Δ -36.1 [-40.7, -31.9]) | 53.7% (acc 0; rej-good 4) | 33.1% (acc 0; rej-good 5; Δ -21.4 [-25.7, -17.2]) |
| All internals | 32.0% (acc 0; rej-good 11; Δ -26.3 [-31.0, -21.3]) | 14.0% (acc 0; rej-good 1) | 48.9% (acc 0; rej-good 12; Δ -37.3 [-41.9, -32.7]) | 56.7% (acc 0; rej-good 5) | 38.5% (acc 1/25; rej-good 3; Δ -16.0 [-19.8, -12.0]) |
| All internals + grounding | 30.6% (acc 0; rej-good 11; Δ -27.7 [-32.6, -23.3]) | 12.2% (acc 0; rej-good 0) | 49.5% (acc 0; rej-good 11; Δ -36.7 [-41.1, -32.1]) | 56.7% (acc 0; rej-good 5) | 41.5% (acc 1/28; rej-good 6; Δ -13.0 [-16.4, -9.0]) |
| Token confidence | 3.2% (acc 0; rej-good 1; Δ -55.1 [-59.5, -50.6]) | 0.0% (acc 0; rej-good 0) | 19.2% (acc 0; rej-good 4; Δ -66.9 [-71.1, -62.9]) | 36.6% (acc 0; rej-good 9) | 23.4% (acc 0; rej-good 7; Δ -31.1 [-35.5, -26.9]) |
