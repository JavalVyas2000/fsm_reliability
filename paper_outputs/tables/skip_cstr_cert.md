# CSTR: validator calls skipped per signal (cert)

Cell: share of validator calls skipped (accepted: failures/accepted; cert = certified failure bound <= X; rej-good = good proposals rejected unchecked; Δ = change vs observables, points, paired bootstrap 95% CI).

## X = 10%

| signal | Qwen2.5-1.5B | Qwen2.5-3B* | Qwen2.5-7B (4-bit) | Llama-3.2-3B |
|---|---|---|---|---|
| Observables | 62.8% (acc 14/121; rej-good 6) | 41.2% (acc 0; rej-good 16) | 85.5% (acc 2/54; rej-good 14) | 70.7% (acc 0; rej-good 16) |
| Plant readings only | 59.3% (acc 14/113; rej-good 5; Δ -3.5 [-6.1, -1.0]) | 1.8% (acc 0; rej-good 0) | 79.5% (acc 3/44; rej-good 15; Δ -6.0 [-9.4, -2.8]) | 4.0% (acc 0; rej-good 1; Δ -66.7 [-70.9, -62.2]) |
| Obs. + grounding | 60.0% (acc 7/102; rej-good 5; Δ -2.8 [-5.5, -0.4]) | 38.8% (acc 0; rej-good 15) | 85.7% (acc 4/57; rej-good 15; Δ +0.2 [-1.2, +1.4]) | 74.9% (acc 0; rej-good 16; Δ +4.2 [+0.8, +7.8]) |
| Obs. + all internals | 61.4% (acc 13/110; rej-good 7; Δ -1.4 [-4.7, +1.8]) | 50.6% (acc 4/38; rej-good 18) | 82.7% (acc 3/46; rej-good 20; Δ -2.8 [-5.6, +0.0]) | 71.5% (acc 0; rej-good 8; Δ +0.8 [-2.3, +4.2]) |
| Obs. + internals + grounding | 60.6% (acc 14/111; rej-good 6; Δ -2.2 [-5.7, +1.2]) | 49.8% (acc 5/43; rej-good 15) | 79.9% (acc 2/35; rej-good 19; Δ -5.6 [-8.6, -2.6]) | 74.1% (acc 0; rej-good 10; Δ +3.4 [-0.2, +7.2]) |
| Region grounding | 27.4% (acc 0; rej-good 3; Δ -35.4 [-39.8, -30.9]) | 10.0% (acc 0; rej-good 7) | 46.8% (acc 0; rej-good 10; Δ -38.8 [-43.4, -34.1]) | 37.6% (acc 0; rej-good 7; Δ -33.1 [-38.2, -28.3]) |
| Attention (regions) | 30.1% (acc 0; rej-good 7; Δ -32.7 [-37.4, -27.8]) | 15.5% (acc 0; rej-good 7) | 49.4% (acc 0; rej-good 11; Δ -36.1 [-40.6, -31.5]) | 47.7% (acc 0; rej-good 7; Δ -23.0 [-27.6, -18.1]) |
| Hidden states | 28.9% (acc 0; rej-good 4; Δ -33.9 [-38.4, -29.1]) | 13.5% (acc 0; rej-good 3) | 49.8% (acc 0; rej-good 11; Δ -35.7 [-40.4, -31.1]) | 47.9% (acc 0; rej-good 6; Δ -22.8 [-27.4, -18.1]) |
| All internals | 29.5% (acc 0; rej-good 8; Δ -33.3 [-37.8, -28.7]) | 14.7% (acc 0; rej-good 3) | 47.0% (acc 0; rej-good 9; Δ -38.6 [-43.2, -33.9]) | 53.2% (acc 0; rej-good 8; Δ -17.5 [-22.4, -12.7]) |
| All internals + grounding | 28.3% (acc 0; rej-good 4; Δ -34.6 [-39.0, -30.1]) | 12.2% (acc 0; rej-good 3) | 47.8% (acc 0; rej-good 9; Δ -37.8 [-42.2, -33.1]) | 52.7% (acc 0; rej-good 6; Δ -17.9 [-22.4, -13.3]) |
| Token confidence | 3.7% (acc 0; rej-good 3; Δ -59.1 [-63.4, -54.7]) | 0.0% (acc 0; rej-good 0) | 16.7% (acc 0; rej-good 2; Δ -68.9 [-73.1, -64.9]) | 33.8% (acc 0; rej-good 10; Δ -36.9 [-42.0, -31.9]) |

## X = 5%

| signal | Qwen2.5-1.5B | Qwen2.5-3B* | Qwen2.5-7B (4-bit) | Llama-3.2-3B |
|---|---|---|---|---|
| Observables | 54.3% (acc 3/79; rej-good 6) | 41.2% (acc 0; rej-good 16) | 83.7% (acc 2/45; rej-good 14) | 70.7% (acc 0; rej-good 16) |
| Plant readings only | 49.2% (acc 0/63; rej-good 5; Δ -5.1 [-7.7, -2.6]) | 1.8% (acc 0; rej-good 0) | 70.7% (acc 0; rej-good 15; Δ -13.1 [-17.1, -9.4]) | 4.0% (acc 0; rej-good 1; Δ -66.7 [-71.1, -62.4]) |
| Obs. + grounding | 53.0% (acc 0/68; rej-good 5; Δ -1.2 [-3.3, +0.8]) | 38.8% (acc 0; rej-good 15) | 78.7% (acc 1/22; rej-good 15; Δ -5.0 [-7.2, -3.0]) | 74.9% (acc 0; rej-good 16; Δ +4.2 [+1.0, +7.6]) |
| Obs. + all internals | 49.8% (acc 2/53; rej-good 7; Δ -4.5 [-7.9, -1.2]) | 43.0% (acc 0; rej-good 18) | 81.3% (acc 3/39; rej-good 20; Δ -2.4 [-5.0, +0.4]) | 71.5% (acc 0; rej-good 8; Δ +0.8 [-2.5, +4.4]) |
| Obs. + internals + grounding | 44.3% (acc 0/31; rej-good 6; Δ -10.0 [-13.4, -6.5]) | 41.2% (acc 0; rej-good 15) | 78.9% (acc 2/30; rej-good 19; Δ -4.8 [-7.8, -2.0]) | 74.1% (acc 0; rej-good 10; Δ +3.4 [-0.6, +7.4]) |
| Region grounding | 27.4% (acc 0; rej-good 3; Δ -26.8 [-31.1, -23.0]) | 10.0% (acc 0; rej-good 7) | 46.8% (acc 0; rej-good 10; Δ -36.9 [-41.6, -32.3]) | 37.6% (acc 0; rej-good 7; Δ -33.1 [-38.0, -27.8]) |
| Attention (regions) | 30.1% (acc 0; rej-good 7; Δ -24.2 [-28.7, -19.7]) | 15.5% (acc 0; rej-good 7) | 49.4% (acc 0; rej-good 11; Δ -34.3 [-39.0, -29.9]) | 47.7% (acc 0; rej-good 7; Δ -23.0 [-27.6, -18.4]) |
| Hidden states | 28.9% (acc 0; rej-good 4; Δ -25.4 [-29.9, -21.1]) | 13.5% (acc 0; rej-good 3) | 49.8% (acc 0; rej-good 11; Δ -33.9 [-38.6, -29.3]) | 47.9% (acc 0; rej-good 6; Δ -22.8 [-27.4, -18.1]) |
| All internals | 29.5% (acc 0; rej-good 8; Δ -24.8 [-29.3, -20.1]) | 14.7% (acc 0; rej-good 3) | 47.0% (acc 0; rej-good 9; Δ -36.7 [-41.0, -32.1]) | 53.2% (acc 0; rej-good 8; Δ -17.5 [-22.2, -12.7]) |
| All internals + grounding | 28.3% (acc 0; rej-good 4; Δ -26.0 [-30.7, -21.5]) | 12.2% (acc 0; rej-good 3) | 47.8% (acc 0; rej-good 9; Δ -35.9 [-40.6, -31.5]) | 52.7% (acc 0; rej-good 6; Δ -17.9 [-22.4, -13.5]) |
| Token confidence | 3.7% (acc 0; rej-good 3; Δ -50.6 [-54.9, -46.3]) | 0.0% (acc 0; rej-good 0) | 16.7% (acc 0; rej-good 2; Δ -67.1 [-71.5, -62.7]) | 33.8% (acc 0; rej-good 10; Δ -36.9 [-42.0, -31.9]) |
