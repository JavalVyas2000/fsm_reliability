# FSM: validator calls skipped per signal (cert)

Cell: share of validator calls skipped (accepted: failures/accepted; cert = certified failure bound <= X; rej-good = good proposals rejected unchecked; Δ = change vs observables, points, paired bootstrap 95% CI).

## X = 10%

| signal | Qwen2.5-3B | Llama-3.2-3B | Qwen2.5-1.5B | SmolLM2-1.7B | Qwen2.5-7B (4-bit) |
|---|---|---|---|---|---|
| Observables | 20.5% (acc 15/335, cert; rej-good 12) | 43.0% (acc 26/333; rej-good 62) | 40.4% (acc 7/284, cert; rej-good 45) | 84.6% (acc 15/166; rej-good 134) | 15.7% (acc 13/258, cert; rej-good 17) |
| Obs. + grounding | 20.8% (acc 11/339, cert; rej-good 9; Δ +0.2 [-0.8, +1.3]) | 58.6% (acc 20/444, cert; rej-good 77; Δ +15.5 [+13.8, +17.2]) | 45.5% (acc 1/228, cert; rej-good 56; Δ +5.1 [+3.6, +6.7]) | 85.2% (acc 0; rej-good 146; Δ +0.6 [-0.8, +2.1]) | 22.2% (acc 41/513; rej-good 4; Δ +6.5 [+5.2, +7.7]) |
| Obs. + all internals | 31.7% (acc 35/599, cert; rej-good 17; Δ +11.2 [+9.7, +12.7]) | 46.8% (acc 7/308, cert; rej-good 78; Δ +3.7 [+2.2, +5.2]) | 36.7% (acc 6/262, cert; rej-good 37; Δ -3.7 [-5.1, -2.3]) | 79.0% (acc 0; rej-good 136; Δ -5.6 [-7.0, -4.2]) | 15.4% (acc 25/368; rej-good 4; Δ -0.3 [-1.4, +0.8]) |
| Obs. + internals + grounding | 34.5% (acc 27/603, cert; rej-good 17; Δ +14.0 [+12.3, +15.6]) | 58.1% (acc 29/532, cert; rej-good 68; Δ +15.0 [+13.2, +16.9]) | 39.9% (acc 5/259, cert; rej-good 40; Δ -0.4 [-2.0, +1.1]) | 90.7% (acc 7/146; rej-good 165; Δ +6.2 [+4.9, +7.5]) | 15.2% (acc 17/375, cert; rej-good 2; Δ -0.5 [-1.6, +0.7]) |
| Region grounding | 14.4% (acc 18/367, cert; rej-good 11; Δ -6.1 [-7.7, -4.5]) | 31.4% (acc 2/102, cert; rej-good 51; Δ -11.7 [-14.0, -9.5]) | 30.7% (acc 5/178, cert; rej-good 29; Δ -9.7 [-11.6, -7.7]) | 79.6% (acc 0; rej-good 120; Δ -5.0 [-6.5, -3.5]) | 3.2% (acc 0/88, cert; rej-good 1; Δ -12.5 [-13.8, -11.3]) |
| Attention (regions) | 21.4% (acc 19/351, cert; rej-good 22; Δ +0.9 [-0.3, +2.0]) | 16.5% (acc 0; rej-good 30; Δ -26.5 [-28.3, -24.8]) | 34.6% (acc 6/195, cert; rej-good 42; Δ -5.8 [-7.3, -4.3]) | 75.5% (acc 0; rej-good 125; Δ -9.1 [-10.5, -7.6]) | 13.3% (acc 24/319; rej-good 2; Δ -2.5 [-3.4, -1.6]) |
| Hidden states | 23.9% (acc 26/506, cert; rej-good 7; Δ +3.3 [+2.0, +4.8]) | 26.5% (acc 13/204; rej-good 40; Δ -16.5 [-18.1, -15.0]) | 22.2% (acc 0; rej-good 29; Δ -18.2 [-19.8, -16.5]) | 74.4% (acc 0; rej-good 108; Δ -10.2 [-11.6, -8.7]) | 9.6% (acc 15/234; rej-good 2; Δ -6.1 [-7.0, -5.1]) |
| All internals | 25.4% (acc 31/526, cert; rej-good 11; Δ +4.9 [+3.4, +6.5]) | 30.7% (acc 8/240, cert; rej-good 46; Δ -12.3 [-14.1, -10.6]) | 25.5% (acc 6/196, cert; rej-good 17; Δ -14.9 [-16.5, -13.4]) | 76.1% (acc 0; rej-good 125; Δ -8.5 [-10.0, -7.1]) | 8.9% (acc 12/252, cert; rej-good 1; Δ -6.8 [-7.8, -5.6]) |
| All internals + grounding | 22.6% (acc 21/483, cert; rej-good 4; Δ +2.0 [+0.4, +3.7]) | 35.8% (acc 13/371, cert; rej-good 31; Δ -7.2 [-9.1, -5.3]) | 38.0% (acc 8/240, cert; rej-good 45; Δ -2.4 [-4.1, -0.6]) | 78.8% (acc 0; rej-good 112; Δ -5.8 [-7.3, -4.2]) | 10.8% (acc 12/303, cert; rej-good 0; Δ -4.9 [-6.1, -3.7]) |
| Token confidence | 17.9% (acc 28/329; rej-good 19; Δ -2.6 [-4.1, -1.0]) | 14.2% (acc 12/140; rej-good 20; Δ -28.8 [-30.7, -26.9]) | 13.4% (acc 0; rej-good 19; Δ -27.0 [-28.8, -25.2]) | 55.8% (acc 0; rej-good 77; Δ -28.8 [-30.5, -27.0]) | 0.0% (acc 0; rej-good 0; Δ -15.7 [-17.1, -14.4]) |

## X = 5%

| signal | Qwen2.5-3B | Llama-3.2-3B | Qwen2.5-1.5B | SmolLM2-1.7B | Qwen2.5-7B (4-bit) |
|---|---|---|---|---|---|
| Observables | 19.3% (acc 3/299, cert; rej-good 12) | 38.2% (acc 0/187, cert; rej-good 62) | 37.4% (acc 0/195, cert; rej-good 45) | 78.9% (acc 0; rej-good 134) | 7.0% (acc 0; rej-good 17) |
| Obs. + grounding | 9.4% (acc 0; rej-good 9; Δ -9.9 [-11.1, -8.7]) | 51.2% (acc 1/223, cert; rej-good 77; Δ +13.0 [+11.4, +14.7]) | 44.7% (acc 0/204, cert; rej-good 56; Δ +7.3 [+5.9, +8.7]) | 85.2% (acc 0; rej-good 146; Δ +6.2 [+5.2, +7.4]) | 4.9% (acc 0; rej-good 4; Δ -2.1 [-2.7, -1.5]) |
| Obs. + all internals | 20.8% (acc 4/274, cert; rej-good 17; Δ +1.5 [+0.3, +2.8]) | 44.7% (acc 3/246, cert; rej-good 78; Δ +6.5 [+5.1, +8.0]) | 34.6% (acc 2/201, cert; rej-good 37; Δ -2.7 [-4.0, -1.3]) | 79.0% (acc 0; rej-good 136; Δ +0.1 [-1.0, +1.3]) | 3.0% (acc 0; rej-good 4; Δ -4.0 [-4.8, -3.3]) |
| Obs. + internals + grounding | 21.3% (acc 1/207, cert; rej-good 17; Δ +1.9 [+0.6, +3.4]) | 51.3% (acc 3/329, cert; rej-good 68; Δ +13.1 [+11.3, +14.9]) | 31.1% (acc 0; rej-good 40; Δ -6.2 [-8.0, -4.4]) | 85.8% (acc 0; rej-good 165; Δ +6.9 [+5.6, +8.0]) | 12.8% (acc 10/303; rej-good 2; Δ +5.8 [+4.4, +7.2]) |
| Region grounding | 2.1% (acc 0; rej-good 11; Δ -17.2 [-18.7, -15.7]) | 27.9% (acc 0; rej-good 51; Δ -10.2 [-12.3, -8.2]) | 24.6% (acc 0; rej-good 29; Δ -12.7 [-14.7, -10.7]) | 79.6% (acc 0; rej-good 120; Δ +0.6 [-0.7, +2.0]) | 0.2% (acc 0; rej-good 1; Δ -6.8 [-7.7, -6.0]) |
| Attention (regions) | 15.4% (acc 3/173; rej-good 22; Δ -3.9 [-5.1, -2.7]) | 16.5% (acc 0; rej-good 30; Δ -21.6 [-23.3, -20.0]) | 27.9% (acc 0; rej-good 42; Δ -9.4 [-11.1, -7.8]) | 75.5% (acc 0; rej-good 125; Δ -3.4 [-4.5, -2.3]) | 2.5% (acc 0; rej-good 2; Δ -4.5 [-5.3, -3.7]) |
| Hidden states | 17.9% (acc 10/328; rej-good 7; Δ -1.4 [-2.6, -0.1]) | 19.7% (acc 0; rej-good 40; Δ -18.5 [-20.1, -16.8]) | 22.2% (acc 0; rej-good 29; Δ -15.1 [-16.8, -13.5]) | 74.4% (acc 0; rej-good 108; Δ -4.5 [-5.8, -3.4]) | 1.7% (acc 0; rej-good 2; Δ -5.3 [-6.1, -4.4]) |
| All internals | 14.1% (acc 3/189; rej-good 11; Δ -5.2 [-6.5, -3.8]) | 22.7% (acc 0; rej-good 46; Δ -15.5 [-17.1, -13.8]) | 18.8% (acc 0; rej-good 17; Δ -18.5 [-20.1, -16.9]) | 76.1% (acc 0; rej-good 125; Δ -2.9 [-4.1, -1.6]) | 0.5% (acc 0; rej-good 1; Δ -6.6 [-7.4, -5.7]) |
| All internals + grounding | 6.4% (acc 0; rej-good 4; Δ -12.9 [-14.4, -11.4]) | 30.7% (acc 2/217, cert; rej-good 31; Δ -7.5 [-9.4, -5.7]) | 29.8% (acc 0; rej-good 45; Δ -7.5 [-9.5, -5.7]) | 78.8% (acc 0; rej-good 112; Δ -0.1 [-1.4, +1.3]) | 0.6% (acc 0; rej-good 0; Δ -6.4 [-7.3, -5.5]) |
| Token confidence | 6.9% (acc 0; rej-good 19; Δ -12.4 [-13.8, -11.0]) | 9.5% (acc 0; rej-good 20; Δ -28.6 [-30.4, -26.8]) | 13.4% (acc 0; rej-good 19; Δ -23.9 [-25.7, -22.1]) | 55.8% (acc 0; rej-good 77; Δ -23.1 [-24.8, -21.4]) | 0.0% (acc 0; rej-good 0; Δ -7.0 [-8.0, -6.1]) |
