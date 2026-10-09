# Closed-loop CSTR: outcomes, compute and savings per model

Compute per episode = generation + internals pass + grounding pass + validator calls x validator cost. Savings are against always-validate on the same episodes; the 30 s / 60 s columns re-cost the same logged decisions for a slower validator.

## Qwen2.5-1.5B (400 episodes; measured validator 10.5 s/call)

| policy | recovered | failing executed unchecked | validator calls / ep (avoided) | compute / ep: gen + int + grd + val | saving @ 10.5 s | @ 30 s | @ 60 s | break-even validator s |
|---|---|---|---|---|---|---|---|---|
| Always validate | 34.5% | 0 | 1.05 | 48.1 s = 37.0 + 0.0 + 0.0 + 10.5 | +0% | +0% | +0% |  |
| Observables probe | 33.0% | 12 | 0.35 (−67%) | 42.6 s = 38.9 + 0.0 + 0.0 + 4.1 | +11% | +28% | +40% | 2.7 |
| Obs. + internals + grounding probe | 33.2% | 9 | 0.38 (−64%) | 45.6 s = 39.9 + 0.5 + 1.2 + 4.3 | +5% | +23% | +36% | 6.8 |
| Internals-only probe | 33.2% | 0 | 0.69 (−35%) | 45.4 s = 36.5 + 0.5 + 1.2 + 7.4 | +6% | +14% | +21% | 3.1 |
| Random routing | 33.8% | 186 | 0.42 (−60%) | 33.1 s = 28.6 + 0.0 + 0.0 + 4.3 | +31% | +40% | +46% | any (cheaper even at 0 s) |
| Never validate | 34.5% | 259 | 0.00 (−100%) | 10.8 s = 10.8 + 0.0 + 0.0 + 0.0 | +78% | +84% | +89% |  |

## Qwen2.5-3B (400 episodes; measured validator 8.1 s/call)

| policy | recovered | failing executed unchecked | validator calls / ep (avoided) | compute / ep: gen + int + grd + val | saving @ 8.1 s | @ 30 s | @ 60 s | break-even validator s |
|---|---|---|---|---|---|---|---|---|
| Always validate | 43.8% | 0 | 2.45 | 75.0 s = 55.1 + 0.0 + 0.0 + 20.1 | +0% | +0% | +0% |  |
| Observables probe | 41.0% | 0 | 1.30 (−47%) | 73.6 s = 63.0 + 0.0 + 0.0 + 10.4 | +2% | +21% | +30% | 6.9 |
| Obs. + internals + grounding probe | 37.0% | 1 | 0.69 (−72%) | 77.9 s = 67.2 + 2.7 + 2.4 + 5.1 | -4% | +28% | +44% | 9.8 |
| Internals-only probe | 30.0% | 0 | 0.86 (−65%) | 89.4 s = 78.3 + 1.2 + 2.9 + 8.5 | -19% | +16% | +34% | 17.2 |
| Random routing | 37.8% | 110 | 1.31 (−47%) | 67.4 s = 56.8 + 0.0 + 0.0 + 9.8 | +10% | +25% | +33% | 1.5 |
| Never validate | 30.0% | 274 | 0.00 (−100%) | 16.5 s = 16.5 + 0.0 + 0.0 + 0.0 | +78% | +87% | +92% |  |

## Llama-3.2-3B (400 episodes; measured validator 8.1 s/call)

| policy | recovered | failing executed unchecked | validator calls / ep (avoided) | compute / ep: gen + int + grd + val | saving @ 8.1 s | @ 30 s | @ 60 s | break-even validator s |
|---|---|---|---|---|---|---|---|---|
| Always validate | 53.8% | 0 | 1.80 | 181.3 s = 166.8 + 0.0 + 0.0 + 14.8 | +0% | +0% | +0% |  |
| Observables probe | 35.2% | 0 | 0.68 (−62%) | 254.0 s = 248.5 + 0.0 + 0.0 + 5.4 | -40% | -22% | -5% | 73.0 |
| Obs. + internals + grounding probe | 44.5% | 0 | 0.99 (−45%) | 253.0 s = 230.8 + 6.8 + 7.5 + 7.6 | -40% | -24% | -11% | 96.7 |
| Internals-only probe | 49.8% | 0 | 1.27 (−29%) | 233.5 s = 210.1 + 6.0 + 7.1 + 10.2 | -29% | -18% | -9% | 107.5 |
| Random routing | 29.5% | 0 | 0.58 (−68%) | 305.1 s = 300.5 + 0.0 + 0.0 + 4.8 | -68% | -44% | -22% | 109.8 |
| Never validate | 15.2% | 312 | 0.00 (−100%) | 56.5 s = 56.5 + 0.0 + 0.0 + 0.0 | +69% | +74% | +79% |  |

## Qwen2.5-7B (4-bit) (400 episodes; measured validator 7.8 s/call)

| policy | recovered | failing executed unchecked | validator calls / ep (avoided) | compute / ep: gen + int + grd + val | saving @ 7.8 s | @ 30 s | @ 60 s | break-even validator s |
|---|---|---|---|---|---|---|---|---|
| Always validate | 34.0% | 0 | 2.50 | 71.4 s = 51.9 + 0.0 + 0.0 + 19.3 | +0% | +0% | +0% |  |
| Observables probe | 25.0% | 8 | 0.24 (−90%) | 59.3 s = 57.3 + 0.0 + 0.0 + 2.2 | +17% | +49% | +64% | 2.4 |
| Obs. + internals + grounding probe | 26.5% | 165 | 0.62 (−75%) | 49.8 s = 40.5 + 1.1 + 3.4 + 4.5 | +30% | +50% | +59% | any (cheaper even at 0 s) |
| Internals-only probe | 30.0% | 0 | 1.41 (−44%) | 72.5 s = 55.4 + 1.5 + 4.6 + 11.8 | -1% | +18% | +28% | 8.8 |
| Random routing | 19.2% | 107 | 0.55 (−78%) | 59.4 s = 55.1 + 0.0 + 0.0 + 4.0 | +17% | +43% | +56% | 1.7 |
| Never validate | 19.2% | 318 | 0.00 (−100%) | 11.9 s = 11.9 + 0.0 + 0.0 + 0.0 | +83% | +91% | +94% |  |

## SmolLM2-1.7B (400 episodes; measured validator 11.6 s/call)

| policy | recovered | failing executed unchecked | validator calls / ep (avoided) | compute / ep: gen + int + grd + val | saving @ 11.6 s | @ 30 s | @ 60 s | break-even validator s |
|---|---|---|---|---|---|---|---|---|
| Always validate | 36.5% | 0 | 1.01 | 28.5 s = 16.7 + 0.0 + 0.0 + 10.9 | +0% | +0% | +0% |  |
| Observables probe | 35.2% | 13 | 0.39 (−62%) | 28.7 s = 24.2 + 0.0 + 0.0 + 4.8 | -1% | +24% | +39% | 12.0 |
| Obs. + internals + grounding probe | 35.0% | 237 | 0.34 (−66%) | 16.2 s = 10.6 + 0.7 + 0.9 + 4.3 | +43% | +52% | +58% | any (cheaper even at 0 s) |
| Internals-only probe | 35.5% | 243 | 0.56 (−44%) | 17.2 s = 9.1 + 0.6 + 0.9 + 7.0 | +40% | +42% | +43% | any (cheaper even at 0 s) |
| Random routing | 34.0% | 172 | 0.36 (−65%) | 24.0 s = 19.9 + 0.0 + 0.0 + 4.0 | +16% | +35% | +47% | 4.9 |
| Never validate | 36.0% | 254 | 0.00 (−100%) | 4.5 s = 4.5 + 0.0 + 0.0 + 0.0 | +84% | +90% | +94% |  |

## Rule: accept unchecked only on first proposals (validate every retry)

Derived from the logged trajectories (see script docstring): unsafe executions and the minimum extra validator calls are exact; recovery is bounded because a blocked failing retry would have continued.

| model | probe | failing executed: logged → rule | validator calls / ep: logged → rule (min) | recovered: logged → rule (bounds) |
|---|---|---|---|---|
| Qwen2.5-1.5B | Observables probe | 12 → **12** | 0.35 → 0.35 | 33.0% → [33.0%, 33.0%] |
| Qwen2.5-1.5B | Obs. + internals + grounding probe | 9 → **9** | 0.38 → 0.38 | 33.2% → [33.2%, 33.2%] |
| Qwen2.5-1.5B | Internals-only probe | 0 → **0** | 0.69 → 0.69 | 33.2% → [33.2%, 33.2%] |
| Qwen2.5-3B | Observables probe | 0 → **0** | 1.30 → 1.30 | 41.0% → [41.0%, 41.0%] |
| Qwen2.5-3B | Obs. + internals + grounding probe | 1 → **1** | 0.69 → 0.70 | 37.0% → [37.0%, 37.0%] |
| Qwen2.5-3B | Internals-only probe | 0 → **0** | 0.86 → 0.86 | 30.0% → [30.0%, 30.0%] |
| Llama-3.2-3B | Observables probe | 0 → **0** | 0.68 → 0.68 | 35.2% → [35.2%, 35.2%] |
| Llama-3.2-3B | Obs. + internals + grounding probe | 0 → **0** | 0.99 → 0.99 | 44.5% → [44.5%, 44.5%] |
| Llama-3.2-3B | Internals-only probe | 0 → **0** | 1.27 → 1.27 | 49.8% → [49.8%, 49.8%] |
| Qwen2.5-7B (4-bit) | Observables probe | 8 → **4** | 0.24 → 0.32 | 25.0% → [25.0%, 26.0%] |
| Qwen2.5-7B (4-bit) | Obs. + internals + grounding probe | 165 → **4** | 0.62 → 1.10 | 26.5% → [26.5%, 66.8%] |
| Qwen2.5-7B (4-bit) | Internals-only probe | 0 → **0** | 1.41 → 1.41 | 30.0% → [30.0%, 30.0%] |
| SmolLM2-1.7B | Observables probe | 13 → **13** | 0.39 → 0.39 | 35.2% → [35.2%, 35.2%] |
| SmolLM2-1.7B | Obs. + internals + grounding probe | 237 → **11** | 0.34 → 0.92 | 35.0% → [35.0%, 91.5%] |
| SmolLM2-1.7B | Internals-only probe | 243 → **2** | 0.56 → 1.18 | 35.5% → [35.5%, 95.8%] |