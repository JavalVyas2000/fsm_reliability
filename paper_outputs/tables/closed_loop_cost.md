# Closed loop: outcomes and compute vs validator cost (measured validator 7.9 s/call)

| policy | recovered % | validator calls/ep | unchecked failing executions | compute s/ep @ 7.9 s | @ 30 s | @ 60 s | break-even validator s |
|---|---|---|---|---|---|---|---|
| Always validate | 43.8 | 2.45 | 0 | 74.4 | 128.6 | 202.1 |  |
| Observables probe | 41.0 | 1.3 | 0 | 73.2 | 102.0 | 141.1 | 6.9 |
| Obs. + internals + grounding probe | 37.0 | 0.68 | 1 | 77.8 | 92.9 | 113.5 | 9.8 |
| Random routing | 37.8 | 1.31 | 110 | 67.1 | 96.0 | 135.2 | 1.5 |
| Never validate | 30.0 | 0.0 | 274 | 16.5 | 16.5 | 16.5 |  |