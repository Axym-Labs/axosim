# Post-I200 dual AxoMamba mechanism ablation

## Design

The fresh panel trained ten 91,451-parameter arms for 300,000 presentations
and 9,375 optimizer steps from identical initialization on one frozen,
five-morphology schedule. Scale-zero controls retained execution and parameter
count. All diagnostics use paired traces and 500 bootstrap replicates.

A second experiment directly lesioned the immutable 960k checkpoint. Those
results measure checkpoint dependence, not the quality of a retrained smaller
architecture. A final common-start 300k continuation tested structural
removal after adaptation.

## Fresh 300k ablation

Positive values mean that retaining the mechanism helped. SERA signs are
oriented so positive is better.

| Mechanism removed | Component params | Voltage | Dynamics | Response | F1 area | Exact F1 | 5 ms F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Route occupancy | 176 | +0.42337 | +0.15255 | +0.23389 | +0.08701 | +0.06588 | +0.08076 |
| Morphology embedding | 290 | +0.20689 | +0.08337 | +0.14712 | +0.01696 | +0.01929 | +0.00801 |
| Local TCN | 14,326 | +0.12533 | +0.06045 | +0.06242 | +0.01564 | +0.00356 | +0.02307 |
| Branch saturation | 236 | +0.10491 | +0.01635 | +0.06989 | +0.00916 | +0.00656 | +0.00968 |
| Spike-voltage coupling | 0 | +0.09135 | +0.01536 | +0.02935 | +0.01567 | +0.00278 | +0.01710 |
| Morphology synaptic gain | 6,490 | +0.02173 | +0.01154 | +0.02205 | +0.00440 | -0.00008 | +0.00283 |
| Shared synaptic gain | 1,298 | +0.00296 | -0.00044 | +0.00414 | -0.00110 | +0.00065 | -0.00095 |
| Bilinear interaction | 1,176 | +0.00233 | +0.00056 | -0.00066 | -0.00095 | -0.00018 | -0.00219 |
| Soma gate | 119 | -0.02466 | -0.00504 | -0.01282 | -0.00984 | -0.00546 | -0.01065 |

The full fresh model reached voltage/dynamics/response SERA
16.13863/21.07888/17.11449 and F1 area/exact/5 ms
0.36902/0.19222/0.44586.

## Mature continuation

All arms started from compatible tensors in the same 960k checkpoint, then
received an identical 300k continuation. Deltas are candidate minus full;
lower SERA and higher F1 are preferred.

| Architecture | Params | Voltage | Dynamics | Response | F1 area | Exact F1 | 5 ms F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full continuation | 91,451 | 15.90715 | 20.94133 | 16.93946 | 0.42466 | 0.22192 | 0.50700 |
| No soma gate | 91,332 | 15.90561 | 20.94659 | 16.93934 | 0.42399 | 0.22453 | 0.50821 |
| No gate or bilinear | 90,156 | 15.91434 | 20.95333 | 16.93859 | 0.42430 | 0.22168 | 0.50942 |
| No gate, bilinear, or shared gain | 88,858 | 15.92107 | 20.96145 | 16.94398 | 0.42173 | 0.21456 | 0.50689 |

The 90,156-parameter arm had no significant core regression. The 88,858-
parameter arm significantly regressed dynamics SERA and was rejected.

## Compute

Median CUDA-event timings use length-500 sequences on an RTX 5090. Inference
uses 100 repeats; forward/backward uses 50 repeats.

| Architecture | Batch-1 inference | Batch-32 inference | Batch-32 fwd/bwd |
| --- | ---: | ---: | ---: |
| Full 91,451 | 0.93555 ms | 1.48795 ms | 4.14110 ms |
| Promoted 90,156 | 0.82824 ms | 1.43144 ms | 3.93888 ms |
| Change | -11.47% | -3.80% | -4.88% |

The shared-loader training panel measured 68.41 GPU seconds for the promoted
arm versus 75.72 for full, but the isolated kernel timing is the less
order-sensitive compute estimate.

## Full-training estimates

Adding each fresh 300k paired ablation delta to the measured 960k endpoint
gives a conditional projection, not a direct result. For the removed bilinear
interaction the projected metrics were voltage/dynamics/response SERA
15.95448/20.98539/16.95961 and F1 area/exact/5 ms
0.41702/0.21556/0.49938. For the removed soma gate they were
15.92749/20.97978/16.94745 and 0.42591/0.22084/0.50784.

The direct mature continuation supersedes those additive projections for the
selected combined architecture. No metric projection is presented as a
measured full-training endpoint.
