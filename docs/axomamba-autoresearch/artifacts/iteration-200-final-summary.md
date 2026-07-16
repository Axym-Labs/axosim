# AxoMamba autoresearch final summary through iteration 200

Date: 2026-07-13

## Final deployed checkpoint

| field | value |
| --- | --- |
| promoted iteration | I180 raw-target adaptation |
| checkpoint | `checkpoints/axomamba.pt` |
| SHA-256 | `f6ab42ad5d4a408ead74c7b8332b8ee0bc56f0c7db4a91dd88226bb6409c84f2` |
| parameters | 91,451 |
| cumulative presentations | 960,000 |
| voltage SERA | 15.95215 mV |
| dynamics SERA | 20.98482 mV/ms |
| intervention response SERA | 16.96027 mV |
| F1 area 0-5 ms | 0.41607 |
| exact F1 | 0.21538 |
| 5 ms F1 | 0.49719 |

The later 300k balanced continuation reached 15.91603 mV voltage SERA and
0.43185 F1 area, but its incremental continuation effect remained below the
promotion bar and was not used to overwrite the reproducible stable checkpoint.

## Iterations 191-200

| iteration | intervention | voltage delta vs matched control | key non-voltage result | decision |
| ---: | --- | ---: | --- | --- |
| 191 | learned fractional output delays | -0.00123 mV | exact F1 -0.00079; tied dense kernel | reject |
| 192 | sparse delayed branch impulses | -0.00300 mV | exact F1 -0.00164; no sparse-over-dense effect | reject |
| 193 | continuous quotient embedding | +0.00131 mV | all assessments uncertain; generic residual better | reject |
| 194 | activity-gated local TCN | -0.00173 mV | dynamics/response worsen; static gate only -0.00495 mV | reject |
| 195 | raw 1/5/20 ms future supervision | +0.00113 mV | dynamics +0.00478; old clipped-target F1 signal vanishes | reject |
| 196 | linear routed-current bypass | -0.00442 mV | exact F1 +0.00470 but F1 area -0.00271 | reject |
| 197 | nonlinear routed-current bypass | -0.00006 mV | latent capacity only -0.00238 mV | reject |
| 198 | detach both spike gradient paths | -0.00773 mV | F1 area -0.01777; 5 ms -0.02626 | reject |
| 199 | SERA-dominant objective | -0.03616 mV | F1 area -0.01662; 5 ms -0.03004 | reject |
| 200 | SERA-dominant objective, 300k | -0.04186 mV | F1 area -0.03880; 5 ms -0.05711 | reject long |

Negative deltas are improvements for SERA. No I191-I200 intervention reaches
the requested 0.1 mV magnitude, and the two largest voltage gains significantly
collapse tolerant spike fidelity.

## Conclusion

Within the current five-morphology, 500 ms, deterministic surrogate setup,
the evidence does not support another local AxoMamba addon. Raw-target
correction remains the only large late-stage improvement. Delays, local gates,
hybrid embeddings, future auxiliaries, forcing bypasses, gradient surgery, and
aggressive SERA weighting all fail either practical magnitude or balanced
event performance.

Future work should be a new task arc rather than I201: first test a joint
branch-time state-space neural operator, then cross-morphology operator
adaptation if held-out-cell generalization is part of the deployment target.
See `../literature/iteration-200-final-scan.md`.

## Verification

- AxoSim: 105 passed.
- AxoBench: 131 passed, 1 skipped.
- Stable checkpoint loads as AxoMamba with 91,451 parameters and 960k metadata.
- `git diff --check` passes.

The promoted/long-control voltage trajectory is plotted in
`promoted-voltage-progress.png`; its source values are in
`promoted-voltage-progress.csv`. The final I200 point is shown for training-
curve context but was not promoted.
