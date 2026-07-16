# Initial-state premise audit

Date: 2026-07-13

Before implementing the iteration-170 literature scan's observed-state model,
the premise was tested on the 120 morphology-balanced held-out traces and the
I163 stable-model prediction cache.

- Initial raw soma voltage: mean -76.0 mV, standard deviation 0.0 mV, range
  [-76.0, -76.0] mV.
- Correlation with model residual at 50, 100, 250, 500, 750, 1,000, and 2,000
  ms: undefined because the initial-state feature has zero variance.
- Oracle screen: fit a separate linear coefficient at every time step from
  centered initial voltage to model residual on a fixed 25% calibration split,
  then evaluate the correction on the other 75% after the standard 500 ms
  exclusion. Baseline and corrected voltage SERA are both
  17.1155550831 mV, delta 0.0 mV.

Conclusion: initial voltage contains no information in this dataset, so adding
it would only create a constant feature and cannot test partial-state
estimation. No model run is warranted.
