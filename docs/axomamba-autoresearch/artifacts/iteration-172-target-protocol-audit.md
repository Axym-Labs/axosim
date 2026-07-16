# Training/evaluation target protocol audit

Date: 2026-07-13

The v1 dataset stores two soma views:

- `targets[..., 1]`: voltage clipped above -55 mV, then normalized as
  `(v + 67.7) * 0.1`;
- `soma_mv`: the raw low-resolution simulator trace.

The autoresearch objective currently trains every voltage loss on the clipped
channel. The current AxoBench diagnostics instead pass `soma_mv` to voltage,
dynamics, intervention-response, and failure-mode scoring.

On the first 40 held-out traces, converting the clipped training channel back
to millivolts and comparing it with raw soma gives:

- maximum absolute difference: 98.66021 mV;
- mean signed difference: -0.44820 mV;
- RMSE: 5.42640 mV;
- clipped target range after inversion: -80.50211 to -55.00000 mV;
- raw range: -80.50211 to 43.66021 mV.

This is an intentional historical dataset encoding but a mismatch with the new
raw-voltage optimization target. The next controlled experiment must load raw
soma once per shared batch and optimize the exact scored voltage view, with the
old clipped target and a 50/50 blend as matched controls.
