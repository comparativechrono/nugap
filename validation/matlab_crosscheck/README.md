# MATLAB `gapmetric` cross-check

This compares the package's `nu_gap` against MATLAB's `gapmetric` (Robust
Control Toolbox), the canonical reference implementation of the Vinnicombe
nu-gap. It is the one fully *external* correctness check (the other checks in
`../validate_numerics.py` are self-contained).

## Files

- `panel.json` — a panel of 13 SISO system pairs (continuous and discrete),
  stored as numerator/denominator coefficients in descending powers. **Shared**
  by both sides so they compare identical systems. `dt: null` means continuous;
  a number means a discrete system with that sample time.
- `reference_values.m` — MATLAB script: reads `panel.json`, builds each pair
  with `tf`, runs `[gap, nugap] = gapmetric(P1, P2)`, and writes
  `reference_values.csv`. The Vinnicombe nu-gap is the **second** output.
- `crosscheck_matlab.py` — Python harness: rebuilds the same systems, computes
  `nu_gap`, and asserts agreement with the MATLAB `nugap` column (tolerance
  5e-3; `gapmetric`'s default relative accuracy is 1e-3).

## Run

MATLAB (needs Control System Toolbox + Robust Control Toolbox, R2016b+ for
`jsondecode`), from this folder:

```matlab
>> reference_values        % writes reference_values.csv
```

Then Python, from this folder:

```bash
python crosscheck_matlab.py
```

Without `reference_values.csv` present, the Python script prints the package
values only, alongside the command to generate the MATLAB reference.

## Panel coverage

Identical systems (→ 0); constant-gain pairs (closed form); a far
stable-vs-unstable pair (→ 1); the near-boundary `1/(s-0.001)` vs `1/(s+0.001)`
case from the MathWorks documentation (stable vs unstable yet a *small* nu-gap,
which exercises the winding-number term); the MathWorks doc example
`1/(s+1)` vs `(s+1)/(s^2+3s+10)`; first- and second-order pairs of varying
separation and damping; an all-pass-vs-unit boundary case; and discrete-time
pairs (stable/stable, biproper, and stable/unstable), since sampled data is the
package's primary use case.

## Note on discrete systems

`gapmetric` operates on `tf` objects in either domain; the discrete pairs are
built with `tf(num, den, Ts)`. If your MATLAB release objects to a particular
discrete case, comment that row out of `panel.json` (and re-run both sides) —
the continuous panel alone already exercises every code path.

## Verified reference included

`reference_values.csv` in this folder is the captured output of `reference_values.m` run under MATLAB R2024-era Robust Control Toolbox. Anyone (with or without MATLAB) can therefore run `python crosscheck_matlab.py` and reproduce the agreement: all 13 pairs match the package to within 2.3e-6 (max), well below MATLAB's 1e-3 default accuracy. Regenerate it yourself with `reference_values.m` to re-verify on your own toolbox version.
