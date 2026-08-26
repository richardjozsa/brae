# Task: CFD Critic OpenFOAM v2406 compatibility and observability

## Status

Proposed evaluation work for the `richardjozsa/brae` fork. This is not a
production-integration task. Negative or numerically discrepant results must be
retained and reported.

## Objective

Make Brae capable of evaluating retained CFD Critic Ahmed-body cases with the
same convergence and reporting semantics used by OpenFOAM v2406. The immediate
goal is to measure whether Brae reproduces converged drag coefficients and
steady-solver limit cycles on identical meshes, while preserving Brae's GPU
speed advantage.

Do not tune solver settings to reproduce the known OpenFOAM answers.

## Existing evidence

Evaluation commit: `8d5a70613cd0cc7335962ce61bd09155bc5cdf5e`.

- Brae builds on the RTX 3090 using CUDA Toolkit 12.6. CUDA 12.2 fails because
  `cudaGraphConditionalHandle` is unavailable.
- Three adapted Issue 37 cases completed 2,000 iterations with identical cell
  counts and approximately 10.5x to 10.7x raw wall-time speedup:

  | Case | OpenFOAM Cd | Brae final Cd | Difference | OpenFOAM wall | Brae wall |
  |---|---:|---:|---:|---:|---:|
  | `ahmed_run_0005` | 0.278713071 | 0.276840900 | -0.672% | 970.84 s | 90.73 s |
  | `ahmed_run_0001` | 0.291239963 | 0.288150600 | -1.061% | 1121.35 s | 106.85 s |
  | `ahmed_run_0002` | 0.413077043 | 0.413497600 | +0.102% | 1190.28 s | 112.43 s |

- These were diagnostic adapted inputs, not byte-identical reproductions.
- An unmodified retained case fails with:

  ```text
  brae ERROR: brae: unsupported RASModel ''
  ```

  The OpenFOAM v2406 dictionary uses `RAS { model kOmegaSST; }`, while Brae
  expects `RASModel`.
- Steady `simpleFoam` currently reports only a final Cd, so the result cannot be
  evaluated with a convergence window or used to measure a limit cycle.
- No usable Brae yPlus function-object/output path was found.
- `k` and `omega` residual controls were not honored in the adapted runs.
- The test suite registered 235 tests: 197 passed, 30 skipped, and 8 failed.
  Most failures are missing fixtures or unsupported cluster launch on Ampere,
  but `gpu_divdevreff` reported a material component error of approximately
  `4.933e-02` and requires investigation.

## Required work

### 1. OpenFOAM v2406 turbulence dictionary compatibility

- Accept both `model kOmegaSST` and the existing `RASModel` spelling.
- Preserve existing input behavior.
- Add parser tests covering both forms and clear errors for unsupported models.
- A retained CFD Critic case must start without editing its dictionaries.

### 2. Per-iteration force-coefficient history

- For steady `simpleFoam`, evaluate configured `forceCoeffs` every iteration.
- At minimum output iteration, Cd, force components, residuals, and elapsed wall
  time in a stable machine-readable format.
- Use the case's configured reference area, density, velocity, drag direction,
  patches, and moment reference. Do not silently substitute defaults.
- Keep the current final summary, but derive it from the recorded history.
- Document coefficient normalization and compare it directly with OpenFOAM.

### 3. Convergence-control semantics

- Parse and apply `residualControl` for `U`, `p`, `k`, and `omega`.
- Document whether values represent initial, final, or relative residuals.
- Record why a run stopped: convergence, iteration limit, numerical failure, or
  user interruption.
- Never label an iteration-limited run as converged.

### 4. yPlus field and patch summaries

- Calculate yPlus using the active turbulence/wall-function model.
- Export the raw wall-face field with unambiguous iteration/time provenance.
- Report area-weighted summaries for named patches, including `body` and
  `lowerWall`.
- Never reuse pre-existing OpenFOAM yPlus output from an input bundle.
- Validate face-level values and patch summaries against OpenFOAM on the same
  mesh, fields, and iteration.

### 5. Numerical test investigation

- Reproduce and diagnose `gpu_divdevreff` on RTX 3090 compute capability 8.6.
- Determine whether its failing path is exercised by `simpleFoam` or
  `pimpleFoam` for the retained cases.
- Add a regression test for any correction.
- Classify unsupported cluster-launch tests separately from numerical failures;
  do not make a red test green merely by weakening its tolerance.

## Validation cases

Use retained case bundles read-only. Do not modify CFD Critic database rows,
object-store objects, or `/home/rj/development/cfd`.

### Primary: converged Issue 37 L2 cases

| Case | Expected OpenFOAM Cd | Cells |
|---|---:|---:|
| `ahmed_run_0005` | 0.278713071 | 313625 |
| `ahmed_run_0001` | 0.291239963 | 369823 |
| `ahmed_run_0002` | 0.413077043 | 383092 |

### Secondary: nonstationary Issue 42 steady cases

| Case | OpenFOAM window mean | Peak-to-peak | Cells |
|---|---:|---:|---:|
| `ahmed_run_0005` | 0.306467 | 20.5% | 895675 |
| `ahmed_run_0001` | 0.308364 | 15.8% | 982401 |
| `ahmed_run_0002` | 0.429176 | 15.1% | 979764 |

For Issue 42, compare window mean, amplitude, period, residual histories, and
yPlus. A final endpoint Cd is not a valid comparison with a window mean.

## Experiment protocol

1. Run the byte-identical input first and retain any failure.
2. Record Brae commit, case-bundle digest, CUDA/toolchain versions, GPU model,
   cell count, iteration count, stopping reason, wall time, and peak GPU memory.
3. Run 2,000 iterations, then 4,000 and 8,000 only when the history has not met
   a declared terminal condition.
4. Calculate the Cd estimand from the final 25% of at least 100 finite samples.
5. A stationary result requires absolute drift <= 0.005 and relative range
   <= 0.02. A nonstationary run has no converged scalar Cd; report its window
   mean and oscillation characteristics instead.
6. Repeat at least one Issue 37 case and one Issue 42 case to measure numerical
   repeatability.
7. Report accuracy beside speed for every comparison.

## Acceptance gates

### PASS for internal cohort evaluation

- All three Issue 37 meshes load byte-identically and report the expected cell
  counts.
- Their window Cd values are within 2% of OpenFOAM and meet the stationary-run
  criteria.
- Repeated histories and reported summaries are reproducible within a declared
  numerical tolerance.
- yPlus fields and area-weighted patch summaries agree sufficiently with
  OpenFOAM to support the same wall-treatment conclusion.
- The `gpu_divdevreff` failure is fixed or demonstrated not to affect the
  exercised solver path, with evidence.
- Wall time and GPU measurements are recorded on the same meshes and iteration
  budgets as their OpenFOAM comparisons.

### REFORMULATE

- Issue 37 remains within 2%, but one or more cases exhibit a repeatable limit
  cycle rather than stationary convergence; or
- Issue 42 produces a repeatable limit cycle whose mean or amplitude differs
  materially enough to require a solver-semantics investigation.

### STOP

- Any Issue 37 Cd discrepancy exceeds 5% after estimator and normalization
  equivalence are verified.
- Mesh cell counts differ.
- Results depend materially on arbitrary iteration endpoint selection.
- yPlus or force normalization cannot be reconciled with OpenFOAM.
- A numerical kernel failure affects the retained-case solver path.

## Agent workflow

Keep changes reviewable and avoid overlapping ownership:

1. Parser compatibility and tests.
2. Force-coefficient history and normalization tests.
3. Residual-control semantics and stopping-reason tests.
4. yPlus implementation and OpenFOAM comparison fixtures.
5. GPU numerical failure diagnosis.
6. Integrated retained-case benchmark and evidence report.

For every change, inspect the full diff, run focused tests, then build with at
most two jobs unless the host constraints are explicitly changed. Check GPU
availability before running a case. Do not stop containers or workers without
current explicit authorization.

## Out of scope

- Production deployment or CFD Critic scheduler integration.
- Supporting arbitrary OpenFOAM function objects or turbulence models.
- Changing the retained OpenFOAM evidence or expected values.
- Tuning coefficients or discretization specifically toward known Cd targets.
- Multi-GPU execution.
