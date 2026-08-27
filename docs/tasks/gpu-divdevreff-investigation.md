# GPU `divDevReff` investigation

Date: 2026-08-27
Branch: `investigate/gpu-divdevreff`
Accepted baseline: `62984d4` (`Fix gpuSimpleFoam residual control semantics`)
Tolerance: `STAGE_TOL = 1e-12` (unchanged)

## Verdict

The original discrepancy was a real empty-patch contribution defect in the
gradient path, not a GPU z-indexing or tensor-contraction defect. Both
`fvc::gaussGrad` and the device scalar Gauss-gradient kernel accumulated
values from faces on an `empty` patch. `fvc::div`, `fvc::gradUBoundary`, and
the device tensor-divergence kernel already excluded those faces. The four
production fixes are retained; this document records the evidence and the
regression coverage.

The original first divergent stage was cell `grad(U)`. With the original
source behavior, the z-row measurements were:

| component | maximum absolute difference | maximum host magnitude | maximum GPU magnitude | relative difference | cell |
|---|---:|---:|---:|---:|---:|
| `zx` | `3.788e-13` | `3.103e-11` | `3.105e-11` | `1.221e-02` | `5816` |
| `zy` | `3.022e-14` | `1.503e-12` | `1.508e-12` | `2.011e-02` | `4579` |
| `zz` | `1.517e-14` | `8.337e-13` | `8.369e-13` | `1.819e-02` | `3532` |

The final z source was structurally near zero, which made its relative error
ill-conditioned. The accepted-baseline instrumented result was:

| source component | maximum absolute difference | maximum host magnitude | maximum GPU magnitude | relative difference | cell / face classification |
|---|---:|---:|---:|---:|---|
| `x` | `1.694066e-21` | `6.609412e-06` | `6.609412e-06` | `2.563111e-16` | `342`; internal, inlet, empty |
| `y` | `2.223461e-21` | `9.964522e-07` | `9.964522e-07` | `2.231378e-15` | `12222`; internal, wall, empty |
| `z` | `3.152534e-22` | `6.390980e-21` | `6.417854e-21` | `4.932786e-02` | `8464`; internal, empty |

Thus `4.933e-02` was noise on a quantity whose host magnitude was only
`6.391e-21`, not evidence of a meaningful z force or a GPU tensor defect.

## Reproduction and gate proof

The unchanged accepted-baseline command failed as reported:

```text
GPU divDevReff source (nCells=12225):
  V*div(sigma).x : 2.563e-16
  V*div(sigma).y : 2.231e-15
  V*div(sigma).z : 4.933e-02
FAIL
```

The legacy varying-`Uz` pitzDaily configuration is now a gate. It is retained
deliberately: it exercises the empty-face production path, even though the
field is not physically compatible with pitzDaily's empty front/back
patches. The corrected result is:

```text
legacy pitzDaily varying-Uz gated regression (nCells=12225, nInternalFaces=24170, nBoundaryFaces=25010):
  V*div(sigma).x abs 1.694e-21 host 6.609e-06 gpu 6.609e-06 rel 2.563e-16 cell 342 touches[internal=yes wall=no inlet=yes outlet=no empty=yes]
  V*div(sigma).y abs 2.012e-21 host 9.965e-07 gpu 9.965e-07 rel 2.019e-15 cell 12222 touches[internal=yes wall=yes inlet=no outlet=no empty=yes]
  V*div(sigma).z abs 6.771e-36 host 2.362e-20 gpu 2.362e-20 rel 2.867e-16 cell 12222 touches[internal=yes wall=yes inlet=no outlet=no empty=yes]
  first stage beyond relative 1e-12: none
PASS
```

### Reverting the device skip

I temporarily removed only `if (bndIsEmpty[kk]) continue;` from
`src/cuda/device_fvc.cu:100`, rebuilt `test_gpu_divdevreff` with `-j2`, and
ran the pitzDaily gate. It failed with exit code 1. The first vector-path
divergence was cell `grad(U)`; the final source was:

```text
cell grad(U), zx: abs 5.541e-11 host 5.506e-11 gpu 3.105e-11 rel 1.006e+00 at 12173
cell grad(U), zy: abs 2.858e-12 host 2.838e-12 gpu 1.508e-12 rel 1.007e+00 at 10877
cell grad(U), zz: abs 2.425e-12 host 2.269e-12 gpu 8.369e-13 rel 1.069e+00 at 12173
V*div(sigma).z abs 2.353e-20 host 2.362e-20 gpu 6.418e-21 rel 9.964e-01 cell 11531 touches[internal=yes wall=no inlet=no outlet=no empty=yes]
first stage beyond relative 1e-12: scalar Gauss gradient
FAIL
```

The harness names the scalar check first because it also guards the scalar
kernel. The first divergent stage in the requested tensor pipeline remains
cell `grad(U)`, as shown above. The device skip was then restored and rebuilt.

### Reverting the host scalar skip

I temporarily removed only `if (fp.type == "empty") continue;` from the
scalar overload at `src/finiteVolume/finiteVolume/fvc.cu:35`, rebuilt, and ran
the same command. It failed with exit code 1 because the independent scalar
Gauss check caught the reversion:

```text
scalar Gauss gradient (empty-face ordering, n=12225):
  z abs 3.207e-11 host 2.601e-12 gpu 3.207e-11 rel 1.233e+01 at 12173
legacy ...
  V*div(sigma).z abs 6.771e-36 host 2.362e-20 gpu 2.362e-20 rel 2.867e-16 ...
  first stage beyond relative 1e-12: scalar Gauss gradient
FAIL
```

The vector `divDevReff` portion stays green in this isolated reversion,
because the vector host overload at `fvc.cu:69` was not reverted. That is why
the scalar guard is present; the host-scalar reversion is not silently
reported as a vector-kernel failure. The host skip was restored, rebuilt, and
the registered `gpu_divdevreff` test passed again.

## Formulation and layout audit

The host and device implementations use the same conventions:

- `grad(U)` is packed in row-major tensor slots
  `xx, xy, xz, yx, yy, yz, zx, zy, zz`, with slot `q = 3*row + column`.
  Each scalar `gaussGrad(U_i)` is `(dU_i/dx, dU_i/dy, dU_i/dz)`, and
  `outer(Sf,U)` produces the OpenFOAM convention `G_ij = dU_j/dx_i`.
- `sigma = nuEff * dev2(transpose(grad(U)))`. `dev2` subtracts two-thirds of
  the trace from the diagonal; off-diagonal slots are unchanged. The test
  compares every one of the nine slots at cell and boundary stages.
- On internal faces, the owner contribution is positive and the neighbour
  contribution is negative. Face stress is linearly interpolated as
  `w*owner + (1-w)*neighbour`. Boundary contribution is positive and uses
  `dot(Sf, sigmaFace)`.
- `buildDeviceMesh` skips coupled-interface patches from the ordinary
  boundary-face list, appends the remaining patch faces in patch/face order,
  and uses `bndPerm` to gather them per cell. `boundaryTensorsSoA` applies the
  same coupled-interface exclusion. The test now asserts that each host
  tensor boundary snapshot has exactly `9 * dm.nBndFaces` scalars before any
  `metric()` call.
- Empty faces remain addressable in that common boundary ordering but carry an
  empty flag. `deviceGaussGrad`, `gradBKernel`, and `tensorDivKernel` exclude
  them. The host `fvc::gradUBoundary` and `fvc::div` also exclude them.
- For non-empty boundary faces, both implementations use
  `n = Sf/magSf`, `snGrad = (U_b-U_c)*deltaCoeffs`, and
  `gradB = gradCell + outer(n, snGrad - dot(n, gradCell))`. Empty patches
  return no boundary gradient. The scalar Gauss-gradient path uses evaluated
  face values directly; it does not apply the tensor boundary `snGrad`
  replacement.
- No `deltaCoeffs` or `snGrad` correction is used by the final tensor
  divergence itself; those quantities enter only the non-empty boundary
  gradient stage. There is no owner/neighbour sign or boundary-face indexing
  discrepancy in the stage comparisons.

Changing the host `fvc.cu` exclusion is therefore a faithfulness correction,
not moving the oracle. OpenFOAM's `emptyFvPatch` is a zero-size,
reduced-dimensional constraint, so an empty patch must not contribute a
finite-volume face sum. The local implementation already encoded that rule
in `fvc.cu:201` (`gradUBoundary`), `fvc.cu:239` (`div`), and
`device_divdevreff.cu:167` (`tensorDivKernel`) before this investigation.
The corresponding OpenFOAM references are
[emptyFvPatchField](https://github.com/OpenFOAM/OpenFOAM-dev/blob/master/src/finiteVolume/fields/fvPatchFields/constraint/empty/emptyFvPatchField.H)
and
[gaussGrad](https://github.com/OpenFOAM/OpenFOAM-dev/blob/master/src/finiteVolume/finiteVolume/gradSchemes/gaussGrad/gaussGrad.C).

## Validation configurations

The test prints all nine tensor components for cell `grad(U)`, boundary
`grad(U)`, cell `sigma`, and boundary `sigma`, plus all three source
components. The following are the exact final-source records (each row has
absolute difference, host magnitude, GPU magnitude, relative difference, and
the maximizing cell with touch classification):

| configuration | source component | absolute difference | host magnitude | GPU magnitude | relative difference | result |
|---|---|---:|---:|---:|---:|---|
| legacy pitzDaily, varying `Uz` | x | `1.694e-21` | `6.609e-06` | `6.609e-06` | `2.563e-16` | pass; cell 342, internal/inlet/empty |
| legacy pitzDaily, varying `Uz` | y | `2.012e-21` | `9.965e-07` | `9.965e-07` | `2.019e-15` | pass; cell 12222, internal/wall/empty |
| legacy pitzDaily, varying `Uz` | z | `6.771e-36` | `2.362e-20` | `2.362e-20` | `2.867e-16` | pass; cell 12222, internal/wall/empty |
| valid 2D pitzDaily, `Uz=0` | x | `1.694e-21` | `6.609e-06` | `6.609e-06` | `2.563e-16` | pass; cell 342, internal/inlet/empty |
| valid 2D pitzDaily, `Uz=0` | y | `1.716e-21` | `9.965e-07` | `9.965e-07` | `1.723e-15` | pass; cell 144, internal/inlet/empty |
| valid 2D pitzDaily, `Uz=0` | z | `6.019e-36` | `2.362e-20` | `2.362e-20` | `2.548e-16` | pass; cell 10832, internal/empty |
| structured 3D box | x | `2.168e-19` | `4.160e-04` | `4.160e-04` | `5.212e-16` | pass; cell 40, internal/wall/inlet |
| structured 3D box | y | `3.253e-19` | `1.723e-04` | `1.723e-04` | `1.887e-15` | pass; cell 23, internal/wall |
| structured 3D box | z | `2.711e-19` | `3.009e-04` | `3.009e-04` | `9.008e-16` | pass; cell 49, internal/wall/outlet |
| retained Issue 37 Ahmed mesh | x | `3.388e-20` | `5.741e-05` | `5.741e-05` | `5.902e-16` | pass; cell 183273, internal |
| retained Issue 37 Ahmed mesh | y | `3.632e-20` | `4.847e-05` | `4.847e-05` | `7.494e-16` | pass; cell 104637, internal |
| retained Issue 37 Ahmed mesh | z | `8.132e-20` | `1.242e-04` | `1.242e-04` | `6.546e-16` | pass; cell 237907, internal/wall |

The valid 2D case has empty patches and uses `Uz=0`; its z source magnitude
is `2.362e-20`, so z non-vacuity is not required. The box field has spatially
varying nonzero `Ux`, `Uy`, and `Uz`, with `fixedValue` inlet, `zeroGradient`
outlet, `noSlip` walls, and a symmetry-like patch. Its injected `1e-6` z
perturbation gives relative `3.323e-03` and is detected. The retained Ahmed
mesh's same negative control gives relative `8.050e-03` and is detected.

For `../validation/kEpsCorrect`, the retained branch reports and accepts its
structure rather than applying a constant 3D requirement:

```text
mesh structure: emptyPatches=yes zExtent 1.000e-03 zNonDegenerate=yes -> structurally-2D
structurally-2D bundle: z-source non-vacuity is not required
V*div(sigma).z abs 4.702e-38 host 3.262e-22 gpu 3.262e-22 rel 1.442e-16 ...
PASS
```

## Retained Ahmed path and solver impact

The archive `/home/rj/brae-eval/case-issue37-0005-L2.tar` was extracted
read-only into `/tmp/brae-task4-ahmed-review.Qriq99`. The archive was not
modified. Its mesh has `313625` cells, `932581` internal faces, `45016`
ordinary boundary faces, no empty patches, and z extent `1.400e+00`; it is
genuinely 3D. The exact retained-mesh stage output included:

```text
scalar Gauss gradient (n=313625):
  x abs 3.144e-13 host 6.169e+01 gpu 6.169e+01 rel 5.096e-15
  y abs 5.684e-13 host 6.240e+01 gpu 6.240e+01 rel 9.109e-15
  z abs 2.793e-13 host 6.910e+01 gpu 6.910e+01 rel 4.042e-15
cell grad(U), all nine components: maximum abs 9.095e-13, maximum host 3.583e+03, maximum GPU 3.583e+03, maximum relative 2.896e-16
boundary grad(U), all nine components: maximum abs 9.095e-13, maximum host 4.159e+03, maximum GPU 4.159e+03, maximum relative 2.495e-16
cell sigma, all nine components: maximum abs 8.882e-16, maximum host 3.583e+00, maximum GPU 3.583e+00, maximum relative 4.302e-16
boundary sigma, all nine components: maximum abs 8.882e-16, maximum host 4.159e+00, maximum GPU 4.159e+00, maximum relative 3.704e-16
final tensor-divergence source:
  x abs 3.388e-20 host 5.741e-05 gpu 5.741e-05 rel 5.902e-16
  y abs 3.632e-20 host 4.847e-05 gpu 4.847e-05 rel 7.494e-16
  z abs 8.132e-20 host 1.242e-04 gpu 1.242e-04 rel 6.546e-16
first stage beyond relative 1e-12: none
3D z negative control: injected 1e-6 gives relative 8.050e-03 -> detected
PASS
```

The maximum values above are reductions over the nine records; the binary
also prints each `xx` through `zz` record individually. The full captured
run was performed against the extracted `constant/polyMesh`, using a
deterministic field with nonzero, spatially varying `Ux`, `Uy`, and `Uz`.

The captured all-nine stage records for that run were:

```text
cell grad(U):
  xx abs 9.095e-13 host 3.141e+03 gpu 3.141e+03 rel 2.896e-16 at 312090
  xy abs 1.137e-13 host 8.278e+02 gpu 8.278e+02 rel 1.373e-16 at 257381
  xz abs 1.137e-13 host 8.512e+02 gpu 8.512e+02 rel 1.336e-16 at 257561
  yx abs 9.095e-13 host 3.583e+03 gpu 3.583e+03 rel 2.538e-16 at 278581
  yy abs 2.274e-13 host 9.453e+02 gpu 9.453e+02 rel 2.405e-16 at 312629
  yz abs 2.274e-13 host 9.638e+02 gpu 9.638e+02 rel 2.359e-16 at 272100
  zx abs 6.821e-13 host 3.418e+03 gpu 3.418e+03 rel 1.996e-16 at 306376
  zy abs 1.137e-13 host 9.040e+02 gpu 9.040e+02 rel 1.258e-16 at 256116
  zz abs 2.274e-13 host 9.422e+02 gpu 9.422e+02 rel 2.413e-16 at 286808
boundary grad(U):
  xx abs 9.095e-13 host 3.646e+03 gpu 3.646e+03 rel 2.495e-16 at 25059
  xy abs 1.705e-13 host 9.608e+02 gpu 9.608e+02 rel 1.775e-16 at 39261
  xz abs 1.137e-13 host 9.887e+02 gpu 9.887e+02 rel 1.150e-16 at 24777
  yx abs 4.547e-13 host 4.159e+03 gpu 4.159e+03 rel 1.093e-16 at 25709
  yy abs 2.274e-13 host 1.097e+03 gpu 1.097e+03 rel 2.072e-16 at 43788
  yz abs 2.274e-13 host 1.140e+03 gpu 1.140e+03 rel 1.994e-16 at 29389
  zx abs 9.095e-13 host 3.975e+03 gpu 3.975e+03 rel 2.288e-16 at 44681
  zy abs 1.137e-13 host 1.051e+03 gpu 1.051e+03 rel 1.081e-16 at 24562
  zz abs 1.137e-13 host 1.096e+03 gpu 1.096e+03 rel 1.038e-16 at 24420
cell sigma:
  xx abs 4.441e-16 host 1.438e+00 gpu 1.438e+00 rel 3.088e-16 at 312090
  xy abs 8.882e-16 host 3.583e+00 gpu 3.583e+00 rel 2.479e-16 at 278581
  xz abs 8.882e-16 host 3.418e+00 gpu 3.418e+00 rel 2.599e-16 at 306376
  yx abs 1.134e-16 host 8.278e-01 gpu 8.278e-01 rel 1.370e-16 at 259654
  yy abs 8.882e-16 host 2.064e+00 gpu 2.064e+00 rel 4.302e-16 at 257561
  yz abs 1.388e-16 host 9.040e-01 gpu 9.040e-01 rel 1.535e-16 at 256457
  zx abs 1.388e-16 host 8.512e-01 gpu 8.512e-01 rel 1.630e-16 at 263848
  zy abs 2.220e-16 host 9.638e-01 gpu 9.638e-01 rel 2.304e-16 at 272100
  zz abs 4.441e-16 host 2.284e+00 gpu 2.284e+00 rel 1.944e-16 at 257381
boundary sigma:
  xx abs 5.551e-16 host 1.669e+00 gpu 1.669e+00 rel 3.325e-16 at 32853
  xy abs 8.882e-16 host 4.159e+00 gpu 4.159e+00 rel 2.136e-16 at 29387
  xz abs 8.882e-16 host 3.975e+00 gpu 3.975e+00 rel 2.235e-16 at 37174
  yx abs 1.665e-16 host 9.608e-01 gpu 9.608e-01 rel 1.733e-16 at 39261
  yy abs 8.882e-16 host 2.398e+00 gpu 2.398e+00 rel 3.704e-16 at 25059
  yz abs 2.220e-16 host 1.051e+00 gpu 1.051e+00 rel 2.112e-16 at 30537
  zx abs 2.220e-16 host 9.887e-01 gpu 9.887e-01 rel 2.246e-16 at 27316
  zy abs 2.220e-16 host 1.140e+00 gpu 1.140e+00 rel 1.948e-16 at 29389
  zz abs 8.882e-16 host 2.651e+00 gpu 2.651e+00 rel 3.350e-16 at 25059
```

These are the raw all-nine records emitted by the test for the retained
Ahmed run; the `scalar Gauss gradient` block immediately above the stage
records contains its three scalar components.

`device_simple_foam.cu:1056` calls `deviceDivDevReff` from
`solveMomentumPredictor`, so the corrected production path is used by
`simpleFoam`. `gpuPimpleFoam.cu:893` calls `DeviceSimpleSolver::pimpleStep`,
which calls the same `solveMomentumPredictor` at
`device_simple_foam.cu:3473`; the path is therefore also used by
`pimpleFoam` when its momentum predictor is enabled.

The retained Ahmed mesh has no empty patches, so the specific empty-face
correction is numerically neutral on that case: all four stages and all three
source components agree to the displayed floating-point roundoff, and the
nonzero z control is detected. This establishes no defect-driven Ahmed Cd
change from this patch, but it is not a claim of a converged Ahmed Cd based
on a synthetic operator test.

For an additional bounded production check, I extracted the same archive
twice without its archived `postProcessing` directory, changed only the
scratch copies to `endTime 3` and `writeInterval 1`, and ran the accepted
baseline binary and the fixed binary with `brae -case`. Both returned 0,
reached the three-iteration limit, and produced three clean coefficient
samples. The fixed-minus-baseline maximum absolute deltas over the three
samples were:

| quantity | maximum absolute delta over iterations 1--3 | final baseline | final fixed |
|---|---:|---:|---:|
| `Cd` | `4.0676e-07` | `-8.667105402889465e+02` | `-8.667105398821867e+02` |
| `Cl` | `2.8386e-07` | `2.707430075161824e+02` | `2.707430078000454e+02` |
| `Cm` | `6.0007e-08` | `-1.873276606389672e+02` | `-1.873276606989743e+02` |
| `Fx` | `2.2785e-08` | `-4.854965762482563e+01` | `-4.854965760204058e+01` |
| `Fy` | `4.1615e-10` | `4.896285878666162e-04` | `4.896290040153406e-04` |
| `Fz` | `1.5901e-08` | `1.516594030902647e+01` | `1.516594032492734e+01` |

The printed U/p residuals were identical; the final omega/k residuals differed
only in the last displayed digits (`0.00413087` vs `0.00413101` and
`0.00962754` vs `0.00962748`). These are the expected small GPU reduction
ordering differences on a genuinely 3D mesh with no empty faces, not an
empty-patch effect. The run stopped at the requested iteration limit and was
not treated as converged.

## Scalar Gauss-gradient scope and neutrality

`device_fvc.cu:100` is shared by every device scalar Gauss gradient. The
repository call sites include pressure gradients in the momentum/pressure
path, gradient-based limiter and convection support, turbulence scalar
gradients, and wall-distance-related gradient consumers. The scalar guard in
this test proves that removing the skip is observable rather than a dead
line.

For compatible 2D vector fields, opposing front/back area vectors cancel and
the z tensor rows that should be zero are zero; the valid 2D source z
magnitude is `2.362e-20` versus x `6.609e-06`. The deliberately varying-`Uz`
legacy case is intentionally not used as a physical 2D neutrality claim; it
is the empty-face regression gate. The full-suite comparison below provides
the solver-level neutrality check for the shared scalar-gradient change.

## Verification output

The required direct commands were run with the MPI runtime library path and
`-j2` build/test limits:

```text
nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader
NVIDIA GeForce RTX 3090, 580.173.02, 1 MiB

nice -n 19 cmake --build build --target test_gpu_divdevreff -j2
[100%] Built target test_gpu_divdevreff

./test_gpu_divdevreff ../validation/pitzDaily
PASS

./test_gpu_divdevreff ../validation/kEpsCorrect
mesh structure: emptyPatches=yes zExtent 1.000e-03 zNonDegenerate=yes -> structurally-2D
PASS

./test_gpu_divdevreff /no/such/path
gpu_divdevreff ERROR: unrecognized or missing case path '/no/such/path' (expected /no/such/path/constant/polyMesh)
exit code 2
```

The registered CTest invocation was also run verbatim in the build directory:

```text
nice -n 19 ctest --test-dir build -R '^gpu_divdevreff$' -V
1/1 Test #214: gpu_divdevreff ...................   Passed    0.41 sec
100% tests passed out of 1
```

The full unfiltered command completed with rc 8:

```text
nice -n 19 ctest -j2 --output-on-failure
80% tests passed, 48 tests failed out of 240
Total Test time (real) = 215.39 sec
```

The focused rows in that run were:

```text
force_history .................... Passed
divdevreff ....................... Passed
simple_run ....................... Passed
gpu_momentum ..................... Passed
gpu_simple ....................... Passed
gpu_resident_turb ............... Passed
divdevreff_gradscheme ........... Passed
gpu_divdevreff .................. Passed
simple_step ...................... Passed
residual_control ................ Passed
force_coeffs .................... Passed
simple_turbulent_full ........... Passed
momentum_predictor .............. Passed
residual_control_driver ......... Failed (AssertionError: 0.0118743)
```

Failure names were compared, not CTest numbers, because the working tree has
additional registrations relative to the retained baseline log. The exact
set difference was:

```text
review minus ~/brae-eval/ctest-cuda126.log:
decompose_exchange_np1 np2 np4 np8
gpu_bind_np1 np2 np4 np8
gpu_decompose_np1 np2 np4 np8
interface_exchange_np1 np2 np4 np8
local_assembly_np1 np2 np4 np8
local_convection_np1 np2 np4 np8
local_mesh_np1 np2 np4 np8
proc_delta_np1 np2 np4 np8
proc_interp_np1 np2 np4 np8
processor_field_np1 np2 np4 np8
residual_control_driver

baseline minus review:
gpu_divdevreff
```

The `*_np{1,2,4,8}` failures all report the environment's
`mpiexec.openmpi: Error: unknown option "--oversubscribe"`. The remaining
`residual_control_driver` failure is the known other-task assertion drift;
the run reports the current initial `Uz` residual as `0.0118743`. There were
no new non-MPI, non-`residual_control_driver` failures. The archived
baseline's `restart_continuity`, `tracer_transport`, `dilu`, `brae_run`,
`gpu_cluster`, `gpu_amg_fused`, and `gpu_cluster_spmv` failures remain.

## Files and remaining uncertainty

The final repository changes are limited to the existing four production
fixes, the investigation test, and this evidence document. `CMakeLists.txt`
was inspected but did not need a change: its registered test continues to
run `validation/pitzDaily`, while the binary accepts an explicitly supplied
retained bundle for the manual Ahmed-path gate.

The production simpleFoam/pimpleFoam operator path is affected on meshes that
actually contain empty patches; the retained Ahmed path is exercised and is
not affected by this specific correction because it has no empty patches.
The bounded retained Ahmed comparison supports no defect-driven Cd change
from this patch, but it is only a three-iteration, non-converged run and
does not establish a final physical Ahmed coefficient.
