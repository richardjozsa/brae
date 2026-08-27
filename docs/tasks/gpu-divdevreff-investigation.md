# GPU `divDevReff` investigation: rebuilt evidence

Date: 2026-08-27
Branch: `investigate/gpu-divdevreff`
Accepted baseline: `62984d4` (`Fix gpuSimpleFoam residual control semantics`)
HEAD under test: `b74820e` (`close gpu divdevreff review findings`)
Tolerance: `STAGE_TOL = 1e-12` (unchanged)

This is a verification refresh. The previous full-suite and Ahmed solver
records were collected with stale executables. The all-target rebuild and all
measurements below were redone after the four empty-face fixes in `b74820e`.
No production code was changed during this refresh.

## Finding

The original `4.933e-02` was caused by empty-patch faces being included in
both the host and device Gauss-gradient paths, while the divergence paths
already excluded those faces. It was not a tensor-layout, owner/neighbour
sign, boundary-index, or GPU z-indexing defect. The first divergent stage was
cell `grad(U)`.

The historical failing result was:

```text
GPU divDevReff source (nCells=12225):
  V*div(sigma).x : 2.563e-16
  V*div(sigma).y : 2.231e-15
  V*div(sigma).z : 4.933e-02
FAIL
```

Its original final-z diagnostic was already structurally tiny:

| component | maximum absolute difference | maximum host magnitude | maximum GPU magnitude | relative difference | cell |
|---|---:|---:|---:|---:|---:|
| `z` | `3.152534e-22` | `6.390980e-21` | `6.417854e-21` | `4.932786e-02` | `8464`, internal/empty |

Thus the relative number was noise on a near-zero quantity, not a meaningful
z source. After the fix, the rebuilt legacy gate reports:

```text
V*div(sigma).x abs 1.694e-21 host 6.609e-06 gpu 6.609e-06 rel 2.563e-16
V*div(sigma).y abs 2.012e-21 host 9.965e-07 gpu 9.965e-07 rel 2.019e-15
V*div(sigma).z abs 6.771e-36 host 2.362e-20 gpu 2.362e-20 rel 2.867e-16
first stage beyond relative 1e-12: none
PASS
```

For the same rebuilt legacy run, the stage maxima were:

| stage | maximum absolute difference | maximum host magnitude | maximum GPU magnitude | relative difference |
|---|---:|---:|---:|---:|
| cell `grad(U)` | `2.132e-12` | `6.289e+03` | `6.289e+03` | `3.389e-16` |
| boundary `grad(U)` | `2.132e-12` | `1.232e+04` | `1.232e+04` | `1.730e-16` |
| cell sigma | `2.137e-15` | `6.289e+00` | `6.289e+00` | `3.398e-16` |
| boundary sigma | `2.137e-15` | `1.232e+01` | `1.232e+01` | `1.734e-16` |

The four production hunks in `b74820e` remain unchanged. The host exclusion
in `fvc.cu` is a faithfulness correction, not an oracle move: OpenFOAM's
`emptyFvPatch` has zero face count in a reduced-dimensional mesh, so an empty
patch contributes no finite-volume faces. This code already had the same
exclusion in `fvc.cu:201` (`gradUBoundary`), `fvc.cu:239` (`div`), and
`device_divdevreff.cu:167` (`tensorDivKernel`) before this investigation.
See [OpenFOAM emptyFvPatchField](https://github.com/OpenFOAM/OpenFOAM-dev/blob/master/src/finiteVolume/fields/fvPatchFields/constraint/empty/emptyFvPatchField.H)
and [OpenFOAM gaussGrad](https://github.com/OpenFOAM/OpenFOAM-dev/blob/master/src/finiteVolume/finiteVolume/gradSchemes/gaussGrad/gaussGrad.C).

## Stage and formulation evidence

The stage comparisons use row-major tensor slots
`xx, xy, xz, yx, yy, yz, zx, zy, zz`, with `q = 3*row + column`.
Each `gaussGrad(U_i)` is `(dU_i/dx, dU_i/dy, dU_i/dz)`, so the host and
device use the OpenFOAM convention `G_ij = dU_j/dx_i`. The stress is
`nuEff * dev2(transpose(grad(U)))`; transpose swaps the off-diagonal slots
and `dev2` subtracts two-thirds of the trace from each diagonal.

Internal-face contributions use positive owner and negative neighbour signs.
Boundary divergence uses `dot(Sf, sigmaFace)`. Non-empty boundary gradients
use `n = Sf/magSf`, `snGrad = (U_b-U_c)*deltaCoeffs`, and the normal correction
to the cell gradient. The final tensor divergence itself does not use
`deltaCoeffs` or `snGrad`; they enter only the boundary-gradient stage.

`buildDeviceMesh` skips coupled-interface patches from the ordinary boundary
list and appends the remaining faces in patch/face order. The test helper
mirrors that rule and now checks that its flattened host boundary tensor size
is exactly `9 * dm.nBndFaces` before calling `metric()`. Empty faces remain
addressable in that common ordering but are excluded from the cell-gradient,
boundary-gradient, and tensor-divergence sums.

The original-device-skip reversion proof remains:

```text
removed device_fvc.cu:100 empty-face skip
cell grad(U), zx: abs 5.541e-11 host 5.506e-11 gpu 3.105e-11 rel 1.006e+00
cell grad(U), zy: abs 2.858e-12 host 2.838e-12 gpu 1.508e-12 rel 1.007e+00
cell grad(U), zz: abs 2.425e-12 host 2.269e-12 gpu 8.369e-13 rel 1.069e+00
V*div(sigma).z abs 2.353e-20 host 2.362e-20 gpu 6.418e-21 rel 9.964e-01
first stage beyond relative 1e-12: scalar Gauss gradient
FAIL (exit 1)
```

The original-host-scalar-skip reversion also failed independently:

```text
removed fvc.cu:35 empty-face skip
scalar Gauss gradient z abs 3.207e-11 host 2.601e-12 gpu 3.207e-11 rel 1.233e+01
V*div(sigma).z abs 6.771e-36 host 2.362e-20 gpu 2.362e-20 rel 2.867e-16
FAIL (exit 1)
```

The second reversion leaves the vector `divDevReff` stage green because only
the scalar host overload was reverted; the independent scalar check is what
proves that hunk is live.

## Metric definition and validation cases

The `Metric::relative` calculation in `tests/test_gpu_divdevreff.cu:53-55`
is a field-norm gate, not a per-cell relative error. For one component it
computes:

```text
hostMag = max over all field entries |host[i]|
absDiff = max over all field entries |gpu[i] - host[i]|
relative = absDiff / hostMag
```

The reported host and GPU magnitudes must therefore be read beside every
relative number. The 3D negative control injects `1e-6` into one z-source
entry. The z field maximum is `3.009e-04`, so the injected signal is
`1e-6 / 3.009e-04 = 3.323e-03`, approximately `0.3%` of the z field max;
the gate detects it.

The rebuilt direct runs produced these final-source records:

| configuration | source | absolute difference | host magnitude | GPU magnitude | relative difference | max cell / touch classification |
|---|---|---:|---:|---:|---:|---|
| legacy pitzDaily, varying `Uz` | x | `1.694e-21` | `6.609e-06` | `6.609e-06` | `2.563e-16` | `342`, internal/inlet/empty |
| legacy pitzDaily, varying `Uz` | y | `2.012e-21` | `9.965e-07` | `9.965e-07` | `2.019e-15` | `12222`, internal/wall/empty |
| legacy pitzDaily, varying `Uz` | z | `6.771e-36` | `2.362e-20` | `2.362e-20` | `2.867e-16` | `12222`, internal/wall/empty |
| valid 2D pitzDaily, `Uz=0` | x | `1.694e-21` | `6.609e-06` | `6.609e-06` | `2.563e-16` | `342`, internal/inlet/empty |
| valid 2D pitzDaily, `Uz=0` | y | `1.716e-21` | `9.965e-07` | `9.965e-07` | `1.723e-15` | `144`, internal/inlet/empty |
| valid 2D pitzDaily, `Uz=0` | z | `6.019e-36` | `2.362e-20` | `2.362e-20` | `2.548e-16` | `10832`, internal/empty |
| structured 3D box | x | `2.168e-19` | `4.160e-04` | `4.160e-04` | `5.212e-16` | `40`, internal/wall/inlet |
| structured 3D box | y | `3.253e-19` | `1.723e-04` | `1.723e-04` | `1.887e-15` | `23`, internal/wall |
| structured 3D box | z | `2.711e-19` | `3.009e-04` | `3.009e-04` | `9.008e-16` | `49`, internal/wall/outlet |
| retained Issue 37 Ahmed mesh | x | `3.388e-20` | `5.741e-05` | `5.741e-05` | `5.902e-16` | `183273`, internal |
| retained Issue 37 Ahmed mesh | y | `3.632e-20` | `4.847e-05` | `4.847e-05` | `7.494e-16` | `104637`, internal |
| retained Issue 37 Ahmed mesh | z | `8.132e-20` | `1.242e-04` | `1.242e-04` | `6.546e-16` | `237907`, internal/wall |

The valid 2D field has `Uz=0` and is solution-direction-compatible. The
legacy field deliberately retains spatially varying `Uz` as the negative
control that catches the shared empty-face defect. The structured 3D box has
non-zero, spatially varying `Ux`, `Uy`, and `Uz`; its boundary families are
`noSlip` wall, symmetry-like/slip plane, fixedValue inlet, and zeroGradient
outlet. All stage checks report no relative error above `1e-12`.

`../validation/kEpsCorrect` is structurally 2D and is not rejected for a
near-zero z source:

```text
mesh structure: emptyPatches=yes zExtent 1.000e-03 zNonDegenerate=yes -> structurally-2D
structurally-2D bundle: z-source non-vacuity is not required
V*div(sigma).x abs 4.235e-22 host 1.248e-06 gpu 1.248e-06 rel 3.394e-16
V*div(sigma).y abs 2.118e-21 host 5.963e-06 gpu 5.963e-06 rel 3.551e-16
V*div(sigma).z abs 4.702e-38 host 3.262e-22 gpu 3.262e-22 rel 1.442e-16
first stage beyond relative 1e-12: none
PASS
```

## OpenFOAM-gated 2D cases

The repository contains no `bump2D` or `turbulentFlatPlate` fixture or test
registration. The actual empty-patch tutorial-like cases and their rebuilt
CTest results were:

| case | registered gates run | result |
|---|---|---|
| `pitzDaily` | `gpu_fvc`, `gpu_pressure`, `gpu_momentum`, `gpu_simple`, `gpu_step`, `gpu_boundary`, `gpu_boundary_vec`, `gpu_bndflux`, `gpu_resident`, `gpu_relax`, `gpu_divdevreff`, `gpu_kepsilon`, `gpu_kepsilon_correct`, `gpu_resident_turb`, `gpu_amg`, `gpu_driver`, `gpu_pimple`, `pimple_run`, `momentum_predictor`, `pimple_dispatch`, `force_history`, `brae_benchmark`, `linear_upwind_const`, `overset_refused`, `gpu_spmv`, `gpu_pcg`, `gpu_fvm`, `gpu_benchmark`, `gpu_amg_graph`, `gpu_loop_graph`, `gpu_crossover` | all listed passed; `gpu_divdevreff` passed in `0.31 s`; `residual_control_driver` failed as recorded below |
| `pitzDaily282` | `gpu_komega_sst`, `forces`, `fields_bc`, `wall_function`, `fvc_grad`, `kepsilon` | all passed |
| `pitzDailyTurb` | `turb_fixedpoint_12k`, `simple_turbulent_full` | both passed; full run `213.06 s` |
| `pitzDailyTurbBig` | `turb_fixedpoint_49k` | passed |
| `kEpsCorrect` | `kepsilon_correct`, `divdevreff` | both passed; the direct divDevReff run reports structurally 2D |
| `lmFlatPlate` | `komegasstlm_flatplate` | passed |
| `bump2D` | no fixture or registration | not present |
| `turbulentFlatPlate` | no fixture or registration | not present |

The MPI topology/field tests (`interface_exchange_np*`,
`gpu_bind_np*`, `gpu_decompose_np*`, `decompose_exchange_np*`,
`local_mesh_np*`, `proc_delta_np*`, `local_assembly_np*`,
`local_convection_np*`, `processor_field_np*`, and `proc_interp_np*`) fail
before numerical validation because this host's
`mpiexec.openmpi` rejects CMake's `--oversubscribe` option. The scalar-gradient
and solver gates above are serial and passed except for the separately
discussed `residual_control_driver`.

The pitzDaily cluster tests `gpu_amg_fused` and `gpu_cluster_spmv` also fail
with the known unsupported-device/cluster-misconfiguration errors; they do
not exercise the empty-face scalar-gradient gate.

The registered OpenFOAM comparison wrappers
`rho_vs_openfoam`, `ctl_vs_openfoam`, `ke_vs_openfoam`, `fr_vs_openfoam`,
`tp_vs_openfoam`, `lu_vs_openfoam`, `suth_vs_openfoam`, `rhoE_vs_openfoam`,
`wc_vs_openfoam`, `turbswitch_vs_openfoam`, `mx_vs_openfoam`,
`rhotiming_vs_openfoam`, `hf_vs_openfoam`, `transonic_vs_openfoam`,
`naca_vs_openfoam`, `sst_vs_openfoam`, `luturb_vs_openfoam`, `sa_vs_openfoam`,
`io_vs_openfoam`, `ti_vs_openfoam`, `ke2_vs_openfoam`, `rx_vs_openfoam`,
`pm_vs_openfoam`, `roundtrip_vs_openfoam`, `flowrate_vs_openfoam`,
`restart_vs_openfoam`, `energy_vs_openfoam`, and `bc_vs_openfoam` were all
skipped because their external OpenFOAM gates are not configured here. No
OpenFOAM-gated result moved in this run.

## Rebuilt solver residual result

The residual driver at `CMakeLists.txt:1122` asserts the final observed `Uz`
initial residual is in `[0.45, 0.65]`. I built `brae` from `62984d4` in a
separate worktree at `/tmp/brae-task4-followup-baseline`, using the same CUDA
12.6 / SM86 configuration and explicit MPI library paths, then ran the same
driver twice per binary:

```text
62984d4 run 1:
converged at iteration 238 (<400): OK
valid U components Ux,Uy and Uz initial residual 0.586352: OK
epsilon/k initial residuals at convergence epsilon=0.00044484, k=0.000986499: OK
unknown-only residualControl reaches endTime (400 iterations): OK

62984d4 run 2:
converged at iteration 238 (<400): OK
valid U components Ux,Uy and Uz initial residual 0.577267: OK
epsilon/k initial residuals at convergence epsilon=0.000444303, k=0.000990942: OK
unknown-only residualControl reaches endTime (400 iterations): OK

b74820e rebuilt HEAD run 1:
Traceback (most recent call last):
  File "<stdin>", line 14, in <module>
AssertionError: 0.0119126

b74820e rebuilt HEAD run 2:
Traceback (most recent call last):
  File "<stdin>", line 14, in <module>
AssertionError: 0.0117552
```

The baseline values vary by `9.085e-03` but stay in the asserted range; HEAD
values vary by `1.574e-04` and are far outside it. This controlled A/B shows
that the large movement is associated with the b74820e code difference, not
the small baseline run-to-run drift. It is a real production `simpleFoam`
behavior change on the pitzDaily empty-patch path. The driver assertion is
not changed in this task.

## Retained Ahmed path and bounded force evidence

The archive `/home/rj/brae-eval/case-issue37-0005-L2.tar` was never modified.
It was extracted read-only into a fresh temporary directory for the manual
diagnostic. Its mesh has `313625` cells, `932581` internal faces, `45016`
ordinary boundary faces, no empty patches, and z extent `1.400e+00`.

The rebuilt `test_gpu_divdevreff` invocation against that extraction returned
zero and printed the following per-stage maxima:

```text
mesh structure: emptyPatches=no zExtent 1.400e+00 zNonDegenerate=yes -> genuinely-3D
scalar Gauss gradient:
  x abs 3.144e-13 host 6.169e+01 gpu 6.169e+01 rel 5.096e-15
  y abs 5.684e-13 host 6.240e+01 gpu 6.240e+01 rel 9.109e-15
  z abs 2.793e-13 host 6.910e+01 gpu 6.910e+01 rel 4.042e-15
cell grad(U): maximum abs 9.095e-13, host 3.583e+03, GPU 3.583e+03, rel 2.896e-16
boundary grad(U): maximum abs 9.095e-13, host 4.159e+03, GPU 4.159e+03, rel 2.495e-16
cell sigma: maximum abs 8.882e-16, host 3.583e+00, GPU 3.583e+00, rel 4.302e-16
boundary sigma: maximum abs 8.882e-16, host 4.159e+00, GPU 4.159e+00, rel 3.704e-16
final tensor-divergence source:
  x abs 3.388e-20 host 5.741e-05 gpu 5.741e-05 rel 5.902e-16
  y abs 3.632e-20 host 4.847e-05 gpu 4.847e-05 rel 7.494e-16
  z abs 8.132e-20 host 1.242e-04 gpu 1.242e-04 rel 6.546e-16
first stage beyond relative 1e-12: none
3D z negative control: injected 1e-6 gives relative 8.050e-03 -> detected
PASS
```

The exact all-nine records emitted by that rebuilt Ahmed invocation were:

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

This retained Ahmed invocation is manual and uses a temporary extraction that
no longer exists as a stable fixture. It is not regression-protected:
`CMakeLists.txt:1567` registers only
`validation/pitzDaily`. The current diagnostic run does not change that
registration status.

`device_simple_foam.cu:1056` calls `deviceDivDevReff` from the `simpleFoam`
momentum predictor. `gpuPimpleFoam.cu:893` reaches the same predictor through
`DeviceSimpleSolver::pimpleStep` at `device_simple_foam.cu:3473`, so the
operator is also on the `pimpleFoam` path. The retained Ahmed mesh has no
empty patches, so the specific empty-face branches are not exercised there.
Thus `divDevReff` is exercised by both solver paths on the retained Ahmed
mesh, but the empty-face condition fixed here is not; that particular defect
cannot contribute an Ahmed source or Cd change through an empty face.

To establish a noise floor, I extracted the same archive twice without its
archived `postProcessing`, changed only each scratch `controlDict` to
`endTime 3` and `writeInterval 1`, and ran the rebuilt HEAD `brae` twice.
The raw final coefficient rows were:

```text
HEAD run 1, time 3:
Cd -866.71053984382888  Cl 270.74300806032460  Cm -187.32766075573130
Fx -48.549657599891923  Fy 0.000489627880674850  Fz 15.165940339507145

HEAD run 2, time 3:
Cd -866.71053998338277  Cl 270.74300794534349  Cm -187.32766073570400
Fx -48.549657607709172  Fy 0.000489627807512569  Fz 15.165940330666361
```

The same-binary maximum absolute spread and the rebuilt-HEAD versus rebuilt-
baseline maximum absolute delta over all three rows were:

| quantity | same-HEAD-run max absolute spread | HEAD-baseline max absolute delta | max relative to baseline |
|---|---:|---:|---:|
| `Cd` | `1.395538902e-07` | `2.843780749e-07` | `3.026859833e-10` |
| `Cl` | `1.149811055e-07` | `1.053500682e-07` | `3.891146406e-10` |
| `Cm` | `2.002730071e-08` | `6.346326131e-08` | `3.387821162e-10` |
| `Fx` | `7.817249070e-09` | `1.592972154e-08` | `3.026860226e-10` |
| `Fy` | `7.316228060e-11` | `4.621022538e-10` | `9.437834490e-07` |
| `Fz` | `6.440783906e-09` | `5.901290123e-09` | `3.891146868e-10` |

For `Cl` and `Fz`, the between-binary delta is inside the same-binary spread
and each is therefore **indistinguishable from run-to-run noise**. `Cd`,
`Cm`, `Fx`, and `Fy` exceed the measured same-HEAD spread. These
three-iteration differences are tiny relative to the displayed coefficients,
but the evidence does not justify claiming zero Cd impact or “no
defect-driven change.” No final-force conclusion is made from the synthetic
unit test alone.

The values `Cd = -866.7` and `Cl = 270.7` are not “clean samples” or physical
coefficients: forceCoeffs reference normalization is evidently not applied,
as noted by contract section 2. They are only bounded-run diagnostic rows.
The run reached the requested three-iteration limit and was not treated as
converged.

## Scalar-gradient scope

`device_fvc.cu:100` is shared by every device scalar Gauss gradient. That
includes pressure gradients, gradient-based limiter/convection support,
turbulence scalar gradients, scalar transport, energy-related paths, and
wall-distance-related consumers. The pitzDaily scalar gate fails when the
skip is removed, while the rebuilt serial pressure, turbulence, energy, force,
MRF, and cyclic gates in the full suite remain green where their fixtures are
available. On a compatible 2D field, the front/back empty-face area vectors
are opposed, so their scalar Gauss contributions cancel; the valid-2D and
serial full-suite gates provide the measured neutral result for that
solution-direction construction. The full-suite name comparison against the
accepted baseline likewise shows no new failure other than the explicitly
recorded residual driver and known MPI/cluster environment cases. The
aggregate b74820e A/B residual movement is nevertheless retained: this suite
does not isolate that movement to the scalar hunk, and it is not honest to
call the complete solver change numerically neutral.

## Rebuilt verification output

GPU availability was checked first:

```text
nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader
NVIDIA GeForce RTX 3090, 580.173.02, 1 MiB
```

The required all-target rebuild succeeded:

```text
nice -n 19 cmake --build build --clean-first -j2
build_rc=0
[100%] Built target brae
```

The rebuild marker was `2026-08-27 07:20:42.143540437 UTC`. All 173
top-level executable targets in `build/` had timestamps after that marker;
the oldest product target was not pre-existing. Older executable bits found
only below `CMakeFiles/` were compiler probes and older embedded benchmark
worktree hook/sample files, not build targets.

The required direct gate results were:

```text
nice -n 19 cmake --build . --target test_gpu_divdevreff -j2
[100%] Built target brae_core
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

The rebuilt full suite was run without filtering:

```text
nice -n 19 ctest -j2 --output-on-failure
80% tests passed, 48 tests failed out of 240
Total Test time (real) = 213.13 sec
ctest_rc=8
```

The exact failure-name comparison was:

```text
rebuilt minus ~/brae-eval/ctest-cuda126.log:
decompose_exchange_np1
decompose_exchange_np2
decompose_exchange_np4
decompose_exchange_np8
gpu_bind_np1
gpu_bind_np2
gpu_bind_np4
gpu_bind_np8
gpu_decompose_np1
gpu_decompose_np2
gpu_decompose_np4
gpu_decompose_np8
interface_exchange_np1
interface_exchange_np2
interface_exchange_np4
interface_exchange_np8
local_assembly_np1
local_assembly_np2
local_assembly_np4
local_assembly_np8
local_convection_np1
local_convection_np2
local_convection_np4
local_convection_np8
local_mesh_np1
local_mesh_np2
local_mesh_np4
local_mesh_np8
proc_delta_np1
proc_delta_np2
proc_delta_np4
proc_delta_np8
proc_interp_np1
proc_interp_np2
proc_interp_np4
proc_interp_np8
processor_field_np1
processor_field_np2
processor_field_np4
processor_field_np8
residual_control_driver

baseline minus rebuilt:
gpu_divdevreff
```

The 32 MPI failures report `mpiexec.openmpi: Error: unknown option
"--oversubscribe"`. The remaining rebuilt-only failure,
`residual_control_driver`, is not excused: its measured baseline/HEAD
comparison is recorded above. The baseline failures
`restart_continuity`, `tracer_transport`, `dilu`, `brae_run`, `gpu_cluster`,
`gpu_amg_fused`, and `gpu_cluster_spmv` remain. The cluster failures report
the known unsupported-device/cluster-misconfiguration errors.

Focused rows from the same rebuilt CTest run were:

```text
force_coeffs ..................... Passed
force_history .................... Passed
divdevreff ....................... Passed
divdevreff_gradscheme ............ Passed
gpu_divdevreff ................... Passed
gpu_resident_turb ................ Passed
simple_run ....................... Passed
simple_step ...................... Passed
gpu_momentum ..................... Passed
gpu_simple ....................... Passed
residual_control ................. Passed
residual_control_driver .......... Failed (AssertionError: 0.0116159)
gpu_pimple ....................... Passed
pimple_run ....................... Passed
momentum_predictor ............... Passed
```

## Files and remaining uncertainty

This follow-up changes only:

```text
docs/tasks/gpu-divdevreff-investigation.md
```

There are no CMake, test-source, or production-code changes in the follow-up
working tree. The registered `gpu_divdevreff` test still covers only
`validation/pitzDaily`; the Ahmed path remains a manual diagnostic.

The exact demonstrated conclusions are:

- the first operator-stage divergence was cell `grad(U)` from empty-face
  accumulation, and the four fixes correct it;
- the fixed operator gate passes valid 2D, invalid legacy 2D negative-control,
  true 3D, and fresh Ahmed-mesh diagnostics;
- rebuilt `simpleFoam` behavior on pitzDaily changed enough to invalidate the
  residual driver's hard-coded `Uz` range, and that discrepancy is retained;
- the three-iteration Ahmed force comparison is bounded but not a converged
  Cd result; some component deltas are indistinguishable from noise and some
  exceed the measured same-binary spread;
- no claim of final Ahmed Cd neutrality is justified without a longer,
  normalized, controlled production comparison.
