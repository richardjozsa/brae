# simpleFoam

Steady-state incompressible solver (SIMPLE / SIMPLEC). This is the solver brae ships today, a faithful GPU port of
OpenFOAM v2412's `simpleFoam`, fully device-resident and validated cell-by-cell against OpenFOAM. If your case uses
only what is listed here, it should run and match OpenFOAM to under 1%.

[<- back to all solvers](../../README.md#-solvers)

## At a glance

| | |
|---|---|
| **Algorithm** | steady incompressible SIMPLE / SIMPLEC |
| **Turbulence (RAS)** | laminar, k-epsilon, k-omega SST, Spalart-Allmaras, realizable k-epsilon, k-omega SST-LM (transition) |
| **Physics** | MRF, cyclic / cyclicAMI, porosity, rotor-disk & fvOptions, wall functions, force/moment coefficients |
| **Schemes** | Gauss upwind / linearUpwind / limitedLinear / vanLeer / MUSCL, bounded div, corrected (non-orthogonal) laplacian |
| **BCs** | the standard incompressible set (fixedValue, zeroGradient, inletOutlet, freestream, slip, symmetry, wall functions, ...) |
| **I/O** | standard OpenFOAM case in, standard time directories out; ASCII & binary mesh |

Legend: ✓ supported & validated · ✗ not yet.

## Solver & algorithm

| feature | status |
|---|---|
| SIMPLE / SIMPLEC pressure-velocity coupling | ✓ |
| Momentum: `div(phi,U) − laplacian(nuEff,U) − div(devReff)` (full incompressible stress) | ✓ |
| Pressure reference handling (closed domains, all-Neumann, `fixedFluxPressure`) | ✓ |
| Under-relaxation, `residualControl` convergence | ✓ |
| Non-orthogonal correction in the Laplacian (`corrected`) | ✓ |
| `nNonOrthogonalCorrectors > 0` (extra pressure correctors) | ✓ (with a `corrected` laplacian scheme) |

## Turbulence (RAS), 5 models, all validated vs OpenFOAM v2412

| model | notes |
|---|---|
| laminar | machine-precision vs OF |
| **k-epsilon** | channel / pitzDaily / backward-facing-step, sub-1% |
| **realizable k-epsilon** | coefficients read from dict |
| **k-omega SST** | pitzDaily settled U 0.10% / p 0.18% / k 0.09% |
| **Spalart-Allmaras** | NACA0012 airfoil; lift within ~1% of OF |
| **k-omega SST-LM** (Langtry-Menter transition) | transitional airfoil; transition captured |
| plain k-omega, RNG k-epsilon, LaunderSharma, RSM, LES | ✗ not yet |

Model coefficients are read from `turbulenceProperties` (defaults match OpenFOAM exactly). Wall functions include
`nutkWallFunction`, `nutUSpaldingWallFunction`, and the standard k/epsilon/omega wall functions. Solver-owned
`yPlus` additionally implements the documented `nutUWallFunction` and `nutUBlendedWallFunction` paths.

For incompressible RAS dictionaries, brae accepts both the older `RASModel kOmegaSST;` spelling and the OpenFOAM
v2406 `model kOmegaSST;` spelling inside `RAS { ... }`. If both keys are present, their values must match; a
conflict is rejected instead of giving either spelling precedence.

## Boundary conditions

The standard incompressible set: `fixedValue`, `zeroGradient`, `noSlip`, `slip`, `symmetry` / `symmetryPlane`,
`inletOutlet`, `outletInlet`, `pressureInletOutletVelocity`, `freestream` / `freestreamPressure`,
`fixedFluxPressure`, `uniformFixedValue`, `atmBoundaryLayerInlet*`, and the wall-function BCs. `cyclic` and
`cyclicAMI` are supported as coupled interfaces. Unsupported BCs are rejected by name at start-up.

## Discretization schemes (`fvSchemes`)

| category | supported |
|---|---|
| `div(phi,U)` | Gauss upwind, linearUpwind, linearUpwindV, LUST, limitedLinear, vanLeer, MUSCL, `bounded` |
| `div(phi,k\|epsilon\|omega\|nuTilda)` | upwind, limitedLinear |
| gradient | Gauss linear, cellLimited |
| laplacian / snGrad | Gauss linear `corrected` (non-orthogonal), `limited` |
| interpolation | linear |

The vector-limited `limitedLinearV` falls back to its scalar form; `leastSquares` gradient and `cellLimited`-corrected
combinations are not yet available.

## Solver-owned yPlus

`yPlus` is implemented only by `simpleFoam` for incompressible RAS `kOmegaSST`. It is not a generic OpenFOAM
function-object lifecycle: Brae computes it after a completed SIMPLE iteration, when device U, k, wall nut and
near-wall distance are mutually consistent, and emits it on the configured supported cadence. `pimpleFoam` yPlus is
not implemented.

The recognized dictionary is, for example:

```text
yPlus
{
    type yPlus;
    libs ("libfieldFunctionObjects.so");  // recognized metadata; no OpenFOAM library is loaded
    executeControl timeStep;
    executeInterval 1;
    writeControl writeTime;
}
```

Supported `executeControl` values are `timeStep` and `writeTime`; supported `writeControl` values are `timeStep`,
`writeTime` and `none`. Intervals must be positive integer iteration intervals. Unsupported controls are refused
explicitly. Brae samples only when both the execute and write schedules are due: for example,
`executeControl timeStep; executeInterval 2; writeControl timeStep; writeInterval 3;` samples at iterations 6,
12, ... rather than at 3, 6, 9. This conjunction is the declared solver-owned cadence and is not an
OpenFOAM-identical lifecycle. A final completed iteration is emitted for a `writeTime` object even when it is the
iteration limit or the residual-control convergence iteration. `writeFields true` is accepted with a notice, but
Brae writes no OpenFOAM `yPlus` volScalarField. No yPlus object means no Brae yPlus directory.

Only geometric patches whose `polyMesh/boundary` type is `wall` are reported. Empty, processor, cyclic, inlet,
outlet, symmetry and slip patches are excluded. `lowerWall` is reported in full; there is no implicit ground ROI.
The active `nut` wall boundary configuration is resolved with the same exact/group/regex precedence as a field
boundary. The currently supported paths are:

1. An ordinary wall patch (no `nutWallFunction`-derived nut BC):
   `yPlus = y * sqrt(nuEff * |snGrad(U)|) / nu`.
2. `nutkWallFunction`: let
   `yPlusInertial = Cmu^0.25 * y * sqrt(k_cell) / nu`. If it is below `yPlusLam`, use
   `y * sqrt(nuEff * |snGrad(U)|) / nu`; otherwise use `yPlusInertial`.
3. `nutUSpaldingWallFunction`: solve the OpenFOAM Spalding Newton iteration for `uTau`, seeded with the wall nut,
   then use `yPlus = y*uTau/nu`.
4. `nutUWallFunction`: perform OpenFOAM's ten-step log-law fixed-point iteration; below `yPlusLam`, use the same
   viscous `nuEff`/`snGrad(U)` branch.
5. `nutUBlendedWallFunction`: use the tangential `|U_cell-U_wall|`, combine the viscous and log-law friction
   velocities with the configured binomial exponent `n` (default 4), under-relax for up to ten iterations, and use
   `yPlus = y*uTau/nu`.

In these expressions `nu` is molecular kinematic viscosity, `nuEff = nu + nut_wall`, `y` is the
`nearWallDist` distance used by the turbulence model, and `snGrad(U)` is the wall-normal velocity gradient
(`deltaCoeff*(U_wall-U_cell)`). `k_cell` is the adjacent cell k. Wall-function defaults are the OpenFOAM
`wallFunctionCoefficients` defaults (`Cmu=0.09`, `kappa=0.41`, `E=9.8`, and path-specific iteration controls);
they are boundary-condition coefficients, not the SST `betaStar`. Brae refuses rough, low-Re, tabulated and
atmospheric nut paths rather than substituting one of these smooth-wall formulas.

The formula provenance is OpenCFD OpenFOAM v2406/v2412 source: `functionObjects::yPlus`, the turbulence-model
`yPlus()` methods, `nutkWallFunction`, `nutUSpaldingWallFunction`, `nutUWallFunction`,
`nutUBlendedWallFunction`, `nearWallDist`, and `wallFunctionCoefficients`. The implementation is in
`src/functionObjects/brae_yplus.cuh`, and the retained no-solve comparison is
`tests/test_yplus_retained.cu`.

### yPlus evidence files

Brae never reads or overwrites an existing OpenFOAM `yPlus` field. Each configured object gets the collision-safe,
Brae-owned namespace:

```text
postProcessing/braeYPlus/<object>/<start-time>/
    faceValues.dat
    patchSummary.dat
    metadata.dat
```

`faceValues.dat` has the documented columns
`solver_time iteration patch patch_local_face global_face face_area near_wall_distance yPlus`.
`patchSummary.dat` has
`solver_time iteration patch face_count total_area area_weighted_mean_yPlus minimum maximum percent_area_yPlus_30_300`.
Both files use deterministic lexical patch ordering, patch-local face ordering, and scientific precision 17.
`metadata.dat` records the solver/model, molecular viscosity, formula path, cadence, the explicit
`openfoam_yPlus_field_read=false` provenance marker, stopping reason, completed iteration and sample count.

The retained v2406/v2412 comparison predeclares `abs(yPlus_Brae-yPlus_OpenFOAM) <= 1e-5` and maximum relative
difference `<= 5e-7`, with `|yPlus_OpenFOAM| <= 1e-12` excluded from the relative denominator. Those limits are set
from the retained fields' `writePrecision 8` before the final differences are inspected.

The namespace is append-only within one run and an existing object/start-time directory is a hard collision: Brae
fails before truncating it. Retained OpenFOAM `2000/yPlus.gz` and
`postProcessing/yPlus/0/yPlus.dat` remain separate evidence and are byte-preserved.

## Physics & function objects

| feature | status |
|---|---|
| MRF (multiple reference frames) | ✓ validated (laminar + turbulent) |
| cyclic / cyclicAMI interfaces (incl. turbulence transport across) | ✓ |
| Porosity (Darcy-Forchheimer) | ✓ |
| `fvOptions` (rotorDiskSource, meanVelocityForce, velocityDamping, ...) | ✓ (subset) |
| `forces` (Cd, Cl, Cm) | ✓ validated (airfoil Cl within ~1% of OF) |
| `forceCoeffs` | ✓ solver-owned history; sampled after each completed SIMPLE iteration |
| `yPlus` | ✓ only for simpleFoam + incompressible RAS `kOmegaSST`, with the paths below |
| wallShearStress, Q, vorticity, mag(U) writers | ✗ |
| general `fvOptions` framework, arbitrary function objects | ✗ (subset only) |

## Mesh & I/O

- Reads OpenFOAM `polyMesh`, **ASCII or binary**, gzipped (`*.gz`) or plain.
- Reads stock dictionaries and `$macro` expansions.
- Writes standard time directories, post-process with `paraFoam` / `postProcess` exactly as usual.
- True 3D and 2D (empty-patch) meshes; validated on structured, graded, and snappyHexMesh meshes.

---

If you hit an unsupported model, BC, or scheme, brae stops with a clear message naming it, it never silently
substitutes something else. See [roadmap.md](../roadmap.md) for what is on the roadmap.
