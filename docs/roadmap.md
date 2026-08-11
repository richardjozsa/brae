# Roadmap

Brae runs steady and transient incompressible single-region flow on a single GPU today:
[`simpleFoam`](solvers/simplefoam.md) and [`pimpleFoam`](solvers/pimplefoam.md). Coming next:

- Adaptive time stepping (`adjustTimeStep` / `maxCo`) and runtime function objects for the transient solver
- MRF and `fvOptions` in the transient solver (the steady solver has them)
- Compressible solvers
- Multiphase solvers
- Multi-region
- Heat transfer (needs a temperature-GRADIENT boundary condition first; see the backlog below)

## Backlog

Ordered by value against cost, not by ambition.

**Boundary conditions** the stock rhoSimpleFoam tutorials use that brae refuses today. All four stop at
start-up naming the type, so no case runs wrong -- they just do not run:

| BC | field | blocks | note |
|---|---|---|---|
| `nutUWallFunction` | nut | `gasMixing` | one more in a family brae already has (nutk / nutUSpalding / nutUBlended / nutLowRe) |
| `coded` | U | `squareBendLiq` | brae has `codedFixedValue` via NVRTC; likely just the shorthand name |
| `expression` | T | `squareBendLiq` | needs an expression parser |
| `functionObjectTrigger` | T | `squareBendLiq` | niche |

**Temperature-gradient boundary conditions** (`fixedGradient`, `externalWallHeatFluxTemperature`,
`wallHeatFlux`). `DeviceBoundary` carries a reference VALUE but no reference GRADIENT, so these cannot be
represented at all and are refused. No stock tutorial needs them; industrial heat-transfer work does, and
this is the prerequisite for the "Heat transfer" item above.

**Overset (`overRhoSimpleFoam`, `overPimpleDyMFoam`, `overInterDyMFoam`).** Deliberately NOT scheduled
against the current solvers -- it belongs on its own branch. It is not a boundary condition; OpenFOAM's
`src/overset` changes the matrix itself:

- `cellCellStencil` -- donor/acceptor search across disconnected meshes, in four flavours
  (inverseDistance, cellVolumeWeight, leastSquares, trackingInverseDistance). Geometric search across
  mesh regions, which is the hard part on a GPU.
- `fvMeshPrimitiveLduAddressing` -- an acceptor cell's equation is REPLACED by an interpolation from its
  donors, so the sparsity pattern changes. brae's device matrix has fixed LDU addressing built from the
  mesh, so this is new matrix structure, not a new patch type.
- `oversetAdjustPhi` / `oversetPatchPhiErr` -- global flux conservation across the overlap.
- Hole cutting and cell classification (calculated / interpolated / hole).

Scale is comparable to or larger than a whole solver port. Its value is in MOVING bodies (floating hulls,
rotors), which is transient, so steady overset is the unusual case.

> **Refused, as of the `overset_refused` gate.** `overset` was briefly classified as a constraint patch
> type, so brae synthesised a default entry and an overset case RAN, converging to a wrong answer. It is
> now removed from `isConstraintPatchType` and refused by name in `buildPatches`, restoring the guarantee
> below. Support remains unscheduled.

**Multi-GPU** is no longer scheduled. The distributed solvers moved to `legacy/` and are out of scope; see
`legacy/README.md`. `brae ... -parallel` refuses at start-up.

## Accuracy notes

- Matches OpenFOAM to under 1% on the fields for the validated cases
- **Near-wall (low y+):** near-wall turbulence quantities can differ ~10-15% on flat plates; bulk fields and forces still track OpenFOAM
- **Extreme aspect ratio (AR ≳ 1000):** the steady solve can settle differently on sliver cells; use the same under-relaxation as OpenFOAM
- On hard steady cases (bluff-body aero), brae plateaus its residual exactly as OpenFOAM does

**Brae never guesses:** if a model, boundary condition, or scheme is not supported, it stops at start-up naming
exactly what it found, so you never get a silently wrong result.
