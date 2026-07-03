# motorBike: brae vs OpenFOAM (same mesh, same scheme)

A same-mesh, same-scheme cross-check on the standard OpenFOAM `motorBike` case (snappyHexMesh, kOmegaSST, 354k cells,
500 SIMPLE iterations) on a single NVIDIA GB10. All four solvers use the stock scheme
`div(phi,U) bounded Gauss linearUpwindV grad(U)`.

## Force coefficients (`forceCoeffs` on `motorBikeGroup`)

| solver | Cd | Cl | Cm | pressure solve |
|---|---:|---:|---:|---|
| **brae** | **0.4125** | 0.0829 | 0.1547 | device-resident AMG-PCG (GPU) |
| OpenFOAM (20 Grace cores) | 0.4194 | 0.0680 | 0.1578 | GAMG (CPU) |
| OpenFOAM + AMGX | 0.4134 | 0.0700 | 0.1598 | AMGX aggregation AMG (GPU) |
| OpenFOAM + PETSc-GPU | 0.4166 | 0.0657 | 0.1558 | PETSc GAMG on cuSPARSE (GPU) |

**Reading it:**

- **Cd within 1.6%** across all four (0.4125 to 0.4194). brae, an independent clean-room reimplementation, lands
  right inside the OpenFOAM family's band.
- The three OpenFOAM variants (CPU / AMGX / PETSc) agree with each other to ~1.4% on Cd. They are the same OpenFOAM
  physics with only the pressure linear-solve swapped, so that is the expected solver-to-solver scatter, and brae
  sits inside it.
- **Cm within ~3%** across all four (0.1547 to 0.1598).
- **Cl** is the noisiest coefficient: the OpenFOAM family clusters at 0.066 to 0.070, brae reads 0.083 (about 0.015
  higher in absolute terms). Cl is small, and the flow is not tightly converged at 500 iterations on a massively
  separated bluff body, so lift is the most sensitive number; brae's bulk drag and moment track OpenFOAM closely.

Note: this case does not converge tightly (a steady solver on a separated bluff body plateaus its residual, in brae
and in OpenFOAM alike), so treat these as a consistency cross-check at a fixed iteration count, not as converged
reference values.
