# cyclicACMI + moving mesh + ddtCorr — state

## Repo state
Branch `feat/rhoSimpleFoam`, ctest **193/193**. The first round is committed (`011334b`); the moving-mesh
and `fvc::ddtCorr` work below is not. `cyclicACMI` is still REFUSED unless `BRAE_ALLOW_ACMI=1`.

## Against stock OpenFOAM v2412, oscillatingInletACMI2D, 10 steps, dt 1e-3
Nothing disabled on either side.

| | at the start | now |
|---|---|---|
| static mesh, laminar + upwind | 1.7e-02 | **6.5e-07** |
| static mesh, kEpsilon + linearUpwind | 1.8e-02 | **2.1e-03** |
| moving mesh, laminar + upwind | 2.3e-01 | **5.1e-03** |
| moving mesh, kEpsilon | 2.3e-01 | **2.5e-02** |
| one step from identical fields | 5.2e-03 | **7.6e-08** |
| interface flux, per face | 7% | **6.2e-08** |

## What was wrong (seven defects, all found by tracing values against OF term by term)

1. **The continuity residual omitted the interface flux** — the famous `contLocal 0.33` was the
   instrument, not the solver. `deviceDiv` cannot see coupled faces; the pEqn's own `div(phiHbyA)` adds
   them and the residual did not, so it measured the flux the pEqn had just driven to zero.
2. **The ACMI coverage was applied twice** — in the face area *and* in the interpolation weights.
   `cyclicACMIPolyPatch::scalePatchFaceAreas` re-normalises the weights back to 1 after scaling `Sf`.
3. **`moveMesh` discarded the interface flux** — `buildDeviceAMI` returns a fresh zeroed `.phi`.
4. **The coupled-patch flux was never written**, so a restart rebuilt it from U — 0.95% on rAU.
5. **Momentum and pressure shared one `ifCoeff` buffer** — from the 2nd corrector on, `UEqn.H()` read the
   pressure laplacian coefficient (~900× the momentum one).
6. **`fvc::makeRelative` was applied at the wrong point** — brae relativised the PREVIOUS step's flux with
   THIS step's meshPhi at the mesh move; OF relativises the flux it just computed, at the end of pEqn.H,
   and only touches phi at the move when `correctPhi` is set (this case says no). At step 1 that turned
   the initial phi (0) into `-meshPhi`. Step-1 velocity 2.2e-02 → 8.6e-04. **And it now includes the
   interface** (`cyc_/ami_.phi`), which brae never reached.
7. **`fvc::ddtCorr` did not exist.** ~1% of the flux on nearly every internal face.

Plus two non-ACMI ones fixed earlier: `pFinal` inert, and the `$macro` expansion swallowing the entry
after it.

## The two traces this round

### makeRelative on `ami_.phi` — measured, and it is zero here
The interface's own meshPhi on oscillatingInletACMI2D is **7.4e-17 across all 136 faces** (the mesh's
largest face value is 1.96e-03): the channel slides *in* the interface plane, so those faces sweep no
volume. The term is correct-in-general and contributes nothing on this fixture — which is why
`test_mesh_flux_relative` also translates through the interface in x, where it does bite.
**The moving-mesh gap was the ordering (6), not the coverage.** With laminar + upwind and ddtCorr matched
on both sides, moving-mesh agreement is **6.8e-07 over 10 steps**.

### fvc::ddtCorr — the formula, its size, and where it lives
Verified against OF face by face (21464 internal faces, 1.2e-10 L2; interface faces 1.6e-10):

    phiCorr = phi_old - (interpolate(U_old) & Sf)
    coeff   = 1 - min(|phiCorr| / (|phi_old| + SMALL), 1)
    phiHbyA += interpolate(rAU) * coeff * (1/deltaT) * phiCorr

- **Size**: 3.5% of the peak flux, 0.31% rms, on 20924 of 21464 faces. `coeff` ran 0.875…1.0.
- **Where**: zero on inlet/outlet/walls/blockage — a boundary face's `phi` *is* `U_b & Sf`, so `phiCorr`
  vanishes there identically. Non-zero on the **coupled** patches (max 1.07e-04), because
  `cyclicACMIFvPatch` derives from `coupledFvPatch`, **not** from `cyclicAMIFvPatch` — so OF's
  `isA<cyclicAMIFvPatch>` exemption does not fire for an ACMI. A plain cyclicAMI *is* exempt.
- **This also closes an older loose end**: OF's ACMI1 term sums to +5.43e-04 and ACMI2's to −4.77e-04.
  That 6.6e-05 imbalance is exactly the "OF loses mass at the interface" measured earlier — it is
  ddtCorr, applied asymmetrically to the two sides, not a conservation defect.
- **No `Uf` field needed.** pimpleFoam passes `ddtCorr(U, phi, Uf)`, which takes the Uf form on a dynamic
  mesh. `fvc::correctUf` builds Uf from the flux *before* `makeRelative`, so
  `(Sf & Uf.oldTime()) == phi.oldTime() + mesh.phi()` — verified to **2.7e-10**. brae reconstructs it
  that way and carries no surface vector field.

## Known limit, and it is announced
The `phi + meshPhi` reconstruction is exact only while `Sf` is unchanged between the two time levels —
OF dots the *current* Sf into the *stored* Uf. An ACMI rescales its coupled areas every step, and a
rotating mesh changes them everywhere. Measured: moving ACMI step 1 exact (4.0e-07), growing to 5.1e-03
by step 10, **entirely on the interface columns**. brae now emits a one-time
`NOTICE [approximated] PIMPLE/ddtCorr (moving mesh)` when it detects the areas moving; static meshes are
silent. Closing it properly means carrying a real `Uf` surfaceVectorField.

## Still open
- **Uf for moving meshes** (above) — the 5.1e-03 on the moving laminar case.
- **kEpsilon**: static laminar is 6.5e-07 and static turbulent 2.1e-03, so the turbulence model is the
  whole difference there. Untouched by any of this.
- **Defect 5 has no regression test** — verified only by the OF comparison (7% → 6.2e-08 per face).
- **Moving-mesh restart** is broken independently: brae reconstructs points from `points0` and `t`, but
  the first step after a restart takes `oldPoints` from `constant/polyMesh`, so its meshPhi spans the
  whole displacement since t=0.

## Tests
`acmi_mask` (coverage applied once), `acmi_area_scaling` (leg 6 inverted — it used to assert the
double-count as correct), `linear_solver_setup` (pFinal through the `$macro` idiom), plus new:
`coupled_phi_roundtrip`, `mesh_flux_relative`, `ddt_corr`. Every one fails when its fix is reverted, and
only on the intended assertions — checked individually.

## Method notes worth carrying
- Compare against OF at the **same time level**, and from **identical inputs** — a free-run comparison
  cannot separate a wrong term from an accumulated one.
- Bisect the physics before the code: laminar + upwind vs turbulent + linearUpwind localised two defects
  in one run each.
- OF's log and its printed diagnostics are not evidence of what it solved with (`sum(weights)` is printed
  before cyclicACMI overrides it; `ddtPhiCoeff` is not even read from fvSchemes in v2412).
- A per-cell continuity residual cannot see this interface's failure mode: both sides balance
  cell-by-cell while transmitting different totals.
