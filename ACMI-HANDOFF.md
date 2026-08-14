# cyclicACMI — the coupling is verified; mesh motion is not

## Repo state
Branch `feat/rhoSimpleFoam`, ctest **191/191**. Nothing committed (per standing rule).
`cyclicACMI` is still REFUSED in `buildPatches` unless `BRAE_ALLOW_ACMI=1` — but for a different and
much narrower reason than before, and the refusal text has been rewritten twice to keep up.

## Where it landed, against OpenFOAM v2412 on oscillatingInletACMI2D

Measured with `ddtCorr no` in OF's PIMPLE dict, so both codes solve the same equations (brae does not
implement `fvc::ddtCorr` — see the open items). 10 steps, fixed dt 1e-3.

| | at the start | now |
|---|---|---|
| static mesh, laminar + upwind, U (L2 rel) | 1.7e-02 | **4.7e-07** |
| static mesh, kEpsilon + linearUpwind, U | 1.8e-02 | **1.5e-03** |
| moving mesh, kEpsilon, U | 2.3e-01 | **3.0e-02** |
| one step from identical fields, U | 5.2e-03 | **7.6e-08** |
| interface flux, per face | 7% | **6.2e-08** |
| contLocal (case's own tolerances) | 0.33 | **2.2e-11** |

The interface coupling is done. What is left is mesh motion and ddtCorr.

## The five defects, and how each was found

The method that worked, after two false starts: **trace one value at a time against OpenFOAM, from
identical inputs.** A small OF utility (`amiProbe`, rebuilt in the scratchpad each time) dumped OF's
own `w`, `deltaCoeffs`, `magSf`, `patchNeighbourField`, `rAU`, `UEqn.H()`, `HbyA`, `fvc::flux(HbyA)`
and `fvc::div(phiHbyA)` at the interface; brae's counterparts came from `BRAE_DUMP_STAGE`.

**1. The continuity residual omitted the interface flux** (`device_simple_foam.cu`, 3 sites).
`deviceDiv` only sees DeviceBoundary faces and coupled faces are not in it. The pressure equation's own
`div(phiHbyA)` adds them (`interfaceAddDiv`); the residual did not. So the pEqn drove `div(phi)`
*including* the interface to zero and the residual then measured the interface flux it had left out —
0.33, "concentrated at the interface". That number and its localisation were both the instrument.

**2. The ACMI coverage was applied twice** (`ami_interface.cuh`). `cyclicACMIPolyPatch::scalePatchFaceAreas`
scales the coupled `Sf` by the mask **and then** re-normalises the AMI weights back to 1 —
*"Re-normalise the weights since the effect of overlap is already accounted for in the area"*. brae did
the first and not the second, so a partially covered face transmitted `mask²`. It must run **before**
the per-face geometry loop, which interpolates the neighbour delta with the same weights.
OF's printed `sum(weights)` cannot settle this — it is emitted from inside `normaliseWeights`, before
the override — and reading it as confirmation was the original error.

**3. `moveMesh` discarded the interface flux** (`device_simple_foam.cu`). `buildDeviceAMI` returns a
fresh zeroed `.phi`, so `ami_.phi` — the interface's convection coefficient — was wiped at the top of
every step after the first. Signature: step 1 matched to 6e-04, step 2 put the channel 1140 above the
duct, and a *restart cured it*.

**4. The coupled-patch flux was never written** (`foam_field_writer.cuh`, `read_surface_field.cuh`,
`gpuPimpleFoam.cu`). OF writes phi's coupled-patch values; brae wrote `type cyclicAMI;` and nothing, and
its reader skipped them, so a restart rebuilt the flux from U. That is not the same number — 1.3168e-04
on face 0 — and through the upwind `max(phi,0)` it landed on the momentum diagonal: **0.9455% of rAU on
all 40 source-side interface cells and nothing else in 10880.** The rAU error matched the flux
difference to five figures.

**5. The momentum and pressure assemblies shared one `ifCoeff` buffer** (`device_simple_foam.cu`).
`deviceAmiAssembleLaplacian` overwrites the same `ami_.ifCoeff` that `deviceAmiAssembleMomentum` wrote
and that `interfaceAddH` reads. On the FIRST pressure corrector H is built before the pressure is
assembled, so it reads the right one; from the SECOND corrector on it read the pressure laplacian
coefficient — 4.7e-05 in place of -5.3e-08, a factor of ~900.
Signature that isolated it: **nCorrectors 1 → 7.6e-08; nCorrectors 2 → 2.6e-03**, appearing as a
*uniform* pressure offset on the upstream block (a uniform shift has no gradient, so the bulk velocity
stayed exact and only the interface-adjacent columns moved).

## Two more defects found on the way, neither ACMI-specific

**`pFinal` was inert** (`linear_solver_setup.cuh`, `solver_controls.cuh`, `device_simple_foam.cu`).
Every corrector was solved to the loose `p` tolerance. Fixed with OF's actual selection, which is two
different rules and the difference is visible in OF's log:
- **p** → `pEqn.solve(p.select(pimple.finalInnerIter()))` = last pressure corrector, last non-orth pass,
  gated on the outer iteration only when `finalOnLastPimpleIterOnly` is set (default false)
- **U, k, epsilon** → the bare `solve()`, which resolves through `fvMatrix::solverDict()` →
  `psi_.select(mesh.data().isFinalIteration())` = anywhere in the final OUTER corrector

**brae's `$macro` expansion swallowed the entry after it** (`foam_dict.cuh`). The expansion is textual
and the captured body ends in its own `;`, so the canonical idiom

    p      { solver GAMG; tolerance 1e-5;  relTol 0.01; }
    pFinal { $p; tolerance 1e-10; relTol 0; }

became `... relTol 0.01 ; ; tolerance 1e-10 ;` and the parser read the second `;` as a KEY holding
`tolerance 1e-10`. The override was not lost, it was *eaten* — so `pFinal` silently became `p`. Fixed at
the parser (a stray `;` is an empty entry, skip it), because an unparseable empty entry is harmless
while one that eats the next entry is a silently ignored input.

## Still open

- **`fvc::ddtCorr` is not implemented at all.** It contributes nothing *at* the interface (OF zeroes the
  coefficient on `cyclicAMIFvPatch` in `ddtScheme::fvcDdtPhiCoeff`) but it is a real brae-wide pimpleFoam
  gap elsewhere: it is the whole difference between the 1.5e-03 above and 1.1e-02 against stock OF.
  Note `ddtPhiCoeff` is NOT read from `fvSchemes` in v2412, so testing it that way proves nothing — use
  the `PIMPLE { ddtCorr no; }` switch.
- **Mesh motion: 3.0e-02.** The static case is at 4.7e-07, so this is motion, not the interface.
  First place to look: `fvc::makeRelative` is applied to `phiInt_`/`phiBnd_` in `moveMesh` but never to
  `ami_.phi`. Zero for this fixture (the slide is tangential, so the swept volume of an interface face
  is ~0) but not in general.
- **Defect 5 has no regression test.** It is verified only by the OpenFOAM comparison above
  (7% → 6.2e-08 per face). A unit test needs a fixture where extra correctors are provably no-ops.
- **Turbulent static is 1.5e-03** where laminar is 4.7e-07 — a kEpsilon difference, not ACMI.

## Tests added / corrected
- `test_acmi_mask` leg 5: the coverage appears exactly once (mask still 0.5 while weights sum to 1),
  with a vacuity guard that the fixture has a genuinely blended face.
- `test_acmi_area_scaling` leg 6: **inverted** — it used to assert the double-count as correct, citing
  OF's log. The old reasoning is kept in the comment because every line of it was true and the
  conclusion was still wrong.
- `test_linear_solver_setup` leg 8: `pFinal`/`UFinal` through OF's `$macro` idiom, with a longhand
  negative control and an absent-entry fallback leg.
- `test_coupled_phi_roundtrip` (new): coupled-patch flux survives write→read exactly; negative control
  for a value-less file; vacuity guard.

All four fail when their fix is reverted, and only on the intended assertions.
