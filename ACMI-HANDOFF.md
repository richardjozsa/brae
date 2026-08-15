# cyclicACMI + moving mesh + ddtCorr — state

## Repo state
Branch `feat/rhoSimpleFoam`, ctest **208/208**. NOTHING IN THIS DOCUMENT IS COMMITTED YET. The first round is committed (`011334b`); the moving-mesh,
`fvc::ddtCorr`, `Uf` and turbulence work below is not. `cyclicACMI` is still REFUSED unless `BRAE_ALLOW_ACMI=1`.

## Against stock OpenFOAM v2412, oscillatingInletACMI2D, 10 steps, dt 1e-3
Nothing disabled on either side.

| | at the start | now |
|---|---|---|
| static mesh, laminar + upwind | 1.7e-02 | **2.3e-09** |
| static mesh, kEpsilon + linearUpwind | 1.8e-02 | **1.1e-03** (k 1.0e-04, eps 8.7e-05) |
| moving mesh, laminar + upwind | 2.3e-01 | **3.4e-07** |
| moving mesh, kEpsilon | 2.3e-01 | **9.0e-03** (k 9.3e-02, eps 4.4e-02) |
| laminar at nu = 1e-3 (a turbulent nut's size) | 4.7e-04 | **6.7e-08** |
| laminar + linearUpwind + cellLimited | 1.9e-03 | **1.4e-07** |
| interface flux, per face | 7% | **6.2e-08** |

All FREE runs: both solvers from `0`, ten steps, nothing disabled on either side, no restart and no
probe. That last point is not incidental -- see the retraction below.

## 500 steps, a QUARTER of the oscillation period (t = 0.5, dt = 1e-3)

Ten steps is a transient snapshot. The period is 2*pi/3.14 = 2.0 s, so t = 0.5 sweeps the channel through
its full half-amplitude. Both solvers free-run from `0`; brae 125 s, OpenFOAM 117 s on the moving case.

| U (L2 rel) | t=0.1 | t=0.2 | t=0.3 | t=0.4 | t=0.5 |
|---|---|---|---|---|---|
| moving, laminar | 5.05e-06 | 1.80e-05 | 3.74e-05 | 5.55e-05 | **6.58e-05** |
| static, kEpsilon | 7.45e-05 | 5.09e-05 | 3.93e-05 | 3.22e-05 | **2.73e-05** |
| moving, kEpsilon (REFUSED) | 1.55e-02 | -- | 2.37e-02 | -- | 2.92e-02 |

The static turbulent case IMPROVES with time -- the 1.1e-03 in the ten-step table is the initial
transient, and by t = 0.5 every field is at 1e-04 or better:

| static kEpsilon at t = 0.5 | L2 rel |
|---|---|
| U | 2.73e-05 |
| p | 1.35e-05 |
| k | 1.19e-04 |
| epsilon | 1.77e-04 |
| nut | 2.29e-04 |

Moving laminar at t = 0.5: U 6.58e-05, p 2.73e-04. Continuity on the last step: contLocal 9.6e-13
(moving), 3.3e-15 (static).

The moving TURBULENT case stays bounded over the same 500 steps (k peaks at 3.4e-01 around t = 0.1 and
decays to 1.2e-01) but 3% in velocity and 12% in k is a wrong answer, not a tolerance -- which is why it
is still refused.

**The flow path is finished**, on both convection schemes now. Laminar, the sliding-ACMI case matches
stock OpenFOAM to 2.3e-09 static and 3.4e-07 moving on free runs, and to 1.4e-07 with `linearUpwind` and
a `cellLimited` gradient -- the combination that had never been compared before this round and that
turned out to hold most of what looked like a turbulence defect. What remains is k and epsilon
themselves; see the trace and "Still open".

The wall gate (defect 8) was measured one step from OpenFOAM's own t=0.01 and took epsilon 3.9e-01 →
2.0e-02, nut 7.9e-02 → 7.6e-04, k 6.9e-03 → 1.3e-03, with U unchanged at 4.6e-03. **Treat those figures
as indicative only**: they came from the restart probe whose reference-time bug is documented in the
retraction below. The fix itself is independently covered by `acmi_wall_gate`, and the free-run numbers
in the table above are not affected by it.

## What was wrong (fifteen defects, all found by tracing values against OF term by term)

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
8. **The ACMI blockage was counted as a wall everywhere.** A cyclicACMI carries a coincident
   nonOverlapPatch of area (1-mask)*A, so on the COVERED part it is a `wall` patch with essentially zero
   area -- a wall face with no wall behind it. brae counted all of them, which cost twice over: the
   near-wall epsilon was imposed on cells nowhere near a wall, AND `deviceSolveScalarTransport` zeroes the
   AMI off-diagonal for wall cells (their value is fixed), so epsilon lost its interface coupling
   entirely. OF gates it -- `cyclicACMIFvPatchField::manipulateMatrix` redirects to the non-overlap patch
   field with weights `1 - mask`, and `epsilonWallFunction` acts only where that exceeds
   `tolerance_ = 1e-5`, blending rather than switching in between.
9. **No `Uf`.** On a moving mesh OF's ddtCorr does not read `phi.oldTime()` at all — it reads
   `Sf & Uf.oldTime()`, the CURRENT face areas dotted into the face velocity stored at the previous time
   level. Reconstructing that as `phi.oldTime() + mesh.phi()` is exact only while Sf holds still; a
   cyclicACMI rescales its coupled areas from the overlap mask every step. Carrying the real field took
   the moving laminar case from 5.1e-03 to **9.7e-07**.

10. **`divDevReff` ignored the case's named `grad(U)` scheme** -- see the retraction section below for
   the measurement and for why a factor of 7000 stayed invisible for so long.
11. **The wall-function geometry was never rebuilt when the mesh moved.** `moveMesh` refreshed the AMI
   weights and left `DeviceWallData` -- built once in the constructor -- beside them. Everything in it is
   a function of the geometry, and one entry is a function of the ACMI mask: the area fractions that
   decide which non-overlap faces count as wall at all (defect 9). A sliding ACMI re-splits those areas
   every move, so from step 2 on the wall gate answered for the mesh at t = 0. Measured on the moving
   turbulent case: some cell's wall weight moves by 0.15 EVERY step and an `isWallCell` entry flips on 2
   of the 10 steps; refreshing changes k by 1.0e-03 (L2 rel) at step 1 and 1.0e-01 by step 10.
   The same call carries the second half: `movingWallVelocity` is assigned into the device boundary by
   `setPatchVelocity` after the move, and the host `U` field is not updated, so the refresh reads the
   wall velocity back out of `dbU_`. On this case the wall slides at 1.57 m/s against a 1 m/s inlet, and
   the wall functions were being handed 0.

12. **`cellLimitedGrad` could not see the coupled patches.** A cyclic/AMI/ACMI face is a boundary face
   to brae's addressing (DeviceBoundary skips it, so it is in neither `dm.bndCellStart` nor `Ubnd`) and
   an INTERNAL one to OF's limiter, which touches it twice: `calcGrad` folds
   `psf.patchNeighbourField()` into maxVsf/minVsf under `if (psf.coupled())`, and its second loop clips
   the extrapolation to EVERY boundary face, coupled ones included. brae did neither, so an interface
   cell was limited as though the interface were not there. See the trace below -- this one defect was
   most of what looked like a turbulence problem for several rounds.

13. **kEpsilon's production gradient ignored the named `grad(U)` scheme** -- `gradULimitK` appeared
   nowhere in `device_kepsilon.cu`, while kOmegaSST had honoured it all along. See the trace below.
14. **The epsilon matrix was constrained to the RAW wall value, not the blended one.** See the trace.

15. **`nearWallDist` searched within one patch instead of across all wall patches.** OF v2412 has both
   branches and `cellDistFuncs::useCombinedWallPatch` DEFAULTS TO TRUE, so it gathers every wall patch
   into one `uindirectPrimitivePatch` and takes point-neighbours there. See the trace below.

Plus two non-ACMI ones fixed earlier: `pFinal` inert, and the `$macro` expansion swallowing the entry
after it.

## Earlier traces (moving mesh / ddtCorr round)

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
- **The Uf form is a DIFFERENT quantity on a moving mesh.** pimpleFoam passes `ddtCorr(U, phi, Uf)`,
  which takes the Uf form whenever the mesh is dynamic, and that form reads `phiUf0 = Sf & Uf.oldTime()`
  for both the correction AND the coefficient's denominator. `fvc::correctUf` builds Uf from the flux
  *before* `makeRelative`, so `(Sf & Uf.oldTime()) == phi.oldTime() + mesh.phi()` — verified to
  **2.7e-10** — but only while Sf holds still. See the Uf section below.

## Uf, and why the shortcut had to go
brae first reconstructed `phiUf0` as `phi.oldTime() + mesh.phi()` — verified against OpenFOAM to
**2.7e-10**, and genuinely exact for a translating mesh. It is wrong the moment a face's area changes
between time levels, because OF dots the *current* Sf into the *stored* Uf. A cyclicACMI does exactly
that every step, from the overlap mask. Measured: moving ACMI step 1 exact (4.0e-07), growing to 5.1e-03
by step 10, all of it on the interface columns. An interim build announced this as
`NOTICE [approximated] PIMPLE/ddtCorr (moving mesh)`; the notice is gone because the approximation is.

brae now carries `Uf` for real, in the four compartments a flux lives in (internal faces, non-coupled
boundary faces, and the two interfaces):
- constructed as `fvc::interpolate(U)` before the first step, when the case has mesh motion — OF's
  `createUfIfPresent.H` makes it whenever `mesh.dynamic()` and NOT otherwise, which is why a static case
  still takes ddtCorr's `phi.oldTime()` form
- `fvc::correctUf` at the end of every pressure corrector, before `makeFluxRelative`, so it is built from
  the ABSOLUTE flux exactly as `pEqn.H` does
- aged into `Uf.oldTime()` at the top of each step, then read as `phiUf0 = Sf_current & Uf.oldTime()`

The defining property, and what `test_correct_uf` pins: after `correctUf`, **`Sf & Uf == phi` exactly**,
with the tangential component untouched. A version that just set `Uf = n*phi/magSf` satisfies the first
half and is wrong; the test catches it on the second, and separately on the zero-area guard (an uncovered
ACMI face has magSf ~1e-13 and would divide to NaN).

## RETRACTED: "the interface's viscous coupling is wrong"

The previous round of this document claimed a viscous interface defect -- error proportional to nuEff,
`HbyA` 0.1-0.2% out at the interface, "the explicit half of `UEqn.H()`". **All of it was an artefact of
the probe**, and the whole chain is withdrawn.

The probe restarted brae from OpenFOAM's written state and compared one step. It selected the reference
directory with `runTime.times().last()`, which picked up `0.011` -- left behind by an earlier one-step
run -- while the comparison was against `0.01`. Compared against `0.011` the difference is
`max|diff| = 0.0000e+00`, which is what gave it away. Re-run as a FREE run (no restart, no probe, both
solvers from `0` with nothing disabled), laminar at `nu = 1e-3`, the two agree to **6.7e-08**. There is
no viscous interface defect.

What was real, and had been noted in that same section as a side finding, is that **`divDevReff` ignored
the case's named `grad(U)` scheme**. OF's `linearViscousStress::divDevReff` is
`-fvc::div(nuEff*dev2(T(fvc::grad(U)))) - fvm::laplacian(nuEff,U)`, and `fvc::grad(U)` resolves
gradSchemes `grad(U)` -- `cellLimited Gauss linear 1` on this case. brae built a plain Gauss gradient
there. Free runs, no probe:

| | unlimited | limited (OF) |
|---|---|---|
| laminar, nu = 1e-6 (the tutorial) | 6.5e-07 | 6.5e-07 |
| laminar, nu = 1e-3 | 4.7e-04 | **6.7e-08** |

a factor of 7000, and invisible at the tutorial's own viscosity -- which is why it survived every
laminar comparison. Fixed, and covered by `divdevreff_gradscheme`.

Do NOT "fix" the non-orthogonal laplacian correction to match: brae leaves that gradient unlimited and
is right to. OF's `correctedSnGrad<vector>::correction` loops the components and calls
`fullGradCorrection(vf.component(cmpt))`, so the scheme lookup is for `grad(U.component(0))`, which no
case defines and which falls back to gradSchemes `default`.

## Trace: the static case, and what "the turbulence is wrong" actually was (defect 12)

The turbulent case sat at 2.2e-03 while the laminar one was at 1.4e-07, and that read as a turbulence
defect for several rounds. It was not. **The two cases also differ in their convection scheme**, and
nobody had looked: `statlam` says `div(phi,U) Gauss upwind`, `statturb` says
`Gauss linearUpwind grad(U)`. Every comparison that ever reached 1e-07 was run on upwind, so linearUpwind
had never been checked against OpenFOAM on this mesh at all.

Bisect 1 -- run the case LAMINAR (no turbulence model whatsoever) with the turbulent case's scheme:

| static, laminar, 10 free steps | vs OF |
|---|---|
| `Gauss upwind` | 1.4e-07 |
| `Gauss linearUpwind grad(U)` | **1.9e-03** |

which reproduces the whole turbulent gap with no turbulence present.

Bisect 2 -- the same laminar linearUpwind case on an UNLIMITED gradient:

| static, laminar, linearUpwind | vs OF |
|---|---|
| `grad(U) Gauss linear` | **2.0e-09** |
| `grad(U) cellLimited Gauss linear 1` | 1.9e-03 |

a factor of a million. That acquits the linearUpwind correction *and* the whole interface flux path --
2.0e-09 over ten steps is not a code with a broken interface -- and convicts the LIMITER. Localised, 91%
of the squared error sat on the 136 interface-adjacent cells, 1.25% of the mesh, from the first step it
appeared.

The cause is defect 12: brae's `deviceCellLimitGrad` walks internal faces and non-coupled boundary faces,
and a coupled face is in neither list. Folding them into both of OF's loops (the range, and the
extrapolation clip) gives:

| | before | after |
|---|---|---|
| laminar + linearUpwind + cellLimited | 1.9e-03 | **1.4e-07** |
| static laminar + upwind (the tutorial) | 1.4e-07 | **2.3e-09** |
| static turbulent | 2.2e-03 | **1.2e-03** |

The laminar tutorial improved 62x as a side effect, because its `divDevReff` gradient is cellLimited too
(defect 10) -- the same limiter, the same blind spot. The fix is applied everywhere that gradient is
built: the momentum predictor's linearUpwind gradient, `divDevReff`, the kOmegaSST strain, and the
startup `correctNut`.

**Method note worth keeping**: two cases that differ in more than one way cannot be compared. "laminar vs
turbulent" was really "upwind + laminar vs linearUpwind + turbulent", and the entire hunt for a
turbulence defect followed from not separating those.

## The scalar-transport path, brought up to the momentum one

`deviceSolveScalarTransport` -- k, epsilon, omega, nuTilda, he -- had NONE of the coupled-patch terms the
momentum predictor has carried since the interface work. Four, all now added:

| term | what was missing |
|---|---|
| `deviceGaussGrad` | the gradient stopped at the interface (no `interfaceAddGrad`) |
| `cellLimitedGrad` | defect 12, the same blind spot |
| `linearUpwind` correction | OF reaches it through `if (pSfCorr.coupled())` in `linearUpwind::correction()` |
| non-orth laplacian correction | no `interfaceAddLapCorr` |

The two correction kernels existed but only in a VECTOR form -- component index, three-buffer gradient,
neighbour rotated by forwardT. A scalar has one gradient and is never transformed across a rotational
interface, so both got a scalar overload.

**These are inert on the tutorial, and the run is bit-identical.** That is correct, not a failure: the
mesh is orthogonal so the non-orth correction is identically zero, and the tutorial convects k/epsilon
with `Gauss upwind` so no linearUpwind gradient is ever built. Re-run with
`div(phi,k) Gauss linearUpwind grad(k)` and cellLimited `grad(k)`/`grad(epsilon)` they engage:

| linearUpwind + cellLimited turbulence, 10 steps | without | with |
|---|---|---|
| nut, worst relative | 1.27e-01 | **6.83e-02** |
| nut, L2 rel | 1.15e-02 | **9.90e-03** |
| U | 1.1933e-03 | 1.1927e-03 |
| k | 1.841e-02 | 1.856e-02 |
| epsilon | 2.387e-02 | 2.429e-02 |

**Read that honestly.** nut improves clearly and U marginally; k and epsilon move slightly the WRONG way.
They are not a regression so much as noise on top of a larger separate error: k is 1.8e-02 out *away from
the interface* as well as on it (only 6.5% of its squared error sits on the interface ring, which is 1.25%
of the cells). A correct term added to a field that is already 2% wrong for another reason can move the
bottom line either way. The evidence these terms are right is `scalar_interface_corr` and
`cell_limit_interface`, not this table.

## Trace: the k/epsilon equations term by term in the inlet channel (defects 13, 14)

**Method.** pimpleFoam with `nOuterCorrectors 1` runs momentum predictor -> pressure correctors ->
turbulence correct() -> write. So the U that `correct()` sees at step n IS the U written at step n, while
the nut/k/epsilon it starts from are those written at step n-1. Every input is on disk, and OF's own
production term can be reconstructed from written fields with no instrumentation of the solver
(`turbTerms`, in the scratchpad; it takes its time from `-time` and prints it, after last round).

One catch found immediately: `0/nut` is `uniform 0`, and reading it gave G == 0 identically. OF's
`turbulence->validate()` runs `correctNut()` at STARTUP and never writes the result, so the nut the first
`correct()` actually uses is `Cmu k^2/epsilon` from the initial fields -- 7.03125e-04 here, which is
exactly what brae's dump showed.

**Defect 13, the production gradient.** `gradULimitK` appeared nowhere in `device_kepsilon.cu`. OF's
`kEpsilon::correct()` opens with `fvc::grad(U)`, which resolves gradSchemes `grad(U)` = `cellLimited
Gauss linear 1`; brae's kOmegaSST path had honoured that for a long time, kEpsilon's never did. Step 1,
against OF's reconstructed production:

| GbyNu | before | after |
|---|---|---|
| inlet channel | 2.06e+01 | **1.32e-02** |
| duct | 3.42e+00 | **7.54e-08** |

The duct going exact to 7.5e-08 is the proof: the gradient OPERATOR was always right, only the scheme was
wrong. The channel's residual 1.3e-02 is the U difference propagating -- U differs by 6.2e-03 there over
~1e-2 cells, giving a gradient difference of 6.16e-01, which is what is measured.

**Defect 14, the epsilon constraint value.** With production fixed, the remaining epsilon error was
**100% on the 136 ACMI cells** (wall cells exact to 2.0e-05, interior to 5.2e-04) -- and the median ACMI
cell was already right to 1.7e-07, so it was a handful of cells at the edge of the overlap.
`epsilonWallFunction::updateCoeffs(weights)` blends `epsilon[celli] = (1-w)*epsilon[celli] +
w*epsilon0[celli]`, and `manipulateMatrix` then appends **that blended value** to `matrix.setValues`.
brae blended the FIELD but constrained the MATRIX to `eps0`, the raw wall value. On an ordinary wall
w = 1 and the blend is the identity, so it was invisible everywhere except the cells a cyclicACMI
partially covers.

| 10 steps | after defect 13 | after defect 14 |
|---|---|---|
| static k | 6.92e-04 | 7.37e-04 |
| static epsilon | 1.93e-02 | 1.92e-02 |
| moving k | 9.76e-02 | **9.27e-02** |
| moving epsilon | 4.25e-01 | **5.46e-02** |
| moving epsilon, worst cell | 9.72e-01 | **7.70e-02** |

Nearly inert on the static case and 7.8x on the moving one, which is the expected shape: a sliding
interface produces partially covered faces continuously, a static one has a fixed and mostly 0/1 mask.

Together the two took the static case from k 1.76e-02 to 7.4e-04 and the moving case from k 1.26 to
9.3e-02, epsilon 2.69 to 5.5e-02, U 2.47e-02 to 9.0e-03.

## Trace: the four edge cells (defect 15)

They were cells 79/3199 and 3200/10800 -- the FIRST and LAST faces of each side's coupled patch. All four
touch a `walls` face AND a blockage face; every other interface cell touches only the blockage. Ruled out
in order, each by a measurement:

- **the corner weighting is not it.** A probe replicating OF's `createAveragingWeights` says OF counts
  **2** epsilonWallFunction faces at those cells and 1 elsewhere -- the same as brae's `invNw` of 0.5.
- **the blend is not it.** Defect 14 left those cells bit-identical.
- **the AMI mask is not it.** brae's `wallW` there is 1, from the real wall, and the arithmetic below
  needs no mask.

What it was: with k uniform at 3.75e-03 each wall face contributes `Cmu^.75 k^1.5/(kappa y)` =
9.2035e-05/y, so the distances can be read straight off the values. brae's eps0 at cell 3200 was
0.5*(9.2035e-05/1.25e-02 + 9.2035e-05/5.20833e-03); OF's was 9.2035e-05/5.20833e-03. Dumping OF's own
`nearWallDist` settled it:

|  | OF | brae (per-patch) |
|---|---|---|
| y(ACMI2_blockage) at cell 3200 | 5.20833e-03 | 1.25e-02 |
| y(walls) at cell 3200 | 5.20833e-03 | 5.20833e-03 |
| y(ACMI2_blockage) at cell 3280 (not a corner) | 1.25e-02 | 1.25e-02 |

OF gives BOTH faces of a corner cell the distance to the NEAREST wall, across patches. All four cells
then match to 1.000000, and cell 159 (0.9965 before) with them.

| static turbulent, 10 steps | before | after |
|---|---|---|
| epsilon | 1.92e-02 | **8.70e-05** |
| k | 7.37e-04 | **1.03e-04** |
| nut | 1.15e-03 | **9.12e-05** |
| U | 1.069e-03 | 1.069e-03 |

## pimpleFoam tutorial coverage (OpenFOAM v2412, 30 cases)

Each meshed with its own scripts, then OpenFOAM and brae run SERIALLY over the same 20-step window.

| Runs in both | n | U (L2 rel) |
|---|---|---|
| RAS/pitzDaily | 12225 | 3.48e-03 |
| laminar/mixerVesselAMI2D | 3072 | 1.01e-02 |
| RAS/oscillatingInletACMI2D | 10880 | 1.33e-02 |
| RAS/TJunctionFan | 3875 | 2.77e-02 |
| LES/periodicPlaneChannel | 60000 | 4.75e-02 |
| RAS/TJunction (+Arrhenius) | 3875 | 1.01e-01 (declared: `limitedLinearV` has no exact kernel) |
| laminar/planarPoiseuille | 40 | **8.07e-08** (200 steps; the flow IS the fvOptions source) |

**What still blocks the rest**, brae-side:

| Missing | Cases |
|---|---|
| `interfaceTrackingFvMesh` (VOF/film) | 2 (contactAngleCavity, sloshing2D) |
| motion solvers other than `solidBody` | 2 (movingCone, oscillatingInletPeriodicAMI2D) |
| `binary` writeFormat field files | 1 (NACA4412) |
| `fixedMean` BC | 1 (wallMountedHump) |
| `Maxwell` viscoelastic laminar model | 1 (planarContraction) |
| `velocityFilmShell` BC (finite-area film) | 1 (inclinedPlaneFilm) |
| divSchemes `default` for div(phi,U) | 1 (cylinder2D) -- deliberate |
| time-varying cyclicACMI `scale` | 1 (TJunctionSwitching) -- deliberate |

**DEShybrid did NOT on its own unblock the two LES tutorials**, which is worth stating plainly because it
was the reason for doing it. It removed one of TWO blockers from each: NACA4412 now gets through scheme
parsing and builds its SpalartAllmarasDDES model before dying on a `binary` field file, and
wallMountedHump reaches its boundary conditions and stops on `fixedMean`. Neither refusal mentions
DEShybrid any more.

The remaining 12 are OpenFOAM-side or meshing failures in the harness, not brae refusals.

## Still open
- **U on the turbulent case, now the laggard at 1.07e-03** while k, epsilon and nut are all ~1e-04 and
  the same case run laminar is 2.3e-09. nut agrees to 9.1e-05, so this is no longer a turbulence-field
  error being passed into the momentum equation -- it is something in the momentum equation that only
  appears when nut is non-zero. That is the next trace, and it is NOT the stress term (measured at 2%
  earlier) nor the convection scheme (fixed).
- **The moving case's k, 9.3e-02 at ten steps -- but it is ACCUMULATED, not a term error.** Re-measured
  by zone after all of the above, step 1:

  | moving case, step 1 | moving zone | stationary |
  |---|---|---|
  | k | 9.08e-05 | 3.68e-07 |
  | epsilon | 6.53e-05 | 2.99e-07 |
  | U | 8.89e-04 | 4.53e-07 |

  k was 3.1e-01 in the moving zone before this round and is 9.1e-05 now. What is left at step 1 is U,
  8.89e-04 in the moving zone -- **unchanged by every turbulence fix**, so it is not turbulence-driven.
  It grows into k over the ten steps (1.13e-01 by t = 0.01).

  So both remaining items are the same one: **a ~1e-03 velocity error that appears only when nut is
  non-uniform.** The same case laminar is 2.3e-09, laminar with linearUpwind 1.4e-07, and laminar with a
  large UNIFORM nu = 1e-3 is 6.7e-08 -- so it is neither the scheme nor the magnitude of the viscosity,
  it is its VARIATION. First suspect: nut at the boundary FACES (`nutBndAll_`, the nutkWallFunction
  values), which is a separate field from the internal nut and is not covered by the 9.1e-05 agreement
  on the internal one. Compare it face by face against OF's `nut.boundaryField()` before anything else.
- **The old note that the k/epsilon error was "in the BULK, not the interface" was true only of the
  pre-fix state.** The production defect was everywhere and masked this; with it gone the residue is
  entirely at the interface. Re-localise after every fix, not once.
 With the momentum path clean, what is left is the turbulence transport:
  static k 1.8e-02 and epsilon 2.4e-02, and the moving case far worse (k 1.3, epsilon 2.7). U follows nut, so the static U residue of 1.2e-03 is downstream of
  these. **It is not an interface defect**: with the interface ring separated out, k is 1.8e-02 away from
  the interface too, and the ring holds only 6.5% of the squared error. Combined with the earlier zone
  split -- the inlet channel 1.1e-02 against the duct's 1.2e-03 at step 1, mesh held still -- the next
  trace is a term-by-term comparison of the k/epsilon equations in the CHANNEL on the STATIC case:
  production G, the wall functions, and the reaction terms. Not the interface, not the motion, not the
  convection scheme.
  The moving case additionally has k 31% out in the moving zone at step 1, which nothing so far touches.
- **The covered/uncovered transition cells.** After the wall gate, epsilon's residual (max 2.9e-01,
  median 2.6e-05 near the interface) sits on the cells either side of the blended faces. OF blends the
  wall function there by `1 - mask`; brae now does too, taking the max over a cell's wall faces, but the
  two need not agree cell for cell when a cell has several. Small, and next after the momentum item.
- **`Uf` is not written**, so a restart re-initialises it from `interpolate(U)` where OF resumes the
  stored field (AUTO_WRITE). Needs a surface VECTOR writer, which brae does not have. Moot until the
  moving-mesh restart below is fixed.
- **Defect 5 has no regression test** — verified only by the OF comparison (7% → 6.2e-08 per face).
- **Moving-mesh restart** is broken independently: brae reconstructs points from `points0` and `t`, but
  the first step after a restart takes `oldPoints` from `constant/polyMesh`, so its meshPhi spans the
  whole displacement since t=0.

## Tests
`acmi_mask` (coverage applied once), `acmi_area_scaling` (leg 6 inverted — it used to assert the
double-count as correct), `linear_solver_setup` (pFinal through the `$macro` idiom), plus new:
`coupled_phi_roundtrip`, `mesh_flux_relative`, `ddt_corr`, `correct_uf`, `acmi_wall_gate`,
`divdevreff_gradscheme`, `wall_data_refresh`, `cell_limit_interface`, `scalar_interface_corr`,
`kepsilon_production_and_blend`. Every one fails when its fix is reverted, and only on the
intended assertions — checked individually. Two of them caught their own fixture on the way: the
gradscheme test's first field made `divDevReff` identically zero (a step in y leaves the x-component
blind to it) and its vacuity guard said so; `wall_data_refresh` needs the wall-free cells to be TWO
rings in, because a cell reads its neighbours' gradients through the Gauss divergence.

## Method notes worth carrying
- Compare against OF at the **same time level**, and from **identical inputs** — a free-run comparison
  cannot separate a wrong term from an accumulated one.
- Bisect the physics before the code: laminar + upwind vs turbulent + linearUpwind localised two defects
  in one run each.
- OF's log and its printed diagnostics are not evidence of what it solved with (`sum(weights)` is printed
  before cyclicACMI overrides it; `ddtPhiCoeff` is not even read from fvSchemes in v2412).
- A per-cell continuity residual cannot see this interface's failure mode: both sides balance
  cell-by-cell while transmitting different totals.
- **Never let a probe choose its own reference time.** `runTime.times().last()` picked up a `0.011`
  directory left by an earlier run and compared it against `0.01`, and the resulting phantom sent a
  whole round of work after a viscous interface defect that does not exist. Name the time explicitly,
  and when a probe and a free run disagree, believe the free run.
- **Change one thing.** The laminar case that reached 1e-07 and the turbulent one that did not differed
  in their CONVECTION SCHEME as well as their physics, and several rounds went into the physics before
  anyone diffed the two fvSchemes files. Before attributing a gap to the obvious difference, list all of
  them.
- Localise by REGION before by term. Splitting the error over the moving cellZone and the stationary
  duct took "the turbulent case is 2.2e-03 out" to "the inlet channel is 6.9e-04 out while the duct is
  2.7e-08" in one measurement, and moved the search off the interface entirely.
- A relative error is only meaningful against its own region's magnitude: normalise per zone, or a
  region where the field is still near zero will look clean for the wrong reason.
