# squareBend, configured to what brae actually implements

The B1 (transonic) debugging case. **Not a gate — brae fails it.** It is here so the next session starts
from the diagnostic that made the defect unambiguous, instead of rediscovering it.

## Why it exists

The stock tutorial hides the defect. It runs `relTol 0.1` on every field, so each outer iteration only
reduces its linear residual 10x — and brae substitutes its own linear solvers anyway (AMG-PCG for GAMG,
Jacobi for DILU; `dict_audit` reports all of it, findings E2/E3). Two codes then take different paths
through the same fixed-point iteration, and it is fair to ask whether a disagreement is the assembly or
just the solver.

So this case removes the question: solvers brae genuinely implements, and tolerances tight enough that
every linear system is solved essentially exactly on both sides.

    p                 PBiCGStab + DILU   tolerance 1e-12   relTol 0
    "(U|e|k|epsilon)"  PBiCGStab + DILU   tolerance 1e-12   relTol 0
    functions {}                                  (the tutorial's sampled.plane1 needs Allrun.pre)

`PBiCGStab`/`DILU` because the transonic pressure matrix is ASYMMETRIC — OF refuses to start with a
symmetric-only smoother on it, and refuses DILU on the subsonic (symmetric) one. That constraint is real
and OF enforces it in both directions.

## What it showed

| setup | OpenFOAM | brae |
|---|---|---|
| stock (`relTol 0.1`, GAMG) | converges, 156 iters | stable but WRONG — p 72%, U 264% off |
| **this case** (matched, `relTol 0`) | **converges, 144 iters** | **diverges to NaN** |

The loose tolerance was DAMPING the inconsistency into a stable wrong answer. Solving exactly exposes the
real fixed-point iteration, and it diverges. That is the more useful failure: it rules out "different
linear solver, different path" and leaves only the assembly.

## Running it

    (cd validation/sbMatched && blockMesh && rhoSimpleFoam)          # OF: converges at 144
    BRAE_TRANSONIC=1 build/brae_rhoSimpleFoam -case <copy>           # brae: NaN

`BRAE_TRANSONIC=1` is required — transonic is refused by default precisely because of this.

## Where to look next

The residual STALLS (7e-2) under damping rather than climbing, and diverges without it. That signature is
an equation inconsistent with the flux/thermo update around it, not a bad matrix per se. The untested link
is `phi = phiHbyA + pEqn.flux()`: the `fvm::div(phid,p)` coefficients are folded into `pU_/pL_/pIC/pBC` so
`deviceMatrixFluxInternal` (which IS `lduMatrix::faceH`) picks them up — but that reconstruction has never
been compared against OF. `tools/dumpScalarMatrix` + `tests/bcoeff_compare.cu` are the instrument; they
currently cover a scalar transport and would need extending to the pressure equation.

---

## Iteration-1 comparison (added after `tools/dumpPEqn` was built)

`tools/dumpPEqn` is OF's own rhoSimpleFoam with the transonic pressure equation's INPUTS written at
iteration 1 (`phid`, `phiHbyA`, `rhorAtU`, `rAtU`, `psi`, `rho`, and `pEqn.flux()`). brae dumps the same
summary under `BRAE_DUMP_PEQN=1`. Comparing them on this case:

| field | brae | OF | verdict |
|---|---|---|---|
| `phiHbyA` sum | −2.6e-13 | 2.0e-16 | ✅ both ≈ 0 — the `interp(psi*p)/interp(rho)` subtraction is right |
| `phid` max | 8.145e-09 | 7.703e-09 | ❌ **+5.7%** |
| `phid` sum | 6.046e-06 | 5.752e-06 | ❌ **+5.1%** |
| `rho` | mean 0.3591, min 0.3307 | mean 0.3823, min 0.3716 | ❌ **L2rel 7.0e-02** |

`phiHbyA` cancelling to zero on both sides confirms the transonic subtraction. But `phid` is
`(interp(psi)/interp(rho)) * phiHbyA_orig`, and **rho is already 7% off before the pressure equation is
even assembled** — so the leading discrepancy is UPSTREAM of the transonic branch, in the thermo/energy
path (`rho = psi*p = p/(R*T)`), not in the `fvm::div(phid,p)` term.

That reframes the remaining work: chasing the pressure matrix further is premature while its inputs differ
by 5-7%. Compare `rho`/`T`/`psi` at iteration 1 first.

**Caveat on the comparison point:** OF writes `rho_` inside pcEqn (after that iteration's EEqn and
`thermo.correct()`); brae's `rho` is written at the end of the step. rhoSimpleFoam's phase order is
UEqn → EEqn → pEqn → thermo → turbulence, so confirm brae is not evaluating the pressure equation against
a thermo from the PREVIOUS iteration before drawing conclusions from the 7%.

---

## Iteration-1 narrowing (second pass, after the rho-timing fix)

`tools/dumpPEqn` now also writes `phiHbyA0` — phiHbyA BEFORE the SIMPLEC term and before the psi*p
subtraction, which is what `phid` is actually built from. brae dumps the same under `BRAE_DUMP_PEQN=1`.

| quantity @ iteration 1 | brae vs OF | verdict |
|---|---|---|
| `rAtU` | sum and max **identical** | ✅ SIMPLEC and the momentum diagonal are right |
| `T` | 9.89e-04 | ✅ energy is right |
| `rho` | 9.52e-04 | ✅ after the heRhoThermo timing fix (was 7.02e-02) |
| `phiHbyA0` | **+5.11%** | ❌ the EARLIEST discrepancy |
| `phid` | +5.1% — the SAME ratio | ❌ inherited from phiHbyA0, not its own bug |
| `p` after the solve | 10.8%, and the ranges are DISJOINT | ❌ |
| `U` after correction | \|U\| mean 1.70x | ❌ |

`phid = (interp(psi)/interp(rho)) * phiHbyA0`, and the phid error ratio equals the phiHbyA0 error ratio to
three digits — so the transonic term is faithfully propagating an error it did not create.
`phiHbyA0 = interpolate(rho)*fvc::flux(HbyA)` with rho matching to 9.5e-04 and rAtU exact, which leaves
**HbyA** — i.e. the momentum predictor, UPSTREAM of anything transonic.

### The pressure goes the wrong way

squareBend pins `outlet p = fixedValue 110000` (`$internalField`). brae pins it correctly — the written
outlet values are 110000 with spread exactly 0. But:

    OF   internal p : [110012, 117904]   entirely ABOVE the pinned outlet
    brae internal p : [ 96395, 109992]   entirely BELOW it

Not a perturbed answer — the pressure responds in the opposite direction. **Flipping the sign of the
`fvm::div(phid,p)` contribution was tested and is NOT the cause**: it gives [101700, 109995], still below.

### What has not been checked yet

With `phiHbyA` driven to ~0 by the transonic subtraction, the pressure equation is driven ENTIRELY by its
BOUNDARY coefficients — the inlet mass enters through `div(phid,p)`'s boundary contribution, since
squareBend's inlet is `zeroGradient` on p. `phidB` (the boundary phid) has never been compared against OF,
and it is the one remaining input on the path. Dump it next; `bc_vs_openfoam`/`bcoeff_compare` is the
existing instrument for boundary coefficients and would need extending to the pressure equation.

---

## phidB and the boundary: RULED OUT

`phid` at the BOUNDARY was the last unchecked input on the path — with `phiHbyA` driven to ~0 by the
transonic subtraction, the inlet mass enters the pressure equation entirely through `div(phid,p)`'s
boundary contribution (squareBend's inlet is `zeroGradient` on p). It is correct:

| quantity @ iteration 1 | brae | OF | verdict |
|---|---|---|---|
| `phid` boundary sum | −4.54545e-06 | −4.54544e-06 | ✅ 6 significant figures |
| `phiHbyA0` boundary sum | **−0.5** | **−0.5** | ✅ exact — and that IS `massFlowRate 0.5 kg/s` |
| `phiHbyA0` INTERNAL sum | 0.665109 | 0.632766 | ❌ **+5.11%** |

So the inlet mass flux is exactly right on both sides, the boundary phid is right, `rAtU` is exact, and
rho and T match. The error is confined to **`phiHbyA0` on INTERNAL faces** — `interpolate(rho)*fvc::flux(HbyA)`
— which with rho and rAtU accounted for leaves **`HbyA`**, i.e. the momentum operator's `UEqn.H()`.

That is upstream of every transonic term, and it explains the disjoint pressure ranges: the boundary mass
flux is pinned correctly while the internal flux distribution is 5% wrong, so the pressure field has to
compensate in the wrong direction.

### Next probe

Check whether the SUBSONIC squareBend shows the same +5% in `phiHbyA0`. If it does, this is not a transonic
defect at all — it is the momentum predictor on this mesh (3D, non-orthogonal, `consistent yes`), and B1
has been chasing a symptom. `tools/dumpPEqn` writes `phiHbyA0` in both branches, so the comparison is one
run per side.
