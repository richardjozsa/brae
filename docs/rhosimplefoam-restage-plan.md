# rhoSimpleFoam: re-grounding the port on OpenFOAM's source, stage by stage

## The ground rule

**OpenFOAM's own source is the reference for every term. `gpuSimpleFoam` is not.**

That rule exists because the opposite produced most of what this audit found. `linear_solver_setup.cuh`
records it plainly — the compressible driver was *"ported by copying only the parts of gpuSimpleFoam it
needed, and left fifteen controls behind."* A1 was the same shape: `dbHe_` carried the inletOutlet mask
because the incompressible code had one, but nothing ever called `deviceUpdateInletOutlet` on it, and T
came out 276% wrong. Copying transfers the *shape* of a solution without its *reasons*.

## What this plan is, and what it is not

It is **not** a rewrite. The momentum, energy and pressure equations already agree with OF to 1e-6…1e-8 on
SIX gated cases (`rho_vs_openfoam`, `rhoE_vs_openfoam`, `io_vs_openfoam`, `ti_vs_openfoam`,
`hf_vs_openfoam`, `mx_vs_openfoam`). Retyping code that is provably correct on six cases will not surface
a defect that appears only on the seventh.

What is missing is the **instrument**: a way to compare each stage of one SIMPLE iteration against OF, on
any case, in the order the data actually flows. Build that, run it on squareBend, and the wrong stage names
itself. Then — and only then — re-derive that stage from OF's source.

## The ordering, and why it is the right one

SIMPLE is **segregated**: each equation is solved separately and coupled by outer iteration. So an error in
one stage appears *downstream* in the next, never upstream. Verifying in dependency order therefore
localises a defect to a single stage instead of a whole solver:

```
    rAU, rAtU        <- UEqn.A(), and SIMPLEC's 1/(1/rAU - UEqn.H1())
        |
    HbyA             <- rAU*UEqn.H()                      [the current suspect]
        |
    phiHbyA          <- interpolate(rho)*fvc::flux(HbyA)
        |
    phid             <- (interpolate(psi)/interpolate(rho))*phiHbyA     [transonic only]
        |
    p                <- the pressure equation solve
        |
    U                <- HbyA - rAtU*grad(p)
        |
    he               <- EEqn, on the corrected U
        |
    k, epsilon|omega <- turbulence, last
```

This is exactly the chain the squareBend investigation walked backwards, and it is why the answer landed on
`HbyA`: everything downstream inherited its +5.11% unchanged.

## Phase 0 — the stage harness (build this first)

Extend the two tools that already exist rather than inventing a third.

| exists | does |
|---|---|
| `tools/dumpScalarMatrix` | OF's assembled `fvScalarMatrix`: diag, source, internalCoeffs, boundaryCoeffs |
| `tools/dumpPEqn` | OF's rhoSimpleFoam writing `phid`, `phiHbyA0`, `rAtU`, `rho`, `psi`, `pEqn.flux()` at iteration 1 |
| `tests/bcoeff_compare.cu` | brae's boundary coefficients vs OF's, per patch, to machine precision |
| `BRAE_DUMP_PEQN=1` | brae's `phid`/`phiHbyA0`/`rAtU`/`phidB` summary at iteration 1 |

**To add:**

1. `dumpPEqn` also writes `rAU`, `UEqn.A()`, `UEqn.H()`, `HbyA` and `phi` at iteration 1. These are the
   stages upstream of what it currently covers, and `HbyA` is the present suspect.
2. brae grows `BRAE_DUMP_STAGE=<n>` writing the same quantities as **OpenFOAM-format fields**, not console
   summaries. Summaries were enough to see "these differ"; localising *where* on a 112k-cell mesh needs the
   field, so the comparison can report which cells and whether the error is at walls, at the inlet, or
   interior.
3. `tests/stage_compare.cu` — reads both sides, reports per-stage L2 **and** the spatial distribution
   (worst cells, and whether they are boundary-adjacent). One tool, all stages, any case.

**Phase 0 is the whole leverage.** Every phase below is then a measurement, not an investigation.

## Phases 1–4 — verify, then re-derive only what fails

Each phase: run the harness on **all six tutorials plus the existing gate cases**, find the first stage that
disagrees, re-derive that stage from the OF file named below, gate it, move on.

### Phase 1 — momentum: `rAU`, `rAtU`, `HbyA`, `phiHbyA`
Reference: `UEqn.H`, `pEqn.H`/`pcEqn.H` (top), `constrainHbyA.H`.
Known state: `rAtU` is EXACT on squareBend; `phiHbyA0` is **+5.11% on internal faces** while its boundary
is exactly −0.5 (= `massFlowRate`). So the defect is interior, in `UEqn.H()` or `fvc::flux(HbyA)`.
**First question to settle, one run per side:** does SUBSONIC squareBend show the same +5%? If yes, this was
never a transonic defect and B1 has been chasing a symptom.
Watch for: the `dev2(T(grad(U)))` term, `constrainHbyA` at fixed-value patches, non-orthogonal correction,
and MRF (absent here but on the same path).

### Phase 2 — energy: `he`, `T`, `alphaEff`
Reference: `EEqn.H`, and `docs/of-energy-trace.md`, which already traces this end to end with a 15-row
checklist. 13 of 15 rows are ✅; the open ones are `fvOptions(rho,he)` (refused, A2) and
`fvc::div(MRF.phi(), p)` (absent).
Known state: T matches to 9.89e-04 on squareBend at iteration 1 — this phase is in good shape.

### Phase 3 — pressure: the matrix, then the flux
Reference: `pEqn.H` (subsonic), `pcEqn.H` (SIMPLEC), and note **OF picks the file on `consistent`** —
`rhoSimpleFoam.C:78-84`.
Extend `bcoeff_compare` from the scalar transport to the pressure equation; it is already the instrument
that settled D2 and B5 to machine precision, and the pressure equation's boundary path has never been
compared.
Known state: transonic B1 is implemented with three defects found and fixed (asymmetric matrix → BiCGStab;
relax ordering; relax **sign convention** — brae's matrix is `+laplacian` with a NEGATIVE diagonal where
OF's is `-laplacian` with a positive one). Still refused, because it is stable-but-wrong on squareBend, and
the cause now looks upstream in Phase 1.

### Phase 4 — turbulence
Reference: the `TurbulenceModels` sources; `RAS/turbulence` on/off is already gated (E5b), as is the
startup `validate()` → `correctNut` ordering (E7).
Do this last: it reads the converged U and feeds back only through `nut`, so it cannot be judged while an
upstream stage is 5% out.

## Rules for re-deriving a stage

1. **Open OF's file. Do not open `gpuSimpleFoam`.** If the incompressible driver already does it right,
   that is a coincidence to be re-confirmed against OF, not a source.
2. **One term at a time, measured.** Every fix in this audit that stuck came with a number attached
   (T 7.97e-03 → 4.94e-08; nut 8.41e-02 → 2.41e-09; rho 7.02e-02 → 9.52e-04).
3. **Mutation-test every gate**, and include a negative control. The `(1-vf)` refGrad weight and the
   solver-notice false positive were both caught only by their negative controls.
4. **Match the case to what brae implements** before blaming the assembly — `validation/sbMatched` is the
   worked example. It converted "brae disagrees" into "brae's assembly is wrong", and also showed a loose
   `relTol` was *damping an inconsistency into a stable wrong answer*.
5. **Silence is the bug.** `dict_audit` reports every input read and ignored; it found all eight group-E
   items, two of which were only reachable because an earlier fix in the same group exposed them.

## What NOT to redo

16 group-A defects, groups C and E in full, D1/D2/D4/D5, B3/B5/B5b — all measured against OF and gated.
**32 gates currently pass.** Re-deriving them from scratch would risk regressions with no expected gain.
The `heRhoThermo` rho-timing fix (74×) is the model: a small, exactly-located correction found by
comparing against OF at one iteration — not by rewriting.

## Honest cost

- Phase 0: the real work, and the only phase that is pure construction.
- Phase 1: likely where squareBend's defect is; possibly small once the harness names the cell set.
- Phases 2–4: mostly confirmation, given the existing gates.
- The plan will find *more* defects than B1. That is the point — every previous sweep did, and the ones it
  found were invisible until the instrument existed.

---

# Phase 0 — DONE. Phase 1 — first result.

## Phase 0: what was built

| piece | what it does |
|---|---|
| `tools/dumpPEqn` | now also writes `stage_rAU`, `stage_rAtU`, `stage_HbyA`, `stage_phi` at iteration 1, in **both** `pcEqn.H` and `pEqn.H`, so the harness works with `consistent` either way |
| `src/applications/solvers/common/stage_dump.cuh` | `BRAE_DUMP_STAGE=<dir>`; off and free by default. Plain text on purpose — only ONE side has to be a real OpenFOAM field, and an instrument that needs debugging is worse than none |
| `tests/stage_compare.cu` | reads both sides, reports L2 **and the spatial distribution** (worst entries, and what fraction of cells carry half the error) |

Usage:

    (cd <case> && dumpPEqn)                                     # OF side, writes <case>/1/stage_*
    BRAE_DUMP_STAGE=<dir> brae_rhoSimpleFoam -case <case>        # brae side
    stage_compare <case>/1 <dir> HbyA 1e-6

## Phase 1: `HbyA` is the first stage that breaks, and it is NOT a transonic defect

Walking the chain on `validation/sbMatched` (squareBend, Mach 0.958):

| stage | L2rel | verdict |
|---|---|---|
| `rAU` | 1.69e-05 | ✅ |
| `rAtU` | 1.88e-06 | ✅ |
| **`HbyA`** | **4.76e-02** | ❌ **origin** |
| `phiHbyA0` | 5.47e-02 | ❌ inherited |
| `phid` | 5.47e-02 | ❌ inherited — *identical to phiHbyA0* |

`phiHbyA0` and `phid` carry **identical** L2 and identical error concentration: the downstream stages
transport `HbyA`'s error without adding any of their own. **The transonic term is fully exonerated.**

### It is not transonic-specific

Re-run with `transonic no` (and `PCG`/`DIC`, since the subsonic matrix is symmetric): **every number
identical** — `HbyA` 4.7636e-02, the same worst cells, the same distribution. B1 has been chasing a
symptom for several sessions. The momentum predictor is wrong on squareBend with the transonic branch
entirely out of the picture.

### It is a uniform SCALE error, not a regional one

The first read of `stage_compare` said "half the error is carried by 356 of 112000 cells", which looks
regional. It is not — those are simply the cells where `|HbyA|` is largest, because the error scales with
the value:

| cell | OF | brae | ratio |
|---|---|---|---|
| 4200 (wall, near the bend) | 470.694 | 493.972 | 1.04945 |
| 200 (interior) | 470.625 | 493.891 | 1.04944 |

Across all 3196 cells with `|HbyA| > 1` the ratio is **bimodal**:

    group A  2796 cells  ratio 1.041724   rAU mean 4.48e-03
    group B   400 cells  ratio 1.049455   rAU mean 1.25e-05

**Group B is exactly the 400 inlet-adjacent cells** (the inlet patch has exactly 400 faces, and the group's
indices run 0, 20, 40, … at stride 20). Its tiny `rAU` is the signature of a fixed-value boundary's large
diagonal. So: a ~4.17% scale error everywhere, ~4.95% adjacent to the inlet.

### Why every existing gate still passes

On `rhoBoxE` — a passing gate case — the same chain gives `rAU` 1.48e-07 ✅, `rAtU` 1.48e-07 ✅ and
`HbyA` **1.26e-04**: the same defect, **400x smaller**, which washes out before convergence (that case
matches OF to 1e-7). The defect is real and systematic; the simple cases are just insensitive to it.

### Ruled out

- Transonic branch (identical subsonic).
- `rAU`/`rAtU`, i.e. the momentum DIAGONAL including boundary `internalCoeffs` — correct on both cases.
- The relaxation factor: brae resolves `U 0.9`, matching the case.
- The relaxation SOURCE mechanism: brae builds `relaxSrc = (D - D0)*U` into `H`, as OF does.
- A multi-patch corner: the worst cells have exactly ONE boundary face each.
- Turbulence: `nut` is uniform 0 at iteration 1, so `muEff = mu`.

### Next

`HbyA = rAU*UEqn.H()` with `rAU` correct, so **`H` is ~4.2% high**. `H` is
`(offdiag . U + source)/V` plus `addBoundarySource(boundaryCoeffs)`. What differs between rhoBoxE (1.3e-04)
and squareBend (4.8e-02) is the candidate list: 3D vs 2D, a non-orthogonal bend, `sutherland` vs `const`
transport, and `flowRateInletVelocity` vs a fixed-value inlet. Dump `H` itself and the momentum
`boundaryCoeffs` next — `bcoeff_compare` already compares boundary coefficients and would need pointing at
the momentum equation.

## Phase 1, second pass: the origin is `Upred`, not `HbyA`

Adding the momentum predictor's U as its own stage (it was missing — `UEqn.H()` uses the JUST-SOLVED U,
not the initial field, which is why a 4% `HbyA` error was inexplicable on a case whose `0/U` is uniform
zero):

| stage | L2rel | |
|---|---|---|
| `rAU` | 1.69e-05 | ✅ |
| `rAtU` | 1.88e-06 | ✅ |
| **`Upred`** | **4.7636e-02** | ❌ **ORIGIN** |
| `HbyA` | 4.7636e-02 | inherited — bit-identical to Upred |
| `phiHbyA0` / `phid` | 5.4725e-02 | inherited |

`HbyA` adds NOTHING. `H` faithfully propagates a predictor solution that is already 4.8% wrong. Combined
with `rAU` being correct — and `rAU = 1/A` includes the boundary `internalCoeffs` — **the momentum
DIAGONAL is right and the momentum SOURCE is wrong.**

### What the source can be at iteration 1

`0/U` is uniform `(0 0 0)` and `0/p` is uniform 110000, so at iteration 1:

- `grad(p)` = 0 (uniform p, zeroGradient inlet, fixedValue outlet at the same value) -> no pressure source
- the relaxation source `(D - D0)*U` = 0 (U is zero)
- `dev2(T(grad(U)))` = 0 in the interior
- `div(phi,U)` internal = 0 (phi is zero internally)

So the momentum source is **almost entirely the BOUNDARY contribution** — the inlet's `U_b` through
`boundaryCoeffs`, plus the noSlip walls.

### The inlet is PART of it but not all

Replacing `flowRateInletVelocity` with a plain `fixedValue (26 0 0)` (so `U_b` is identical on both sides
by construction) drops the error 4.76e-02 -> **3.42e-02**. Real, but the majority survives.

This also shows why "the boundary mass flux matches exactly" was NOT proof the inlet is right: with
`flowRateInletVelocity` the mass flux is pinned to `mdot` on both sides *by construction*, so
`U_b = mdot/(rho_b*A)` can still differ if `rho_b` does, while `phiHbyA0` at the boundary stays exactly
-0.5 on both. That measurement excluded nothing.

### Next

Dump the momentum equation's `boundaryCoeffs` and `internalCoeffs` per patch and compare. `bcoeff_compare`
already does exactly this for a scalar transport equation and needs pointing at the momentum equation;
`tools/dumpScalarMatrix` is the OF-side counterpart. With the diagonal proven correct and the source
proven wrong, that comparison is the whole remaining question.

## Harness bug found and fixed while doing this

`stageDumpFirstOnly()` used a single global latch. Adding a second dump site made the FIRST one silently
stop writing, and `stage_compare` reported "missing" for stages being computed perfectly well. Now latched
per call site. Noted because an instrument that fails silently is the exact defect class this project
exists to eliminate, and it is worth knowing the instrument had it too.

## Phase 1, third pass: everything upstream of the momentum SOLVE is verified correct

Reading OF's source for each term (the ground rule), then measuring it:

| input, at iteration 1 | brae | OF | verdict |
|---|---|---|---|
| `muEff` (OF: `rho*nuEff()`, `linearViscousStress.C:130`) | 2.139698e-04 | 2.1397e-04 | ✅ ratio 0.999999 |
| `rAU`, `rAtU` (the momentum DIAGONAL, incl. boundary `internalCoeffs`) | — | — | ✅ 1.7e-05 / 1.9e-06 |
| inlet `U_b` | 523.0871 | 523.0870 | ✅ |
| inlet mass flux | −0.5 | −0.5 | ✅ exact (= `massFlowRate`) |
| **`Upred`** | — | — | ❌ **4.7636e-02** |

brae's `muEff` assembly is right and cites the OF file: `muEff = rho*nut + mu`, because
`nuEff() = nut + mu/rho` so `rho*nuEff() = rho*nut + mu`. Confirmed numerically, not just by reading.

**A trap worth recording:** `nut` is NOT zero at iteration 1 even though `0/nut` is `uniform 0` — the
turbulence model's `validate()`/`correctNut` sets it from `k` and `epsilon` at construction. OF's `muEff`
is 2.14e-04 against a laminar `mu` of 4.19e-05, so the turbulent part is 5x the laminar one from the very
first momentum solve. Reasoning "nut is zero at iteration 1, so turbulence cannot be involved" is wrong,
and that assumption was made earlier in this investigation.

**Also recorded:** comparing brae's `nuConst_` against OF's `muEff()` is not a comparison — those are the
laminar part and the total. The first attempt did exactly that and produced a spurious 5.1x "discrepancy".
The dump now takes the assembled `nuEff` from where it is built.

### So the fault is in the momentum SOURCE or its OFF-DIAGONALS

The diagonal is right, the diffusivity is right, the inlet is right. What is left in
`fvm::div(phi,U) + turbulence->divDevRhoReff(U) == -fvc::grad(p)`:

- **the noSlip WALLS** — 22400 faces against the inlet's 400, and completely unverified. The obvious
  candidate purely on surface area.
- the EXPLICIT stress `-fvc::div((rho*nuEff)*dev2(T(grad(U))))` — brae carries this in `relaxSrc`
- the initial `phi`: OF builds it in `compressibleCreatePhi.H` as `linearInterpolate(rho*U) & Sf`, brae as
  `fvc::rhoFlux(rho0, U, ...)`. **Interpolation does not commute with the product**, so
  `interp(rho*U)` and `interp(rho)*interp(U)` are not the same field — worth checking even though both
  give zero internally at iteration 1 with `U0 = 0`.
- `grad(p)` is zero at iteration 1 (p uniform, zeroGradient inlet, fixedValue outlet at the same value)

### Next

Compare the momentum equation's `internalCoeffs`/`boundaryCoeffs` per patch, walls first.
`tests/bcoeff_compare.cu` already does this for a scalar transport equation and `tools/dumpScalarMatrix`
is the OF-side counterpart; both need pointing at the momentum equation. With the diagonal proven right
and the source proven wrong, that is the whole remaining question.

### Harness limitation to fix

`stage_compare` reports "missing" when OF writes a field as `uniform` (n=1) against brae's n=112000. The
muEff comparison had to be done by hand because of it. Expand a uniform OF field to the brae count.

## Phase 1, FOURTH pass: the defect is the INLET's boundary diffusivity

Momentum boundary coefficients per patch, at iteration 1, brae vs OF (`tools/dumpPEqn` now writes
`stage_UIC`/`stage_UBC` after `UEqn.relax()` + `constrain()`, the same state brae's `iC`/`bCb` are in):

| patch | faces | quantity | brae | OF | verdict |
|---|---|---|---|---|---|
| **walls** | 22400 | sum\|iC\| | 0.00474195 | 0.00474195 | ✅ **EXACT** |
| walls | 22400 | sum\|bC\| | 0 | 0 | ✅ |
| outlet | 400 | both | 0 | 0 | ✅ |
| **inlet** | 400 | sum\|iC\| | **0.0671659** | **0.00042794** | ❌ **157x** |
| **inlet** | 400 | sum\|bC\| | 235.134 | 261.767 | ❌ 0.898 |

22400 wall faces match EXACTLY on the same formula with the same diffusivity. Only the 400 inlet faces are
wrong. Dumping the boundary diffusivity itself:

| patch | brae `nuEffBnd` | OF `muEff` boundary | |
|---|---|---|---|
| **inlet** | **3.358293e-02** | **2.1397e-04** | ❌ **157x** |
| outlet | 2.139698e-04 | 2.1397e-04 | ✅ |
| walls | 4.191435e-05 | 4.19143e-05 | ✅ |

That single wrong value explains the entire chain: inlet `iC` 157x -> momentum source wrong -> `Upred`
4.76e-02 -> `HbyA` (bit-identical) -> `phiHbyA0` -> `phid` -> `p` -> `U`.

### The likely mechanism (inferred, not yet proven)

brae's implied inlet `nut` is 0.0877 against OF's 4.5e-04. squareBend's `0/nut` inlet is `calculated`, and
for kEpsilon brae computes such patches as `Cmu*k_b^2/eps_b`. Its inlet `k_b` looks like the REFRESHED
turbulent-inlet value: OF's `t=1` inlet k is 1026.08, and `1.5*(0.05*523.087)^2 = 1026` exactly — i.e. k
recomputed from the CURRENT inlet velocity and `intensity 0.05`.

OF's `turbulentIntensityKineticEnergyInlet::updateCoeffs()` runs when the k EQUATION is assembled, which in
rhoSimpleFoam is AFTER the momentum predictor. brae refreshes its turbulent inlets before it. So the
momentum equation sees a boundary `nut` built from this iteration's refreshed k, where OF's sees the
previous iteration's.

If that is right, it is the same shape as E7 (validate ran before the thermo existed) and the heRhoThermo
rho timing: **the right formula at the wrong point in the iteration**. A3 fixed "turbulent inlets frozen at
set-up"; this would be that fix over-corrected, refreshing one phase too early.

### Proven vs inferred

**Proven by measurement:** the inlet `nuEffBnd` is 157x OF's at the momentum solve; walls and outlet are
exact; inlet `iC` is 157x and `bC` is 10% low; `Upred` is 4.76e-02 and every downstream stage inherits it
unchanged.

**Inferred, still to confirm:** that the cause is the turbulent-inlet refresh ordering. Confirm by
computing `Cmu*k_b^2/eps_b` from brae's refreshed inlet k/eps and checking it reproduces 0.0877, then by
moving the refresh after the momentum predictor and re-running this comparison.

### A trap this pass walked into

OF's `t=1` field files are written AFTER that iteration's turbulence solve, but the momentum equation runs
FIRST. Comparing brae's momentum-solve state against OF's `t=1` nut gave a 32000x "discrepancy" that was
purely a difference of MOMENT. The comparison only became meaningful against `stage_muEff`, which
`dumpPEqn` writes at the momentum solve. Any stage comparison must pin the moment, not just the quantity.

## Chain state after the Phase 1 fix — the next defect is the PRESSURE EQUATION

Full chain, transonic squareBend, iteration 1, both sides at the same moment:

| stage | L2rel | verdict |
|---|---|---|
| `rAU` / `rAtU` | 4.58e-05 / 5.00e-06 | ✅ |
| `Upred` | 7.04e-05 | ✅ |
| `HbyA` | 7.04e-05 | ✅ |
| `phiHbyA0` | 7.60e-05 | ✅ |
| `phid` | 7.59e-05 | ✅ |
| `T` | 1.40e-06 | ✅ |
| `rho` | 1.87e-06 | ✅ |
| **`p`** | **1.07e-01** | ❌ **NEXT** |
| `U` (after correction) | 2.66e+00 | inherited from p |
| `k` / `epsilon` / `nut` | 0.91 / 9.46 / 0.99 | inherited — solved last, on the wrong U |

Every input to the pressure equation now agrees with OF to **7e-05**, and `p` still comes out **10.7%**
wrong. That isolates the pressure equation itself for the first time: before the momentum fix its inputs
were 5% off, so input error and equation error were indistinguishable.

It also matches the run behaviour — squareBend is stable but its PRESSURE residual stalls at ~7.5e-02 while
Ux fell to 4.1e-03.

### Phase 3 (pressure) is therefore next, and the instrument already exists

Give the pressure equation the same treatment the momentum equation just had:

1. `tools/dumpPEqn` writes `stage_pIC`/`stage_pBC` per patch, the way it now writes `stage_UIC`/`stage_UBC`
   (that comparison is what found the inlet defect — walls exact, inlet 157x).
2. brae dumps its `pIC`/`pBC` at the same point.
3. Compare per patch. squareBend's p BCs are `zeroGradient` inlet + `fixedValue` outlet, so the patches
   carry very different coefficients and a vacuous pass is unlikely.
4. Then `pEqn.flux()` — `dumpPEqn` already writes it — against brae's reconstructed flux. That link has
   never been compared and is the one place where a matrix can be right and the FLUX still wrong, which is
   exactly the signature of a stalling residual.

Order matters here too: compare the COEFFICIENTS before the flux, because the flux is built from them.

## Phase 3, first pass: the pressure BOUNDARY coefficients (fixed, but not the cause)

`tools/dumpPEqn` now writes `stage_pIC`/`stage_pBC` per patch (after `pEqn.relax()`, which does not touch
either — it moves the diagonal and the source); brae dumps the same under `BRAE_DUMP_STAGE`.

| patch | quantity | OF | brae, before | brae, after |
|---|---|---|---|---|
| inlet (zeroGradient p) | iC | −4.54545e-06 | +4.54545e-06 | ✅ exact |
| walls | iC / bC | 0 / 0 | 0 / 0 | ✅ |
| **outlet** (fixedValue p) | iC | 0.0373221 | **0.0976136** | **0.0373221** ✅ |
| **outlet** | bC | 4105.43 | **10737.5** | **4105.43** ✅ |

(The sign flip on the inlet is expected and correct: brae's pressure matrix is the NEGATIVE of OF's.)

**The defect:** OF's pressure laplacian is `- fvm::laplacian(rhorAtU, p)` with `rhorAtU = rho*rAtU`
(`pcEqn.H:13`), and an fvMatrix's boundary coefficients are built from the SAME diffusivity as its internal
ones. brae passed the unweighted `rAtU` to `deviceBCLaplacianCoeffs` while its internal coefficients used
`interpolate(rho*rAtU)` — so every pressure boundary coefficient was too large by exactly **1/rho**.
Measured 2.6154x against 1/0.3823 = 2.6157.

Why the inlet and walls hid it: squareBend's inlet is `zeroGradient` on p (coefficients ~0) and the walls
likewise. Only a `fixedValue` pressure patch carries a non-trivial coefficient, and there is exactly one.

### It is NOT the cause of the remaining 10.7%

`p` is unchanged at 1.066e-01 after the fix, and that is explainable rather than surprising: for a
`fixedValue` patch the imposed boundary value is `bC/iC`, so scaling BOTH by 1/rho leaves the imposed
pressure identical. The fix is still correct and necessary — the coefficients now match OF exactly, and
they feed the flux reconstruction (`deviceMatrixFluxBoundary` reads pIC/pBC) where the ratio does NOT
cancel.

### Next: the INTERNAL pressure coefficients and the source

With every boundary coefficient exact and every input at 7e-05, what is left in the pressure equation:

1. the internal laplacian coefficients `pD_`/`pU_`/`pL_` — `rAUf = interpolate(rho*rAtU)`, which looks
   right but has never been compared
2. the implicit `fvm::div(phid,p)` internal coefficients
3. the source `divPhiH = div(phiHbyA)` — note that in the transonic branch phiHbyA -> 0 on BOTH sides, so
   the equation is driven almost entirely by the boundary and the div term
4. `pEqn.flux()` vs brae's reconstruction — `dumpPEqn` already writes OF's

Dump the assembled diagonal and off-diagonals next. That is the same comparison `dumpScalarMatrix` +
`bcoeff_compare` already do for a scalar transport equation (D2), pointed at the pressure equation.

## Phase 3, second pass: the pressure system resolved into its parts

`dumpPEqn` now also writes `stage_pD` (the diagonal INCLUDING internalCoeffs, `pEqn.D()`) and `stage_pSrc`
(the full RHS = `source()` plus the boundary contribution, applied explicitly since `addBoundarySource()`
is protected). brae dumps `diagCp_`/`bp_` after `deviceFoldPressure`, which are the like-for-like
counterparts. brae's matrix is the NEGATIVE of OF's, so the comparison is against the negation.

| component | L2rel | distribution |
|---|---|---|
| diagonal `D()` | **3.67e-06** | ✅ essentially exact (sum 25.921 on both, sign flipped) |
| boundary `pIC`/`pBC` | exact | ✅ after the 1/rho fix above |
| **RHS** | **2.45e-04** | ❌ half the error in **269 of 112000 cells (0.24%)** — CONCENTRATED |
| resulting `p` | 1.07e-01 | |

So: the matrix is right, the boundary coefficients are right, and the SOURCE is wrong in ~269 cells.
`bp_ = divPhiH + pBC` and `pBC` now matches exactly, so the error is in **`divPhiH = div(phiHbyA)`**.

### Why 2.45e-04 in the source becomes 10.7% in p

The transonic formulation drives `phiHbyA` to ~zero on both sides (for perfectGas `psi*p == rho`, so the
subtraction cancels it), which means the pressure system is nearly homogeneous — driven almost entirely by
its boundary. Such a system is ill-conditioned, and a small, SPATIALLY CONCENTRATED source error can
produce a large solution error. A 300x amplification from 2.4e-04 to 1.1e-01 is consistent with that, and
it also explains the stalling residual: the iteration is stable but converging to a different fixed point.

### Next

Find why `div(phiHbyA)` differs in those 269 cells when `phiHbyA` itself is ~0 on both sides.

- Compare `phiHbyA` AFTER the subtraction (brae zeroes the boundary exactly with `phiHb -= phiHb0`; OF's
  measured internal sum is 2.0e-16, but its BOUNDARY values were never checked)
- Map the 269 cells: are they inlet-adjacent (400 cells), outlet-adjacent (400), or at the bend?
- Note the ordering trap already recorded twice: `phid` is built from the ORIGINAL phiHbyA and the SIMPLEC
  term is added BESIDE the subtraction, not folded in first (`pcEqn.H:26-29`)

`stage_compare` already reports the worst-cell indices, and `writeCellCentres` maps them to coordinates —
that is how the momentum defect was localised to the inlet.

## Phase 3, third pass: the pressure SOURCE, localised to the inlet cells

Mapping the RHS error cells (via `writeCellCentres` + the polyMesh owner/boundary lists):

    269-cell error set, by adjacency:
       inlet          209
       inlet+walls     60
       interior         0

**Every single one is inlet-adjacent.** And the values are not subtly different, they are present vs absent:

| | OF `stage_pSrc` | brae `stage_pSrc` |
|---|---|---|
| every inlet-adjacent cell | **0.0025** | **-3.9e-11** (zero) |

400 inlet cells x 0.0025 = 1.0, against the measured RHS sum difference of 1.34.

### What is ruled out

- It is not the boundary coefficients: `pIC`/`pBC` now match OF exactly on all three patches, and the
  inlet's pEqn `bC` is 0 on both sides (its `iC` is `phid_b`, matching to 6 s.f.).
- It is not `phiHbyA` at the boundary: OF writes `phiHbyA` inlet as **`uniform 0`** after the transonic
  subtraction — exactly what brae produces. Both boundary fluxes are zero.
- It is not the diagonal: 3.67e-06.

So OF's `source()` carries a term at inlet cells that brae's `divPhiH` does not, even though the flux
field it is supposedly the divergence of is zero on both sides at that patch.

### Where to look next

`source()` for `fvc::div(phiHbyA) + fvm::div(phid,p) - fvm::laplacian(rhorAtU,p) == fvOptions(psi,p,rho)`
is `-fvc::div(phiHbyA)*V` plus whatever `fvOptions` contributes. Candidates, in order:

1. **`fvOptions(psi, p, rho.name())`** — squareBend declares no fvOptions, but the term is in the equation
   and OF may still contribute through it. Check `constant/fvOptions` and whether OF instantiated any.
2. `fvc::div(phiHbyA)` on a field whose INTERNAL faces near the inlet are not exactly zero even though the
   boundary face is. brae's `phiHi -= fac*phiHi0` gives EXACTLY zero because for perfectGas
   `interp(psi*p) == interp(rho)` identically; OF's may leave rounding-level residue that is nonetheless
   0.0025 after `/V` on a small near-inlet cell. Compare `phiHbyA` INTERNAL faces adjacent to the inlet
   cells, not just the aggregate (its total is 2.0e-16, which can hide per-face structure).
3. The cell volume: 0.0025 is suspiciously round. Check `V` for those cells and whether the source is
   really `div*V` or `div` on one side and not the other.

Item 3 is the cheapest and would be a units/scaling error rather than a physics one; do it first.

## Phase 3, fourth pass: the pressure source resolved to ONE line

The 0.0025-per-inlet-cell gap is the sum of TWO terms that cancel in brae and do not in OF.

**OF's relaxation is the source of the 0.0025, and the arithmetic pins it exactly:**
`fvMatrix::relax()` computes `D = |D0 + sumMag(iC)|/alpha - sum(iC)`; with the inlet's pEqn `iC` negative
(-4.54545e-06 over 400 faces = -1.136e-08 per face) that is `D0 + |iC| + |iC|`, so
`delta = 2|iC| = 2.2727e-08` and `source += delta*p = 2.2727e-08 * 110000 = ` **0.0025**. Measured: 0.0025.

**brae computes that term correctly.** Dumping `deltaM` and `delta*p` directly:
`delta = -2.272727e-08`, `delta*p = -2.500000e-03` at every inlet cell — the exact negation of OF's, which
is right for brae's sign convention (its pressure matrix is the negative of OF's).

**But brae's final `bp_` at those cells is -3.9e-11, not -0.0025.** With `pBC(inlet) = 0` (measured) and
`bp_ = divPhiH + pBC`, that forces `div(phiHbyA)_brae = +0.0025` at inlet cells — where OF's is exactly
zero, because OF's `phiHbyA` after the transonic subtraction is `uniform 0` on the inlet patch and
`max|.| = 8.6e-18` on internal faces (zero faces above 1e-12).

### The suspect line

OF's SIMPLEC correction is `phiHbyA += interp(rho*(rAtU - rAU))*snGrad(p)*magSf` (`pcEqn.H:27`). At a
**zeroGradient-p inlet, `snGrad(p) = 0`**, so OF adds nothing there. brae's boundary handling of the same
term does not appear to honour that: the transonic block deliberately keeps the SIMPLEC boundary
contribution with

    deviceAxpy(-1.0, phiHb0, phiHb);   // "phiHb -= phiHb0" rather than zeroing

which was written to preserve the SIMPLEC term OF also keeps. If brae's boundary SIMPLEC correction is not
built from `snGrad(p)` — which is identically zero on a zeroGradient patch — it injects a flux OF never
has, exactly +0.0025 per inlet cell.

### To confirm and fix

1. Dump `divPhiH` BEFORE the relax adds to it and check it is +0.0025 at inlet cells (the inference above
   is arithmetic, not yet a direct measurement).
2. Then check how brae builds the boundary half of the SIMPLEC flux correction and whether it uses
   `snGrad(p)`. On a zeroGradient patch that is zero by construction, so the correction must vanish there.

This is the same shape as the pressure boundary-diffusivity defect found earlier in this phase: the
INTERNAL face treatment was right and the BOUNDARY treatment was not, and the case hid it because only one
patch type exercises the term.
