# OpenFOAM v2412 rhoSimpleFoam — the energy equation, traced end to end

A reference for comparing brae against OpenFOAM **term by term**, read off OF's source rather than
reconstructed from memory. Every claim below cites the file and line it came from, so it can be re-checked
when OF changes. Written because the recurring defect in this project has never been "brae cannot do X" —
it is "brae does X slightly differently and converges anyway", and that is only findable against an exact
reference.

Scope: the `he` equation and everything it touches — schemes, boundary conditions, diffusivity, the kinetic
term, and the thermo update that closes the loop.

---

## 1. The equation

`applications/solvers/compressible/rhoSimpleFoam/EEqn.H`:

```cpp
volScalarField& he = thermo.he();

fvScalarMatrix EEqn
(
    fvm::div(phi, he)
  + (
        he.name() == "e"
      ? fvc::div(phi, volScalarField("Ekp", 0.5*magSqr(U) + p/rho))   // sensibleInternalEnergy
      : fvc::div(phi, volScalarField("K",   0.5*magSqr(U)))           // sensibleEnthalpy
    )
  - fvm::laplacian(turbulence->alphaEff(), he)
 ==
    fvOptions(rho, he)
);

if (MRF.active()) { EEqn += fvc::div(MRF.phi(), p); }

EEqn.relax();
fvOptions.constrain(EEqn);
EEqn.solve();
fvOptions.correct(he);
thermo.correct();
```

Five things to note, each a place brae could differ silently:

1. **The solved variable is `he`, not `T`.** `T` is a *result* of `thermo.correct()`.
2. **The kinetic term is EXPLICIT** (`fvc::div`), while the convection of `he` is implicit (`fvm::div`).
   They are separate fvSchemes entries — see §2.
3. **The kinetic term's CONTENT depends on the energy variable**: `Ekp = 0.5|U|² + p/ρ` for `e`, but
   `K = 0.5|U|²` for `h`. The `p/ρ` piece is not small — order 40% of `e` for air at STP.
4. **`EEqn.relax()`** — the equation, not the field. Uses `relaxationFactors/equations/(e|h)`.
5. **`thermo.correct()` closes the loop**: `he → T → ρ, ψ, μ, α`. Everything downstream reads those.

---

## 2. Which fvSchemes entries it reads

`fvm::div(phi, he)` and `fvc::div(phi, K|Ekp)` resolve through `mesh.divScheme(name)`
(`finiteVolume/finiteVolume/fvm/fvmDiv.C:58`), keyed by the **term name string**:

| term | fvSchemes key | notes |
|---|---|---|
| `fvm::div(phi, he)` | `div(phi,h)` or `div(phi,e)` | named for the energy variable |
| `fvc::div(phi, K\|Ekp)` | `div(phi,K)` or `div(phi,Ekp)` | **separate entry**, explicit |
| `fvm::laplacian(alphaEff, he)` | `laplacian(alphaEff,h\|e)` → usually `default` | |
| `fvc::div(MRF.phi(), p)` | `div(phid,p)`-style, MRF only | |

The lookup is generic: exact key → regex → `default`. A solver adding a new transported scalar needs no
new code in the scheme layer.

> **brae divergence (C7, open):** brae resolves schemes into hardcoded per-field booleans
> (`luHe`, `luKin`, …) rather than a name-keyed table, so every new scalar needs a new flag AND a new
> parser branch, and `default` is never resolved. C4 fixed the specific case of `div(phi,K)` being merged
> into `div(phi,h)`'s slot; the shape is still wrong.

---

## 3. Boundary conditions — `T`'s BC decides `he`'s

`basicThermo::heBoundaryTypes()` (`thermophysicalModels/basic/basicThermo.C:206-228`) maps each `T` patch
to an `he` patch type. **This is the single most defect-prone place in the whole energy path**, because the
`he` BC is never written by the user and never appears in the case.

| `T` patch type | → `he` patch type |
|---|---|
| `fixedValue` | `fixedEnergy` |
| `zeroGradient` **or** `fixedGradient` | `gradientEnergy` |
| `mixed` | `mixedEnergy` |
| `fixedJump` | `energyJump` |
| `fixedJumpAMI` | `energyJumpAMI` |

And the conversions themselves (`.../derivedFvPatchFields/*/`):

| BC | what it sets | source |
|---|---|---|
| `fixedEnergy` | `value = thermo.he(pw, Tw, patchi)` | value only |
| `gradientEnergy` | `gradient() = thermo.Cpv(pw, Tw, patchi) * Tw.snGrad()` | `gradientEnergyFvPatchScalarField.C:111` |
| `mixedEnergy` | `valueFraction() = Tw.valueFraction()`; `refValue() = thermo.he(pw, Tw.refValue())`; `refGrad() = Cpv * Tw.refGrad()` | `mixedEnergyFvPatchScalarField.C:120-123` |

**The `Cpv` factor is the trap.** A gradient copied across unscaled is wrong by ~718 J/kg/K (air, `e`) —
the difference between a heat-flux wall and a nearly adiabatic one. This is exactly defect **B5**, and
brae's `fixedGradient` T support was measured at T 7.97e-03 → 4.94e-08 once the scaling was applied.

`mixedEnergy` is now **PORTED and gated** (`mx_vs_openfoam`). `MixedPatchField` carries a per-face
`refGrad` handed to `DeviceBoundary::refGrad` through the same hook `fixedGradient` uses, and the reader
accepts `refGradient` alongside `refValue` and `valueFraction`.

**The subtlety `fixedGradient` never exposes:** OF weights the refGrad term by **(1-vf)**, not 1 — every
one of `valueBoundaryCoeffs`, `gradientBoundaryCoeffs` and `evaluate()` is a `lerp`
(`mixedFvPatchField.C:239-310`). With `vf = 0` that reduces to 1, which is exactly why B5 was correct
without it. Measured cost of assuming 1: boundaryCoeffs **9.81e-02** on the `bc_vs_openfoam` mixed patch,
and T/p/rho all failing `mx_vs_openfoam`.

---

## 4. The diffusivity `alphaEff`

Chain: `turbulence->alphaEff()` → `EddyDiffusivity::alphaEff()` (`EddyDiffusivity.H:136-146`)
→ `transport_.alphaEff(alphat())` → `heThermo::alphaEff` (`heThermo.C:874`):

```cpp
alphaEff = this->CpByCpv() * (this->alpha_ + alphat)
```

Two separate factors, both easy to drop:

- **`CpByCpv`** = 1 for `sensibleEnthalpy`, but **γ = Cp/Cv (~1.4)** for `sensibleInternalEnergy`. So an
  `e` case diffuses ~40% faster than an `h` case with the same `alpha`. Dropping it still converges.
- **`alphat`** is the turbulent part, owned by the turbulence model and set by
  `alphatWallFunction` (default `Prt = 0.85`) — which differs from the *model's* `Prt` (1.0). That
  mismatch was defect **A6**, measured at T 5.3e-03 → 2.7e-07.

There is a **patch-wise overload** (`alphaEff(patchi)`), and OF's laplacian uses the PATCH value at a
boundary face, not the adjacent cell's. That distinction is invisible with `transport const` and worth
~60% of the wall heat flux under `sutherland` with a hot wall. brae supplies it via `deviceAlphaEffBoundary`
and the face-diffusivity laplacian variant `deviceBCLaplacianCoeffsFace` — the variant B5's first attempt
forgot, leaving the heat-flux wall completely unheated.

---

## 5. `thermo.correct()` — closing the loop

For `hePsiThermo` / `heRhoThermo` with `perfectGas` + `hConst`:

```
T   = T(he)                     inverse of he = Cp*T + Hf  (h)  or  Cv*T  (e)
psi = 1/(R T)
rho = p*psi          (hePsiThermo)   or   mixture.rho(p,T)  (heRhoThermo) -- identical for perfectGas
mu  = sutherland(T)  or  const
alpha = mu/Pr
```

`R = RR/molWeight`, and **`RR` is not a literal**: OF builds it as `NA*k` from `DimensionedConstants` in
`etc/controlDict`, merged project → site → user with later winning. OF v2412 lands on
**8314.47006650545**, not the CODATA-2018 8314.46261815324. That 8.958e-07 gap reaches ρ in every
compressible case (finding **E6**).

---

## 6. Ordering within the SIMPLE step

`rhoSimpleFoam.C`: `UEqn.H` → `EEqn.H` → (`pEqn.H` | `pcEqn.H`) → turbulence.

The energy equation therefore runs on the **momentum predictor's** velocity and the **previous**
iteration's pressure. `thermo.correct()` at the end of `EEqn.H` means the pressure equation sees a thermo
consistent with the just-solved `he`. Getting this order wrong is not a convergence problem — it converges
to a different answer.

Also note (`pcEqn.H:120`): **`rho.relax()` is SKIPPED in the transonic branch**, and `adjustPhi` is only
called in the non-transonic one.

---

## 7. Checklist for comparing brae

Each row is a place a silent difference can hide. Status as of this writing.

| # | Check | brae |
|---|---|---|
| 1 | `div(phi,he)` scheme honoured | ✅ A4, gate `ke2_vs_openfoam` |
| 2 | `div(phi,K\|Ekp)` its OWN scheme, not he's | ✅ C4 (right result, wrong shape — C7) |
| 3 | `Ekp` includes `p/ρ` for `e` | ✅ |
| 4 | `fixedEnergy` from fixedValue T | ✅ |
| 5 | `gradientEnergy` = **Cpv** × snGrad(T) | ✅ B5, gate `hf_vs_openfoam` |
| 6 | `mixedEnergy` (valueFraction + refValue + **refGrad**) | ✅ **PORTED** — gate `mx_vs_openfoam` (T 8.75e-08 end-to-end, evaluated wall value 3.36e-09) + `bc_vs_openfoam` (coefficients to 5.6e-15); 3 mutations |
| 7 | `alphaEff` includes `CpByCpv` | ✅ |
| 8 | `alphaEff` uses the PATCH value at boundary faces | ✅ `deviceBCLaplacianCoeffsFace` |
| 9 | `alphat` wall `Prt` 0.85 ≠ model `Prt` 1.0 | ✅ A6, gate `rx_vs_openfoam` |
| 10 | `EEqn.relax()` uses `equations/(e\|h)` | ✅ |
| 11 | `thermo.correct()` after the he solve, before pEqn | ✅ |
| 12 | `RR` resolved from DimensionedConstants | ✅ E6, test `foam_constants` |
| 13 | boundary `internalCoeffs`/`boundaryCoeffs` match OF | ✅ D2, gate `bc_vs_openfoam` (machine precision) |
| 14 | `fvOptions(rho, he)` sources/constraints | ❌ **refused** (A2) |
| 15 | `fvc::div(MRF.phi(), p)` for MRF | ❌ not in the compressible driver |

---

## 8. How to re-derive any of this

```bash
# the equation
cat $FOAM_APP/solvers/compressible/rhoSimpleFoam/EEqn.H
# T -> he BC dispatch
grep -n "heBoundaryTypes" -A 40 $FOAM_SRC/thermophysicalModels/basic/basicThermo/basicThermo.C
# the three energy BCs
find $FOAM_SRC -name "gradientEnergyFvPatchScalarField.C" -o -name "mixedEnergyFvPatchScalarField.C"
# alphaEff
grep -rn "alphaEff" $FOAM_SRC/thermophysicalModels/basic/lnInclude/heThermo.C
# and the matrix itself, term by term, against brae:
(cd tools/dumpScalarMatrix && wmake) && ctest -R bc_vs_openfoam
```

`tools/dumpScalarMatrix` dumps OF's assembled `fvScalarMatrix` — diagonal, source, `internalCoeffs` and
`boundaryCoeffs` per patch. `tests/bcoeff_compare.cu` reads those back and compares brae's. That pair is
the instrument for rows 5, 8 and 13, and is how B5 was confirmed in OF's own coefficient terms
(`sum|BC| = 0.025 = 40 faces × γ·|Sf|·g`) rather than by a converged-field comparison.
