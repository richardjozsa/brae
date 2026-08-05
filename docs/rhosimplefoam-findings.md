# rhoSimpleFoam findings — status board

From a source-level audit of brae against the six stock OpenFOAM v2412 rhoSimpleFoam tutorials, plus
direct matrix comparison against OpenFOAM (`tools/dumpScalarMatrix`).

**Ordered by expected field error, not by effort.** Fix top-down.

The dangerous class is not "missing feature" — it is **present, parsed, and never applied**, because that
converges to a plausible wrong answer and says nothing. Every fix in group A must either apply the input
or refuse the case; see `src/applications/solvers/common/brae_notice.cuh`.

## A. Silent wrong behaviour — runs, converges, wrong

| # | Finding | Where | Impact | Status |
|---|---|---|---|---|
| A1 | `inletOutlet` on T never resolved. **Measured cost: T off by 276% (L2rel 2.8e+00) vs OpenFOAM.** `dbHe_` carried the mask but `deviceUpdateInletOutlet` was never called on it, so outlet enthalpy stayed clamped at `fixedValue(inletValue)` with full convection + laplacian coupling. Propagates to `rhoBnd_` → outlet density → mass flux → pressure. **All 6 tutorials hit this.** | `device_simple_foam.cu` ~421 | highest | ✅ **FIXED** |
| A2 | `fvOptions` never read by the compressible driver (`grep -c fvOption` → 0, vs 5 in simpleFoam). `angledDuctExplicitFixedCoeff` silently dropped `explicitPorositySource` (dominates its momentum balance), `fixedTemperatureConstraint`, `scalarFixedValueConstraint`. | `gpuRhoSimpleFoam.cu` | very high | ✅ **REFUSED** (apply later) |
| A3 | Turbulent-inlet BCs frozen at set-up, not recomputed per iteration. **Measured: k 1.95e-01 frozen vs 4.4e-06 refreshed — a factor of 44,000.** OF recomputes `k = 1.5(I·\|U\|)²` and `ε = Cmu75·k^1.5/L` every `updateCoeffs`. Harmless with a fixedValue U inlet; with `flowRateInletVelocity` the seed density differs from the converged one → **k ≈ +41%, ε ≈ +67%** on angledDuct. | `turbulent_inlet.cuh:39,70` | high | ✅ **FIXED** + gate `ti_vs_openfoam` |
| A4 | `div(phi,K)` / `div(phi,Ekp)` ignores `linearUpwind` and runs upwind. **Measured: T 2.0e-02 -> 1.9e-06 (10,600x).** `deviceEnergyKineticSource` has no linearUpwind parameter. On `sensibleInternalEnergy`, `Ekp = 0.5\|U\|² + p/ρ` is large (p/ρ ≈ 86 kJ/kg vs e ≈ 215 kJ/kg), so first-ordering it is not cosmetic. | `device_energy.cuh:79`, `device_simple_foam.cu:1271` | medium-high | ✅ **FIXED** + gate `ke2_vs_openfoam` |
| A5 | `uniformFixedValue` with a Function1/expression `uniformValue` silently reuses a stale scalar (squareBendLiq T walls → constant 350 K instead of the expression). The reader relied on "dispatch throws when there is no value", which fails when an OVERRIDDEN `type fixedValue` entry left one behind. Now recorded by name and refused at construction; the constant forms are unaffected. | `foam_field_reader.cuh:334` | medium | ✅ **REFUSED** + gate `uniform_function1` (mutation-tested) |
| A6 | Per-face `Prt` for `alphatWallFunction` matched by **exact patch name only**. **Measured: T 5.3e-03 -> 2.7e-07 (19,000x).** Fixed via a shared `findPatchEntry` (exact -> group -> regex, last wins) now used by all four sites, including the mask builder added for A3; a regex key (`"(?i).*walls"`) falls back to model Prt 1.0 instead of 0.85 → wall `alphat` and heat flux ~15% low. Same limitation in `turbulent_inlet.cuh:30,58`. | `gpuRhoSimpleFoam.cu:217` | medium | ✅ **FIXED** + gate `rx_vs_openfoam` |
| A7 | `freestreamVelocity`/`freestreamPressure` valueFraction uses `(φ_b/(ρ_b·\|Sf\|))/\|U_cell\|` where OF uses `(U_patch·n)/\|U_patch\|` — mixed numerator/denominator, so not a true cosine. Small at a true far field. | `device_boundary_flow.cu:66` | low-medium | ☐ open |
| A8 | `pMaxFactor`/`pMinFactor` reference scans **every** patch; OF scans only patches that fix a value and errors if none does. Identical for a uniform initial p; diverges for a non-uniform one. | `gpuRhoSimpleFoam.cu:99` | low | ☐ open |
| A9 | `tangentialVelocity` on `pressureInletOutletVelocity` silently ignored (header says unsupported, nothing enforces it). | `fv_patch_field.cuh:406` | low | ☐ open |
| A10 | `overset` classified as a constraint patch type, so an overset case runs instead of being refused. | `foam_dict.cuh:41` | high if hit | ✅ **FIXED** — removed from `isConstraintPatchType`, refused in `buildPatches`; gate `overset_refused` (2 mutations) |
| A12 | `grad(K)` for the kinetic term took its boundary from `deviceBCValue(dbHe, K, ...)` — the ENERGY field's BC descriptor applied to the K array, returning he's refValue at a fixedValue patch instead of K's boundary value. Masked under limitedLinear (the gradient only feeds a limiter clamped to [0,1]); fatal under linearUpwind. **Enabling A4 on the bad gradient made T 4x WORSE (9.3e-02) than upwind** — that is how it was found. | `device_energy.cu` | high (blocks A4) | ✅ **FIXED** |
| A11 | Written `boundaryField` is a PASS-THROUGH of the input entry, not the computed boundary value — for every field, not just `nut`. brae's own solve is unaffected, but anything post-processing brae's output (yPlus, wall shear, forces, a restart) reads stale boundary values. Confirmed on `nut` (input vs written difference 0.0) and again on T in the new `io_vs_openfoam` gate. | `foam_field_writer.cuh` | medium (output only) | ☐ open |

## B. Honest refusals — limit scope, never wrong

| # | Refusal | Blocks | Status |
|---|---|---|---|
| B1 | `transonic yes` | `squareBend` | ☐ Phase 7 |
| B2 | non-perfectGas / liquid thermo | `squareBendLiq`, `squareBendLiqNoNewtonian` | ☐ backlog |
| B3 | `nutUWallFunction` | `gasMixing` | ☐ Phase 1 |
| B4 | `coded` (U), `expression` (T), `functionObjectTrigger` (T) | `squareBendLiq` | ☐ Phase 1/3 |
| B5 | temperature-GRADIENT BCs (`fixedGradient`, `externalWallHeatFluxTemperature`) — `DeviceBoundary` has no `refGrad` | heat-transfer cases | ☐ Phase 2 |
| B6 | `alphatJayatillekeWallFunction`, `totalPressure` isentropic, `flowRateInletVelocity extrapolateProfile` | — | ☐ backlog |
| B7 | SA + realizableKE compressible (not ρ-weighted) | — | ☐ backlog |

**Mis-shaped refusal:** `gasMixing` dies with `TokenStream: expected '(' got '0'` on `0/U` — the reader
parses `inletValue` unconditionally, but OF's `pressureInletOutletVelocity` never reads that entry. brae
fails on a legitimate-to-OF file, before reaching its real `nutUWallFunction` refusal. ☐ open

## C. No effect today, but silent

| # | Gap | Status |
|---|---|---|
| C1 | `interpolationSchemes` never parsed (zero hits in `src/`); a non-`linear` entry would be ignored | ☐ |
| C2 | `grad(k)`/`grad(omega) cellLimited` parsed by nothing — only `grad(U)` lines are scanned | ☐ |
| C3 | `hasWord(ln,"limited")` fires on a `div` line and sets `nonOrth` — wrong if the laplacian is `orthogonal` | ☐ |
| C4 | `div(phi,{h,e,K,Ekp})` flags OR-accumulated into one slot; different schemes per term silently merge | ☐ |
| C5 | No `residualControl` in the compressible driver — always runs to `endTime` | ☐ |
| C6 | `startFrom latestTime` ignored; hardcoded `caseDir + "/0"` | ☐ |

## D. Verification gaps — why these survived

| # | Gap | Status |
|---|---|---|
| D1 | **No gate case uses `inletOutlet` on T**, so A1 was invisible to all 9 compressible gates | ✅ **FIXED** — `validation/rhoIO` + `io_vs_openfoam.sh`; mutation-tested (reverting A1 takes T from 2.7e-07 to 2.8e+00) |
| D2 | Boundary coefficients (`internalCoeffs`/`boundaryCoeffs`) never compared against OF for real BC types. The matrix comparison used plain fixedValue/zeroGradient only. | ☐ in progress |
| D3 | Only **2 of 6** tutorials start, so 4 are untested end-to-end | ☐ |
| D4 | No gate exercised a turbulent inlet whose set-up value differs from the converged one (all used fixedValue U). | ✅ **FIXED** — `validation/rhoTI` + `ti_vs_openfoam.sh` |

## Notes

- Matrix assembly itself is verified: diagonal **1.76e-06** and source **6.66e-07** vs OF's
  `fvScalarMatrix` (`tools/dumpScalarMatrix`). The transport operator is not the problem.
- Compressible-specific BCs verified correct: `totalPressure` (ρ-weighted), `flowRateInletVelocity`
  (mass and volumetric), `freestream` base, `compressible::alphatWallFunction`, model `Prt` default 1.0.
