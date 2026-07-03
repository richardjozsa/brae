#pragma once
// Turbulent-inlet BCs (OpenFOAM): the inflow value is computed from the inlet velocity (and k), then the BC
// behaves as inletOutlet. For steady fixedValue-U inlets the inflow value is constant, so it is computed once at
// setup from the read fields. Mirrors v2412:
//   turbulentIntensityKineticEnergyInlet:        k     = 1.5*(intensity*|U|)^2
//   turbulentMixingLengthDissipationRateInlet:   eps   = Cmu^0.75 * k^1.5 / mixingLength
//   turbulentMixingLengthFrequencyInlet:         omega = sqrt(k) / (Cmu^0.25 * mixingLength)
#include "cf_types.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "foam_field_reader.cuh"
#include "fv_patch.cuh"
#include <cmath>
#include <memory>
#include <vector>

namespace brae {

// k's turbulentIntensityKineticEnergyInlet patches -> inletOutlet(1.5*(intensity*|U|)^2). Rebuilds k.boundary.
inline void applyTurbulentInletK(GeometricField<scalar>& k, const FieldData<scalar>& kFD,
                                 const GeometricField<vector>& U, const std::vector<FvPatch>& fvp) {
    for (std::size_t pi = 0; pi < fvp.size(); ++pi) {
        const PatchFieldData<scalar>* b = nullptr;
        for (const auto& bd : kFD.boundary) if (bd.name == fvp[pi].name) { b = &bd; break; }
        if (!b || b->type != "turbulentIntensityKineticEnergyInlet") continue;
        const std::vector<vector>& Uin = U.boundary[pi]->value();
        std::vector<scalar> kin(fvp[pi].size);
        for (label i = 0; i < fvp[pi].size; ++i) kin[i] = 1.5 * (b->intensity * b->intensity) * magSqr(Uin[i]);
        k.boundary[pi] = std::make_unique<InletOutletPatchField<scalar>>(fvp[pi], false, scalar{0}, kin);
    }
    k.evaluateBoundary();
}

// 2nd-scalar (epsilon|omega) turbulentMixingLength*Inlet patches, from the (already-computed) inlet k.
inline void applyTurbulentInletSecond(GeometricField<scalar>& second, const FieldData<scalar>& sFD,
                                      const GeometricField<scalar>& k, scalar Cmu, const std::vector<FvPatch>& fvp) {
    const scalar Cmu75 = std::pow(Cmu, 0.75), Cmu25 = std::pow(Cmu, 0.25);
    for (std::size_t pi = 0; pi < fvp.size(); ++pi) {
        const PatchFieldData<scalar>* b = nullptr;
        for (const auto& bd : sFD.boundary) if (bd.name == fvp[pi].name) { b = &bd; break; }
        if (!b) continue;
        const bool eps = (b->type == "turbulentMixingLengthDissipationRateInlet");
        const bool om  = (b->type == "turbulentMixingLengthFrequencyInlet");
        if (!eps && !om) continue;
        const std::vector<scalar>& kin = k.boundary[pi]->value();
        std::vector<scalar> sin(fvp[pi].size);
        for (label i = 0; i < fvp[pi].size; ++i)
            sin[i] = eps ? Cmu75 * std::pow(kin[i], 1.5) / b->mixingLength
                         : std::sqrt(kin[i]) / (Cmu25 * b->mixingLength);
        second.boundary[pi] = std::make_unique<InletOutletPatchField<scalar>>(fvp[pi], false, scalar{0}, sin);
    }
    second.evaluateBoundary();
}

} // namespace brae
