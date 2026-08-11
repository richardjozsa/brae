#pragma once
// Host-side boundary muEff -- the first function of the CPU-first rhoSimpleFoam reference.
// See ../../../../../../rhosimplefoam-ground-truth-port.md for the port protocol this follows.
//
// OF (compressibleTurbulenceModel / eddyViscosity):
//
//     muEff() = rho*nuEff()  and  nuEff() = nu() + nut()
//     =>  muEff = mu + rho*nut                    on the internal field AND on every patch
//
// A GeometricField's boundaryField is evaluated PER PATCH from that patch's own BC. OF never
// extrapolates a cell value onto a boundary face to build muEff: it takes mu, rho and nut each from
// their own boundaryField, which for `calculated` nut is the value in the case file (commonly 0 at an
// inlet) and for a wall-function nut is what the wall function wrote via setValue().
//
// THIS IS WHERE THE LEGACY GPU PATH IS WRONG. It assigns non-wall boundary faces the extrapolated
// CELL nuEff (device_kepsilon.cu:797,831,922 -- `if (!isWall[i]) { nutBnd[i] = nutCell[c]; }`), which
// on gasMixing/injectorPipe gives muEff_b = 3.755785e-02 at inlet_air where OF gives 6.181948e-05
// -- 607x. That feeds internalCoeffs/boundaryCoeffs and accounts for the entire iteration-1 diagonal
// difference against OF's own assembled matrix (ratio 1.0020 over 120 cells).
//
// GENERAL BY CONSTRUCTION -- no case, patch name, or patch type is referenced anywhere below. Patch
// identity is resolved by GeometricField's own name/group/regex matching when the field is read, and
// the wall-vs-inlet distinction is carried by the BC objects themselves. A case with no nut field
// (laminar) is handled by passing nut = nullptr, which is the laminar muEff = mu.

#include "cf_types.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include <vector>

namespace brae {
namespace cpu {

// muEff on one patch: mu_b + rho_b*nut_b, all three from their own boundaryField.
// nut == nullptr => laminar (nut is absent from the case), so muEff_b = mu_b.
inline std::vector<scalar> boundaryMuEff(
    label patchi,
    const GeometricField<scalar>& mu,
    const GeometricField<scalar>& rho,
    const GeometricField<scalar>* nut)
{
    const std::vector<scalar>& muB  = mu.boundary[patchi]->value();
    const std::vector<scalar>& rhoB = rho.boundary[patchi]->value();

    std::vector<scalar> muEff(muB.size());
    if (!nut)
    {
        for (std::size_t i = 0; i < muB.size(); ++i) muEff[i] = muB[i];
        return muEff;
    }

    const std::vector<scalar>& nutB = nut->boundary[patchi]->value();
    for (std::size_t i = 0; i < muB.size(); ++i)
    {
        muEff[i] = muB[i] + rhoB[i]*nutB[i];
    }
    return muEff;
}

// muEff over every patch, in mesh patch order -- the shape fvm::laplacian consumes as its boundary
// gamma. Sizes follow each patch's own face count, so an empty/zero-size patch stays zero-size.
inline std::vector<std::vector<scalar>> boundaryMuEff(
    const std::vector<FvPatch>& patches,
    const GeometricField<scalar>& mu,
    const GeometricField<scalar>& rho,
    const GeometricField<scalar>* nut)
{
    std::vector<std::vector<scalar>> out(patches.size());
    for (std::size_t p = 0; p < patches.size(); ++p)
    {
        out[p] = boundaryMuEff(static_cast<label>(p), mu, rho, nut);
    }
    return out;
}

}   // namespace cpu
}   // namespace brae
