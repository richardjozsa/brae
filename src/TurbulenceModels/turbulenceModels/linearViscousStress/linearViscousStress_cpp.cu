// _cpp REFERENCE implementation -- see linearViscousStress_cpp.cuh for the OpenFOAM provenance.
#include "linearViscousStress_cpp.cuh"

namespace brae {
namespace cpu {

namespace {

// nuEff interpolated to faces, the way OpenFOAM's fvm::laplacian(volScalarField, U) does it.
//
// Internal faces are the linear (weight-based) interpolation. BOUNDARY faces take the boundary field of
// nuEff -- NOT the owner cell's value. On a wall with a nut wall function those differ by the whole of
// nut_wall, and using the cell value silently under-predicts wall shear. brae has been bitten by exactly
// this before (validation/bc_vs_openfoam.sh, the boundary_mu_eff gate), which is why the boundary array is
// a required argument here rather than something this function is allowed to invent.
SurfaceScalarField interpolateEff(
    const std::vector<scalar>& nuEff,
    const std::vector<std::vector<scalar>>& nuEffBnd,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    SurfaceScalarField gf = fvc::interpolate(nuEff, m, g, patches);
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (pi < nuEffBnd.size() && nuEffBnd[pi].size() == gf.boundary[pi].size())
        {
            gf.boundary[pi] = nuEffBnd[pi];
        }
    }
    return gf;
}

} // namespace


std::vector<vector> divDevReffExplicit(
    const GeometricField<vector>& U,
    const std::vector<scalar>&    nuEff,
    const std::vector<std::vector<scalar>>& nuEffBnd,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches)
{
    // fvc::grad(U) -- cell tensors, then the boundary tensors with OpenFOAM's gaussGrad boundary
    // correction (wall-normal component replaced by snGrad(U)).
    const std::vector<tensor> gradU    = fvc::gaussGrad(U, m, g, patches);
    const std::vector<std::vector<tensor>> gradUb =
        fvc::gradUBoundary(U, gradU, m, g, patches);

    // nuEff*dev2(T(grad(U))) as a volTensorField (cells + boundary faces).
    std::vector<tensor> tCell(gradU.size());
    for (std::size_t c = 0; c < gradU.size(); ++c)
    {
        tCell[c] = nuEff[c] * dev2(transpose(gradU[c]));
    }

    std::vector<std::vector<tensor>> tBnd(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const std::size_t n = gradUb[pi].size();
        tBnd[pi].resize(n);
        for (std::size_t f = 0; f < n; ++f)
        {
            const scalar nu =
                (pi < nuEffBnd.size() && f < nuEffBnd[pi].size()) ? nuEffBnd[pi][f] : 0.0;
            tBnd[pi][f] = nu * dev2(transpose(gradUb[pi][f]));
        }
    }

    // OpenFOAM: -fvc::div(...). fvc::div already divides by the cell volume.
    std::vector<vector> d = fvc::div(tCell, tBnd, m, g, patches);
    for (auto& v : d)
    {
        v = {-v.x, -v.y, -v.z};
    }
    return d;
}


void addDivDevReff(
    FvVectorMatrix&               UEqn,
    const GeometricField<vector>& U,
    const std::vector<scalar>&    nuEff,
    const std::vector<std::vector<scalar>>& nuEffBnd,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches)
{
    // Implicit half: OpenFOAM writes `- fvm::laplacian(nuEff, U)` inside divDevReff, and UEqn.H adds
    // divDevReff to the equation -- so the laplacian enters with coefficient -1.
    const SurfaceScalarField gammaf = interpolateEff(nuEff, nuEffBnd, m, g, patches);
    addEqual(UEqn, fvm::laplacian<vector>(gammaf, U, m, g, patches), -1.0);

    // Explicit half. OpenFOAM's equation reads  M(U) + divDevReff(U) == 0, i.e. the explicit term sits on
    // the LEFT. brae's FvMatrix solves  M.psi = source, so a left-hand explicit term crosses to the right
    // with a sign change: source -= (explicit contribution) becomes source += (-...)*V.
    //
    // Two conversions happen here and both are easy to get silently wrong, so both are stated:
    //   1. the side change (left -> right) flips the sign;
    //   2. fvc::div returns a per-VOLUME quantity, while FvMatrix::source is an EXTENSIVE per-cell
    //      quantity, so it must be multiplied back by V.
    // tests/test_divdevreff_cpp.cu pins the result against an OpenFOAM dump; do not "simplify" the signs
    // here without re-running it.
    const std::vector<vector> expl = divDevReffExplicit(U, nuEff, nuEffBnd, m, g, patches);
    const std::vector<scalar>& V = g.V();
    for (std::size_t c = 0; c < expl.size(); ++c)
    {
        UEqn.source[c].x -= expl[c].x * V[c];
        UEqn.source[c].y -= expl[c].y * V[c];
        UEqn.source[c].z -= expl[c].z * V[c];
    }
}

} // namespace cpu
} // namespace brae
