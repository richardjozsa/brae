// _cpp REFERENCE implementation -- see UEqn_cpp.cuh for the OpenFOAM provenance and the refusal contract.
#include "UEqn_cpp.cuh"
#include "fvm.cuh"
#include "fv_matrix_ops.cuh"
#include "linearViscousStress_cpp.cuh"
#include <stdexcept>

namespace brae {
namespace cpu {

namespace {

void refuseUnsupported(const MomentumInput& in)
{
    // UEqn.H reaches MRF twice (correctBoundaryVelocity, DDt) and fvOptions three times (the source, the
    // constraint, the correction). Neither is optional when the case declares it. Refusing here is the
    // point: brae has already shipped a compressible path that read MRFProperties, ignored it, converged,
    // and reported nothing wrong.
    if (in.hasMRF)
        throw std::runtime_error(
            "UEqn_cpp: the case declares MRF, which UEqn.H applies via MRF.correctBoundaryVelocity(U) and "
            "MRF.DDt(U) (simpleFoam/UEqn.H:3,8). The _cpp reference does not implement it; refusing rather "
            "than silently solving a different equation.");
    if (in.hasFvOptions)
        throw std::runtime_error(
            "UEqn_cpp: the case declares fvOptions, which UEqn.H applies as fvOptions(U), "
            "fvOptions.constrain(UEqn) and fvOptions.correct(U) (simpleFoam/UEqn.H:11,17,23). The _cpp "
            "reference does not implement it; refusing rather than silently solving a different equation.");
}

} // namespace


FvVectorMatrix momentumCore(
    const GeometricField<vector>& U,
    const MomentumInput&          in,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches)
{
    refuseUnsupported(in);

    // fvm::div(phi, U) -- the convection operator. Its implicit weights come from the div SCHEME, which is
    // why the scheme is a first-class part of the port manifest rather than a detail.
    FvVectorMatrix M = fvm::div(*in.phi, *in.phiBnd, U, m, patches);

    // - fvm::laplacian(nuEff, U), the implicit half of divDevReff. Face nuEff takes the BOUNDARY field on
    // boundary faces, not the owner cell value; see interpolateEff in linearViscousStress_cpp.cu.
    addEqual(M, fvm::laplacian<vector>(
                    effectiveFaceViscosity(*in.nuEff, *in.nuEffBnd, m, g, patches), U, m, g, patches),
             -1.0);
    return M;
}


FvVectorMatrix assembleUEqn(
    const GeometricField<vector>& U,
    const MomentumInput&          in,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches)
{
    refuseUnsupported(in);

    FvVectorMatrix M = fvm::div(*in.phi, *in.phiBnd, U, m, patches);

    // `bounded`: - fvm::Sp(fvc::div(phi), U). Applied BEFORE relax, as OpenFOAM does -- it is part of the
    // matrix the relaxation then acts on, not a correction bolted on afterwards.
    if (in.bounded)
    {
        SurfaceScalarField phis;
        phis.internal = *in.phi;
        phis.boundary = *in.phiBnd;
        const std::vector<scalar> divPhi = fvc::div(phis, m, g, patches);
        const std::vector<scalar>& V = g.V();
        for (std::size_t c = 0; c < M.diag.size(); ++c) M.diag[c] -= divPhi[c] * V[c];
    }

    // turbulence->divDevReff(U): implicit -laplacian(nuEff,U) into the matrix AND the explicit
    // -div(nuEff*dev2(T(grad U))) into the source. Both halves, one call, so they cannot drift apart.
    addDivDevReff(M, U, *in.nuEff, *in.nuEffBnd, m, g, patches, in.correctedLaplacian);

    // UEqn.relax(). OpenFOAM guards this with if(relaxEquation(name)); a factor of 1 is the identity, and
    // relaxMatrix already early-returns on alpha <= 0.
    if (in.relaxU > 0.0 && in.relaxU < 1.0)
    {
        relaxMatrix<vector>(M, U, m, patches, in.relaxU);
    }
    return M;
}


void addPressureGradient(
    FvVectorMatrix&               UEqn,
    const GeometricField<scalar>& p,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches)
{
    // solve(UEqn == -fvc::grad(p)). The right-hand side of an fvMatrix equation is its source, and
    // fvc::grad returns a per-volume quantity, so the extensive form is -grad(p)*V.
    const std::vector<vector> gradP = fvc::gaussGrad(p, m, g, patches);
    const std::vector<scalar>& V = g.V();
    for (std::size_t c = 0; c < gradP.size(); ++c)
    {
        UEqn.source[c].x -= gradP[c].x * V[c];
        UEqn.source[c].y -= gradP[c].y * V[c];
        UEqn.source[c].z -= gradP[c].z * V[c];
    }
}

} // namespace cpu
} // namespace brae
