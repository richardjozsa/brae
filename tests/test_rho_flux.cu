// Phase 2 unit check: the mass flux reduces exactly to the volumetric flux at constant density.
//
// This is the property the whole compressible path is built on. Gate 2 asserts that a rho=const
// compressible run reproduces the incompressible answer to 1e-10; that can only hold if the flux itself
// reduces exactly first. Checking it here, on its own, means a Gate 2 failure later points at the
// pressure equation or the momentum predictor rather than at the flux.
//
//   rhoFlux(rho=1, U)  ==  flux(U)              bitwise
//   rhoFlux(rho=c, U)  ==  c * flux(U)          to rounding
//
// The second case matters because a scale factor that is silently dropped somewhere still passes the
// first: rho=1 makes multiplication by rho invisible.

#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvc.cuh"
#include "parallel_simple.cuh"
#include "field_distribute.cuh"
#include "box_mesh.cuh"
#include <cstdio>
#include <cmath>
#include <vector>

using namespace brae;

namespace {

int failures = 0;

void report(
    const char* what,
    scalar worst,
    scalar tol)
{
    const bool ok = (worst <= tol);
    std::printf("  %-44s worst %.3e  tol %.1e  %s\n", what, worst, tol, ok ? "OK" : "FAIL");
    if (!ok) failures++;
}

}   // namespace

int main()
{
    const label Nx = 12;
    const label Ny = 6;
    const label Nz = 4;
    const label nC = Nx * Ny * Nz;

    // A sheared mesh, so the face normals are not axis-aligned and the interpolation weights are not all
    // 0.5. On an orthogonal box a weight bug is invisible.
    const PrimitiveMesh gm = boxtest::boxMesh(Nx, Ny, Nz, 0.3);
    const std::vector<label> cellToPart(nC, 0);
    const Partition P(gm, cellToPart, 0);
    const FvGeometry& lg = P.lg;
    const std::vector<FvPatch>& lp = P.lp;

    // A non-uniform velocity, so the flux varies face to face.
    FieldData<vector> Ufd;
    Ufd.internalUniform = true;
    Ufd.internalUniformValue = vector{1.0, 0.25, -0.4};
    Ufd.boundary.push_back(boxtest::pfd<vector>("inlet", "fixedValue", vector{1.0, 0.25, -0.4}, true));
    Ufd.boundary.push_back(boxtest::pfd<vector>("outlet", "zeroGradient", vector{0, 0, 0}, false));
    for (const char* w : {"wallYmin", "wallYmax", "wallZmin", "wallZmax"})
    {
        Ufd.boundary.push_back(boxtest::pfd<vector>(w, "fixedValue", vector{0, 0, 0}, true));
    }
    GeometricField<vector> U = distributeField<vector>(Ufd, gm.patches(), P.Lm, lp, P.procW, 0);
    U.evaluateBoundary();

    const SurfaceScalarField phiV = fvc::flux(U, P.Lm.mesh, lg, lp);

    // rho = 1 : must be bitwise identical
    {
        const std::vector<scalar> rho(nC, 1.0);
        const SurfaceScalarField phiM = fvc::rhoFlux(rho, U, P.Lm.mesh, lg, lp);
        scalar worst = 0.0;
        for (std::size_t f = 0; f < phiV.internal.size(); ++f)
        {
            worst = std::max(worst, std::abs(phiM.internal[f] - phiV.internal[f]));
        }
        for (std::size_t p = 0; p < phiV.boundary.size(); ++p)
        {
            for (std::size_t i = 0; i < phiV.boundary[p].size(); ++i)
            {
                worst = std::max(worst, std::abs(phiM.boundary[p][i] - phiV.boundary[p][i]));
            }
        }
        report("rho=1: rhoFlux == flux (absolute)", worst, 0.0);
    }

    // rho = c : must be exactly c times the volumetric flux
    {
        const scalar c = 1.225;
        const std::vector<scalar> rho(nC, c);
        const SurfaceScalarField phiM = fvc::rhoFlux(rho, U, P.Lm.mesh, lg, lp);
        scalar worst = 0.0;
        scalar ref = 0.0;
        for (std::size_t f = 0; f < phiV.internal.size(); ++f)
        {
            worst = std::max(worst, std::abs(phiM.internal[f] - c * phiV.internal[f]));
            ref = std::max(ref, std::abs(c * phiV.internal[f]));
        }
        report("rho=c: rhoFlux == c*flux (relative)", ref > 0.0 ? worst / ref : worst, 1e-15);
    }

    std::printf("rho_flux: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
