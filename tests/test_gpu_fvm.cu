// GPU offload G4: fvm matrix assembly on device. Validate the device-assembled raw lduMatrix coefficients
// (diag/upper/lower) against the CPU fvm::laplacian (face-varying gamma) and fvm::div (upwind convection)
// on pitzDaily. The boundary internalCoeffs/boundaryCoeffs are folded by the solver (G5), not tested here.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace brae;

static scalar relMax(const std::vector<scalar>& a, const std::vector<scalar>& b) {
    scalar n = 0, d = 0; for (std::size_t i = 0; i < a.size(); ++i) { n = std::fmax(n, std::fabs(a[i]-b[i])); d = std::fmax(d, std::fabs(b[i])); }
    return d > 0 ? n / d : n;
}

int main(int argc, char** argv) {
    const std::string caseDir = argc > 1 ? argv[1] : "validation/pitzDaily";
    PrimitiveMesh m; m.read(caseDir + "/constant/polyMesh");
    FvGeometry g; g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();

    auto mkScalar = [&]() { GeometricField<scalar> f; f.internal.assign(nC, 0.0);
        for (const FvPatch& q : fvp) { if (q.type == "empty") f.boundary.push_back(std::make_unique<EmptyPatchField<scalar>>(q));
            else f.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q)); }
        f.evaluateBoundary(); return f; };
    GeometricField<scalar> p = mkScalar();

    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);

    // --- laplacian(gammaf, p) ---
    std::vector<scalar> gammaCell(nC); for (label c = 0; c < nC; ++c) gammaCell[c] = 1.0 + 0.3 * std::sin(0.01 * c);
    const SurfaceScalarField gammaf = fvc::interpolate(gammaCell, m, g, fvp);
    const FvScalarMatrix Mlap = fvm::laplacian(gammaf, p, m, g, fvp);
    DeviceBuffer<scalar> dgamma(gammaf.internal), lapDiag, lapUp, lapLo;
    deviceLaplacianCoeffs(dm, dgamma, lapDiag, lapUp, lapLo);
    const scalar eLapD = relMax(lapDiag.host(), Mlap.diag), eLapU = relMax(lapUp.host(), Mlap.upper), eLapL = relMax(lapLo.host(), Mlap.lower);

    // --- div(phi, U) upwind ---
    GeometricField<vector> U; U.internal.resize(nC);
    for (label c = 0; c < nC; ++c) U.internal[c] = { std::sin(0.01*c) - 0.3, std::cos(0.017*c), 0.0 };
    for (const FvPatch& q : fvp) { if (q.type == "empty") U.boundary.push_back(std::make_unique<EmptyPatchField<vector>>(q));
        else U.boundary.push_back(std::make_unique<ZeroGradientPatchField<vector>>(q)); }
    U.evaluateBoundary();
    const SurfaceScalarField phi = fvc::flux(U, m, g, fvp);
    const FvVectorMatrix Mdiv = fvm::div(phi.internal, phi.boundary, U, m, fvp);
    DeviceBuffer<scalar> dphi(phi.internal), divDiag, divUp, divLo;
    deviceDivUpwindCoeffs(dm, dphi, divDiag, divUp, divLo);
    const scalar eDivD = relMax(divDiag.host(), Mdiv.diag), eDivU = relMax(divUp.host(), Mdiv.upper), eDivL = relMax(divLo.host(), Mdiv.lower);

    std::printf("GPU G4 fvm assembly (nCells=%d):\n", nC);
    std::printf("  laplacian  diag %.3e  upper %.3e  lower %.3e\n", eLapD, eLapU, eLapL);
    std::printf("  div upwind diag %.3e  upper %.3e  lower %.3e\n", eDivD, eDivU, eDivL);
    const bool pass = eLapD < 1e-12 && eLapU < 1e-14 && eLapL < 1e-14 && eDivD < 1e-12 && eDivU < 1e-14 && eDivL < 1e-14;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
