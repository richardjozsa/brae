// GPU offload: on-device VECTOR BC evaluation (momentum). A vector field's boundary is three scalar
// boundaries; validate the per-component value() and the laplacian/div boundary coefficients against the
// CPU fvm assembly, across all BC categories (fixedValue inlet, noSlip walls, zeroGradient outlet, empty).
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "device_buffer.cuh"
#include "device_boundary.cuh"
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
    const scalar nu = 1e-3;

    GeometricField<vector> U; U.internal.resize(nC);
    for (label c = 0; c < nC; ++c) U.internal[c] = { 1.0 + 0.2*std::sin(0.01*c), 0.1*std::cos(0.013*c), 0.0 };
    for (const FvPatch& q : fvp) {
        if (q.type == "empty")      U.boundary.push_back(std::make_unique<EmptyPatchField<vector>>(q));
        else if (q.name == "inlet") U.boundary.push_back(std::make_unique<FixedValuePatchField<vector>>(q, true, vector{10,0,0}, std::vector<vector>{}));
        else if (q.type == "wall")  U.boundary.push_back(std::make_unique<NoSlipPatchField<vector>>(q));
        else                        U.boundary.push_back(std::make_unique<ZeroGradientPatchField<vector>>(q));
    }
    U.evaluateBoundary();
    const SurfaceScalarField phi = fvc::flux(U, m, g, fvp);

    const FvVectorMatrix Mlap = fvm::laplacian(U, nu, m, g, fvp);
    const FvVectorMatrix Mdiv = fvm::div(phi.internal, phi.boundary, U, m, fvp);

    const DeviceVectorBoundary db = buildDeviceVectorBoundary(U, fvp, g);
    DeviceBuffer<scalar> gammaC(std::vector<scalar>(nC, nu)), phiB(([&]{ std::vector<scalar> o; for (auto& a : phi.boundary) o.insert(o.end(), a.begin(), a.end()); return o; }()));

    scalar maxVal = 0, maxLapIC = 0, maxLapBC = 0, maxDivIC = 0, maxDivBC = 0;
    for (int k = 0; k < 3; ++k) {
        std::vector<scalar> internalK(nC); for (label c = 0; c < nC; ++c) internalK[c] = component(U.internal[c], k);
        // CPU per-component flattened.
        std::vector<scalar> valC, lapICc, lapBCc, divICc, divBCc;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi) for (label i = 0; i < fvp[pi].size; ++i) {
            valC.push_back(component(U.boundary[pi]->value()[i], k));
            lapICc.push_back(component(Mlap.internalCoeffs[pi][i], k)); lapBCc.push_back(component(Mlap.boundaryCoeffs[pi][i], k));
            divICc.push_back(component(Mdiv.internalCoeffs[pi][i], k)); divBCc.push_back(component(Mdiv.boundaryCoeffs[pi][i], k));
        }
        DeviceBuffer<scalar> dIntK(internalK), dVal; deviceBCValue(db.comp[k], dIntK, dVal);
        DeviceBuffer<scalar> dLapIC, dLapBC; deviceBCLaplacianCoeffs(db.comp[k], gammaC, dLapIC, dLapBC);
        DeviceBuffer<scalar> dDivIC, dDivBC; deviceBCDivCoeffs(db.comp[k], phiB, dDivIC, dDivBC);
        maxVal   = std::fmax(maxVal,   relMax(dVal.host(),   valC));
        maxLapIC = std::fmax(maxLapIC, relMax(dLapIC.host(), lapICc)); maxLapBC = std::fmax(maxLapBC, relMax(dLapBC.host(), lapBCc));
        maxDivIC = std::fmax(maxDivIC, relMax(dDivIC.host(), divICc)); maxDivBC = std::fmax(maxDivBC, relMax(dDivBC.host(), divBCc));
    }

    std::printf("GPU on-device VECTOR BC eval (nBndFaces=%d):\n", db.n);
    std::printf("  value()  (max over xyz)        : %.3e\n", maxVal);
    std::printf("  laplacian internalC/boundaryC  : %.3e / %.3e\n", maxLapIC, maxLapBC);
    std::printf("  div       internalC/boundaryC  : %.3e / %.3e\n", maxDivIC, maxDivBC);
    const bool pass = maxVal < 1e-14 && maxLapIC < 1e-14 && maxLapBC < 1e-14 && maxDivIC < 1e-14 && maxDivBC < 1e-14;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
