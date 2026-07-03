// GPU offload: on-device BC evaluation. Validate the device boundary value() and the matrix boundary
// coefficients (laplacian internalCoeffs/boundaryCoeffs from a diffusivity, div internalCoeffs/
// boundaryCoeffs from a face flux) against the CPU fvPatchField + fvm assembly, across all 3 BC
// categories (fixedValue outlet, calculated inlet, zeroGradient walls, empty front/back).
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

static std::vector<scalar> flat(const std::vector<std::vector<scalar>>& v) { std::vector<scalar> o; for (auto& a : v) o.insert(o.end(), a.begin(), a.end()); return o; }
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

    // scalar field exercising all 3 BC categories.
    GeometricField<scalar> p; p.internal.resize(nC);
    for (label c = 0; c < nC; ++c) p.internal[c] = std::sin(0.013 * c) + 0.4;
    for (const FvPatch& q : fvp) {
        if (q.type == "empty")        p.boundary.push_back(std::make_unique<EmptyPatchField<scalar>>(q));          // 0
        else if (q.name == "outlet")  p.boundary.push_back(std::make_unique<FixedValuePatchField<scalar>>(q, true, 1.5, std::vector<scalar>{})); // 1
        else if (q.name == "inlet")   p.boundary.push_back(std::make_unique<CalculatedPatchField<scalar>>(q, true, 3.0, std::vector<scalar>{}));  // 2
        else                          p.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q));  // 0
    }
    p.evaluateBoundary();

    // diffusivity + flux for the coeff checks.
    std::vector<scalar> gammaCell(nC); for (label c = 0; c < nC; ++c) gammaCell[c] = 0.5 + 0.2 * std::cos(0.01 * c);
    const SurfaceScalarField gammaf = fvc::interpolate(gammaCell, m, g, fvp);
    GeometricField<vector> U; U.internal.resize(nC);
    for (label c = 0; c < nC; ++c) U.internal[c] = { std::sin(0.01*c), 0.2*std::cos(0.01*c), 0.0 };
    for (const FvPatch& q : fvp) { if (q.type == "empty") U.boundary.push_back(std::make_unique<EmptyPatchField<vector>>(q)); else U.boundary.push_back(std::make_unique<ZeroGradientPatchField<vector>>(q)); }
    U.evaluateBoundary();
    const SurfaceScalarField phi = fvc::flux(U, m, g, fvp);

    // CPU references.
    const std::vector<scalar> valCpu = flat([&]{ std::vector<std::vector<scalar>> v; for (std::size_t pi = 0; pi < fvp.size(); ++pi) v.push_back(p.boundary[pi]->value()); return v; }());
    const FvScalarMatrix Mlap = fvm::laplacian(gammaf, p, m, g, fvp);
    const std::vector<scalar> lapIC = flat(Mlap.internalCoeffs), lapBC = flat(Mlap.boundaryCoeffs);
    const FvScalarMatrix Mdiv = fvm::div(phi.internal, phi.boundary, p, m, fvp);
    const std::vector<scalar> divIC = flat(Mdiv.internalCoeffs), divBC = flat(Mdiv.boundaryCoeffs);

    // Device.
    const DeviceBoundary db = buildDeviceBoundary(p, fvp, g);
    DeviceBuffer<scalar> dInternal(p.internal), dValue; deviceBCValue(db, dInternal, dValue);
    DeviceBuffer<scalar> dGamma(gammaCell), dLapIC, dLapBC; deviceBCLaplacianCoeffs(db, dGamma, dLapIC, dLapBC);
    DeviceBuffer<scalar> dPhiB(flat(phi.boundary)), dDivIC, dDivBC; deviceBCDivCoeffs(db, dPhiB, dDivIC, dDivBC);

    std::printf("GPU on-device BC eval (nBndFaces=%d):\n", db.n);
    std::printf("  value()              : %.3e\n", relMax(dValue.host(), valCpu));
    std::printf("  laplacian internalC  : %.3e   boundaryC : %.3e\n", relMax(dLapIC.host(), lapIC), relMax(dLapBC.host(), lapBC));
    std::printf("  div internalC        : %.3e   boundaryC : %.3e\n", relMax(dDivIC.host(), divIC), relMax(dDivBC.host(), divBC));
    const bool pass = relMax(dValue.host(), valCpu) < 1e-14 && relMax(dLapIC.host(), lapIC) < 1e-14 && relMax(dLapBC.host(), lapBC) < 1e-14
                   && relMax(dDivIC.host(), divIC) < 1e-14 && relMax(dDivBC.host(), divBC) < 1e-14;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
