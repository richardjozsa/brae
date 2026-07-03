// GPU offload (#2): the explicit divDevReff stress source on device. Validate V*fvc::div(nuEff*dev2(T(grad
// U))) (the source the faithful momentum adds) against the CPU fvc, across the momentum BCs.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "fvc.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "device_divdevreff.cuh"
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
    for (label c = 0; c < nC; ++c) U.internal[c] = { 1.0 + 0.2*std::sin(0.01*c), 0.1*std::cos(0.013*c), 0.05*std::sin(0.02*c) };
    for (const FvPatch& q : fvp) {
        if (q.type == "empty")      U.boundary.push_back(std::make_unique<EmptyPatchField<vector>>(q));
        else if (q.name == "inlet") U.boundary.push_back(std::make_unique<FixedValuePatchField<vector>>(q, true, vector{10,0,0}, std::vector<vector>{}));
        else if (q.type == "wall")  U.boundary.push_back(std::make_unique<NoSlipPatchField<vector>>(q));
        else                        U.boundary.push_back(std::make_unique<ZeroGradientPatchField<vector>>(q));
    }
    U.evaluateBoundary();

    // CPU divDevReff source = V * fvc::div(nu * dev2(T(grad U))).
    const std::vector<tensor> gradC = fvc::gaussGrad(U, m, g, fvp);
    const std::vector<std::vector<tensor>> gradB = fvc::gradUBoundary(U, gradC, m, g, fvp);
    std::vector<tensor> sigC(nC); for (label c = 0; c < nC; ++c) sigC[c] = nu * dev2(transpose(gradC[c]));
    std::vector<std::vector<tensor>> sigB(fvp.size());
    for (std::size_t pi = 0; pi < fvp.size(); ++pi) { sigB[pi].resize(fvp[pi].size); for (label i = 0; i < fvp[pi].size; ++i) sigB[pi][i] = nu * dev2(transpose(gradB[pi][i])); }
    const std::vector<vector> divSig = fvc::div(sigC, sigB, m, g, fvp);
    std::vector<scalar> sx(nC), sy(nC), sz(nC);
    for (label c = 0; c < nC; ++c) { sx[c] = g.V()[c]*divSig[c].x; sy[c] = g.V()[c]*divSig[c].y; sz[c] = g.V()[c]*divSig[c].z; }

    // Device.
    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    const DeviceVectorBoundary dbU = buildDeviceVectorBoundary(U, fvp, g);
    std::vector<scalar> ux(nC), uy(nC), uz(nC); for (label c = 0; c < nC; ++c) { ux[c]=U.internal[c].x; uy[c]=U.internal[c].y; uz[c]=U.internal[c].z; }
    DeviceBuffer<scalar> dUx(ux), dUy(uy), dUz(uz), nuCell(std::vector<scalar>(nC, nu)), nuBnd(std::vector<scalar>(dm.nBndFaces, nu));
    DeviceBuffer<scalar> srcX, srcY, srcZ;
    deviceDivDevReff(dm, dbU, dUx, dUy, dUz, nuCell, nuBnd, srcX, srcY, srcZ);

    std::printf("GPU divDevReff source (nCells=%d):\n", nC);
    std::printf("  V*div(sigma).x : %.3e\n", relMax(srcX.host(), sx));
    std::printf("  V*div(sigma).y : %.3e\n", relMax(srcY.host(), sy));
    std::printf("  V*div(sigma).z : %.3e\n", relMax(srcZ.host(), sz));
    const bool pass = relMax(srcX.host(), sx) < 1e-12 && relMax(srcY.host(), sy) < 1e-12 && relMax(srcZ.host(), sz) < 1e-12;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
