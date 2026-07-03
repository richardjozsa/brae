// GPU offload G1: device lduMatrix SpMV (Amul) on a REAL matrix (a Gauss Laplacian on pitzDaily).
// Validate the device per-cell gather against the CPU face-scatter Amul. The gather sums each row in a
// different order than the CPU scatter, so they match to machine precision (FP non-associativity).
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "fvm.cuh"
#include "ldu_spmv.cuh"
#include "device_buffer.cuh"
#include "device_ldu.cuh"
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace brae;

int main(int argc, char** argv) {
    const std::string caseDir = argc > 1 ? argv[1] : "validation/pitzDaily";
    PrimitiveMesh m; m.read(caseDir + "/constant/polyMesh");
    FvGeometry g; g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells(), nIf = m.nInternalFaces();

    // A real lduMatrix: Gauss laplacian(1, f) on the mesh.
    GeometricField<scalar> f; f.internal.assign(nC, 1.0);
    for (const FvPatch& p : fvp) {
        if (p.type == "empty") f.boundary.push_back(std::make_unique<EmptyPatchField<scalar>>(p));
        else                   f.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(p));
    }
    f.evaluateBoundary();
    const FvScalarMatrix M = fvm::laplacian(f, 1.0, m, g, fvp);

    // test vector.
    std::vector<scalar> psi(nC);
    for (label c = 0; c < nC; ++c) psi[c] = std::sin(0.013 * c) + 0.5 * std::cos(0.007 * c);

    // CPU oracle.
    const std::vector<scalar> Acpu = Amul(M, psi, m);

    // GPU: internal-face owner/neighbour + the SpMV gather.
    const std::vector<label> ownerInt(m.owner().begin(), m.owner().begin() + nIf);
    const DeviceLduMatrix dM = buildDeviceLdu(M.diag, M.upper, M.lower, ownerInt, m.neighbour(), nC);
    DeviceBuffer<scalar> dpsi(psi), dApsi(nC);
    deviceAmul(dM.view(), dpsi, dApsi);
    const std::vector<scalar> Agpu = dApsi.host();

    scalar nE = 0, dE = 0;
    for (label c = 0; c < nC; ++c) { nE = std::fmax(nE, std::fabs(Agpu[c] - Acpu[c])); dE = std::fmax(dE, std::fabs(Acpu[c])); }
    const scalar relErr = nE / dE;

    std::printf("GPU G1 SpMV (nCells=%d, nInternalFaces=%d):\n", nC, nIf);
    std::printf("  device Amul vs CPU Amul : %.3e\n", relErr);
    const bool pass = relErr < 1e-12;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
