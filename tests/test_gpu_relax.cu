// GPU offload: device matrix relaxation. Validate deviceRelaxDiag (relaxed diagonal + the per-component
// source adjustment delta*psi) against the CPU fvMatrix::relax on a momentum matrix (div - laplacian),
// across the momentum BCs (fixedValue inlet, noSlip walls, zeroGradient outlet, empty).
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "fv_matrix_ops.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_ldu.cuh"
#include "device_blas.cuh"
#include "device_boundary.cuh"
#include "device_simple.cuh"
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
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    const scalar nu = 1e-3, alpha = 0.7;

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

    // CPU: assemble UEqn = div - laplacian, then relax.
    FvVectorMatrix UEqn = fvm::div(phi.internal, phi.boundary, U, m, fvp);
    addEqual(UEqn, fvm::laplacian(U, nu, m, g, fvp), -1.0);
    relaxMatrix(UEqn, U, m, fvp, alpha);
    const std::vector<scalar> relaxedDiagCpu = UEqn.diag;   // raw diag was overwritten by relax

    // Device: assemble + relax.
    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    const DeviceVectorBoundary dbU = buildDeviceVectorBoundary(U, fvp, g);
    DeviceBuffer<scalar> dphi(phi.internal), nuU(std::vector<scalar>(nIf, nu)), nuCell(std::vector<scalar>(nC, nu));
    DeviceBuffer<scalar> mDiag, mUp, mLo, lD, lU, lL;
    deviceDivUpwindCoeffs(dm, dphi, mDiag, mUp, mLo);
    deviceLaplacianCoeffs(dm, nuU, lD, lU, lL);
    deviceAxpy(-1.0, lD, mDiag); deviceAxpy(-1.0, lU, mUp); deviceAxpy(-1.0, lL, mLo);   // raw UEqn = div - laplacian
    // combined boundary internalCoeffs (component 0, isotropic): div iC - laplacian iC.
    DeviceBuffer<scalar> phiB(([&]{ std::vector<scalar> o; for (auto& a : phi.boundary) o.insert(o.end(),a.begin(),a.end()); return o; }()));
    DeviceBuffer<scalar> dIC, dBC, lIC, lBC; deviceBCDivCoeffs(dbU.comp[0], phiB, dIC, dBC); deviceBCLaplacianCoeffs(dbU.comp[0], nuCell, lIC, lBC);
    deviceAxpy(-1.0, lIC, dIC);   // dIC = combined internalCoeffs (comp 0)
    DeviceBuffer<scalar> relaxedDiag, delta;
    deviceRelaxDiag(deviceLduView(dm, mDiag, mUp, mLo), dm, dIC, alpha, relaxedDiag, delta);

    const scalar eDiag = relMax(relaxedDiag.host(), relaxedDiagCpu);

    // per-component source adjustment = delta * U_k vs CPU UEqn.source_k.
    scalar eSrc = 0;
    for (int k = 0; k < 3; ++k) {
        std::vector<scalar> Uk(nC); for (label c = 0; c < nC; ++c) Uk[c] = component(U.internal[c], k);
        DeviceBuffer<scalar> dUk(Uk), src; deviceHadamard(src, delta, dUk);
        std::vector<scalar> srcCpu(nC); for (label c = 0; c < nC; ++c) srcCpu[c] = component(UEqn.source[c], k);
        eSrc = std::fmax(eSrc, relMax(src.host(), srcCpu));
    }

    std::printf("GPU device relaxMatrix (nCells=%d, alpha=%.2f):\n", nC, alpha);
    std::printf("  relaxed diag         : %.3e\n", eDiag);
    std::printf("  source adj (delta*U) : %.3e\n", eSrc);
    const bool pass = eDiag < 1e-13 && eSrc < 1e-13;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
