// GPU offload: the boundary-flux kernels needed to close the resident loop on device, phiHbyA.boundary
// (HbyA.boundary & Sf) and pEqn.flux().boundary (internalCoeffs*p - boundaryCoeffs). Validated vs CPU.
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
#include "device_boundary.cuh"
#include "device_simple.cuh"
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
    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);

    // HbyA-like vector field with constrainHbyA boundaries (fixedValue inlet/noSlip walls/zeroGrad outlet).
    GeometricField<vector> HbyA; HbyA.internal.resize(nC);
    for (label c = 0; c < nC; ++c) HbyA.internal[c] = { 1.0 + 0.2*std::sin(0.01*c), 0.1*std::cos(0.013*c), 0.0 };
    for (const FvPatch& q : fvp) {
        if (q.type == "empty")      HbyA.boundary.push_back(std::make_unique<EmptyPatchField<vector>>(q));
        else if (q.name == "inlet") HbyA.boundary.push_back(std::make_unique<FixedValuePatchField<vector>>(q, true, vector{10,0,0}, std::vector<vector>{}));
        else if (q.type == "wall")  HbyA.boundary.push_back(std::make_unique<NoSlipPatchField<vector>>(q));
        else                        HbyA.boundary.push_back(std::make_unique<ZeroGradientPatchField<vector>>(q));
    }
    HbyA.evaluateBoundary();
    const SurfaceScalarField phiHbyA = fvc::flux(HbyA, m, g, fvp);

    // device phiHbyA.boundary = (HbyA.boundary & Sf).
    const DeviceVectorBoundary dbH = buildDeviceVectorBoundary(HbyA, fvp, g);
    DeviceBuffer<scalar> uxb, uyb, uzb;
    { std::vector<scalar> hx(nC), hy(nC), hz(nC); for (label c=0;c<nC;++c){ hx[c]=HbyA.internal[c].x; hy[c]=HbyA.internal[c].y; hz[c]=HbyA.internal[c].z; }
      DeviceBuffer<scalar> dhx(hx), dhy(hy), dhz(hz); deviceBCValue(dbH.comp[0], dhx, uxb); deviceBCValue(dbH.comp[1], dhy, uyb); deviceBCValue(dbH.comp[2], dhz, uzb); }
    DeviceBuffer<scalar> dphiB; deviceBoundaryFlux(dm, uxb, uyb, uzb, dphiB);
    const scalar eFlux = relMax(dphiB.host(), flat(phiHbyA.boundary));

    // pEqn.flux().boundary.
    GeometricField<scalar> p; p.internal.resize(nC);
    for (label c = 0; c < nC; ++c) p.internal[c] = std::sin(0.008*c);
    for (const FvPatch& q : fvp) { if (q.type=="empty") p.boundary.push_back(std::make_unique<EmptyPatchField<scalar>>(q));
        else if (q.name=="outlet") p.boundary.push_back(std::make_unique<FixedValuePatchField<scalar>>(q,true,0.0,std::vector<scalar>{}));
        else p.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q)); }
    p.evaluateBoundary();
    std::vector<scalar> rAU(nC); for (label c = 0; c < nC; ++c) rAU[c] = 0.1 + 0.05*std::sin(0.01*c);
    const SurfaceScalarField rAUf = fvc::interpolate(rAU, m, g, fvp);
    const FvScalarMatrix pEqn = fvm::laplacian(rAUf, p, m, g, fvp);
    const SurfaceScalarField pflux = matrixFlux(pEqn, p, m, fvp);

    const DeviceBoundary dbP = buildDeviceBoundary(p, fvp, g);
    DeviceBuffer<scalar> dRAU(rAU), pIC, pBC; deviceBCLaplacianCoeffs(dbP, dRAU, pIC, pBC);
    DeviceBuffer<scalar> dp(p.internal), dFluxB; deviceMatrixFluxBoundary(dbP, pIC, pBC, dp, dFluxB);
    const scalar eMF = relMax(dFluxB.host(), flat(pflux.boundary));

    std::printf("GPU boundary fluxes:\n");
    std::printf("  phiHbyA.boundary (HbyA & Sf) : %.3e\n", eFlux);
    std::printf("  pEqn.flux().boundary         : %.3e\n", eMF);
    const bool pass = eFlux < 1e-13 && eMF < 1e-13;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
