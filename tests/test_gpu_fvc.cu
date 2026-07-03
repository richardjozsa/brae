// GPU offload G3: fvc explicit operators on device (interpolate / div / gaussGrad), validated against the
// CPU fvc on pitzDaily. The cell-gather operators (div, gaussGrad) use the ownerStart/losort + boundary
// gather; they match the CPU scatter to machine precision (FP order).
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
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
    const label nC = m.nCells(), nIf = m.nInternalFaces();

    // scalar field p: fixedValue inlet/outlet (=2), zeroGradient walls, empty front/back.
    GeometricField<scalar> p; p.internal.resize(nC);
    for (label c = 0; c < nC; ++c) p.internal[c] = std::sin(0.013 * c) + 0.4 * std::cos(0.006 * c);
    for (const FvPatch& q : fvp) {
        if (q.type == "empty")     p.boundary.push_back(std::make_unique<EmptyPatchField<scalar>>(q));
        else if (q.type == "wall") p.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q));
        else                       p.boundary.push_back(std::make_unique<FixedValuePatchField<scalar>>(q, true, 2.0, std::vector<scalar>{}));
    }
    p.evaluateBoundary();

    // vector field U for the flux/div test.
    GeometricField<vector> U; U.internal.resize(nC);
    for (label c = 0; c < nC; ++c) U.internal[c] = { std::sin(0.01*c), std::cos(0.013*c), 0.0 };
    for (const FvPatch& q : fvp) {
        if (q.type == "empty") U.boundary.push_back(std::make_unique<EmptyPatchField<vector>>(q));
        else                   U.boundary.push_back(std::make_unique<ZeroGradientPatchField<vector>>(q));
    }
    U.evaluateBoundary();

    // CPU oracle.
    const SurfaceScalarField sfCpu  = fvc::interpolate(p.internal, m, g, fvp);
    const std::vector<vector> gCpu  = fvc::gaussGrad(p, m, g, fvp);
    const SurfaceScalarField phiCpu = fvc::flux(U, m, g, fvp);
    const std::vector<scalar> divCpu = fvc::div(phiCpu, m, g, fvp);

    // Device.
    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    std::vector<std::vector<scalar>> pbnd; for (std::size_t pi = 0; pi < fvp.size(); ++pi) pbnd.push_back(p.boundary[pi]->value());
    DeviceBuffer<scalar> dp(p.internal), dpbval(flattenBoundary(pbnd));

    DeviceBuffer<scalar> dsf; deviceInterpolate(dm, dp, dsf);
    DeviceBuffer<scalar> dgx, dgy, dgz; deviceGaussGrad(dm, dp, dpbval, dgx, dgy, dgz);
    DeviceBuffer<scalar> dphiInt(phiCpu.internal), dphiBval(flattenBoundary(phiCpu.boundary)), ddiv;
    deviceDiv(dm, dphiInt, dphiBval, ddiv);

    // Compare.
    std::vector<scalar> sfRef(nIf); for (label f = 0; f < nIf; ++f) sfRef[f] = sfCpu.internal[f];
    const scalar eInterp = relMax(dsf.host(), sfRef);
    std::vector<scalar> gxRef(nC), gyRef(nC); for (label c = 0; c < nC; ++c) { gxRef[c] = gCpu[c].x; gyRef[c] = gCpu[c].y; }
    const scalar eGx = relMax(dgx.host(), gxRef), eGy = relMax(dgy.host(), gyRef);
    const scalar eDiv = relMax(ddiv.host(), divCpu);

    std::printf("GPU G3 fvc (nCells=%d):\n", nC);
    std::printf("  interpolate(scalar) : %.3e\n", eInterp);
    std::printf("  gaussGrad(scalar).x : %.3e   .y : %.3e\n", eGx, eGy);
    std::printf("  div(surfaceScalar)  : %.3e\n", eDiv);
    const bool pass = eInterp < 1e-13 && eGx < 1e-12 && eGy < 1e-12 && eDiv < 1e-12;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
