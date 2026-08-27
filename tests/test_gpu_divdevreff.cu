// GPU divDevReff diagnosis and regression coverage.
//
// The original pitzDaily fixture used a spatially varying Uz on a mesh with empty front/back patches. That
// makes the z source a near-zero, non-solution-direction quantity, but it remains a gated regression for the
// empty-face fix. A compatible 2D case and a genuinely 3D case are also gated independently; the 3D case is the
// component-wise negative control for a real z-indexing/contraction defect.
#include "box_mesh.cuh"
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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace brae;

namespace {

constexpr scalar STAGE_TOL = 1e-12;

struct Metric
{
    scalar absDiff = 0.0;
    scalar hostMag = 0.0;
    scalar gpuMag = 0.0;
    scalar relative = 0.0;
    label index = -1;
};

Metric metric(const std::vector<scalar>& host, const std::vector<scalar>& gpu, label n, int q)
{
    Metric r;
    for (label i = 0; i < n; ++i)
    {
        const scalar h = host[static_cast<std::size_t>(q) * n + i];
        const scalar d = gpu[static_cast<std::size_t>(q) * n + i];
        r.hostMag = std::fmax(r.hostMag, std::fabs(h));
        r.gpuMag = std::fmax(r.gpuMag, std::fabs(d));
        const scalar e = std::fabs(d - h);
        if (e > r.absDiff) { r.absDiff = e; r.index = i; }
    }
    r.relative = r.hostMag > 0.0 ? r.absDiff / r.hostMag : r.absDiff;
    return r;
}

std::vector<scalar> tensorsSoA(const std::vector<tensor>& v)
{
    std::vector<scalar> out(static_cast<std::size_t>(9) * v.size());
    for (std::size_t c = 0; c < v.size(); ++c)
    {
        const scalar q[9] = {v[c].xx, v[c].xy, v[c].xz, v[c].yx, v[c].yy, v[c].yz,
                             v[c].zx, v[c].zy, v[c].zz};
        for (int i = 0; i < 9; ++i) out[static_cast<std::size_t>(i) * v.size() + c] = q[i];
    }
    return out;
}

std::vector<scalar> boundaryTensorsSoA(const std::vector<std::vector<tensor>>& v,
                                       const std::vector<FvPatch>& patches)
{
    std::vector<scalar> faceMajor;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (isCoupledInterfaceType(patches[pi].type)) continue;
        for (label i = 0; i < patches[pi].size; ++i)
        {
            const tensor& t = v[pi][i];
            const scalar q[9] = {t.xx, t.xy, t.xz, t.yx, t.yy, t.yz, t.zx, t.zy, t.zz};
            for (int j = 0; j < 9; ++j) faceMajor.push_back(q[j]);
        }
    }
    // Host boundary fields are naturally patch/face-major; DeviceBuffer snapshots are component-major.
    const std::size_t n = faceMajor.size() / 9;
    std::vector<scalar> soa(faceMajor.size());
    for (std::size_t i = 0; i < n; ++i)
        for (int q = 0; q < 9; ++q)
            soa[static_cast<std::size_t>(q) * n + i] = faceMajor[i * 9 + q];
    return soa;
}

std::vector<scalar> vectorsSoA(const std::vector<vector>& v)
{
    std::vector<scalar> out(static_cast<std::size_t>(3) * v.size());
    for (std::size_t i = 0; i < v.size(); ++i)
    {
        out[i] = v[i].x;
        out[v.size() + i] = v[i].y;
        out[2 * v.size() + i] = v[i].z;
    }
    return out;
}

scalar maxAbs(const std::vector<vector>& v, int comp)
{
    scalar r = 0.0;
    for (const vector& x : v) r = std::fmax(r, std::fabs(comp == 0 ? x.x : (comp == 1 ? x.y : x.z)));
    return r;
}

std::vector<vector> volumeSource(const std::vector<vector>& div, const FvGeometry& g)
{
    std::vector<vector> out(div.size());
    for (std::size_t c = 0; c < div.size(); ++c) out[c] = g.V()[c] * div[c];
    return out;
}

const char* tensorName(int q)
{
    static const char* names[9] = {"xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"};
    return names[q];
}

bool printTensorStage(const char* name, const std::vector<scalar>& host,
                      const std::vector<scalar>& gpu, label n)
{
    std::printf("  %s (all tensor components, n=%d):\n", name, n);
    bool ok = true;
    for (int q = 0; q < 9; ++q)
    {
        const Metric m = metric(host, gpu, n, q);
        std::printf("    %-2s abs %.3e host %.3e gpu %.3e rel %.3e at %d\n",
                    tensorName(q), (double)m.absDiff, (double)m.hostMag, (double)m.gpuMag,
                    (double)m.relative, m.index);
        if (m.relative > STAGE_TOL) ok = false;
    }
    return ok;
}

struct Touch
{
    bool internal = false;
    bool wall = false;
    bool inlet = false;
    bool outlet = false;
    bool empty = false;
};

Touch cellTouch(const PrimitiveMesh& m, const std::vector<FvPatch>& patches, label c)
{
    Touch t;
    for (label f = 0; f < m.nInternalFaces(); ++f)
        if (m.owner()[f] == c || m.neighbour()[f] == c) t.internal = true;
    for (const FvPatch& p : patches)
        for (label i = 0; i < p.size; ++i)
            if (p.faceCells[i] == c)
            {
                t.wall = t.wall || p.type == "wall";
                t.inlet = t.inlet || p.name == "inlet";
                t.outlet = t.outlet || p.name == "outlet";
                t.empty = t.empty || p.type == "empty";
            }
    return t;
}

bool printSource(const char* name, const std::vector<scalar>& host, const std::vector<scalar>& gpu,
                 const PrimitiveMesh& mesh, const std::vector<FvPatch>& patches)
{
    const label n = mesh.nCells();
    static const char* names[3] = {"x", "y", "z"};
    bool ok = true;
    std::printf("  %s:\n", name);
    for (int q = 0; q < 3; ++q)
    {
        const Metric m = metric(host, gpu, n, q);
        const Touch t = m.index >= 0 ? cellTouch(mesh, patches, m.index) : Touch{};
        std::printf("    V*div(sigma).%-1s abs %.3e host %.3e gpu %.3e rel %.3e cell %d "
                    "touches[internal=%s wall=%s inlet=%s outlet=%s empty=%s]\n",
                    names[q], (double)m.absDiff, (double)m.hostMag, (double)m.gpuMag, (double)m.relative,
                    m.index, t.internal ? "yes" : "no", t.wall ? "yes" : "no",
                    t.inlet ? "yes" : "no", t.outlet ? "yes" : "no", t.empty ? "yes" : "no");
        if (m.relative > STAGE_TOL) ok = false;
    }
    return ok;
}

GeometricField<vector> makePitzField(const std::vector<FvPatch>& patches, label nC, bool compatible2D)
{
    GeometricField<vector> U;
    U.internal.resize(nC);
    for (label c = 0; c < nC; ++c)
        U.internal[c] = {1.0 + 0.2*std::sin(0.01*c), 0.1*std::cos(0.013*c),
                         compatible2D ? 0.0 : 0.05*std::sin(0.02*c)};
    for (const FvPatch& q : patches)
    {
        if (q.type == "empty")      U.boundary.push_back(std::make_unique<EmptyPatchField<vector>>(q));
        else if (q.name == "inlet") U.boundary.push_back(std::make_unique<FixedValuePatchField<vector>>(
                                      q, true, vector{10,0,0}, std::vector<vector>{}));
        else if (q.type == "wall")  U.boundary.push_back(std::make_unique<NoSlipPatchField<vector>>(q));
        else                         U.boundary.push_back(std::make_unique<ZeroGradientPatchField<vector>>(q));
    }
    U.evaluateBoundary();
    return U;
}

GeometricField<scalar> makeScalarField(const std::vector<FvPatch>& patches,
                                       const FvGeometry& g, label nC)
{
    GeometricField<scalar> f;
    f.internal.resize(nC);
    for (label c = 0; c < nC; ++c)
        f.internal[c] = 0.7 + 0.13*std::sin(0.017*c) + 0.04*g.C()[c].x;
    for (const FvPatch& q : patches)
    {
        if (q.type == "empty")
            f.boundary.push_back(std::make_unique<EmptyPatchField<scalar>>(q));
        else if (q.name == "inlet")
            f.boundary.push_back(std::make_unique<FixedValuePatchField<scalar>>(
                q, true, scalar{2.0}, std::vector<scalar>{}));
        else
            f.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q));
    }
    f.evaluateBoundary();
    return f;
}

bool checkScalarGaussGrad(const PrimitiveMesh& mesh, const FvGeometry& g,
                          const std::vector<FvPatch>& patches, const DeviceMesh& dm)
{
    const label nC = mesh.nCells();
    GeometricField<scalar> f = makeScalarField(patches, g, nC);
    const std::vector<vector> host = fvc::gaussGrad(f, mesh, g, patches);
    const DeviceBoundary db = buildDeviceBoundary(f, patches, g);
    std::vector<scalar> cell(nC);
    for (label c = 0; c < nC; ++c) cell[c] = f.internal[c];
    DeviceBuffer<scalar> dCell(cell), bval, gx, gy, gz;
    deviceBCValue(db, dCell, bval);
    deviceGaussGrad(dm, dCell, bval, gx, gy, gz);

    std::vector<scalar> hostSoA(static_cast<std::size_t>(3) * nC), gpuSoA;
    for (label c = 0; c < nC; ++c)
    {
        hostSoA[c] = host[c].x;
        hostSoA[nC + c] = host[c].y;
        hostSoA[2*nC + c] = host[c].z;
    }
    const std::vector<scalar> hx = gx.host(), hy = gy.host(), hz = gz.host();
    gpuSoA.resize(static_cast<std::size_t>(3) * nC);
    for (label c = 0; c < nC; ++c)
    {
        gpuSoA[c] = hx[c];
        gpuSoA[nC + c] = hy[c];
        gpuSoA[2*nC + c] = hz[c];
    }

    static const char* names[3] = {"x", "y", "z"};
    bool ok = true;
    std::printf("  scalar Gauss gradient (empty-face ordering, n=%d):\n", nC);
    for (int q = 0; q < 3; ++q)
    {
        const Metric m = metric(hostSoA, gpuSoA, nC, q);
        std::printf("    %-1s abs %.3e host %.3e gpu %.3e rel %.3e at %d\n",
                    names[q], (double)m.absDiff, (double)m.hostMag, (double)m.gpuMag,
                    (double)m.relative, m.index);
        if (m.relative > STAGE_TOL) ok = false;
    }
    return ok;
}

GeometricField<vector> makeBoxField(const std::vector<FvPatch>& patches, const FvGeometry& g, label nC)
{
    GeometricField<vector> U;
    U.internal.resize(nC);
    for (label c = 0; c < nC; ++c)
    {
        const vector& x = g.C()[c];
        U.internal[c] = {1.2 + 0.17*std::sin(0.31*x.x + 0.13*x.y) + 0.04*x.z,
                         -0.4 + 0.21*std::cos(0.23*x.x - 0.19*x.z) + 0.03*x.y,
                         0.3 + 0.16*std::sin(0.17*x.x + 0.11*x.y + 0.29*x.z)};
    }
    for (const FvPatch& q : patches)
    {
        if (q.name == "inlet")
            U.boundary.push_back(std::make_unique<FixedValuePatchField<vector>>(
                q, true, vector{2.0, -0.5, 0.7}, std::vector<vector>{}));
        else if (q.name == "outlet")
            U.boundary.push_back(std::make_unique<ZeroGradientPatchField<vector>>(q));
        else if (q.name == "wallYmax")
            U.boundary.push_back(std::make_unique<SymmetryPlanePatchField<vector>>(q));
        else if (q.type == "wall")
            U.boundary.push_back(std::make_unique<NoSlipPatchField<vector>>(q));
        else if (q.type == "empty")
            U.boundary.push_back(std::make_unique<EmptyPatchField<vector>>(q));
        else
            U.boundary.push_back(std::make_unique<ZeroGradientPatchField<vector>>(q));
    }
    U.evaluateBoundary();
    return U;
}

GeometricField<vector> makeRetainedDiagnosticField(const std::vector<FvPatch>& patches,
                                                   const FvGeometry& g, label nC)
{
    GeometricField<vector> U;
    U.internal.resize(nC);
    for (label c = 0; c < nC; ++c)
    {
        const vector& x = g.C()[c];
        U.internal[c] = {1.1 + 0.07*std::sin(0.41*x.x + 0.17*x.y) + 0.02*x.z,
                         0.2 + 0.09*std::cos(0.29*x.x - 0.23*x.z) + 0.01*x.y,
                         -0.3 + 0.11*std::sin(0.19*x.x + 0.31*x.y + 0.37*x.z)};
    }
    for (const FvPatch& q : patches)
    {
        if (q.type == "empty")
            U.boundary.push_back(std::make_unique<EmptyPatchField<vector>>(q));
        else if (q.type == "wall")
            U.boundary.push_back(std::make_unique<NoSlipPatchField<vector>>(q));
        else if (q.name == "inlet")
            U.boundary.push_back(std::make_unique<FixedValuePatchField<vector>>(
                q, true, vector{1.0, 0.0, 0.0}, std::vector<vector>{}));
        else if (q.name == "outlet")
            U.boundary.push_back(std::make_unique<ZeroGradientPatchField<vector>>(q));
        else
            U.boundary.push_back(std::make_unique<ZeroGradientPatchField<vector>>(q));
    }
    U.evaluateBoundary();
    return U;
}

struct MeshStructure
{
    bool hasEmptyPatch = false;
    bool zNonDegenerate = false;
    bool structurally2D = false;
    scalar zExtent = 0.0;
};

MeshStructure classifyMesh(const PrimitiveMesh& mesh, const std::vector<FvPatch>& patches)
{
    MeshStructure s;
    for (const FvPatch& p : patches) s.hasEmptyPatch = s.hasEmptyPatch || p.type == "empty";
    if (mesh.points().empty()) throw std::runtime_error("cannot classify a mesh with no points");

    scalar minX = mesh.points().front().x, maxX = minX;
    scalar minY = mesh.points().front().y, maxY = minY;
    scalar minZ = mesh.points().front().z, maxZ = minZ;
    for (const vector& p : mesh.points())
    {
        minX = std::fmin(minX, p.x); maxX = std::fmax(maxX, p.x);
        minY = std::fmin(minY, p.y); maxY = std::fmax(maxY, p.y);
        minZ = std::fmin(minZ, p.z); maxZ = std::fmax(maxZ, p.z);
    }
    s.zExtent = maxZ - minZ;
    const scalar scale = std::max({scalar{1.0}, maxX - minX, maxY - minY, s.zExtent});
    s.zNonDegenerate = s.zExtent > 1e-12 * scale;
    s.structurally2D = s.hasEmptyPatch || !s.zNonDegenerate;
    return s;
}

struct RunResult
{
    bool pass = false;
    bool stagePass = false;
};

RunResult runCase(const char* caseLabel, PrimitiveMesh& mesh, GeometricField<vector>& U,
                  scalar nu, bool gate, bool requireNonzeroSource)
{
    FvGeometry g;
    g.build(mesh);
    const std::vector<FvPatch> patches = buildPatches(mesh, g);
    const label nC = mesh.nCells();
    const std::vector<tensor> gradC = fvc::gaussGrad(U, mesh, g, patches);
    const std::vector<std::vector<tensor>> gradB = fvc::gradUBoundary(U, gradC, mesh, g, patches);
    std::vector<tensor> sigC(nC);
    for (label c = 0; c < nC; ++c) sigC[c] = nu * dev2(transpose(gradC[c]));
    std::vector<std::vector<tensor>> sigB(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        sigB[pi].resize(patches[pi].size);
        for (label i = 0; i < patches[pi].size; ++i)
            sigB[pi][i] = nu * dev2(transpose(gradB[pi][i]));
    }
    const std::vector<vector> hostDiv = volumeSource(fvc::div(sigC, sigB, mesh, g, patches), g);
    const std::vector<scalar> hostGradC = tensorsSoA(gradC);
    const std::vector<scalar> hostGradB = boundaryTensorsSoA(gradB, patches);
    const std::vector<scalar> hostSigmaC = tensorsSoA(sigC);
    const std::vector<scalar> hostSigmaB = boundaryTensorsSoA(sigB, patches);

    const DeviceMesh dm = buildDeviceMesh(mesh, g, patches);
    const std::size_t expectedBoundaryScalars = static_cast<std::size_t>(9) * dm.nBndFaces;
    if (hostGradB.size() != expectedBoundaryScalars || hostSigmaB.size() != expectedBoundaryScalars)
        throw std::runtime_error(
            std::string(caseLabel) + ": host/device boundary ordering size mismatch: expected "
            + std::to_string(expectedBoundaryScalars) + " scalars for each tensor stage, got grad="
            + std::to_string(hostGradB.size()) + " sigma=" + std::to_string(hostSigmaB.size()));
    const bool scalarGradOk = checkScalarGaussGrad(mesh, g, patches, dm);
    const DeviceVectorBoundary dbU = buildDeviceVectorBoundary(U, patches, g);
    std::vector<scalar> ux(nC), uy(nC), uz(nC);
    for (label c = 0; c < nC; ++c) { ux[c] = U.internal[c].x; uy[c] = U.internal[c].y; uz[c] = U.internal[c].z; }
    DeviceBuffer<scalar> dUx(ux), dUy(uy), dUz(uz);
    DeviceBuffer<scalar> nuCell(std::vector<scalar>(nC, nu));
    DeviceBuffer<scalar> nuBnd(std::vector<scalar>(dm.nBndFaces, nu));
    DeviceBuffer<scalar> srcX, srcY, srcZ;
    DeviceBuffer<scalar> gradCell, gradBoundary, sigmaCell, sigmaBoundary;
    DeviceDivDevReffStages stages{&gradCell, &gradBoundary, &sigmaCell, &sigmaBoundary};
    deviceDivDevReff(dm, dbU, dUx, dUy, dUz, nuCell, nuBnd, srcX, srcY, srcZ,
                     nullptr, nullptr, nullptr, 0.0, &stages);

    const std::vector<scalar> gpuGradC = gradCell.host();
    const std::vector<scalar> gpuGradB = gradBoundary.host();
    const std::vector<scalar> gpuSigmaC = sigmaCell.host();
    const std::vector<scalar> gpuSigmaB = sigmaBoundary.host();
    const std::vector<scalar> gpuSource = [&]() {
        const std::vector<scalar> x = srcX.host(), y = srcY.host(), z = srcZ.host();
        std::vector<scalar> v(static_cast<std::size_t>(3) * nC);
        for (label c = 0; c < nC; ++c) { v[c] = x[c]; v[nC+c] = y[c]; v[2*nC+c] = z[c]; }
        return v;
    }();

    std::printf("\n%s (nCells=%d, nInternalFaces=%d, nBoundaryFaces=%d):\n",
                caseLabel, nC, dm.nInternalFaces, dm.nBndFaces);
    const bool gradCellOk = printTensorStage("cell grad(U)", hostGradC, gpuGradC, nC);
    const bool gradBoundaryOk = printTensorStage("boundary grad(U)", hostGradB, gpuGradB, dm.nBndFaces);
    const bool sigmaCellOk = printTensorStage("cell sigma", hostSigmaC, gpuSigmaC, nC);
    const bool sigmaBoundaryOk = printTensorStage("boundary sigma", hostSigmaB, gpuSigmaB, dm.nBndFaces);
    const bool sourceOk = printSource("final tensor-divergence source", vectorsSoA(hostDiv), gpuSource, mesh, patches);
    const bool stageOk = scalarGradOk && gradCellOk && gradBoundaryOk && sigmaCellOk && sigmaBoundaryOk;
    const char* firstBad = !scalarGradOk ? "scalar Gauss gradient"
        : !gradCellOk ? "cell grad(U)"
        : !gradBoundaryOk ? "boundary grad(U)"
        : !sigmaCellOk ? "cell sigma"
        : !sigmaBoundaryOk ? "boundary sigma"
        : !sourceOk ? "final tensor-divergence source" : "none";
    std::printf("  first stage beyond relative %.0e: %s\n", (double)STAGE_TOL, firstBad);

    std::vector<tensor> zeroCell(nC, tensor{0,0,0,0,0,0,0,0,0});
    std::vector<std::vector<tensor>> zeroBoundary(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        zeroBoundary[pi].assign(patches[pi].size, tensor{0,0,0,0,0,0,0,0,0});
    const std::vector<vector> hostInterior = volumeSource(fvc::div(sigC, zeroBoundary, mesh, g, patches), g);
    const std::vector<vector> hostBoundary = volumeSource(fvc::div(zeroCell, sigB, mesh, g, patches), g);
    DeviceBuffer<scalar> zC(std::vector<scalar>(static_cast<std::size_t>(9) * nC, 0.0));
    DeviceBuffer<scalar> zB(std::vector<scalar>(static_cast<std::size_t>(9) * dm.nBndFaces, 0.0));
    DeviceBuffer<scalar> intX, intY, intZ, bndX, bndY, bndZ;
    deviceTensorDivSource(dm, sigmaCell, zB, intX, intY, intZ);
    deviceTensorDivSource(dm, zC, sigmaBoundary, bndX, bndY, bndZ);
    const std::vector<scalar> gpuInterior = [&]() {
        const auto x=intX.host(), y=intY.host(), z=intZ.host(); std::vector<scalar> v(3*nC);
        for (label c=0;c<nC;++c) { v[c]=x[c]; v[nC+c]=y[c]; v[2*nC+c]=z[c]; } return v;
    }();
    const std::vector<scalar> gpuBoundary = [&]() {
        const auto x=bndX.host(), y=bndY.host(), z=bndZ.host(); std::vector<scalar> v(3*nC);
        for (label c=0;c<nC;++c) { v[c]=x[c]; v[nC+c]=y[c]; v[2*nC+c]=z[c]; } return v;
    }();
    const bool interiorOk = printSource("interior-only contribution", vectorsSoA(hostInterior), gpuInterior, mesh, patches);
    const bool boundaryOk = printSource("boundary-only contribution", vectorsSoA(hostBoundary), gpuBoundary, mesh, patches);
    std::printf("  contribution magnitudes: interior %.3e, boundary %.3e\n",
                (double)std::max({maxAbs(hostInterior,0), maxAbs(hostInterior,1), maxAbs(hostInterior,2)}),
                (double)std::max({maxAbs(hostBoundary,0), maxAbs(hostBoundary,1), maxAbs(hostBoundary,2)}));

    bool negativeControl = true;
    if (requireNonzeroSource)
    {
        const scalar sx = maxAbs(hostDiv,0), sy = maxAbs(hostDiv,1), sz = maxAbs(hostDiv,2);
        std::printf("  3D non-vacuity: host source maxima x %.3e y %.3e z %.3e\n", (double)sx, (double)sy, (double)sz);
        if (sx <= 1e-12 || sy <= 1e-12 || sz <= 1e-12) negativeControl = false;
        std::vector<scalar> perturbed = gpuSource;
        perturbed[2*nC] += 1e-6;  // deliberately wrong z value; the component-wise gate must see it.
        const Metric z = metric(vectorsSoA(hostDiv), perturbed, nC, 2);
        const bool detected = z.relative > STAGE_TOL;
        negativeControl = negativeControl && detected;
        std::printf("  3D z negative control: injected 1e-6 gives relative %.3e -> %s\n",
                    (double)z.relative, detected ? "detected" : "NOT DETECTED");
    }

    RunResult result;
    result.pass = !gate || (stageOk && sourceOk && interiorOk && boundaryOk && negativeControl);
    result.stagePass = stageOk;
    return result;
}

} // namespace

int main(int argc, char** argv)
{
    const std::string requested = argc > 1 ? argv[1] : "validation/pitzDaily";
    bool allOk = true;

    const std::filesystem::path casePath = std::filesystem::path(requested).lexically_normal();
    const std::filesystem::path polyMeshPath = casePath / "constant" / "polyMesh";
    if (!std::filesystem::is_directory(polyMeshPath))
    {
        std::fprintf(stderr,
                     "gpu_divdevreff ERROR: unrecognized or missing case path '%s' (expected %s)\n",
                     requested.c_str(), polyMeshPath.string().c_str());
        return 2;
    }

    if (casePath.filename() == "pitzDaily")
    {
        PrimitiveMesh pitz;
        pitz.read(requested + "/constant/polyMesh");
        FvGeometry pg;
        pg.build(pitz);
        const std::vector<FvPatch> pp = buildPatches(pitz, pg);

        // Preserve the original failing construction as a gated regression: the production empty-face fix is
        // exercised by this varying-Uz field even though the field is not solution-direction-compatible with
        // pitzDaily's empty front/back patches.
        GeometricField<vector> invalid2D = makePitzField(pp, pitz.nCells(), false);
        const RunResult legacy = runCase("legacy pitzDaily varying-Uz gated regression", pitz, invalid2D, 1e-3, true, false);
        allOk = allOk && legacy.pass;

        // The same 2D geometry with solution-direction-compatible fields is a required independent validation.
        GeometricField<vector> valid2D = makePitzField(pp, pitz.nCells(), true);
        const RunResult valid = runCase("valid 2D pitzDaily empty-direction case", pitz, valid2D, 1e-3, true, false);
        allOk = allOk && valid.pass;

        // A structured 3D case exercises non-zero spatial variation in all U components and all four requested
        // BC families: fixedValue inlet, zeroGradient outlet, noSlip walls, and a symmetry-like plane.
        PrimitiveMesh box = boxtest::boxMesh(5, 4, 3);
        FvGeometry bg;
        bg.build(box);
        const std::vector<FvPatch> bp = buildPatches(box, bg);
        GeometricField<vector> valid3D = makeBoxField(bp, bg, box.nCells());
        const RunResult threeD = runCase("genuinely 3D validation case", box, valid3D, 1e-3, true, true);
        allOk = allOk && threeD.pass;
    }
    else
    {
        // A supplied extraction is an identical-mesh diagnostic, not a substituted 2D fixture.
        PrimitiveMesh retained;
        retained.read(requested + "/constant/polyMesh");
        FvGeometry rg;
        rg.build(retained);
        const std::vector<FvPatch> rp = buildPatches(retained, rg);
        const MeshStructure structure = classifyMesh(retained, rp);
        std::printf("  mesh structure: emptyPatches=%s zExtent %.3e zNonDegenerate=%s -> %s\n",
                    structure.hasEmptyPatch ? "yes" : "no", (double)structure.zExtent,
                    structure.zNonDegenerate ? "yes" : "no",
                    structure.structurally2D ? "structurally-2D" : "genuinely-3D");
        const bool requireNonzeroSource = !structure.structurally2D;
        if (structure.structurally2D)
            std::printf("  structurally-2D bundle: z-source non-vacuity is not required\n");
        GeometricField<vector> retainedU = makeRetainedDiagnosticField(rp, rg, retained.nCells());
        const RunResult ahmed = runCase("retained mesh deterministic diagnostic", retained, retainedU, 1e-3, true, requireNonzeroSource);
        allOk = allOk && ahmed.pass;
    }

    std::printf("%s\n", allOk ? "PASS" : "FAIL");
    return allOk ? 0 : 1;
}
