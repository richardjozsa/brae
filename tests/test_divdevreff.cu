// Validate cf's new turbulent-stress operators against OpenFOAM (dumpDivDevReff): the explicit
// part of turbulence->divDevReff(U) = fvc::div(nuEff*dev2(T(grad(U)))). Isolates gradUBoundary
// (snGrad-corrected boundary face gradient) + fvc::div(volTensorField). Run: test_divdevreff <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvc.cuh"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace brae;

static scalar relErrV(const std::vector<vector>& a, const std::vector<vector>& b) {
    scalar mx = 0, mg = 0;
    for (std::size_t i = 0; i < b.size(); ++i) {
        mx = std::fmax(mx, std::fmax(std::fabs(a[i].x - b[i].x), std::fmax(std::fabs(a[i].y - b[i].y), std::fabs(a[i].z - b[i].z))));
        mg = std::fmax(mg, std::fmax(std::fabs(b[i].x), std::fmax(std::fabs(b[i].y), std::fabs(b[i].z))));
    }
    return mg > 0 ? mx / mg : mx;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: %s <caseDir>\n", argv[0]); return 2; }
    const std::string caseDir = argv[1];
    const scalar nu = 1e-5;

    PrimitiveMesh m;  m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;     g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);

    GeometricField<vector> U   = buildField<vector>(readField<vector>(caseDir + "/282/U"),   patches, m.nCells());  U.evaluateBoundary();
    GeometricField<scalar> nut = buildField<scalar>(readField<scalar>(caseDir + "/282/nut"), patches, m.nCells());

    // nuEff = nu + nut (cell + boundary, using nut's file boundary values as OF does).
    std::vector<scalar> nuEffC(m.nCells());
    for (label c = 0; c < m.nCells(); ++c) nuEffC[c] = nu + nut.internal[c];
    std::vector<std::vector<scalar>> nuEffB(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi) {
        const std::vector<scalar>& nv = nut.boundary[pi]->value();
        nuEffB[pi].resize(patches[pi].size);
        for (label i = 0; i < patches[pi].size; ++i) nuEffB[pi][i] = nu + nv[i];
    }

    // sigma = nuEff * dev2(T(grad U)), cell + boundary; then fvc::div(sigma).
    const std::vector<tensor> gradC = fvc::gaussGrad(U, m, g, patches);
    const std::vector<std::vector<tensor>> gradB = fvc::gradUBoundary(U, gradC, m, g, patches);
    std::vector<tensor> sigC(m.nCells());
    for (label c = 0; c < m.nCells(); ++c) sigC[c] = nuEffC[c] * dev2(transpose(gradC[c]));
    std::vector<std::vector<tensor>> sigB(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi) {
        sigB[pi].resize(patches[pi].size);
        for (label i = 0; i < patches[pi].size; ++i) sigB[pi][i] = nuEffB[pi][i] * dev2(transpose(gradB[pi][i]));
    }
    const std::vector<vector> divPart = fvc::div(sigC, sigB, m, g, patches);

    // Reference.
    std::ifstream in(caseDir + "/divdevreff.dat");
    if (!in) { std::printf("FAIL no divdevreff.dat\n"); return 1; }
    label nC; in >> nC;
    std::vector<vector> dof(nC); std::vector<scalar> nuEffof(nC); std::vector<tensor> gof(nC);
    for (auto& v : dof) in >> v.x >> v.y >> v.z;
    for (auto& v : nuEffof) in >> v;
    for (auto& t : gof) in >> t.xx >> t.xy >> t.xz >> t.yx >> t.yy >> t.yz >> t.zx >> t.zy >> t.zz;

    // Sanity: same nuEff + same cell grad as OF (isolates the div/boundary operator).
    scalar nuMx = 0, nuMg = 0, gMx = 0, gMg = 0;
    for (label c = 0; c < nC; ++c) { nuMx = std::fmax(nuMx, std::fabs(nuEffC[c] - nuEffof[c])); nuMg = std::fmax(nuMg, std::fabs(nuEffof[c])); }
    for (label c = 0; c < nC; ++c) {
        const scalar e = std::fabs(gradC[c].xx-gof[c].xx)+std::fabs(gradC[c].xy-gof[c].xy)+std::fabs(gradC[c].yx-gof[c].yx)+std::fabs(gradC[c].yy-gof[c].yy);
        gMx = std::fmax(gMx, e); gMg = std::fmax(gMg, std::fabs(gof[c].xy));
    }
    const scalar rd = relErrV(divPart, dof);
    const scalar tol = 1e-6;
    const bool ok = rd < tol;
    std::printf("test_divdevreff: div(nuEff*dev2(T(gradU))) rel=%.3e (tol %.0e)  [nuEff rel=%.2e gradU rel=%.2e]\n",
                rd, tol, nuMx / nuMg, gMx / gMg);
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
