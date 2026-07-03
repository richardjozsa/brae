// Headline gate: cf turbulent simpleFoam end-to-end vs OpenFOAM. Reads a case's initial (0/)
// fields, runs the SIMPLE + k-epsilon loop to convergence (matching OpenFOAM simpleFoam:
// UEqn divDevReff, pEqn, turbulence->correct()), and compares the converged U/p/k/epsilon/nut
// to OpenFOAM's converged fields. Run: test_simple_turbulent <caseDir> <ofTimeDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "simple_foam.cuh"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace brae;

static scalar relErrS(const std::vector<scalar>& a, const std::vector<scalar>& b) {
    scalar mx = 0, mg = 0;
    for (std::size_t i = 0; i < b.size(); ++i) { mx = std::fmax(mx, std::fabs(a[i] - b[i])); mg = std::fmax(mg, std::fabs(b[i])); }
    return mg > 0 ? mx / mg : mx;
}
static scalar relErrVec(const std::vector<vector>& a, const std::vector<vector>& b) {
    scalar mx = 0, mg = 0;
    for (std::size_t i = 0; i < b.size(); ++i) {
        mx = std::fmax(mx, mag(a[i] - b[i])); mg = std::fmax(mg, mag(b[i]));
    }
    return mg > 0 ? mx / mg : mx;
}
static scalar l2rel(const std::vector<scalar>& a, const std::vector<scalar>& b) {
    scalar n = 0, d = 0;
    for (std::size_t i = 0; i < b.size(); ++i) { n += (a[i]-b[i])*(a[i]-b[i]); d += b[i]*b[i]; }
    return d > 0 ? std::sqrt(n/d) : std::sqrt(n);
}

int main(int argc, char** argv) {
    if (argc < 4) { std::printf("usage: %s <caseDir> <startTime> <refTime> [fixedIters]\n", argv[0]); return 2; }
    const std::string caseDir = argv[1], startT = argv[2], ofT = argv[3];
    const int fixedIters = (argc > 4) ? std::atoi(argv[4]) : 0;   // >0: run exactly N iters; 1 = fixed-point

    PrimitiveMesh m;  m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;     g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);

    const FieldData<vector> Udata = readField<vector>(caseDir + "/" + startT + "/U");
    GeometricField<vector> U   = buildField<vector>(Udata, patches, m.nCells());                       U.evaluateBoundary();
    GeometricField<scalar> p   = buildField<scalar>(readField<scalar>(caseDir + "/" + startT + "/p"), patches, m.nCells());  p.evaluateBoundary();
    GeometricField<scalar> k   = buildField<scalar>(readField<scalar>(caseDir + "/" + startT + "/k"), patches, m.nCells());  k.evaluateBoundary();
    GeometricField<scalar> eps = buildField<scalar>(readField<scalar>(caseDir + "/" + startT + "/epsilon"), patches, m.nCells());  eps.evaluateBoundary();
    GeometricField<scalar> nut = buildField<scalar>(readField<scalar>(caseDir + "/" + startT + "/nut"), patches, m.nCells());

    // phi: read OF's conservative flux if present (fixed-point from a converged state needs it),
    // else createPhi = flux(U) as OpenFOAM does at t=0.
    SurfaceScalarField phi;
    {
        std::ifstream pf(caseDir + "/" + startT + "/phi");
        if (pf.good()) {
            const FieldData<scalar> phiF = readField<scalar>(caseDir + "/" + startT + "/phi");
            phi.internal = phiF.internalField; phi.boundary.resize(patches.size());
            for (std::size_t pi = 0; pi < patches.size(); ++pi) {
                phi.boundary[pi].assign(patches[pi].size, 0.0);
                for (const auto& b : phiF.boundary)
                    if (b.name == patches[pi].name && b.hasValue && (label)b.values.size() == patches[pi].size) phi.boundary[pi] = b.values;
            }
        } else {
            phi = fvc::flux(U, m, g, patches);
        }
    }

    SimpleControls ctl;   // nu 1e-5, relaxU 0.7, relaxP 0.3
    TurbControls   tctl;  // relaxK/relaxEps 0.7
    scalar firstP = 0, lastP = 0;
    // fixedIters>0: run exactly N outer iters (resControl=-1 never triggers) to match OF's iteration
    // count for a fair fixed-iteration comparison; else run to convergence.
    const int    maxOuter   = fixedIters > 0 ? fixedIters : 4000;
    const scalar resControl = fixedIters > 0 ? -1.0       : 1e-6;
    const int iters = runSimpleTurbulent(U, p, phi, k, eps, nut, Udata, m, g, patches,
                                         ctl, tctl, maxOuter, resControl, &firstP, &lastP);
    std::printf("cf ran %d outer iters (p res %.2e -> %.2e)\n", iters, firstP, lastP);

    // OpenFOAM converged reference.
    const std::vector<vector> Uof = readField<vector>(caseDir + "/" + ofT + "/U").internalField;
    const std::vector<scalar> pof = readField<scalar>(caseDir + "/" + ofT + "/p").internalField;
    const std::vector<scalar> kof = readField<scalar>(caseDir + "/" + ofT + "/k").internalField;
    const std::vector<scalar> eof = readField<scalar>(caseDir + "/" + ofT + "/epsilon").internalField;
    const std::vector<scalar> nof = readField<scalar>(caseDir + "/" + ofT + "/nut").internalField;

    // Per-component U L2 (Ux/Uy) for a scalar L2 measure.
    std::vector<scalar> Ux(U.internal.size()), Uxof(Uof.size()), Uy(U.internal.size()), Uyof(Uof.size());
    for (std::size_t i = 0; i < Uof.size(); ++i) { Ux[i]=U.internal[i].x; Uxof[i]=Uof[i].x; Uy[i]=U.internal[i].y; Uyof[i]=Uof[i].y; }

    const scalar rU = relErrVec(U.internal, Uof), rp = relErrS(p.internal, pof);
    const scalar rk = relErrS(k.internal, kof), re = relErrS(eps.internal, eof), rn = relErrS(nut.internal, nof);
    std::printf("converged-field agreement vs OpenFOAM:\n");
    std::printf("  U   max-rel=%.3e  Ux L2-rel=%.3e  Uy L2-rel=%.3e\n", rU, l2rel(Ux, Uxof), l2rel(Uy, Uyof));
    std::printf("  p   max-rel=%.3e  L2-rel=%.3e\n", rp, l2rel(p.internal, pof));
    std::printf("  k   max-rel=%.3e  L2-rel=%.3e\n", rk, l2rel(k.internal, kof));
    std::printf("  eps max-rel=%.3e  L2-rel=%.3e\n", re, l2rel(eps.internal, eof));
    std::printf("  nut max-rel=%.3e  L2-rel=%.3e\n", rn, l2rel(nut.internal, nof));
    // Locate the max U and p errors (step corner / recirculation are singular -> expect there).
    auto worst = [&](auto valFn, const std::vector<scalar>& ref) {
        label w = -1; scalar we = -1;
        for (label c = 0; c < m.nCells(); ++c) { scalar e = std::fabs(valFn(c) - ref[c]); if (e > we) { we = e; w = c; } }
        return w;
    };
    const label wp = worst([&](label c){ return p.internal[c]; }, pof);
    const label wu = worst([&](label c){ return U.internal[c].x; }, Uxof);  // vs OF Ux
    std::printf("  worst p cell %d at (%.4f %.4f): cf=%.4e of=%.4e\n", (int)wp, g.C()[wp].x, g.C()[wp].y, p.internal[wp], pof[wp]);
    std::printf("  worst Ux cell %d at (%.4f %.4f): cf=%.4e of=%.4e   (step corner ~ (0,0))\n", (int)wu, g.C()[wu].x, g.C()[wu].y, U.internal[wu].x, Uxof[wu]);
    int nU5 = 0; for (std::size_t i = 0; i < Uof.size(); ++i) if (mag(U.internal[i]-Uof[i]) > 0.05*10.0) ++nU5;
    std::printf("  cells with |dU|>0.5 m/s: %d / %d\n", nU5, (int)m.nCells());

    const scalar tol = 2e-2;   // few-% bulk agreement is the target (FP accumulation over a full run)
    const bool ok = l2rel(Ux, Uxof) < tol && l2rel(p.internal, pof) < tol && l2rel(k.internal, kof) < tol && l2rel(eps.internal, eof) < tol;
    std::printf("%s\n", ok ? "PASS" : "CHECK");
    return 0;
}
