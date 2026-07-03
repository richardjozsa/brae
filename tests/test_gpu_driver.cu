// GPU offload (#4): the device-resident driver (DeviceSimpleSolver) end-to-end. Read a real case from disk
// (mesh + 0/U + 0/p with their actual BCs), run the device-resident SIMPLE loop, and validate against the
// from-source CPU oracle (simpleStep, simple_foam.cuh) stepped in lockstep. Run LAMINAR: nuEff = nu
// everywhere, so the device loop is fully faithful (no boundary-nut convention gap) and must reproduce the
// CPU solver iteration for iteration, all the way to the steady state. Prints both residual traces.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvc.cuh"
#include "simple_foam.cuh"            // CPU oracle: simpleStep
#include "device_simple_foam.cuh"    // device-resident solver
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

int main(int argc, char** argv) {
    const std::string caseDir = argc > 1 ? argv[1] : "validation/pitzDaily";
    PrimitiveMesh m; m.read(caseDir + "/constant/polyMesh");
    FvGeometry g; g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();
    const int N = 150;
    const scalar nu = 1e-2, relaxU = 0.7, relaxP = 0.3, tol = 1e-5;   // tol matches a real fvSolution (1e-5)

    const FieldData<vector> Ufd = readField<vector>(caseDir + "/0/U");
    const FieldData<scalar> pfd = readField<scalar>(caseDir + "/0/p");

    // ---- device-resident loop (laminar), capture residual trace ----
    GeometricField<vector> Ud = buildField<vector>(Ufd, fvp, nC); Ud.evaluateBoundary();
    GeometricField<scalar> pd = buildField<scalar>(pfd, fvp, nC); pd.evaluateBoundary();
    SurfaceScalarField phid = fvc::flux(Ud, m, g, fvp);
    DeviceSimpleControls dc; dc.nu = nu; dc.relaxU = relaxU; dc.relaxP = relaxP; dc.tolU = tol; dc.tolP = tol; dc.turbulent = false;
    DeviceSimpleSolver solver(m, g, fvp, Ud, pd, phid, dc);

    // ---- CPU oracle (simpleStep) in lockstep ----
    GeometricField<vector> Uc = buildField<vector>(Ufd, fvp, nC); Uc.evaluateBoundary();
    GeometricField<scalar> pc = buildField<scalar>(pfd, fvp, nC); pc.evaluateBoundary();
    SurfaceScalarField phic = fvc::flux(Uc, m, g, fvp);
    SimpleControls cc; cc.nu = nu; cc.relaxU = relaxU; cc.relaxP = relaxP; cc.tolU = tol; cc.tolP = tol; cc.maxIter = 5000;

    std::printf("GPU driver vs CPU oracle, %d laminar steps, nCells=%d (residual trace):\n", N, nC);
    for (int it = 1; it <= N; ++it) {
        const DeviceSimpleResidual rd = solver.step();
        const StepResidual rc = simpleStep(Uc, pc, phic, Ufd, m, g, fvp, cc);
        if (it == 1 || it % 25 == 0) {
            const scalar fU = [&]{ const auto a=solver.U(); scalar n=0,d=0; for(label c=0;c<nC;++c){n=std::fmax(n,mag(a[c]-Uc.internal[c]));d=std::fmax(d,mag(Uc.internal[c]));} return d>0?n/d:n; }();
            std::printf("  iter %3d:  GPU Ux %.3e p %.3e | CPU Ux %.3e p %.3e | fieldU diff %.3e\n", it, rd.Ux, rd.p, rc.Ux, rc.p, fU);
        }
    }

    const std::vector<vector> Udev = solver.U(); const std::vector<scalar> pdev = solver.p();
    auto relV = [&](const std::vector<vector>& a, const std::vector<vector>& b){ scalar n=0,d=0;
        for (label c=0;c<nC;++c){ n=std::fmax(n,mag(a[c]-b[c])); d=std::fmax(d,mag(b[c])); } return d>0?n/d:n; };
    auto relS = [&](const std::vector<scalar>& a, const std::vector<scalar>& b){ scalar n=0,d=0;
        for (label c=0;c<nC;++c){ n=std::fmax(n,std::fabs(a[c]-b[c])); d=std::fmax(d,std::fabs(b[c])); } return d>0?n/d:n; };
    const scalar eU = relV(Udev, Uc.internal), eP = relS(pdev, pc.internal);
    std::printf("  final field diff (device vs CPU):  U %.3e  p %.3e\n", eU, eP);
    const bool pass = eU < 1e-3 && eP < 1e-3;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
