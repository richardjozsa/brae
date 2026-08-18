// DIAGNOSTIC: run the _cpp driver and the CUDA driver for N iterations from the same start and print
// both residual trajectories, plus the per-iteration field divergence between them.
//
// Why this shape. One iteration of the CUDA driver matches the _cpp driver to 4e-9 (test_simple_step_cuda)
// and the _cpp driver matches OpenFOAM to 2.5e-11 (test_simple_step_cpp), yet over 20 iterations the
// rebuilt path's residual plateaus where the existing solver's falls. So the error is per-iteration and
// accumulating, and the first question is which half owns it:
//
//   _cpp plateaus too      -> the defect is in the SHARED driver logic both paths run
//   _cpp converges, CUDA not -> the defect is CUDA-specific, and the divergence trace says from when
//
// Run: diag_simple_loop <caseDir> <timeDir> <iters>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_dict.cuh"
#include "fvc.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "createFields_cpp.cuh"
#include "simpleControl_cpp.cuh"
#include "simpleFoam_cpp.cuh"
#include "linearViscousStress_cpp.cuh"
#include "simpleFoam.cuh"
#include "simple_foam.cuh"        // the OLD HOST driver
#include "device_simple_foam.cuh" // the OLD GPU driver -- the one that converges
#include "linear_solver_setup.cuh"  // readLinearSolverControls
#include "scheme_parse.cuh"          // parseFvSchemesControls --
                                    // the SAME setup the brae binary runs, so the comparison is equal

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace brae;

static scalar relV(const std::vector<vector>& a, const std::vector<scalar>& x,
                   const std::vector<scalar>& y, const std::vector<scalar>& z, scalar scale)
{
    scalar mx = 0;
    for (std::size_t c = 0; c < a.size(); ++c)
        mx = std::fmax(mx, std::fmax(std::fabs(a[c].x - x[c]),
                       std::fmax(std::fabs(a[c].y - y[c]), std::fabs(a[c].z - z[c]))));
    return scale > 0 ? mx / scale : mx;
}

static scalar relS(const std::vector<scalar>& a, const std::vector<scalar>& b)
{
    scalar mx = 0, mg = 0;
    for (std::size_t i = 0; i < b.size(); ++i)
    {
        mx = std::fmax(mx, std::fabs(a[i] - b[i]));
        mg = std::fmax(mg, std::fabs(b[i]));
    }
    return mg > 0 ? mx / mg : mx;
}

int main(int argc, char** argv)
{
    if (argc < 4) { std::printf("usage: %s <caseDir> <timeDir> <iters>\n", argv[0]); return 2; }
    const std::string caseDir = argv[1], t = argv[2];
    const int iters = std::atoi(argv[3]);
    const scalar nu = 1e-5;

    PrimitiveMesh m;  m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;     g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();

    FoamDict fvSolution = readDict(caseDir + "/system/fvSolution");
    const FoamDict* simpleDict = fvSolution.subDict("SIMPLE");

    scalar relaxU = 0.7, relaxP = 0.3;
    if (const FoamDict* rf = fvSolution.subDict("relaxationFactors"))
    {
        if (const FoamDict* eq = rf->subDict("equations")) relaxU = eq->scalarOr("U", relaxU);
        if (const FoamDict* fl = rf->subDict("fields"))    relaxP = fl->scalarOr("p", relaxP);
    }
    std::printf("relaxU=%.3g relaxP=%.3g  iters=%d  (laminar, nuEff constant)\n", relaxU, relaxP, iters);

    cpu::SimpleControlDict cd = cpu::readSimpleControl(fvSolution);
    cd.consistent = false;

    // ---- host state ---------------------------------------------------------------------------
    cpu::SimpleFields f = cpu::createFields(caseDir + "/" + t, simpleDict, m, g, fvp);
    cpu::SimpleControl ctlC(cd);

    std::vector<scalar> nuEffC(nC, nu);
    std::vector<std::vector<scalar>> nuEffB(fvp.size());
    for (std::size_t pi = 0; pi < fvp.size(); ++pi) nuEffB[pi].assign(fvp[pi].size, nu);

    cpu::StepInput cin;
    cin.nu = nu; cin.nuEff = nuEffC; cin.nuEffBnd = nuEffB;
    cin.relaxU = relaxU; cin.relaxP = relaxP;
    // Solver tolerances from the CASE, not the struct defaults -- the existing solver uses
    // fvSolution/solvers/{p,U}/tolerance (p 1e-6, U 1e-5 on pitzDaily) and the rebuilt runner was
    // hardcoding 1e-10 for both.
    if (const FoamDict* sv = fvSolution.subDict("solvers"))
    {
        if (const FoamDict* ps = sv->subDict("p"))
        { cin.tolP = ps->scalarOr("tolerance", cin.tolP); cin.relTolP = ps->scalarOr("relTol", cin.relTolP); }
        if (const FoamDict* us = sv->subDict("U"))
        { cin.tolU = us->scalarOr("tolerance", cin.tolU); cin.relTolU = us->scalarOr("relTol", cin.relTolU); }
    }
    std::printf("tolU=%.1e relTolU=%.2g   tolP=%.1e relTolP=%.2g\n",
                cin.tolU, cin.relTolU, cin.tolP, cin.relTolP);

    // ---- device state -------------------------------------------------------------------------
    cpu::SimpleFields f0 = cpu::createFields(caseDir + "/" + t, simpleDict, m, g, fvp);
    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    const DeviceVectorBoundary dbU = buildDeviceVectorBoundary(f0.U, fvp, g);
    const DeviceBoundary dbP = buildDeviceBoundary(f0.p, fvp, g);

    gpu::SolverFields gf;
    {
        std::vector<scalar> ux(nC), uy(nC), uz(nC);
        for (label c = 0; c < nC; ++c)
        { ux[c] = f0.U.internal[c].x; uy[c] = f0.U.internal[c].y; uz[c] = f0.U.internal[c].z; }
        gf.Ux.copyFrom(ux); gf.Uy.copyFrom(uy); gf.Uz.copyFrom(uz);
        gf.p.copyFrom(f0.p.internal);
        gf.phiInt.copyFrom(f0.phi.internal);
        std::vector<scalar> pb;
        for (const auto& v : f0.phi.boundary) for (scalar x : v) pb.push_back(x);
        pb.resize(dm.nBndFaces, 0.0);
        gf.phiBnd.copyFrom(pb);
    }

    const SurfaceScalarField nuFace = cpu::effectiveFaceViscosity(nuEffC, nuEffB, m, g, fvp);
    std::vector<scalar> nuBndFlat;
    for (const auto& v : nuEffB) for (scalar x : v) nuBndFlat.push_back(x);
    nuBndFlat.resize(dm.nBndFaces, nu);
    DeviceBuffer<scalar> dNuCell(nuEffC), dNuFace(nuFace.internal), dNuBnd(nuBndFlat);

    std::vector<label> takeU, adjustable;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            takeU.push_back(f0.U.boundary[pi]->assignable() ? 0 : 1);
            adjustable.push_back(f0.U.boundary[pi]->fixesValue() ? 0 : 1);
        }
    takeU.resize(dm.nBndFaces, 0);
    adjustable.resize(dm.nBndFaces, 0);
    DeviceBuffer<label> dTakeU(takeU), dAdjust(adjustable);

    gpu::StepInput gin;
    gin.nuEffCell = &dNuCell; gin.nuEffFace = &dNuFace; gin.nuEffBndFace = &dNuBnd;
    gin.relaxU = relaxU; gin.relaxP = relaxP;
    gin.momentumPredictor = cd.momentumPredictor;
    gin.nNonOrthogonalCorrectors = cd.nNonOrthogonalCorrectors;
    gin.pRefCell = f0.pRefCell; gin.pRefValue = f0.pRefValue;
    gin.takeUAtBoundary = &dTakeU; gin.adjustable = &dAdjust;
    // SAME solver tolerances as the _cpp column. Leaving the struct defaults (1e-10, relTol 0) here made
    // the two columns run different linear solves -- the fourth unequal comparison in this investigation.
    gin.tolU = cin.tolU; gin.relTolU = cin.relTolU;
    gin.tolP = cin.tolP; gin.relTolP = cin.relTolP;
    gpu::SolverWorkspace ws;

    // The OLD host driver, on the same start. Both are host code, so any difference between it and the
    // _cpp driver is pure solver LOGIC -- no linear-solver or precision difference can be blamed.
    cpu::SimpleFields fo = cpu::createFields(caseDir + "/" + t, simpleDict, m, g, fvp);
    const FieldData<vector> Udata = readField<vector>(caseDir + "/" + t + "/U");
    SimpleControls oc;
    oc.nu = nu; oc.relaxU = relaxU; oc.relaxP = relaxP;

    // The OLD GPU driver, same start, same controls. This is the one that converges, so putting all four
    // in one process on identical inputs is what turns "they behave differently" into a per-iteration diff.
    cpu::SimpleFields fd = cpu::createFields(caseDir + "/" + t, simpleDict, m, g, fvp);
    // Build the controls the way gpuSimpleFoam.cu does -- parseFvSchemesControls + readLinearSolverControls
    // -- rather than by hand. Hand-filling is how the previous three comparisons in this investigation ended
    // up unequal (different relaxation, different turbulence, different scheme), and an unequal comparison
    // is worse than none: it produces a difference that looks like a finding.
    DeviceSimpleControls dctl;
    dctl.caseDir = caseDir;
    dctl.nu = nu;
    dctl.turbulent = false;
    parseFvSchemesControls(caseDir, dctl);
    readLinearSolverControls(fvSolution, "epsilon", dctl);
    std::printf("GPU ctl: relaxU=%.3g relaxP=%.3g tolU=%.1e tolP=%.1e bounded=%d nonOrth=%d "
                "linearUpwind=%d consistent=%d gsU=%d\n",
                dctl.relaxU, dctl.relaxP, dctl.tolU, dctl.tolP,
                (int)dctl.bounded, (int)dctl.nonOrth, (int)dctl.linearUpwind,
                (int)dctl.consistent, (int)dctl.gsU);
    DeviceSimpleSolver dev(m, g, fvp, fd.U, fd.p, fd.phi, dctl);

    std::printf("\n it |  OLD host U |  OLD GPU U |    _cpp U |   cuda U  | dU(host vs GPU)\n");
    std::printf("----+-------------+------------+-----------+-----------+----------------\n");
    for (int it = 1; it <= iters; ++it)
    {
        const cpu::Residuals cr = cpu::simpleStep(f, ctlC, cin, m, g, fvp);
        const gpu::Residuals gr = gpu::simpleStep(gf, ws, dm, dbU, dbP, gin);

        const std::vector<scalar> ux = gf.Ux.host(), uy = gf.Uy.host(), uz = gf.Uz.host();
        scalar Umag = 0;
        for (label c = 0; c < nC; ++c)
            Umag = std::fmax(Umag, std::fmax(std::fabs(f.U.internal[c].x),
                             std::fmax(std::fabs(f.U.internal[c].y), std::fabs(f.U.internal[c].z))));
        const StepResidual orr = simpleStep(fo.U, fo.p, fo.phi, Udata, m, g, fvp, oc, nullptr);

        std::vector<scalar> ox(nC), oy(nC), oz(nC);
        for (label c = 0; c < nC; ++c)
        { ox[c] = fo.U.internal[c].x; oy[c] = fo.U.internal[c].y; oz[c] = fo.U.internal[c].z; }

        // With BRAE_DIAG_SYNC=all every iteration starts from IDENTICAL inputs, so these three numbers
        // are "one CUDA iteration vs one _cpp iteration from the same fields" -- measured fresh each time
        // rather than once on a converged state. Whichever output is biased names the stage.
        if (std::getenv("BRAE_DIAG_FIELDS"))
        {
            const std::vector<scalar> gx2 = gf.Ux.host(), gy2 = gf.Uy.host(), gz2 = gf.Uz.host();
            scalar um = 0;
            for (label c = 0; c < nC; ++c)
                um = std::fmax(um, std::fmax(std::fabs(f.U.internal[c].x),
                     std::fmax(std::fabs(f.U.internal[c].y), std::fabs(f.U.internal[c].z))));
            std::printf("      fields: dU=%.3e  dp=%.3e  dphi=%.3e\n",
                        relV(f.U.internal, gx2, gy2, gz2, um),
                        relS(gf.p.host(), f.p.internal),
                        relS(gf.phiInt.host(), f.phi.internal));
        }

        // BISECT: after each iteration, overwrite ONE carried quantity of the CUDA state with the _cpp
        // value. The two agree to 4e-9 for a single iteration, so whichever field restores convergence is
        // the one the CUDA driver is carrying forward wrongly. BRAE_DIAG_SYNC = U | p | phi | none.
        {
            static const char* syncEnv = std::getenv("BRAE_DIAG_SYNC");
            const std::string sync = syncEnv ? syncEnv : "none";
            if (sync == "U" || sync == "all")
            {
                std::vector<scalar> a(nC), b2(nC), c2(nC);
                for (label c = 0; c < nC; ++c)
                { a[c] = f.U.internal[c].x; b2[c] = f.U.internal[c].y; c2[c] = f.U.internal[c].z; }
                gf.Ux.copyFrom(a); gf.Uy.copyFrom(b2); gf.Uz.copyFrom(c2);
            }
            if (sync == "p" || sync == "all") gf.p.copyFrom(f.p.internal);
            if (sync == "phi" || sync == "all")
            {
                gf.phiInt.copyFrom(f.phi.internal);
                std::vector<scalar> pb;
                for (const auto& v : f.phi.boundary) for (scalar x : v) pb.push_back(x);
                pb.resize(dm.nBndFaces, 0.0);
                gf.phiBnd.copyFrom(pb);
            }
        }

        const DeviceSimpleResidual dr = dev.step();
        const std::vector<vector> dU = dev.U();
        std::vector<scalar> dx(nC), dy(nC), dz(nC);
        for (label c = 0; c < nC; ++c) { dx[c] = dU[c].x; dy[c] = dU[c].y; dz[c] = dU[c].z; }

        std::printf("%3d | %11.4e | %10.3e | %9.2e | %9.2e | %9.2e\n", it,
                    orr.Ux, dr.Ux,
                    cr.count("U") ? cr.at("U") : 0.0,
                    gr.count("U") ? gr.at("U") : 0.0,
                    relV(fo.U.internal, dx, dy, dz, Umag));
    }
    return 0;
}
