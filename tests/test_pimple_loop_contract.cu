// The PIMPLE loop contract -- OF pimpleControl.
//
// The outer loop iterates the PRESSURE-VELOCITY coupling. What it does to everything else is decided by
// a handful of controls whose defaults are OpenFOAM's, not a porter's convenience:
//
//     turbOnFinalIterOnly  true    turbulence->correct() only when finalIter()
//     solveFlow            true    skip the momentum/pressure solve entirely when false
//     residualControl      empty   outer-loop convergence; may end the loop early
//     finalIter()                  = converged || corr == nOuterCorrectors
//
// WHY THIS FILE EXISTS. `turbOnFinalIterOnly` was absent from brae entirely, so turbulence was advanced
// on EVERY outer corrector. On a case with nOuterCorrectors 5 that integrates the turbulence model five
// times per physical step -- a different equation, not a tighter solve. A 30-case tutorial sweep did not
// catch it, because most tutorials use nOuterCorrectors 1, where the correct and incorrect cadences are
// identical. That is the whole argument for testing the CONTRACT rather than sampling cases: a cadence
// is invisible in a converged field, and only shows up when you count.
//
// So every leg here counts calls. Leg 1 is the default and the one that was wrong; Leg 2 is its
// discrimination (a case that asks for the old behaviour must still get it); Leg 4 pins the detail that
// makes residualControl subtle -- the relative test divides by iteration TWO's initial residual, because
// iteration one starts from the previous time step and is not comparable.
#include "box_mesh.cuh"
#include "device_simple_foam.cuh"
#include "cyclic_field.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fvc.cuh"
#include <cmath>
#include <cstdio>
#include <string>
#include <tuple>
#include <vector>

using namespace brae;

namespace {
int failures = 0;

void check(bool ok, const char* what)
{
    if (!ok) { std::printf("  FAIL: %s\n", what); ++failures; }
    else       std::printf("  ok:   %s\n", what);
}
} // namespace

int main()
{
    std::printf("== PIMPLE loop contract ==\n");

    const label N = 5;
    PrimitiveMesh m = boxtest::boxMesh(N, N, N);
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();

    std::vector<scalar> p0(nC, 0.0);
    std::vector<vector> U0(nC, vector{1.0, 0.0, 0.0});

    // A turbulent case, so correctTurbulence() has something to advance and the cadence is meaningful.
    auto makeSolver = [&](DeviceSimpleControls ctl)
    {
        GeometricField<vector> U = buildCyclicField<vector>(U0, fvp, {}, /*wallNoSlip*/true); U.evaluateBoundary();
        GeometricField<scalar> p = buildCyclicField<scalar>(p0, fvp, {});                     p.evaluateBoundary();
        SurfaceScalarField phi = fvc::flux(U, m, g, fvp);
        return std::make_tuple(std::move(U), std::move(p), std::move(phi), ctl);
    };

    const int nOuter = 5, nCorr = 2;

    // ---- Leg 1: the DEFAULT -- turbulence once per time step, on the final outer corrector ----------
    {
        DeviceSimpleControls ctl;
        ctl.nu = 1e-3;
        ctl.turbulent = false;                 // laminar: correctTurbulence() is still called, still counted
        auto f = makeSolver(ctl);
        DeviceSimpleSolver s(m, g, fvp, std::get<0>(f), std::get<1>(f), std::get<2>(f), std::get<3>(f));
        check(std::get<3>(f).turbOnFinalIterOnly, "turbOnFinalIterOnly defaults to TRUE, as OpenFOAM's does");
        s.resetLoopCounters();
        s.pimpleStep(1e-3, nOuter, nCorr);
        check(s.outerIterations() == nOuter, "vacuity guard: all five outer correctors ran");
        check(s.turbCorrections() == 1,
              "turbulence is advanced ONCE per time step, not once per outer corrector");
    }

    // ---- Leg 2: discrimination -- a case that asks for every-iteration still gets it ----------------
    // Without this, hard-coding "correct once" would pass Leg 1 and silently break NACA4412, which sets
    // turbOnFinalIterOnly false on purpose.
    {
        DeviceSimpleControls ctl;
        ctl.nu = 1e-3;
        ctl.turbulent = false;
        ctl.turbOnFinalIterOnly = false;
        auto f = makeSolver(ctl);
        DeviceSimpleSolver s(m, g, fvp, std::get<0>(f), std::get<1>(f), std::get<2>(f), std::get<3>(f));
        s.resetLoopCounters();
        s.pimpleStep(1e-3, nOuter, nCorr);
        check(s.turbCorrections() == nOuter,
              "turbOnFinalIterOnly false: turbulence advanced on EVERY outer corrector");
    }

    // ---- Leg 3: nOuterCorrectors 1 -- where the two cadences coincide ------------------------------
    // This is the configuration most tutorials use, and it is exactly why the defect survived a sweep.
    // Pinning it documents that a passing single-corrector case proves nothing about the cadence.
    {
        for (bool flag : {true, false})
        {
            DeviceSimpleControls ctl;
            ctl.nu = 1e-3;
            ctl.turbulent = false;
            ctl.turbOnFinalIterOnly = flag;
            auto f = makeSolver(ctl);
            DeviceSimpleSolver s(m, g, fvp, std::get<0>(f), std::get<1>(f), std::get<2>(f), std::get<3>(f));
            s.resetLoopCounters();
            s.pimpleStep(1e-3, 1, nCorr);
            check(s.turbCorrections() == 1,
                  flag ? "nOuterCorrectors 1: one correction (flag on)"
                       : "nOuterCorrectors 1: one correction (flag off) -- the cadences coincide here");
        }
    }

    // ---- Leg 4: residualControl ends the loop early, and never on iteration 1 ----------------------
    // Tolerances of 1 are unreachable-to-fail: every residual is below them, so the criteria are met at
    // the first iteration they are ALLOWED to be met. OF forbids iteration 1 (corr_ == 1 returns false)
    // and uses iteration 2 to STORE the reference, so the earliest possible exit runs 3 iterations:
    // store at 2, satisfy at 3, and 3 is then flagged final and is the last one executed.
    {
        DeviceSimpleControls ctl;
        ctl.nu = 1e-3;
        ctl.turbulent = false;
        // Tolerances nothing can fail, so the leg tests the ITERATION ACCOUNTING -- OF's corr_ == 1
        // cannot satisfy, corr_ == 2 only stores the reference, corr_ == 3 is the earliest exit -- and
        // not this synthetic box's residual magnitudes, which are not the contract under test.
        ctl.outerResidualControl.push_back({"U", scalar(1e30), scalar(0)});
        ctl.outerResidualControl.push_back({"p", scalar(1e30), scalar(0)});
        auto f = makeSolver(ctl);
        DeviceSimpleSolver s(m, g, fvp, std::get<0>(f), std::get<1>(f), std::get<2>(f), std::get<3>(f));
        s.resetLoopCounters();
        s.pimpleStep(1e-3, nOuter, nCorr);
        check(s.outerIterations() < nOuter, "residualControl cuts the outer loop short");
        check(s.outerIterations() == 3,
              "...at iteration 3: 1 cannot satisfy, 2 stores the reference, 3 satisfies and is final");
        check(s.turbCorrections() == 1,
              "...and the early-final iteration is where turbulence is corrected");
    }

    // ---- Leg 5: an unreachable tolerance leaves the loop at full length ----------------------------
    {
        DeviceSimpleControls ctl;
        ctl.nu = 1e-3;
        ctl.turbulent = false;
        ctl.outerResidualControl.push_back({"p", scalar(1e-300), scalar(0)});
        auto f = makeSolver(ctl);
        DeviceSimpleSolver s(m, g, fvp, std::get<0>(f), std::get<1>(f), std::get<2>(f), std::get<3>(f));
        s.resetLoopCounters();
        s.pimpleStep(1e-3, nOuter, nCorr);
        check(s.outerIterations() == nOuter,
              "negative control: a tolerance nothing meets runs every outer corrector");
    }

    // ---- Leg 6: residualControl on a field brae cannot track is REFUSED ----------------------------
    // It would otherwise test convergence against a residual of zero and exit the loop immediately.
    {
        DeviceSimpleControls ctl;
        ctl.nu = 1e-3;
        ctl.turbulent = false;
        ctl.outerResidualControl.push_back({"epsilon", scalar(1e-3), scalar(0)});
        auto f = makeSolver(ctl);
        DeviceSimpleSolver s(m, g, fvp, std::get<0>(f), std::get<1>(f), std::get<2>(f), std::get<3>(f));
        bool threw = false;
        try { s.pimpleStep(1e-3, nOuter, nCorr); } catch (const std::exception&) { threw = true; }
        check(threw, "refusal: residualControl on an untracked field, rather than instant convergence");
    }

    // ---- Leg 7: the mesh-update cadence -- OF pimpleFoam.C:140 -------------------------------------
    //     if (pimple.firstIter() || moveMeshOuterCorrectors) mesh.controlledUpdate();
    // Default FALSE: the mesh moves once per time step. With the flag set it moves before EVERY outer
    // corrector, so iteration 2's equations are assembled on different geometry than iteration 1 -- the
    // outer loop converges the mesh position as well as the pressure-velocity coupling. brae used to do
    // the motion in the driver, OUTSIDE the loop, which makes that flag unimplementable rather than
    // merely unimplemented.
    {
        for (bool outerMove : {false, true})
        {
            DeviceSimpleControls ctl;
            ctl.nu = 1e-3;
            ctl.turbulent = false;
            ctl.moveMeshOuterCorrectors = outerMove;
            auto f = makeSolver(ctl);
            DeviceSimpleSolver s(m, g, fvp, std::get<0>(f), std::get<1>(f), std::get<2>(f), std::get<3>(f));
            int moves = 0;
            s.pimpleStep(1e-3, nOuter, nCorr, [&](int){ ++moves; });
            check(moves == (outerMove ? nOuter : 1),
                  outerMove ? "moveMeshOuterCorrectors true: the mesh moves before every outer corrector"
                            : "default: the mesh moves ONCE per time step");
        }
    }

    // ---- Leg 8: a static case is never handed a mesh update ----------------------------------------
    // The callback is optional, and passing none must leave the loop exactly as it was -- every
    // non-moving case and every other solver sharing pimpleStep goes down this path.
    {
        DeviceSimpleControls ctl;
        ctl.nu = 1e-3;
        ctl.turbulent = false;
        auto f = makeSolver(ctl);
        DeviceSimpleSolver s(m, g, fvp, std::get<0>(f), std::get<1>(f), std::get<2>(f), std::get<3>(f));
        s.resetLoopCounters();
        s.pimpleStep(1e-3, nOuter, nCorr);          // no callback at all
        check(s.outerIterations() == nOuter, "negative control: no motion callback, loop unchanged");
    }

    std::printf(failures ? "== FAILED (%d) ==\n" : "== PASSED ==\n", failures);
    return failures ? 1 : 0;
}
