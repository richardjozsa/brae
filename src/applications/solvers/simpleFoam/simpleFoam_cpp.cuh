#pragma once
// _cpp REFERENCE DRIVER -- one simpleFoam SIMPLE iteration, composed of the ported components.
//
// provenance:
//   openfoam: applications/solvers/incompressible/simpleFoam/simpleFoam.C:79-97 (the time loop body)
//   brae:     src/applications/solvers/simpleFoam/simpleFoam_cpp.cu
//   tests:    tests/test_simple_step_cpp.cu   (end-to-end vs OpenFOAM's dumpSimpleStep, step.dat)
//
// THE DRIVER OWNS NO NUMERICS. That is the whole point of the rebuild: everything below is a call into a
// shared component that has its own OpenFOAM provenance and its own test.
//
//     assembleUEqn          UEqn_cpp            (fvm::div + divDevReff + relax)
//     addPressureGradient   UEqn_cpp            (-fvc::grad(p))
//     solveVector           OpenFOAM/matrices   (the momentum solve)
//     pressurePredictor     pEqn_cpp            (rAU, HbyA, phiHbyA, adjustPhi)
//     assemblePEqn          pEqn_cpp            (laplacian == div(phiHbyA), setReference)
//     gamg                  OpenFOAM/matrices   (the pressure solve)
//     correctFlux           pEqn_cpp            (phi = phiHbyA - pEqn.flux())
//     relaxField            pEqn_cpp            (p.relax())
//     correctVelocity       pEqn_cpp            (U = HbyA - rAU*grad(p))
//     correctNonOrthogonal  simpleControl_cpp   (the corrector loop)
//
// Compare with what it replaces: src/applications/solvers/simpleFoam/device_simple_foam.cu, 3578 lines,
// included by pimpleFoam, rhoSimpleFoam and five headers in solvers/common.
//
// ORDER IS OPENFOAM'S. Two orderings matter and neither is obvious from the equations:
//   * the momentum predictor solves a COPY of UEqn with -grad(p) added, because pEqn.H then needs the
//     original UEqn for A() and H(). Adding grad(p) to UEqn itself changes rAU and HbyA.
//   * p is relaxed AFTER the flux correction and BEFORE the velocity correction, so phi is built from the
//     unrelaxed pressure and U from the relaxed one.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "createFields_cpp.cuh"
#include "simpleControl_cpp.cuh"
#include <map>
#include <string>
#include <vector>

namespace brae {
namespace cpu {

struct StepInput
{
    std::vector<scalar>              nuEff;      // cells
    std::vector<std::vector<scalar>> nuEffBnd;   // [patch][face]
    scalar relaxU = 0.7;                         // relaxationFactors/equations U
    scalar relaxP = 0.3;                         // relaxationFactors/fields p
    scalar tolU = 1e-10, relTolU = 0.0;
    scalar tolP = 1e-10, relTolP = 0.0;
    int    maxIter = 2000;
    bool   hasMRF = false;                       // refused downstream
    bool   hasFvOptions = false;                 // refused downstream
};

// field name -> initial residual of its first solve this iteration (what simpleControl checks).
using Residuals = std::map<std::string, scalar>;

// One SIMPLE iteration, in place on f.p / f.U / f.phi.
Residuals simpleStep(
    SimpleFields&               f,
    SimpleControl&              ctl,
    const StepInput&            in,
    const PrimitiveMesh&        m,
    const FvGeometry&           g,
    const std::vector<FvPatch>& patches);

} // namespace cpu
} // namespace brae
