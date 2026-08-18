#!/usr/bin/env python3
"""Generate the OpenFOAM port manifest for one solver.

WHY THIS EXISTS. Porting OpenFOAM has repeatedly failed the same way here: implement what a tutorial
exercises, run it, discover the next missing piece from a wrong answer. Every one of those discoveries
(`relaxationFactors ".*Final"`, MRF silently ignored, the `calculated` nut, LUST's implicit weights) was a
runtime-selected string or a dictionary key that OpenFOAM reads and the port did not. All of them are
statically visible. This emits them as a checklist BEFORE any code is written.

The manifest has two halves and they are kept strictly apart:

  DERIVED   -- produced by querying ofscan's index of the OpenFOAM tree. Never edited by hand. If OpenFOAM
               changes, this half changes on the next run and the diff is the drift.
  CURATED   -- the classification (what kind of component this is) and the brae status (what we intend to
               do about it). This cannot be derived: it is a judgement about OUR code, not about OpenFOAM.

Keeping them apart is the point. A hand-written "OpenFOAM does X" claim rots silently; a derived one cannot.

Usage:
    python3 tools/of_manifest.py simpleFoam > manifest/simpleFoam.yaml
    python3 tools/of_manifest.py simpleFoam --check     # non-zero if DERIVED half has drifted
"""
import argparse
import os
import re
import subprocess
import sys

OFSCAN = os.environ.get("OFSCAN_ROOT", os.path.join(os.path.dirname(__file__),
                                                    "..", "..", "ofscan"))
OF = os.environ.get("FOAM_ROOT", "/usr/lib/openfoam/openfoam2412")

# --------------------------------------------------------------------------------------------------
# CURATED half: classification + status. One entry per component of the solver's closure.
#
# classification (what kind of thing it is):
#   HOST_ONLY            runs once on the host; no GPU form is meaningful
#   CONFIGURATION        dictionary reading / control state
#   DISPATCH             runtime selection of an implementation
#   SHARED_NUMERICAL     finite-volume operator or matrix operation, solver-independent
#   MODEL                turbulence / transport / thermo model
#   BOUNDARY_CONDITION   an fvPatchField implementation
#   LINEAR_SOLVER        lduMatrix solver / preconditioner / smoother
#   GPU_REQUIRED         must be device-resident for the solver to be GPU-native
#   DYNAMIC_OR_UNRESOLVED  ofscan cannot resolve it statically
#
# brae_status (what we do about it):
#   REUSE_EXISTING       lift as-is; it is already solver-independent and validated
#   REVALIDATE_EXISTING  lift, but re-prove against OpenFOAM before trusting it
#   REIMPLEMENT          write again on the new architecture
#   NOT_REQUIRED_ON_GPU  host-side only
#   UNSUPPORTED          explicitly out of scope; the solver must REFUSE, not ignore
# --------------------------------------------------------------------------------------------------
COMPONENTS = {
    "simpleFoam": [
        # ---- orchestration -------------------------------------------------------------------
        dict(name="simpleFoam_main", of_symbol="main",
             of_file="applications/solvers/incompressible/simpleFoam/simpleFoam.C",
             classification="HOST_ONLY", status="REIMPLEMENT",
             brae_reference="src/applications/solvers/simpleFoam/simpleFoam_cpp.cu",
             brae_cuda="src/applications/solvers/simpleFoam/simpleFoam.cu",
             brae_target="src/applications/solvers/simpleFoam/simpleFoam.cu",
             validation="tests/test_simple_step_cpp.cu -- END-TO-END, one SIMPLE iteration composed of the "
                        "_cpp components vs OpenFOAM dumpSimpleStep (validation/matrixDumpSimple/step.dat): "
                        "p 2.5e-11, U 1.6e-12, phi 1.2e-11, every boundary patch <= 3.5e-13. Gate set at "
                        "1e-9, not the 1e-5 the older step test uses. "
                        "CUDA DRIVER: tests/test_simple_step_cuda.cu runs the device driver and the _cpp "
                        "driver for one iteration from the same fields, laminar and turbulent -- U/p/phi "
                        "agree to 4.1e-09 or better, and the pre-solve p residual to 2.6e-12 ABSOLUTE. "
                        "The gate is 1e-7 because the two paths run DIFFERENT Krylov methods (host "
                        "GAMG/BiCGStab vs device AMG-PCG/BiCGStab); the 1e-16 arithmetic gates are "
                        "test_ueqn_cuda and test_peqn_cuda.",
             note="The _cpp driver owns NO numerics -- 9 calls into shared components, each with its own "
                  "OpenFOAM provenance and test. Replaces a 3578-line file that pimpleFoam, rhoSimpleFoam "
                  "and five common/ headers all included. NOTE ON THE FIXTURE: matrixDumpSimple's "
                  "fvSolution sets `consistent yes`, but step.dat was dumped with plain SIMPLE; the test "
                  "now asserts explicitly that the flag is OFF for that comparison, since the refusal that "
                  "used to enforce it is gone. A SIMPLEC oracle is still needed to test SIMPLEC at "
                  "step granularity; end to end it is covered by ctest stock_pitzdaily_vs_openfoam."),
        dict(name="createFields", of_symbol="createFields.H",
             of_file="applications/solvers/incompressible/simpleFoam/createFields.H",
             classification="HOST_ONLY", status="REIMPLEMENT",
             brae_reference="src/applications/solvers/simpleFoam/createFields_cpp.cu",
             brae_target="src/applications/solvers/simpleFoam/createFields.cu",
             validation="tests/test_simple_step_cpp.cu -- phi READ from disk (not recomputed), "
                        "needReference() false on a case whose outlet fixes p, so no reference cell is "
                        "set and adjustPhi does not run.",
             note="p and U are MUST_READ; phi comes from createPhi.H (READ_IF_PRESENT, else "
                  "fvc::flux(U)) -- the read-if-present half was a past defect. setRefCell REFUSES when a "
                  "reference is needed and neither pRefCell nor pRefPoint is given, rather than quietly "
                  "pinning cell 0; pRefPoint is refused outright (needs mesh.findCell)."),
        dict(name="UEqn", of_symbol="UEqn.H",
             of_file="applications/solvers/incompressible/simpleFoam/UEqn.H",
             classification="GPU_REQUIRED", status="REIMPLEMENT",
             brae_reference="src/applications/solvers/simpleFoam/UEqn_cpp.cu",
             brae_cuda="src/applications/solvers/simpleFoam/UEqn.cu",
             brae_target="src/applications/solvers/simpleFoam/UEqn.cu",
             validation="tests/test_ueqn_cpp.cu -- validated by DECOMPOSITION against "
                        "validation/matrixDumpAsym/momentum.dat: the div/laplacian core matches OpenFOAM to "
                        "1e-11 on diag/upper/lower/source; adding divDevReff provably changes only `source`, "
                        "and by exactly the explicit dev2 term; relaxation raises |diag| and leaves the "
                        "off-diagonals alone; MRF and fvOptions both throw. "
                        "CUDA: tests/test_ueqn_cuda.cu compares the device assembly to the reference "
                        "field by field on a laminar AND a turbulent case -- relaxed diag 2.0e-16/2.7e-16, "
                        "upper/lower 0, sources <=1.4e-15, all six boundary-coefficient arrays (25010 "
                        "faces) EXACTLY 0, addPressureGradient <=1.3e-15. MRF/fvOptions refused on the "
                        "device path too.",
             note="24 lines in OpenFOAM. The _cpp reference REFUSES MRF and fvOptions rather than ignoring "
                  "them -- brae has shipped a solver that silently ignored MRFProperties and produced a "
                  "converged wrong answer. The CUDA side (UEqn.cu) mirrors it stage for stage, refusals "
                  "included, and carries `bounded` and the `corrected` laplacian."),
        dict(name="pEqn", of_symbol="pEqn.H",
             of_file="applications/solvers/incompressible/simpleFoam/pEqn.H",
             classification="GPU_REQUIRED", status="REIMPLEMENT",
             brae_reference="src/applications/solvers/simpleFoam/pEqn_cpp.cu",
             brae_cuda="src/applications/solvers/simpleFoam/pEqn.cu",
             brae_target="src/applications/solvers/simpleFoam/pEqn.cu",
             validation="tests/test_peqn_cpp.cu -- stage by stage on validation/matrixDumpAsym: rAU and "
                        "HbyA vs OpenFOAM A()/H() (ops.dat) to 1e-11; the pressure Laplacian incl. all "
                        "patch coefficients vs peqn.dat to 1e-11; source == laplacian source + "
                        "div(phiHbyA)*V; setReference asserted to be exactly fvMatrix.C:1011-1023 (it "
                        "DOUBLES the diagonal, it does not overwrite it); correctFlux analytic at p=0; "
                        "relaxField analytic; MRF/fvOptions/consistent all refused. "
                        "CUDA: tests/test_peqn_cuda.cu compares the device stages to the reference on a "
                        "laminar AND a turbulent case -- rAU 1.7e-16, HbyA <=5.2e-16, phiHbyA int/bnd "
                        "2.9e-16/7.4e-17, laplacian upper/lower 0 diag 1.9e-16, source 1.3e-14, patch "
                        "coeffs <=2.8e-16, setReference 1.9e-16, flux correction 2.9e-16, p.relax "
                        "1.8e-16, corrector <=5.2e-16. All three refusals fire on the device path.",
             note="50 lines. Every intermediate is RETURNED rather than kept local, so the first divergent "
                  "stage can be isolated -- a past investigation ended at `phi = phiHbyA - pEqn.flux()`, "
                  "which is stage 7 here. SIMPLEC (`consistent`) is refused: it needs UEqn.H1() and "
                  "fvc::snGrad, neither ported. The CUDA side (pEqn.cu) mirrors it stage for stage; "
                  "assemblePEqn takes p because the `corrected` laplacian needs grad(p), and REFUSES a "
                  "null p rather than treating it as `no correction`."),

        # ---- control -------------------------------------------------------------------------
        dict(name="simpleControl", of_symbol="Foam::simpleControl",
             of_file="src/finiteVolume/cfdTools/general/solutionControl/simpleControl/simpleControl.C",
             classification="CONFIGURATION", status="REIMPLEMENT",
             brae_reference="src/finiteVolume/cfdTools/general/solutionControl/simpleControl/"
                            "simpleControl_cpp.cu",
             brae_target="src/finiteVolume/cfdTools/general/solutionControl/simpleControl/",
             schema_for="solutionControl",
             validation="tests/test_simple_step_cpp.cu -- parses the real matrixDumpSimple SIMPLE block: "
                        "`consistent yes`, nNonOrthogonalCorrectors 0, three residualControl entries "
                        "including the regex key \"(k|epsilon|omega|f|v2)\" which matches 'epsilon' and "
                        "not 'T'; correctNonOrthogonal runs nNonOrth+1 times and resets.",
             note="loop() = setFirstIterFlag; read(); if(initialised && criteriaSatisfied) writeAndEnd(); "
                  "else storePrevIterFields(); return runTime.loop(). Reads its keys from "
                  "solutionDict().subOrEmptyDict('SIMPLE')."),
        dict(name="relaxationFactors", of_symbol="Foam::solution::relaxField/relaxEquation",
             of_file="src/OpenFOAM/matrices/solution/solution.C",
             classification="CONFIGURATION", status="REVALIDATE_EXISTING",
             brae_existing="src/applications/solvers/common/solver_controls.cuh",
             brae_target="src/OpenFOAM/matrices/solution/",
             note="Legacy flat form promotes only p*/rho* to FIELD relaxation; select() appends 'Final' on "
                  "the final iteration. Both were past defects -- revalidate, do not assume."),

        # ---- shared numerics -----------------------------------------------------------------
        dict(name="fvm_div", of_symbol="Foam::fvm::div",
             of_file="src/finiteVolume/finiteVolume/divSchemes/divScheme/divScheme.C",
             classification="SHARED_NUMERICAL", status="REVALIDATE_EXISTING",
             brae_existing="src/cuda/device_fvm.cu",
             brae_target="src/finiteVolume/finiteVolume/divSchemes/",
             note="Implicit weights come from the SCHEME. limitedSurfaceInterpolationScheme::weights = "
                  "limiter*CDweights + (1-limiter)*pos0(faceFlux); LUST = 0.75*linear + 0.25*upwind. "
                  "Getting the implicit blend wrong is silent -- it was brae's LUST defect."),
        dict(name="fvm_laplacian", of_symbol="Foam::fvm::laplacian",
             of_file="src/finiteVolume/finiteVolume/laplacianSchemes/laplacianScheme/laplacianSchemes.C",
             classification="SHARED_NUMERICAL", status="REVALIDATE_EXISTING",
             brae_existing="src/cuda/device_fvm.cu",
             brae_target="src/finiteVolume/finiteVolume/laplacianSchemes/"),
        dict(name="fvc_grad", of_symbol="Foam::fvc::grad",
             of_file="src/finiteVolume/finiteVolume/gradSchemes/gaussGrad/gaussGrad.C",
             classification="SHARED_NUMERICAL", status="REVALIDATE_EXISTING",
             brae_existing="src/cuda/device_fvc.cu",
             brae_reference="src/finiteVolume/finiteVolume/fvc.cu",
             brae_target="src/finiteVolume/finiteVolume/gradSchemes/gaussGrad/",
             note="A host std::vector reference ALREADY exists (fvc.cu:6 gaussGrad) -- this is the _cpp "
                  "oracle for grad, already written. Move it, do not rewrite it."),
        dict(name="fvc_div", of_symbol="Foam::fvc::div",
             of_file="src/finiteVolume/finiteVolume/fvc/fvcDiv.C",
             classification="SHARED_NUMERICAL", status="REVALIDATE_EXISTING",
             brae_existing="src/cuda/device_fvc.cu",
             brae_reference="src/finiteVolume/finiteVolume/fvc.cu",
             brae_target="src/finiteVolume/finiteVolume/fvc/"),
        dict(name="fvc_flux", of_symbol="Foam::fvc::flux",
             of_file="src/finiteVolume/finiteVolume/fvc/fvcFlux.C",
             classification="SHARED_NUMERICAL", status="REVALIDATE_EXISTING",
             brae_existing="src/cuda/device_fvc.cu",
             brae_target="src/finiteVolume/finiteVolume/fvc/"),
        dict(name="fvc_snGrad", of_symbol="Foam::fvc::snGrad",
             of_file="src/finiteVolume/finiteVolume/snGradSchemes/snGradScheme/snGradScheme.C",
             classification="SHARED_NUMERICAL", status="REIMPLEMENT",
             brae_target="src/finiteVolume/finiteVolume/snGradSchemes/",
             note="Only reached on the `consistent` (SIMPLEC) branch of pEqn.H."),
        dict(name="fvMatrix_A", of_symbol="Foam::fvMatrix<Type>::A",
             of_file="src/finiteVolume/fvMatrices/fvMatrix/fvMatrix.C", of_line=1314,
             classification="SHARED_NUMERICAL", status="REVALIDATE_EXISTING",
             brae_existing="src/cuda/device_simple.cu",
             brae_reference="src/finiteVolume/fvMatrices/fv_matrix_ops.cuh",
             brae_target="src/finiteVolume/fvMatrices/fvMatrix/",
             note="Host reference exists at fv_matrix_ops.cuh:138 (matrixA)."),
        dict(name="fvMatrix_H", of_symbol="Foam::fvMatrix<Type>::H",
             of_file="src/finiteVolume/fvMatrices/fvMatrix/fvMatrix.C", of_line=1333,
             classification="SHARED_NUMERICAL", status="REVALIDATE_EXISTING",
             brae_existing="src/cuda/device_simple.cu",
             brae_reference="src/finiteVolume/fvMatrices/fv_matrix_ops.cuh",
             brae_target="src/finiteVolume/fvMatrices/fvMatrix/",
             note="Host reference exists at fv_matrix_ops.cuh:155 (matrixH)."),
        dict(name="fvMatrix_H1", of_symbol="Foam::fvMatrix<Type>::H1",
             of_file="src/finiteVolume/fvMatrices/fvMatrix/fvMatrix.C", of_line=1385,
             classification="SHARED_NUMERICAL", status="REIMPLEMENT",
             brae_target="src/finiteVolume/fvMatrices/fvMatrix/",
             note="Only used by the `consistent` (SIMPLEC) branch."),
        dict(name="fvMatrix_relax", of_symbol="Foam::fvMatrix<Type>::relax",
             of_file="src/finiteVolume/fvMatrices/fvMatrix/fvMatrix.C", of_line=1102,
             classification="SHARED_NUMERICAL", status="REVALIDATE_EXISTING",
             brae_existing="src/finiteVolume/fvMatrices/fv_matrix_ops.cuh",
             brae_target="src/finiteVolume/fvMatrices/fvMatrix/",
             note="ASYMMETRIC: adds cmptMax(cmptMag(iCoeffs)) to the diagonal and subtracts "
                  "cmptMin(iCoeffs) from the source. Guarded by if(relaxEquation(name))."),
        dict(name="fvMatrix_setReference", of_symbol="Foam::fvMatrix<Type>::setReference",
             of_file="src/finiteVolume/fvMatrices/fvMatrix/fvMatrix.C", of_line=1011,
             classification="SHARED_NUMERICAL", status="REVALIDATE_EXISTING",
             brae_existing="src/applications/solvers/common/solver_controls.cuh",
             brae_target="src/finiteVolume/fvMatrices/fvMatrix/",
             note="Currently lives inside applications/solvers/common -- solver-owned infrastructure, "
                  "exactly the layering defect this rebuild removes."),
        dict(name="fvMatrix_flux", of_symbol="Foam::fvMatrix<Type>::flux",
             of_file="src/finiteVolume/fvMatrices/fvMatrix/fvMatrix.C",
             classification="SHARED_NUMERICAL", status="REVALIDATE_EXISTING",
             brae_existing="src/applications/solvers/simpleFoam/device_simple_foam.cu",
             brae_target="src/finiteVolume/fvMatrices/fvMatrix/",
             note="phi = phiHbyA - pEqn.flux() is the continuity-preserving step; a past investigation "
                  "traced a growing divergence to it."),

        # ---- cfdTools free functions ---------------------------------------------------------
        dict(name="constrainHbyA", of_symbol="Foam::constrainHbyA",
             of_file="src/finiteVolume/cfdTools/general/constrainHbyA/constrainHbyA.C",
             classification="SHARED_NUMERICAL", status="REVALIDATE_EXISTING",
             brae_existing="src/applications/solvers/simpleFoam/simple_foam.cuh",
             brae_target="src/finiteVolume/cfdTools/general/constrainHbyA/",
             note="Solver-owned today; must become shared."),
        dict(name="adjustPhi", of_symbol="Foam::adjustPhi",
             of_file="src/finiteVolume/cfdTools/general/adjustPhi/adjustPhi.C",
             classification="SHARED_NUMERICAL", status="REVALIDATE_EXISTING",
             brae_existing="src/applications/solvers/common/solver_controls.cuh",
             brae_target="src/finiteVolume/cfdTools/general/adjustPhi/"),
        dict(name="constrainPressure", of_symbol="Foam::constrainPressure",
             of_file="src/finiteVolume/cfdTools/general/constrainPressure/constrainPressure.C",
             classification="SHARED_NUMERICAL", status="REVALIDATE_EXISTING",
             brae_existing="src/finiteVolume/fields/fv_patch_field.cuh",
             brae_target="src/finiteVolume/cfdTools/general/constrainPressure/"),
        dict(name="setRefCell", of_symbol="Foam::setRefCell",
             of_file="src/finiteVolume/cfdTools/general/findRefCell/findRefCell.C",
             classification="CONFIGURATION", status="REVALIDATE_EXISTING",
             brae_existing="src/applications/solvers/common/solver_controls.cuh",
             brae_target="src/finiteVolume/cfdTools/general/findRefCell/"),
        dict(name="continuityErrs", of_symbol="continuityErrs.H",
             of_file="src/finiteVolume/cfdTools/incompressible/continuityErrs.H",
             classification="HOST_ONLY", status="REVALIDATE_EXISTING",
             brae_existing="src/applications/solvers/simpleFoam/gpuSimpleFoam.cu",
             brae_target="src/finiteVolume/cfdTools/incompressible/"),

        # ---- models --------------------------------------------------------------------------
        dict(name="turbulenceModel_New", of_symbol="Foam::incompressible::turbulenceModel::New",
             of_file="applications/solvers/incompressible/simpleFoam/createFields.H", of_line=42,
             classification="DISPATCH", status="REIMPLEMENT",
             selection_base="incompressible::turbulenceModel",
             brae_target="src/TurbulenceModels/",
             note="26 implementations in v2412. brae supports a strict subset -- the manifest must say "
                  "which, and the solver must REFUSE the rest rather than substitute."),
        dict(name="divDevReff", of_symbol="Foam::...::divDevReff",
             of_file="src/TurbulenceModels/turbulenceModels/linearViscousStress/"
                     "linearViscousStress.C",
             classification="SHARED_NUMERICAL", status="REVALIDATE_EXISTING",
             brae_existing="src/cuda/device_divdevreff.cu",
             brae_reference="src/TurbulenceModels/turbulenceModels/linearViscousStress/"
                            "linearViscousStress_cpp.cu",
             brae_target="src/TurbulenceModels/turbulenceModels/linearViscousStress/",
             validation="tests/test_divdevreff_cpp.cu -- 7.4e-16 vs OpenFOAM dumpDivDevReff over 12225 "
                        "cells (validation/kEpsCorrect), with a wrong-sign check and a wall-nuEff "
                        "negative control that both fire",
             note="FIRST component extracted onto the mirrored architecture; the template for the rest. "
                  "The _cpp reference is host-only and reuses brae existing transpose/dev2/operator* "
                  "rather than restating them."),
        dict(name="singlePhaseTransportModel", of_symbol="Foam::singlePhaseTransportModel",
             of_file="src/transportModels/incompressible/singlePhaseTransportModel/"
                     "singlePhaseTransportModel.C",
             classification="MODEL", status="REVALIDATE_EXISTING",
             brae_target="src/transportModels/incompressible/",
             note="laminarTransport.correct() is called every SIMPLE iteration; a Newtonian model makes it "
                  "a no-op, a non-Newtonian one does not."),
        dict(name="SpalartAllmaras", of_symbol="Foam::RASModels::SpalartAllmaras",
             of_file="src/TurbulenceModels/turbulenceModels/RAS/SpalartAllmaras/"
                     "SpalartAllmaras.C",
             classification="MODEL", status="REVALIDATE_EXISTING",
             brae_existing="src/cuda/device_spalart.cu",
             schema_for="SpalartAllmarasBase",
             brae_target="src/TurbulenceModels/turbulenceModels/RAS/SpalartAllmaras/",
             note="All 12 coefficients are read via dimensioned<scalar>::getOrAddToDict -- see the derived "
                  "schema below. nut = nuTilda*fv1 on `calculated` patches was a past defect."),
        dict(name="kEpsilon", of_symbol="Foam::RASModels::kEpsilon",
             of_file="src/TurbulenceModels/turbulenceModels/RAS/kEpsilon/kEpsilon.C",
             classification="MODEL", status="REVALIDATE_EXISTING",
             brae_existing="src/cuda/device_kepsilon.cu",
             brae_reference="src/TurbulenceModels/turbulenceModels/RAS/kEpsilon/k_epsilon.cu"
                            "h (host correct(), moved into the mirrored path)",
             brae_target="src/TurbulenceModels/turbulenceModels/RAS/kEpsilon/",
             validation="Coefficients checked against the DERIVED schema below -- brae defaults match "
                        "OpenFOAM exactly (Cmu .09, C1 1.44, C2 1.92, C3 0, sigmak 1.0, sigmaEps 1.3). "
                        "tests/test_simple_turbulent_cpp.cu wires it into the _cpp loop on "
                        "validation/pitzDailyTurb from OpenFOAM's converged 1576: the rebuilt loop is "
                        "BIT-IDENTICAL (rel 0) to the pre-existing OpenFOAM-validated host path on U, p, "
                        "phi, k, epsilon and nut, with a laminar control (drift 1.3e-01) proving nut "
                        "really reaches the momentum equation.",
             note="REUSED, not rewritten: a complete host kEpsilon::correct already existed and is "
                  "validated against OpenFOAM (correct.dat). The new work is the COUPLING -- nuEff = "
                  "nu + nut with boundary values from nut's own boundary field, and the LAGGED ordering "
                  "(turbulence->correct() at the END of the iteration, simpleFoam.C:93-94)."),
        dict(name="kOmegaSST", of_symbol="Foam::RASModels::kOmegaSST",
             of_file="src/TurbulenceModels/turbulenceModels/RAS/kOmegaSST/kOmegaSST.C",
             classification="MODEL", status="REVALIDATE_EXISTING",
             brae_existing="src/cuda/device_komega_sst.cu",
             brae_target="src/TurbulenceModels/turbulenceModels/RAS/kOmegaSST/"),

        # ---- MRF / fvOptions -----------------------------------------------------------------
        dict(name="MRFZoneList", of_symbol="Foam::MRFZoneList",
             of_file="src/finiteVolume/cfdTools/general/MRF/MRFZoneList.C",
             classification="SHARED_NUMERICAL", status="REVALIDATE_EXISTING",
             brae_existing="src/finiteVolume/cfdTools/MRF/device_mrf.cu",
             schema_for="MRFZone",
             brae_target="src/finiteVolume/cfdTools/general/MRF/",
             note="Reached four times in the closure: correctBoundaryVelocity, DDt, makeRelative, update. "
                  "Silently ignoring it produced a converged wrong answer on the compressible path."),
        dict(name="fvOptions", of_symbol="Foam::fv::options",
             of_file="src/finiteVolume/cfdTools/general/fvOptions/fvOptions.C",
             classification="DISPATCH", status="REVALIDATE_EXISTING",
             brae_existing="src/finiteVolume/cfdTools/fvOptions/device_fvoptions.cu",
             selection_base="option",
             brae_target="src/finiteVolume/cfdTools/general/fvOptions/",
             note="Reached three times in UEqn.H/pEqn.H: fvOptions(U), constrain(UEqn), correct(U)."),

        # ---- linear solvers ------------------------------------------------------------------
        dict(name="lduMatrix_solver", of_symbol="Foam::lduMatrix::solver::New",
             of_file="src/OpenFOAM/matrices/lduMatrix/lduMatrix/lduMatrixSolver.C",
             classification="LINEAR_SOLVER", status="REUSE_EXISTING",
             selection_base="lduMatrix::solver",
             brae_existing="src/cuda/device_pcg.cu, src/cuda/device_amg_pcg.cu",
             brae_target="src/matrices/lduMatrix/solvers/",
             note="8 keys in v2412. brae implements PCG (+AMG preconditioning) and PBiCGStab."),
        dict(name="GAMGPreconditioner", of_symbol="Foam::GAMGPreconditioner",
             of_file="src/OpenFOAM/matrices/lduMatrix/preconditioners/GAMGPreconditioner/"
                     "GAMGPreconditioner.C",
             classification="LINEAR_SOLVER", status="REUSE_EXISTING",
             schema_for="GAMGPreconditioner",
             brae_existing="src/cuda/device_amg.cu",
             brae_target="src/matrices/lduMatrix/preconditioners/GAMGPreconditioner/",
             note="brae's AMG is the analogue. Carries the whole-loop conditional-graph PCG."),
        dict(name="DILUPreconditioner", of_symbol="Foam::DILUPreconditioner",
             of_file="src/OpenFOAM/matrices/lduMatrix/preconditioners/DILUPreconditioner/"
                     "DILUPreconditioner.C",
             classification="LINEAR_SOLVER", status="REUSE_EXISTING",
             brae_existing="src/cuda/device_dilu.cu",
             brae_target="src/matrices/lduMatrix/preconditioners/DILUPreconditioner/",
             note="Level-scheduled, bit-identical to OpenFOAM."),
        dict(name="GAMGSolver", of_symbol="Foam::GAMGSolver",
             of_file="src/OpenFOAM/matrices/lduMatrix/solvers/GAMG/GAMGSolver.C",
             classification="LINEAR_SOLVER", status="UNSUPPORTED",
             schema_for="GAMGSolver",
             brae_target="src/matrices/lduMatrix/solvers/GAMG/",
             note="pitzDaily and motorBike BOTH select `GAMG` for p. brae substitutes AMG-preconditioned "
                  "PCG. That is a different algorithm with a different iteration count -- the solver must "
                  "say so, not silently substitute."),

        dict(name="dispatch", of_symbol="controlDict application",
             of_file="applications/solvers/incompressible/simpleFoam/simpleFoam.C",
             classification="DISPATCH", status="REIMPLEMENT",
             brae_cuda="src/applications/solvers/simpleFoam/simpleFoamV2.cu",
             brae_target="src/applications/solvers/simpleFoam/simpleFoamV2.cu",
             validation="CONVERGENCE GATE tests/simplefoam_v2_convergence.sh (ctest: "
                        "simplefoam_v2_convergence, ~90 s) -- the rebuilt path runs pitzDailyTurb from 0/ "
                        "to convergence (1713 iterations; OpenFOAM took 1576) and its converged fields are "
                        "compared with OpenFOAM's own 1576/: U 1.257e-01, p 1.922e-01, k 1.187e-02, "
                        "epsilon 2.512e-02, nut 6.686e-03. The bounds are what brae's ESTABLISHED solver "
                        "reaches against the same reference in the passing simple_turbulent_full gate "
                        "(U 1.311e-01, p 1.989e-01, ...) -- the rebuilt path is marginally better on every "
                        "field. The disagreement is localised (6 of 12225 cells above 0.5 m/s, worst at the "
                        "step corner) and is a property of the comparison, not of the rebuild. Carries a "
                        "control requiring the INITIAL field to exceed the bound. "
                        "ALSO tests/simplefoam_v2_dispatch.sh (ctest: simplefoam_v2_dispatch) -- drives the real "
                        "binary on real cases: OFF changes nothing and stays silent; ON+supported runs to "
                        "endTime and writes; ON+unsupported REFUSES with the reason and exits 1 for each of "
                        "MRF, fvOptions, SIMPLEC, a non-upwind div(phi,U), a transient ddtScheme and RAS; "
                        "the GAMG->AMG-PCG substitution is announced; RAS/kEpsilon RUNS and writes k/epsilon/nut "
                        "with a control that nut changed. Negative control: the guard admits the supported "
                        "case, so it is not refusing unconditionally.",
             note="OPT-IN via BRAE_SIMPLEFOAM_V2=1 while the envelope is small, and there is deliberately "
                  "NO try/catch around it: selected-but-unsupported must stop, never fall through to the "
                  "old solver. A user who asked for the new path and silently got the old one cannot tell "
                  "from the output which algorithm produced the answer. ENVELOPE TODAY: steady, laminar, "
                  "upwind or linearUpwind div(phi,U), orthogonal or `corrected` laplacian, SIMPLE or SIMPLEC, "
                  "no MRF/fvOptions, no fixedFluxPressure, no coupled patches -- stock pitzDaily and "
                  "pitzDailyTurb (RAS/kEpsilon) is inside it. The non-orthogonal check reads "
                  "laplacianSchemes ONLY: `laplacianSchemes Gauss linear orthogonal` builds an orthogonal "
                  "laplacian whatever snGradSchemes says, since snGradSchemes governs explicit fvc::snGrad "
                  "which in simpleFoam appears only in the refused SIMPLEC branch. "
                  "NON-ORTHOGONAL CORRECTION: implemented on the _cpp reference exactly as OpenFOAM defines "
                  "it -- BOTH halves, which is the part that is easy to miss: the implicit face "
                  "coefficient switches to nonOrthDeltaCoeffs = 1/max(n.delta, 0.05|delta|) "
                  "(correctedSnGrad.H:108-119, basicFvGeometryScheme.C:266) AND an explicit deferred "
                  "source -V*div(gamma*magSf*(corrVecs & interpolate(grad(vf)))) is added "
                  "(gaussLaplacianScheme.C), with corrVecs zero on boundary faces. VALIDATED against REAL "
                  "OpenFOAM (ctest: nonorth_vs_openfoam, which generates the reference by running "
                  "simpleFoam itself): on validation/shearedChannel -- genuinely non-orthogonal and using "
                  "only implemented schemes -- U 6.92e-04 and p 3.75e-04 vs OpenFOAM WITH the correction, "
                  "against U 8.47e-02 and p 1.37e-01 without it, i.e. 122x and 365x. The control requiring "
                  "the uncorrected path to be >=20x worse is the substance of that gate: on a "
                  "near-orthogonal mesh like pitzDaily every brae path agrees to 4 digits whether or not "
                  "the term is applied, so a gate there would pass with the code deleted. A SIGN ERROR was "
                  "found by this measurement -- divDevReff returns MINUS the laplacian, so its source "
                  "contribution is +V*div(...) while the pressure laplacian keeps OpenFOAM's own sign; "
                  "writing the laplacian sign in both places made U worse with the correction on (1.69e-01) "
                  "while p improved, and that asymmetry is what localised it. "
                  "THE CUDA SIDE IS NOW DONE TOO and carries both halves: deviceLaplacianCoeffs(..., "
                  "nonOrth=true) for the implicit coefficient and deviceLaplacianCorr for the deferred "
                  "source, in UEqn.cu (momentum, sign -1 because divDevReff carries minus the laplacian) "
                  "and pEqn.cu (pressure, sign +1). Matched to the reference at 2.9e-16 on the relaxed "
                  "diag and <=9.2e-16 on every source component, with controls asserting BOTH halves move "
                  "something (implicit 2.0e-03, explicit 1.7e-03 on U; 3.5e-03 and 7.0e-02 on p) so the "
                  "machine-precision agreement is not vacuous. The nonorth_vs_openfoam gate gained a CUDA "
                  "column that reproduces the reference against real OpenFOAM digit for digit: U 6.9193e-04 "
                  "and p 3.7529e-04, versus 8.4713e-02 / 1.3702e-01 uncorrected. TWO ORDERING TRAPS, both "
                  "of which built and ran clean while being wrong: (1) deviceDivDevReff ASSIGNS the "
                  "momentum source (device_divdevreff.cu, `dX[c] = d[0]`) rather than accumulating, so the "
                  "correction must be added AFTER it -- added before, it is silently discarded; (2) in the "
                  "diagnostic the GPU flags were copied from the host struct at the point `gin` is filled "
                  "in, which runs BEFORE those host flags are assigned, so the device got the struct "
                  "defaults and the CUDA column reproduced the uncorrected answer to five digits. Both "
                  "were caught by the gate's control, not by the build. `limited <coeff>` is REFUSED "
                  "separately: limitedSnGrad caps the correction and only `limited 1` equals `corrected`, "
                  "so accepting the whole family would over-correct. STILL OPEN for stock pitzDaily: "
                  "linearUpwind and SIMPLEC (`consistent yes`), which is what that case ships -- both "
                  "remain blockers. KNOWN NARROW GAP: on COUPLED patches OpenFOAM passes the corrected "
                  "deltaCoeffs to gradientInternalCoeffs (gaussLaplacianScheme.C, the pvf.coupled() "
                  "branch) while uncoupled patches use the plain ones; neither brae path does the coupled "
                  "variant, which is unreachable today because coupled patches are refused outright, and "
                  "must be revisited when they are added. "
                  "ENVELOPE WIDENING: `bounded` (-fvm::Sp(fvc::div(phi),U)) is DONE on both paths and "
                  "matched to 2.9e-16 (tests/test_ueqn_cuda.cu), with a control asserting the term "
                  "actually contributes -- it is ~2e-08 of the diagonal on a converged case because it "
                  "vanishes exactly at convergence, which is why agreement alone cannot prove it was "
                  "applied. The kernel sequence is the existing GPU driver's (device_simple_foam.cu:1150) "
                  "minus the cyclic/AMI additions. An earlier report that the CUDA term was a NO-OP was "
                  "wrong: the test's struct is `gi` and the patch set `gmi`, so the flag was never "
                  "enabled. "
                  "linearUpwind IS DONE on both paths. linearUpwind derives from `upwind`, so the MATRIX "
                  "is unchanged and the entire scheme is a deferred source correction "
                  "corr_f = (Cf - C_up) & grad(vf)_up, entering as `fvm += fvc::surfaceIntegrate(phi*corr)` "
                  "-- which is `source -= faceSum(...)`, since fvMatrix::operator+=(DimensionedField) is "
                  "`source() -= V*su` (fvMatrix.C:1855). Boundary faces contribute NOTHING on uncoupled "
                  "patches: linearUpwind::correction fills only the pvf.coupled() branch, so that is the "
                  "scheme, not an omission. OpenFOAM's `vector` SPECIALISATION uses one tensor grad(U) "
                  "rather than three scalar grads; the device does it per component, which is the same "
                  "field under grad(U)_ij = d(U_j)/dx_i, not an approximation. CUDA matches the reference "
                  "at 6.4e-16/8.8e-16 on the source with a control asserting the term moves the source "
                  "(3.5e-03) AND leaves the matrix EXACTLY alone -- a diagonal that moved would mean the "
                  "implicit weights had been changed, which is not what OpenFOAM does. The named gradient "
                  "is GUARDED: `linearUpwind grad(U)` resolves grad(U) through gradSchemes "
                  "(linearUpwind.H gradSchemeName_ -> mesh.gradScheme(name)), and anything but Gauss "
                  "linear -- cellLimited, leastSquares -- is REFUSED, because this correction does not "
                  "vanish at convergence and a different gradient is a different answer. GATE: "
                  "ctest linearupwind_vs_openfoam generates the oracle by running simpleFoam on "
                  "pitzDaily's own discretisation and CONTROLS with a plain-upwind brae run: "
                  "linearUpwind lands 1.9x (U), 3.2x (k), 2.4x (epsilon), 2.0x (nut) closer to OpenFOAM "
                  "than upwind does, measured in L2 -- pitzDaily's MAX-norm error sits in a few cells at "
                  "the step corner and is ~9.3e-02 whatever the scheme is, so the max norm cannot see "
                  "this term at all. Proven to FAIL with the correction disabled (every ratio collapses "
                  "to exactly 1.00x). "

                  "SIMPLEC (`consistent`) IS DONE on both paths: rAtU = 1/(1/rAU - UEqn.H1()), with "
                  "phiHbyA += interpolate(rAtU-rAU)*snGrad(p)*magSf and HbyA -= (rAU-rAtU)*grad(p), placed "
                  "AFTER adjustPhi as pEqn.H places them. matrixH1 (lduMatrixATmul.C) and fvc::snGrad "
                  "(snGradScheme.C + correctedSnGrad.C) were added for it. On the device the flux term is "
                  "written as a LAPLACIAN FLUX plus its non-orth part rather than as an explicit snGrad, "
                  "because fvm::laplacian(gamma,p).flux() IS gamma_f*magSf_f*snGrad(p)_f -- that keeps the "
                  "correction consistent with the equation it corrects by construction. What is still "
                  "refused is constrainPressure, i.e. a fixedFluxPressure patch: brae maps that type to "
                  "zeroGradient, which is the same BC only when the imposed flux is zero. "

                  "A REAL DEFECT WAS FOUND BY THIS WORK, in the non-orthogonal correction rather than in "
                  "SIMPLEC. fvMatrix::flux() adds faceFluxCorrectionPtr_ (fvMatrix.C:1688), which "
                  "gaussLaplacianScheme stores alongside the source correction (gaussLaplacianScheme.C:"
                  "186-199). brae had the correction in the pressure equation's SOURCE and NOT in "
                  "pEqn.flux(), so `phi = phiHbyA - pEqn.flux()` dropped it and phi was not conservative "
                  "on a non-orthogonal mesh -- silently, because the pressure equation still solved and "
                  "every per-stage test still passed at 1e-16. It was exposed by a control that has "
                  "nothing to do with fluxes: SIMPLEC and plain SIMPLE must converge to the SAME fixed "
                  "point, since SIMPLEC changes the iteration and not the discrete system, and brae's "
                  "differed by 5.3e-03 (U, L2). FIXED by porting faceFluxCorrection onto FvMatrix and "
                  "PressureMatrix, with the per-cell source now computed FROM the face flux so the two "
                  "cannot drift apart. Effect: SIMPLEC-vs-SIMPLE 5.3e-03 -> 2.8e-04; the nonorth gate on "
                  "shearedChannel U 6.92e-04 -> 3.12e-05 and p 3.75e-04 -> 2.69e-06 against real "
                  "OpenFOAM, its discriminating ratio 122x -> 2716x, and its bound tightened 5e-03 -> "
                  "1e-04 accordingly. "

                  "STOCK pitzDaily NOW RUNS AND IS GATED (ctest stock_pitzdaily_vs_openfoam): SIMPLEC + "
                  "bounded Gauss linearUpwind grad(U) + Gauss linear corrected + RAS/kEpsilon, "
                  "unmodified apart from tightening residualControl from the stock `p 1e-2` (at which "
                  "both solvers stop early at different states, so the comparison would measure where "
                  "each stopped). vs real OpenFOAM, L2: U 5.18e-03, p 2.25e-02, k 1.85e-02, "
                  "epsilon 3.82e-02, nut 2.75e-02, and brae is settled -- iteration 749 vs 2000 differs "
                  "by 1.8e-08. The gate also asserts every feature is REPORTED as applied, so a silent "
                  "skip fails it. "
                  "PERFORMANCE, measured on GB10 vs OpenFOAM v2412 on 20 cores, 100 SIMPLE iterations on "
                  "scaled pitzDaily (12k/110k/440k/1.22M): the rebuilt path was NOT reading the case's "
                  "fvSolution solver settings at all -- StepInput hardcoded tolerance 1e-10 / relTol 0 "
                  "while pitzDaily asks for relTol 0.1 -- so every inner solve ran to near machine "
                  "precision every outer iteration. nsys: 614 fine-grid SpMV per SIMPLE iteration against "
                  "the existing GPU driver's 62. FIXED by reading solvers/p and solvers/U (subDict "
                  "resolves OpenFOAM regex keys, so `\"(U|k|epsilon|omega|f|v2)\"` is found by \"U\"); "
                  "maxIter is read too, since an `maxIter 10` caps the answer. Effect: 2.3-2.6x faster at "
                  "every mesh size (1.22M: 166s -> 67s) with the converged answer UNCHANGED -- U L2 vs "
                  "OpenFOAM is 5.1790e-03 before and after, which is the point: the inner tolerance "
                  "changes the path, not the fixed point. Side effect on the gate: brae now plateaus like "
                  "OpenFOAM (U ~9e-06, p ~2e-04, flat from iteration ~700) instead of reaching a tight "
                  "residualControl, so stock_pitzdaily_vs_openfoam now ASSERTS settledness by re-running "
                  "to half the endTime rather than trusting a residual. REMAINING GAP to the existing GPU "
                  "driver is ~2.3x and is now dominated by the MOMENTUM solve: the rebuilt path uses "
                  "Jacobi-BiCGStab (~42 iterations per component per outer step, ~90% of all "
                  "linear-algebra time) where the case asks for smoothSolver/symGaussSeidel, which brae "
                  "already implements and the old driver uses (gsColorT). Also unwired on this path: the "
                  "AMG V-cycle CUDA-graph capture (deviceAMGPCG's captureVcycle defaults false; the old "
                  "driver passes useGraph=true), the PCG residual-read cadence (checkEvery), and the "
                  "turbulence hook's nuEff refresh, which round-trips nut to the HOST and does the face "
                  "interpolation there every iteration when deviceInterpolate would keep it resident. "
                  "The rebuilt driver also writes only its final time -- writeInterval is not honoured. "
                  "RAS/kEpsilon passes the "
                  "envelope check and the hook IS now wired to deviceKEpsilonCorrect: the hook also owns the "
                  "nuEff refresh (nu + nut, boundary value from deviceBoundaryNut's wall function, never "
                  "the owner cell), which is what makes the lagged coupling work without the driver "
                  "knowing the model. "
                  "CONVERGENCE GAP -- FOUND AND FIXED. The CUDA driver allocated its PressureMatrix and "
                  "folded diagonal fresh every iteration while the AMG hierarchy -- and the V-cycle/PCG "
                  "graph caches keyed on that fine matrix -- persisted across iterations. Iteration 1 was "
                  "exact (dU 8.2e-14 from identical inputs) and iteration 2 was wrong by 1.3e-01, with "
                  "EVERY per-stage test still passing at 1e-16. Fixed by holding the pressure buffers in "
                  "SolverWorkspace, matching how device_simple_foam.cu keeps them as members. All four "
                  "drivers (old host, old GPU, _cpp, CUDA) now agree over 60 iterations "
                  "(iter 59: 2.9136e-02 / 2.914e-02 / 2.91e-02 / 2.91e-02) and the rebuilt solver "
                  "converges end-to-end: 1.0 -> 0.391 -> 0.0615 -> 0.0284 over 60 iterations. "
                  "Regression: tests/test_simple_step_cuda.cu gained a multi-iteration mode "
                  "(ctest simple_step_cuda_loop, 10 iterations, tight solves) -- verified to FAIL "
                  "(U 1.5e-01, p 3.7e-01) when the buffers are made transient again. "
                  "TWO EARLIER DIAGNOSES IN THIS INVESTIGATION WERE WRONG and are retracted: the "
                  "host-vs-GPU 'convergence gap' was a sampling artifact (reading lines 1/5/10/20 of an "
                  "oscillating series), and the 'HbyA wrong by 42%' was a reference rebuilt without the "
                  "momentum predictor. Ruled out on the way: bounded, the non-orthogonal correction, "
                  "solver tolerances, deviceGaussGrad (4e-15), AMG face-weight slicing (no effect -- "
                  "internal faces already come first). Found and fixed as genuine silent-substitution "
                  "holes in the envelope guard: `bounded` and the non-orthogonal laplacian correction."),

        dict(name="cuda_vs_reference", of_symbol="(brae-specific)",
             of_file="-",
             classification="GPU_REQUIRED", status="REVALIDATE_EXISTING",
             brae_existing="src/cuda/device_simple.cu, src/cuda/device_fvm.cu",
             brae_target="src/applications/solvers/simpleFoam/",
             validation="tests/test_gpu_vs_cpp.cu -- CUDA against the _cpp reference at STAGE granularity, "
                        "run on BOTH a laminar case (matrixDumpAsym/282) and a TURBULENT one "
                        "(pitzDailyTurb/1576, nuEff varying per cell and per boundary face). "
                        "PRESSURE: rAU 1.5e-16, laplacian upper/lower 0 diag 1.4e-16, pEqn.flux() 0, "
                        "setReference 0. MOMENTUM: div(phi,U) upper/lower 0 diag 4.6e-18, divDevReff "
                        "source 4.8e-15/5.6e-16/2.4e-16, H(U) 3.8e-16/2.3e-16/2.4e-16, phiHbyA 0, "
                        "corrector 0. TURBULENCE: GbyNu 7.6e-17, nut 0. Three controls fire.",
             note="Closes the chain OpenFOAM -> _cpp -> CUDA. Every other GPU test compares the device "
                  "against CPU code written inline in that same test, which proves consistency but not "
                  "correctness. Running on a turbulent case as well is what makes it load-bearing: with "
                  "constant nuEff a kernel that mishandles a per-face diffusivity, or reads the owner "
                  "cell's value on a wall instead of the patch value, still agrees perfectly. NOT yet "
                  "compared this way: the linear solves themselves, the wall functions (G0/eps0), and the "
                  "k/epsilon transport assembly."),

        # ---- determinism ---------------------------------------------------------------------
        dict(name="deterministic_assembly", of_symbol="(brae-specific)",
             of_file="-",
             classification="GPU_REQUIRED", status="REUSE_EXISTING",
             brae_existing="src/matrices/lduMatrix/lduMatrix/reductions.cu",
             brae_target="src/matrices/lduMatrix/lduMatrix/",
             validation="tests/determinism_gate.sh -- pitzDaily (kEpsilon) and pitzDailyKOmega bit-identical "
                        "over two runs; verified to 200 iterations, plus airfoil and backwardFacingStep2D. "
                        "Carries a 1-ULP negative control.",
             note="DONE for the incompressible simpleFoam path. Was 3.6e-02 after 20 iterations, now 0. "
                  "Three scatter sites, all converted to fixed-order gathers: AMG restriction "
                  "(rc[map[c]] += r[c], hit every level of every V-cycle of every PCG iteration), the "
                  "turbulence wall functions (cells with >1 wall face), and the eps setValues constraint "
                  "(cells with >1 constrained face). The last two are RARE -- bit-identical at 1/5/8/10/15 "
                  "iterations and different at 12 -- so intermittency, not just a systematic offset, is "
                  "what the gate has to catch. STILL OPEN: the opt-in BRAE_AMG_SA path still scatters, and "
                  "the cyclic/AMI (42 sites) and distributed (device_halo) paths are untouched."),
    ],
}

SELECTION_NOTE = {
    "incompressible::turbulenceModel": "createFields.H:42",
    "lduMatrix::solver": "fvSolution solvers/<field>/solver",
    "option": "constant/fvOptions or system/fvOptions",
}


def db():
    sys.path.insert(0, os.path.abspath(OFSCAN))
    from ofscan.graph.database import Db
    return Db(os.path.join(os.path.abspath(OFSCAN), "ofscan.db"))


def impls_for(d, base):
    rows = d.q("SELECT DISTINCT selection_key k FROM runtime_types WHERE base=? AND selection_key IS NOT NULL"
               " ORDER BY k", (base,))
    return [r["k"] for r in rows]


def schema_for(d, cls):
    sys.path.insert(0, os.path.abspath(OFSCAN))
    from ofscan.foam import dictionary as dictmod
    out = []
    for r in dictmod.schema_for(d, cls):
        out.append(dict(key=r["key"], required=bool(r["r"]), default=r["d"],
                        op=r["op"], at="%s:%s" % (os.path.basename(r["p"] or "?"), r["l"])))
    return out


def case_selections(d, case):
    sys.path.insert(0, os.path.abspath(OFSCAN))
    from ofscan.foam import case as casemod
    out = []
    for cat, base, key, where in casemod.selections(case):
        r = casemod.resolve(d, base, key)
        out.append(dict(category=cat, keyword=key, where=where,
                        resolved=(r[0] if r else None), how=(r[3] if r else "UNRESOLVED")))
    return out


def y(s):
    """Minimal YAML scalar quoting."""
    if s is None:
        return "null"
    s = str(s)
    if s == "":
        return '""'
    # A leading '-' or '?' is a YAML INDICATOR, not text: an unquoted `file: -` parses as the start of a
    # sequence and makes the whole document unloadable. Quote on indicators anywhere it matters.
    if (re.search(r'[:#\[\]{},&*?|<>=!%@`"\']|^\s|\s$', s)
            or re.match(r'^[-?]', s)
            or s.lower() in ("yes", "no", "true", "false", "null", "~")):
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return s


def verify_provenance(solver):
    """Every OpenFOAM path the manifest claims must EXIST in the OpenFOAM tree.

    Provenance that is not checked is just a comment. This caught a real error on the first run: the
    curated table cited `src/MomentumTransportModels/...` for the turbulence models, which is the path in a
    DIFFERENT OpenFOAM lineage -- v2412 puts them under `src/TurbulenceModels/`. A manifest whose whole
    purpose is mechanical drift detection cannot ship paths that were never there to drift from.
    """
    bad = []
    for c in COMPONENTS[solver]:
        f = c["of_file"]
        if f in ("-", "", None):
            continue
        if not os.path.exists(os.path.join(OF, f)):
            bad.append((c["name"], f))
    return bad


def emit(solver, d, cases):
    L = []
    add = L.append
    add("# GENERATED by tools/of_manifest.py -- do not edit the `derived:` blocks by hand.")
    add("# The `classification:`/`brae_status:`/`note:` fields are CURATED and live in that script.")
    add("solver: %s" % y(solver))
    add("openfoam_root: %s" % y(OF))
    add("")
    add("components:")
    for c in COMPONENTS[solver]:
        add("  %s:" % c["name"])
        add("    openfoam:")
        add("      symbol: %s" % y(c["of_symbol"]))
        add("      file: %s" % y(c["of_file"]))
        if c.get("of_line"):
            add("      line: %d" % c["of_line"])
        add("    classification: %s" % c["classification"])
        add("    brae_status: %s" % c["status"])
        for k in ("brae_existing", "brae_reference", "brae_cuda", "brae_target", "validation"):
            if c.get(k):
                add("    %s: %s" % (k, y(c[k])))
        if c.get("note"):
            add("    note: %s" % y(c["note"]))
        if c.get("selection_base"):
            ks = impls_for(d, c["selection_base"])
            add("    derived_runtime_selection:")
            add("      base: %s" % y(c["selection_base"]))
            add("      selected_at: %s" % y(SELECTION_NOTE.get(c["selection_base"], "-")))
            add("      openfoam_implementations: %d" % len(ks))
            add("      keys: [%s]" % ", ".join(y(k) for k in ks))
        if c.get("schema_for"):
            rows = schema_for(d, c["schema_for"])
            add("    derived_dictionary_schema:")
            add("      class: %s" % y(c["schema_for"]))
            if not rows:
                add("      keys: []")
            else:
                add("      keys:")
                for r in rows:
                    add("        - key: %s" % y(r["key"]))
                    add("          required: %s" % ("true" if r["required"] else "false"))
                    add("          default: %s" % y(r["default"]))
                    add("          read_by: %s" % y(r["op"]))
                    add("          at: %s" % y(r["at"]))
        add("")

    add("validation_cases:")
    for path in cases:
        if not os.path.isdir(path):
            continue
        sels = case_selections(d, path)
        unres = [s for s in sels if s["how"] == "UNRESOLVED"]
        add("  %s:" % y(os.path.basename(path)))
        add("    path: %s" % y(path))
        add("    derived_selection_sites: %d" % len(sels))
        add("    derived_unresolved: %d" % len(unres))
        seen = set()
        add("    derived_required_implementations:")
        for s in sorted(sels, key=lambda x: (x["category"], x["keyword"])):
            tag = (s["category"], s["keyword"])
            if tag in seen:
                continue
            seen.add(tag)
            add("      - {category: %s, keyword: %s, resolves_to: %s}"
                % (s["category"], y(s["keyword"]), y(s["resolved"])))
        add("")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("solver")
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--check", metavar="FILE",
                    help="compare against an existing manifest; non-zero exit if it has drifted")
    a = ap.parse_args()
    if a.solver not in COMPONENTS:
        sys.exit("no curated component table for '%s' (add one to tools/of_manifest.py)" % a.solver)
    bad = verify_provenance(a.solver)
    if bad:
        sys.stderr.write("PROVENANCE ERROR -- these OpenFOAM paths do not exist under %s:\n" % OF)
        for name, f in bad:
            sys.stderr.write("  %-28s %s\n" % (name, f))
        return 2
    cases = a.cases or [
        os.path.join(OF, "tutorials/incompressible/simpleFoam/pitzDaily"),
        os.path.join(OF, "tutorials/incompressible/simpleFoam/airFoil2D"),
        os.path.join(OF, "tutorials/incompressible/simpleFoam/motorBike"),
    ]
    text = emit(a.solver, db(), cases)
    if a.check:
        old = open(a.check, encoding="utf-8").read() if os.path.exists(a.check) else ""
        if old != text:
            sys.stderr.write("manifest DRIFTED: %s is not what tools/of_manifest.py now produces.\n"
                             "Re-generate it and read the diff -- OpenFOAM or ofscan has changed.\n" % a.check)
            return 1
        sys.stderr.write("manifest up to date: %s\n" % a.check)
        return 0
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
