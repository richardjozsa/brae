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
             brae_target="src/applications/solvers/simpleFoam/simpleFoam.cu",
             validation="tests/test_simple_step_cpp.cu -- END-TO-END, one SIMPLE iteration composed of the "
                        "_cpp components vs OpenFOAM dumpSimpleStep (validation/matrixDumpSimple/step.dat): "
                        "p 2.5e-11, U 1.6e-12, phi 1.2e-11, every boundary patch <= 3.5e-13. Gate set at "
                        "1e-9, not the 1e-5 the older step test uses.",
             note="The _cpp driver owns NO numerics -- 9 calls into shared components, each with its own "
                  "OpenFOAM provenance and test. Replaces a 3578-line file that pimpleFoam, rhoSimpleFoam "
                  "and five common/ headers all included. NOTE ON THE FIXTURE: matrixDumpSimple's "
                  "fvSolution sets `consistent yes`, but step.dat was dumped with plain SIMPLE; the test "
                  "asserts the SIMPLEC refusal fires on that case and then compares with SIMPLEC off. A "
                  "SIMPLEC oracle is needed before SIMPLEC can be ported."),
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
             brae_target="src/applications/solvers/simpleFoam/UEqn.cu",
             validation="tests/test_ueqn_cpp.cu -- validated by DECOMPOSITION against "
                        "validation/matrixDumpAsym/momentum.dat: the div/laplacian core matches OpenFOAM to "
                        "1e-11 on diag/upper/lower/source; adding divDevReff provably changes only `source`, "
                        "and by exactly the explicit dev2 term; relaxation raises |diag| and leaves the "
                        "off-diagonals alone; MRF and fvOptions both throw.",
             note="24 lines in OpenFOAM. The _cpp reference REFUSES MRF and fvOptions rather than ignoring "
                  "them -- brae has shipped a solver that silently ignored MRFProperties and produced a "
                  "converged wrong answer. The CUDA side is not written yet."),
        dict(name="pEqn", of_symbol="pEqn.H",
             of_file="applications/solvers/incompressible/simpleFoam/pEqn.H",
             classification="GPU_REQUIRED", status="REIMPLEMENT",
             brae_reference="src/applications/solvers/simpleFoam/pEqn_cpp.cu",
             brae_target="src/applications/solvers/simpleFoam/pEqn.cu",
             validation="tests/test_peqn_cpp.cu -- stage by stage on validation/matrixDumpAsym: rAU and "
                        "HbyA vs OpenFOAM A()/H() (ops.dat) to 1e-11; the pressure Laplacian incl. all "
                        "patch coefficients vs peqn.dat to 1e-11; source == laplacian source + "
                        "div(phiHbyA)*V; setReference asserted to be exactly fvMatrix.C:1011-1023 (it "
                        "DOUBLES the diagonal, it does not overwrite it); correctFlux analytic at p=0; "
                        "relaxField analytic; MRF/fvOptions/consistent all refused.",
             note="50 lines. Every intermediate is RETURNED rather than kept local, so the first divergent "
                  "stage can be isolated -- a past investigation ended at `phi = phiHbyA - pEqn.flux()`, "
                  "which is stage 7 here. SIMPLEC (`consistent`) is refused: it needs UEqn.H1() and "
                  "fvc::snGrad, neither ported. The CUDA side is not written yet."),

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
             brae_target="src/TurbulenceModels/turbulenceModels/RAS/kEpsilon/"),
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
        for k in ("brae_existing", "brae_reference", "brae_target", "validation"):
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
