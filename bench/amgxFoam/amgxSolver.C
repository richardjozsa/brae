/*---------------------------------------------------------------------------*\
    amgxSolver implementation, LDU->CSR + raw AMGX C API.
\*---------------------------------------------------------------------------*/
#include "amgxSolver.H"
#include "addToRunTimeSelectionTable.H"
#include "PrecisionAdaptor.H"

#include <amgx_c.h>
#include <vector>
#include <algorithm>
#include <tuple>
#include <map>
#include <cstdio>

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(amgxSolver, 0);

    lduMatrix::solver::addsymMatrixConstructorToTable<amgxSolver>
        addamgxSymMatrixConstructorToTable_;

    lduMatrix::solver::addasymMatrixConstructorToTable<amgxSolver>
        addamgxAsymMatrixConstructorToTable_;
}


// * * * * * * * * * * * * * Local helpers  * * * * * * * * * * * * * * * * * //

namespace
{
    inline void amgxCheck(AMGX_RC rc, const char* what)
    {
        if (rc != AMGX_RC_OK)
        {
            char msg[512];
            AMGX_get_error_string(rc, msg, sizeof(msg));
            FatalErrorInFunction
                << "AMGX error in " << what << ": " << msg
                << Foam::abort(Foam::FatalError);
        }
    }

    // Initialise AMGX exactly once per process
    void amgxEnsureInit()
    {
        static bool done = false;
        if (!done)
        {
            amgxCheck(AMGX_initialize(), "AMGX_initialize");
            done = true;
        }
    }

    // Per-field cache: build AMGX handles + CSR structure ONCE, then each solve
    // only replaces coefficients + RHS (the petsc4Foam-style caching pattern).
    struct AmgxCache
    {
        AMGX_config_handle    cfg  = nullptr;
        AMGX_resources_handle rsrc = nullptr;
        AMGX_matrix_handle    A    = nullptr;
        AMGX_vector_handle    xv   = nullptr;
        AMGX_vector_handle    bv   = nullptr;
        AMGX_solver_handle    slv  = nullptr;
        std::vector<int>    rowPtr, colIdx;
        std::vector<int>    diagPos, upperPos, lowerPos; // LDU element -> CSR slot
        std::vector<double> val, xh, bh;
        Foam::label nCells = -1, nFaces = -1, nnz = -1;
        bool ready = false;
        bool setupDone = false;
    };
    std::map<Foam::word, AmgxCache> g_amgxCache;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::amgxSolver::amgxSolver
(
    const word& fieldName,
    const lduMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& solverControls
)
:
    lduMatrix::solver
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces,
        solverControls
    )
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::solverPerformance Foam::amgxSolver::scalarSolve
(
    solveScalarField& psi,
    const solveScalarField& source,
    const direction cmpt
) const
{
    solverPerformance solverPerf(typeName, fieldName_);

    const label nCells = psi.size();

    // --- OF-style initial residual (matches PCG) ---
    solveScalarField wA(nCells);
    solveScalarField pA(nCells);
    matrix_.Amul(wA, psi, interfaceBouCoeffs_, interfaces_, cmpt);
    solveScalarField rA(source - wA);

    const solveScalar normFactor = this->normFactor(psi, source, wA, pA);
    solverPerf.initialResidual() =
        gSumMag(rA, matrix().mesh().comm())/normFactor;
    solverPerf.finalResidual() = solverPerf.initialResidual();

    if
    (
        minIter_ > 0
     || !solverPerf.checkConvergence(tolerance_, relTol_, log_)
    )
    {
        const scalarField& diag  = matrix_.diag();
        const scalarField& upper = matrix_.upper();
        const scalarField& lower = matrix_.lower();   // == upper if symmetric
        const labelUList&  lowerAddr = matrix_.lduAddr().lowerAddr();
        const labelUList&  upperAddr = matrix_.lduAddr().upperAddr();
        const label nFaces = upperAddr.size();
        const label nnz = nCells + 2*nFaces;
        const AMGX_Mode mode = AMGX_mode_dDDI;

        amgxEnsureInit();
        AmgxCache& c = g_amgxCache[fieldName_];

        // ---- One-time: CSR structure + AMGX handles for this field ----
        if (!c.ready || c.nCells != nCells || c.nFaces != nFaces)
        {
            c.nCells = nCells; c.nFaces = nFaces; c.nnz = nnz;
            c.rowPtr.assign(nCells+1, 0);
            c.colIdx.assign(nnz, 0); c.val.assign(nnz, 0.0);
            c.diagPos.assign(nCells, 0);
            c.upperPos.assign(nFaces, 0); c.lowerPos.assign(nFaces, 0);
            c.xh.assign(nCells, 0.0); c.bh.assign(nCells, 0.0);

            // per-row entries (col, kind, idx): kind 0=diag,1=upper(f),2=lower(f)
            std::vector<std::vector<std::tuple<int,int,int>>> rows(nCells);
            for (label i = 0; i < nCells; ++i) rows[i].emplace_back(int(i),0,int(i));
            for (label f = 0; f < nFaces; ++f)
            {
                rows[lowerAddr[f]].emplace_back(int(upperAddr[f]),1,int(f));
                rows[upperAddr[f]].emplace_back(int(lowerAddr[f]),2,int(f));
            }
            int pos = 0;
            for (label i = 0; i < nCells; ++i)
            {
                std::sort(rows[i].begin(), rows[i].end(),
                    [](const auto& a, const auto& b){ return std::get<0>(a) < std::get<0>(b); });
                c.rowPtr[i] = pos;
                for (const auto& e : rows[i])
                {
                    c.colIdx[pos] = std::get<0>(e);
                    const int kind = std::get<1>(e), idx = std::get<2>(e);
                    if      (kind == 0) c.diagPos[idx]  = pos;
                    else if (kind == 1) c.upperPos[idx] = pos;
                    else                c.lowerPos[idx] = pos;
                    ++pos;
                }
            }
            c.rowPtr[nCells] = pos;

            // AMGX config: from a JSON file (fvSolution key "amgxConfig") or a
            // tuned default (PCG preconditioned by a classical AMG V-cycle, i.e.
            // NVIDIA's PCG_CLASSICAL_V_JACOBI shape, NOT a bare Jacobi precond).
            const fileName cfgFile
            (
                controlDict_.getOrDefault<fileName>("amgxConfig", fileName())
            );
            if (!cfgFile.empty())
            {
                fileName f(cfgFile); f.expand();
                amgxCheck
                (
                    AMGX_config_create_from_file(&c.cfg, f.c_str()),
                    "config_from_file"
                );
            }
            else
            {
                // AGGREGATION AMG (NVIDIA AGGREGATION_JACOBI shape). NOTE: AMGX
                // 2.5.0's CLASSICAL path throws a Thrust/sm_121 error on GB10,
                // so aggregation is required here.
                amgxCheck
                (
                    AMGX_config_create
                    (
                        &c.cfg,
                        "config_version=2, solver(main)=PCG, "
                        "main:preconditioner(amg)=AMG, "
                        "amg:algorithm=AGGREGATION, amg:selector=SIZE_2, "
                        "amg:smoother=BLOCK_JACOBI, "
                        "amg:presweeps=2, amg:postsweeps=2, "
                        "amg:relaxation_factor=0.75, amg:coarsest_sweeps=2, "
                        "amg:max_iters=1, amg:cycle=V, "
                        "amg:max_levels=50, amg:min_coarse_rows=2, "
                        "main:norm=L2, main:convergence=RELATIVE_INI, "
                        "main:monitor_residual=1"
                    ),
                    "config_create"
                );
            }
            // Override iteration/tolerance controls from fvSolution
            char ov[192];
            std::snprintf
            (
                ov, sizeof(ov),
                "config_version=2, main:max_iters=%d, main:tolerance=%g",
                int(maxIter_ > 0 ? maxIter_ : 1000),
                double(relTol_ > 0 ? relTol_ : (tolerance_ > 0 ? tolerance_ : 1e-6))
            );
            AMGX_config_add_parameters(&c.cfg, ov);
            amgxCheck(AMGX_resources_create_simple(&c.rsrc, c.cfg), "resources");
            amgxCheck(AMGX_matrix_create(&c.A, c.rsrc, mode), "matrix_create");
            amgxCheck(AMGX_vector_create(&c.xv, c.rsrc, mode), "xv_create");
            amgxCheck(AMGX_vector_create(&c.bv, c.rsrc, mode), "bv_create");
            amgxCheck(AMGX_solver_create(&c.slv, c.rsrc, mode, c.cfg), "solver_create");
            // upload structure once (values filled below, replaced thereafter)
            for (label i = 0; i < nCells; ++i) c.val[c.diagPos[i]] = diag[i];
            for (label f = 0; f < nFaces; ++f)
            { c.val[c.upperPos[f]] = upper[f]; c.val[c.lowerPos[f]] = lower[f]; }
            amgxCheck
            (
                AMGX_matrix_upload_all(c.A, nCells, nnz, 1, 1,
                    c.rowPtr.data(), c.colIdx.data(), c.val.data(), nullptr),
                "matrix_upload_all"
            );
            c.ready = true;
        }
        else
        {
            // ---- Subsequent solves: replace coefficients only (no re-sort) ----
            for (label i = 0; i < nCells; ++i) c.val[c.diagPos[i]] = diag[i];
            for (label f = 0; f < nFaces; ++f)
            { c.val[c.upperPos[f]] = upper[f]; c.val[c.lowerPos[f]] = lower[f]; }
            amgxCheck
            (
                AMGX_matrix_replace_coefficients(c.A, nCells, nnz, c.val.data(), nullptr),
                "replace_coefficients"
            );
        }

        for (label i = 0; i < nCells; ++i)
        { c.xh[i] = double(psi[i]); c.bh[i] = double(source[i]); }
        amgxCheck(AMGX_vector_upload(c.bv, nCells, 1, c.bh.data()), "b_upload");
        amgxCheck(AMGX_vector_upload(c.xv, nCells, 1, c.xh.data()), "x_upload");
        if (!c.setupDone)
        {
            amgxCheck(AMGX_solver_setup(c.slv, c.A), "solver_setup");
            c.setupDone = true;
        }
        else
        {
            // reuse the AMG aggregation, recompute operators for new coeffs
            amgxCheck(AMGX_solver_resetup(c.slv, c.A), "solver_resetup");
        }
        amgxCheck(AMGX_solver_solve(c.slv, c.bv, c.xv), "solver_solve");
        amgxCheck(AMGX_vector_download(c.xv, c.xh.data()), "x_download");

        int nIter = 0;
        AMGX_solver_get_iterations_number(c.slv, &nIter);
        for (label i = 0; i < nCells; ++i) psi[i] = solveScalar(c.xh[i]);

        // ---- OF-style final residual ----
        matrix_.Amul(wA, psi, interfaceBouCoeffs_, interfaces_, cmpt);
        rA = source - wA;
        solverPerf.finalResidual() =
            gSumMag(rA, matrix().mesh().comm())/normFactor;
        solverPerf.nIterations() = nIter;
    }

    matrix().setResidualField
    (
        ConstPrecisionAdaptor<scalar, solveScalar>(rA)(),
        fieldName_,
        false
    );

    return solverPerf;
}


Foam::solverPerformance Foam::amgxSolver::solve
(
    scalarField& psi_s,
    const scalarField& source,
    const direction cmpt
) const
{
    PrecisionAdaptor<solveScalar, scalar> tpsi(psi_s);
    return scalarSolve
    (
        tpsi.ref(),
        ConstPrecisionAdaptor<solveScalar, scalar>(source)(),
        cmpt
    );
}


// ************************************************************************* //
