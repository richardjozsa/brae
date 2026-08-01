/* Demo instrumentation for the AMG-PCG showcase (demo/amgpcg).
 *
 * Reads the SAME matrix OpenFOAM dumped (case20/trace/matrix.csv + rhs.csv), converts the LDU
 * form to the CSR that AMGX requires -- which is itself part of the story: every offload has to
 * pay this conversion, brae does not -- and dumps as CSV:
 *   amgx_cycles.csv   the REAL AMGX state after each Krylov iteration (psi + residual)
 *   amgx_levels.txt   AMGX's own grid-hierarchy printout (level sizes, nnz, coarsening rate)
 *
 * The per-iteration trace re-solves from psi = 0 with main:max_iters = 1, 2, 3, ... AMGX's PCG
 * is a deterministic recurrence from a fixed start, so the state a max_iters=k solve stops at is
 * the state the k-th iteration of a long solve passes through.
 *
 * Same solver configuration as the OpenFOAM-AMGX benchmark (bench/amgxFoam/amgxSolver.C:205-215):
 * PCG + AGGREGATION/SIZE_2 + BLOCK_JACOBI, 2 pre / 2 post sweeps, V-cycle.
 */

#include <amgx_c.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_CELLS 1048576
#define MAX_FACES 2097152

static int     nCells = 0;
static int     nFaces = 0;
static double* diagV = NULL;
static double* upperV = NULL;
static double* lowerV = NULL;
static int*    ownerA = NULL;
static int*    neiA = NULL;
static double* rhs = NULL;

/* CSR assembled from the LDU triple. */
static int*    rowPtr = NULL;
static int*    colIdx = NULL;
static double* valCsr = NULL;
static int     nnz = 0;

static void allocArrays(int cells, int faces)
{
    diagV  = malloc((size_t)cells * sizeof(double));
    rhs    = malloc((size_t)cells * sizeof(double));
    upperV = malloc((size_t)faces * sizeof(double));
    lowerV = malloc((size_t)faces * sizeof(double));
    ownerA = malloc((size_t)faces * sizeof(int));
    neiA   = malloc((size_t)faces * sizeof(int));
    rowPtr = malloc(((size_t)cells + 1) * sizeof(int));
    colIdx = malloc(((size_t)cells + 2 * (size_t)faces) * sizeof(int));
    valCsr = malloc(((size_t)cells + 2 * (size_t)faces) * sizeof(double));
}

/* Compact binary written by trace_brae for meshes too large to round-trip through CSV. */
static int readMatrixBinary(const char* path)
{
    FILE* f = fopen(path, "rb");

    if (!f) return 0;

    if (fread(&nCells, sizeof(int), 1, f) != 1) { fclose(f); return 0; }
    if (fread(&nFaces, sizeof(int), 1, f) != 1) { fclose(f); return 0; }

    allocArrays(nCells, nFaces);

    size_t ok = 0;
    ok += fread(diagV,  sizeof(double), (size_t)nCells, f);
    ok += fread(rhs,    sizeof(double), (size_t)nCells, f);
    ok += fread(upperV, sizeof(double), (size_t)nFaces, f);
    ok += fread(lowerV, sizeof(double), (size_t)nFaces, f);
    ok += fread(ownerA, sizeof(int),    (size_t)nFaces, f);
    ok += fread(neiA,   sizeof(int),    (size_t)nFaces, f);
    fclose(f);

    return ok == (size_t)(2 * nCells + 4 * nFaces);
}

static double nowMs(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}

/* ---------------------------------------------------------------------------------------- */

static void readMatrixCsv
(
    const char* matrixPath,
    const char* rhsPath
)
{
    char line[512];
    FILE* f = fopen(matrixPath, "r");

    if (!f)
    {
        fprintf(stderr, "cannot open %s\n", matrixPath);
        exit(1);
    }

    /* CSV path is only used for the small demo cases; a fixed generous allocation is fine. */
    allocArrays(65536, 200000);

    if (fgets(line, sizeof(line), f) == NULL) { fclose(f); return; }

    while (fgets(line, sizeof(line), f))
    {
        char kind[32];
        int  idx, own, nei;
        double val;

        if (sscanf(line, "%31[^,],%d,%d,%d,%lf", kind, &idx, &own, &nei, &val) != 5) continue;

        if (strcmp(kind, "diag") == 0)
        {
            diagV[idx] = val;
            if (idx + 1 > nCells) nCells = idx + 1;
        }
        else
        {
            ownerA[idx] = own;
            neiA[idx]   = nei;

            if (strcmp(kind, "upper") == 0) upperV[idx] = val;
            else                            lowerV[idx] = val;

            if (idx + 1 > nFaces) nFaces = idx + 1;
        }
    }

    fclose(f);

    f = fopen(rhsPath, "r");

    if (!f)
    {
        fprintf(stderr, "cannot open %s\n", rhsPath);
        exit(1);
    }

    if (fgets(line, sizeof(line), f) == NULL) { fclose(f); return; }

    while (fgets(line, sizeof(line), f))
    {
        int cell;
        double b;

        if (sscanf(line, "%d,%lf", &cell, &b) != 2) continue;
        rhs[cell] = b;
    }

    fclose(f);
}

/* ---------------------------------------------------------------------------------------- */
/* LDU -> CSR. This conversion is exactly the per-iteration tax an offload solver pays and
 * brae's LDU-gather matvec avoids (docs/memory-model.md, validation/perf/README.md). */

static void buildCsr(void)
{
    int  c, f, k;
    int* count = malloc((size_t)nCells * sizeof(int));
    int* cursor = malloc((size_t)nCells * sizeof(int));

    for (c = 0; c < nCells; ++c) count[c] = 1;                  /* the diagonal */

    for (f = 0; f < nFaces; ++f)
    {
        count[ownerA[f]] += 1;
        count[neiA[f]]   += 1;
    }

    rowPtr[0] = 0;

    for (c = 0; c < nCells; ++c)
    {
        rowPtr[c + 1] = rowPtr[c] + count[c];
        cursor[c]     = rowPtr[c];
    }

    nnz = rowPtr[nCells];

    for (c = 0; c < nCells; ++c)
    {
        colIdx[cursor[c]] = c;
        valCsr[cursor[c]] = diagV[c];
        cursor[c] += 1;
    }

    for (f = 0; f < nFaces; ++f)
    {
        const int o = ownerA[f];
        const int n = neiA[f];

        colIdx[cursor[o]] = n;
        valCsr[cursor[o]] = upperV[f];
        cursor[o] += 1;

        colIdx[cursor[n]] = o;
        valCsr[cursor[n]] = lowerV[f];
        cursor[n] += 1;
    }

    /* AMGX wants column indices sorted within a row: simple insertion sort, rows are tiny. */
    for (c = 0; c < nCells; ++c)
    {
        for (k = rowPtr[c] + 1; k < rowPtr[c + 1]; ++k)
        {
            const int    kc = colIdx[k];
            const double kv = valCsr[k];
            int          j  = k - 1;

            while (j >= rowPtr[c] && colIdx[j] > kc)
            {
                colIdx[j + 1] = colIdx[j];
                valCsr[j + 1] = valCsr[j];
                j -= 1;
            }

            colIdx[j + 1] = kc;
            valCsr[j + 1] = kv;
        }
    }

    free(count);
    free(cursor);
}

/* ---------------------------------------------------------------------------------------- */

static void residual
(
    const double* x,
    double* r
)
{
    int c, k;

    for (c = 0; c < nCells; ++c)
    {
        double s = 0;

        for (k = rowPtr[c]; k < rowPtr[c + 1]; ++k)
        {
            s += valCsr[k] * x[colIdx[k]];
        }

        r[c] = rhs[c] - s;
    }
}

/* ---------------------------------------------------------------------------------------- */

static void printCallback
(
    const char* msg,
    int length
)
{
    (void)length;
    fputs(msg, stdout);
}

/* ---------------------------------------------------------------------------------------- */

int main(int argc, char** argv)
{
    const char* caseDir   = argc > 1 ? argv[1] : "demo/amgpcg/case20";
    const int   maxCycles = argc > 2 ? atoi(argv[2]) : 30;

    char matrixPath[1024];
    char rhsPath[1024];
    char outPath[1024];

    snprintf(matrixPath, sizeof(matrixPath), "%s/trace/matrix.csv", caseDir);
    snprintf(rhsPath,    sizeof(rhsPath),    "%s/trace/rhs.csv",    caseDir);
    snprintf(outPath,    sizeof(outPath),    "%s/trace/amgx_cycles.csv", caseDir);

    char binPath[1024];
    snprintf(binPath, sizeof(binPath), "%s/trace/matrix.bin", caseDir);

    if (!readMatrixBinary(binPath))
    {
        readMatrixCsv(matrixPath, rhsPath);
    }
    else
    {
        printf("read matrix.bin\n");
    }

    buildCsr();

    printf("read matrix: %d cells, %d internal faces -> CSR nnz = %d\n", nCells, nFaces, nnz);

    AMGX_SAFE_CALL(AMGX_initialize());
    AMGX_SAFE_CALL(AMGX_register_print_callback(&printCallback));
    AMGX_SAFE_CALL(AMGX_install_signal_handler());

    double* x = malloc((size_t)nCells * sizeof(double));
    double* r = malloc((size_t)nCells * sizeof(double));
    int     c, k;

    /* Benchmark mode (maxCycles < 0): time the per-solve cost the way an OFFLOAD really pays it.
     *
     * Two numbers, and the gap between them is the whole argument for a cached hierarchy:
     *   solve        AMGX_solver_solve alone, hierarchy already built
     *   setup+solve  what OpenFOAM+AMGX actually costs per SIMPLE step -- the pressure matrix
     *                changes every step, so AMGX must re-run aggregation and the sparse RAP
     *                (AMGX_solver_setup) before every solve. brae's equivalent is amgGalerkin,
     *                a fixed atomic face scatter over a hierarchy built once per mesh. */
    if (maxCycles < 0)
    {
        const int nRepeat = -maxCycles;
        char cfgStr[1024];

        snprintf
        (
            cfgStr, sizeof(cfgStr),
            "config_version=2, solver(main)=PCG, "
            "main:preconditioner(amg)=AMG, "
            "amg:algorithm=AGGREGATION, amg:selector=SIZE_2, "
            "amg:smoother=BLOCK_JACOBI, "
            "amg:presweeps=2, amg:postsweeps=2, "
            "amg:relaxation_factor=0.75, amg:coarsest_sweeps=2, "
            "amg:max_iters=1, amg:cycle=V, "
            "amg:max_levels=50, amg:min_coarse_rows=2, "
            "main:norm=L1, main:convergence=RELATIVE_INI, "
            "main:monitor_residual=1, main:max_iters=5000, main:tolerance=1e-8"
        );

        AMGX_config_handle    cfg;
        AMGX_resources_handle rsrc;
        AMGX_matrix_handle    A;
        AMGX_vector_handle    bv, xv;
        AMGX_solver_handle    slv;

        AMGX_SAFE_CALL(AMGX_config_create(&cfg, cfgStr));
        AMGX_SAFE_CALL(AMGX_resources_create_simple(&rsrc, cfg));
        AMGX_SAFE_CALL(AMGX_matrix_create(&A,  rsrc, AMGX_mode_dDDI));
        AMGX_SAFE_CALL(AMGX_vector_create(&bv, rsrc, AMGX_mode_dDDI));
        AMGX_SAFE_CALL(AMGX_vector_create(&xv, rsrc, AMGX_mode_dDDI));
        AMGX_SAFE_CALL(AMGX_solver_create(&slv, rsrc, AMGX_mode_dDDI, cfg));

        AMGX_SAFE_CALL(AMGX_matrix_upload_all(A, nCells, nnz, 1, 1, rowPtr, colIdx, valCsr, NULL));
        AMGX_SAFE_CALL(AMGX_vector_upload(bv, nCells, 1, rhs));

        for (c = 0; c < nCells; ++c) x[c] = 0.0;
        AMGX_SAFE_CALL(AMGX_vector_upload(xv, nCells, 1, x));

        /* warm up */
        for (k = 0; k < 2; ++k)
        {
            AMGX_SAFE_CALL(AMGX_solver_setup(slv, A));
            AMGX_SAFE_CALL(AMGX_solver_solve(slv, bv, xv));
            AMGX_SAFE_CALL(AMGX_vector_upload(xv, nCells, 1, x));
        }

        double bestSolve = 1e30;
        double bestBoth  = 1e30;
        int    iters     = 0;

        for (k = 0; k < nRepeat; ++k)
        {
            for (c = 0; c < nCells; ++c) x[c] = 0.0;
            AMGX_SAFE_CALL(AMGX_vector_upload(xv, nCells, 1, x));

            const double t0 = nowMs();
            AMGX_SAFE_CALL(AMGX_matrix_replace_coefficients(A, nCells, nnz, valCsr, NULL));
            AMGX_SAFE_CALL(AMGX_solver_setup(slv, A));
            const double t1 = nowMs();
            AMGX_SAFE_CALL(AMGX_solver_solve(slv, bv, xv));
            const double t2 = nowMs();

            if (t2 - t1 < bestSolve) bestSolve = t2 - t1;
            if (t2 - t0 < bestBoth)  bestBoth  = t2 - t0;

            AMGX_SAFE_CALL(AMGX_solver_get_iterations_number(slv, &iters));
        }

        AMGX_SAFE_CALL(AMGX_vector_download(xv, x));
        residual(x, r);

        double resSum = 0, bSum = 0;

        for (c = 0; c < nCells; ++c)
        {
            resSum += (r[c] < 0 ? -r[c] : r[c]);
            bSum   += (rhs[c] < 0 ? -rhs[c] : rhs[c]);
        }

        printf("BENCH amgx  cells=%d  iters=%d  solve=%.2fms  setup+solve=%.2fms  "
               "ms/cycle=%.3f  final=%.3e\n",
               nCells, iters, bestSolve, bestBoth,
               bestSolve / (iters > 0 ? iters : 1), resSum / bSum);

        AMGX_SAFE_CALL(AMGX_solver_destroy(slv));
        AMGX_SAFE_CALL(AMGX_vector_destroy(xv));
        AMGX_SAFE_CALL(AMGX_vector_destroy(bv));
        AMGX_SAFE_CALL(AMGX_matrix_destroy(A));
        AMGX_SAFE_CALL(AMGX_resources_destroy(rsrc));
        AMGX_SAFE_CALL(AMGX_config_destroy(cfg));
        AMGX_SAFE_CALL(AMGX_finalize());
        return 0;
    }

    FILE* os = fopen(outPath, "w");
    fprintf(os, "cycle,cell,psi,residual,resnorm\n");

    for (k = 0; k <= maxCycles; ++k)
    {
        char cfgStr[1024];

        snprintf
        (
            cfgStr, sizeof(cfgStr),
            "config_version=2, solver(main)=PCG, "
            "main:preconditioner(amg)=AMG, "
            "amg:algorithm=AGGREGATION, amg:selector=SIZE_2, "
            "amg:smoother=BLOCK_JACOBI, "
            "amg:presweeps=2, amg:postsweeps=2, "
            "amg:relaxation_factor=0.75, amg:coarsest_sweeps=2, "
            "amg:max_iters=1, amg:cycle=V, "
            "amg:max_levels=50, amg:min_coarse_rows=2, "
            "main:norm=L2, main:convergence=RELATIVE_INI, "
            "main:monitor_residual=1, main:store_res_history=1, "
            "main:max_iters=%d, main:tolerance=0.0, %s",
            k > 0 ? k : 1,
            k == 0 ? "main:print_grid_stats=1" : "main:print_solve_stats=0"
        );

        AMGX_config_handle    cfg;
        AMGX_resources_handle rsrc;
        AMGX_matrix_handle    A;
        AMGX_vector_handle    bv, xv;
        AMGX_solver_handle    slv;

        AMGX_SAFE_CALL(AMGX_config_create(&cfg, cfgStr));
        AMGX_SAFE_CALL(AMGX_resources_create_simple(&rsrc, cfg));
        AMGX_SAFE_CALL(AMGX_matrix_create(&A,  rsrc, AMGX_mode_dDDI));
        AMGX_SAFE_CALL(AMGX_vector_create(&bv, rsrc, AMGX_mode_dDDI));
        AMGX_SAFE_CALL(AMGX_vector_create(&xv, rsrc, AMGX_mode_dDDI));
        AMGX_SAFE_CALL(AMGX_solver_create(&slv, rsrc, AMGX_mode_dDDI, cfg));

        for (c = 0; c < nCells; ++c) x[c] = 0.0;

        AMGX_SAFE_CALL(AMGX_matrix_upload_all(A, nCells, nnz, 1, 1, rowPtr, colIdx, valCsr, NULL));
        AMGX_SAFE_CALL(AMGX_vector_upload(bv, nCells, 1, rhs));
        AMGX_SAFE_CALL(AMGX_vector_upload(xv, nCells, 1, x));
        AMGX_SAFE_CALL(AMGX_solver_setup(slv, A));

        if (k > 0)
        {
            AMGX_SAFE_CALL(AMGX_solver_solve(slv, bv, xv));
            AMGX_SAFE_CALL(AMGX_vector_download(xv, x));
        }

        residual(x, r);

        double resNorm = 0;
        double bNorm   = 0;

        for (c = 0; c < nCells; ++c)
        {
            resNorm += (r[c] < 0 ? -r[c] : r[c]);
            bNorm   += (rhs[c] < 0 ? -rhs[c] : rhs[c]);
        }

        resNorm /= (bNorm > 0 ? bNorm : 1.0);

        for (c = 0; c < nCells; ++c)
        {
            fprintf(os, "%d,%d,%.16e,%.16e,%.16e\n", k, c, x[c], r[c], resNorm);
        }

        AMGX_SAFE_CALL(AMGX_solver_destroy(slv));
        AMGX_SAFE_CALL(AMGX_vector_destroy(xv));
        AMGX_SAFE_CALL(AMGX_vector_destroy(bv));
        AMGX_SAFE_CALL(AMGX_matrix_destroy(A));
        AMGX_SAFE_CALL(AMGX_resources_destroy(rsrc));
        AMGX_SAFE_CALL(AMGX_config_destroy(cfg));
    }

    fclose(os);
    AMGX_SAFE_CALL(AMGX_finalize());

    printf("wrote %s (%d cycles)\n", outPath, maxCycles);
    return 0;
}
