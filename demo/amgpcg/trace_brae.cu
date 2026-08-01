// Demo instrumentation for the AMG-PCG showcase (demo/amgpcg).
//
// Reads the SAME matrix OpenFOAM dumped (case20/trace/matrix.csv + rhs.csv) so that brae,
// OpenFOAM GAMG and AMGX provably solve one identical linear system, then dumps as CSV:
//   brae_hierarchy.csv   the REAL pairwise agglomeration, level by level, cell by cell
//   brae_levels.csv      the REAL Galerkin coarse matrices (one row per coarse coefficient)
//   brae_cycles.csv      the REAL AMG-PCG state after each Krylov iteration (psi + residual)
//
// The per-iteration trace re-solves from psi = 0 with maxIter = 1, 2, 3, ... CG is a
// deterministic recurrence from a fixed start, so the state a maxIter=k solve stops at is
// exactly the state the k-th iteration of a long solve passes through. Every number in
// brae_cycles.csv is therefore a genuine brae solver output.

#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "fvm.cuh"
#include "device_buffer.cuh"
#include "device_ldu.cuh"
#include "device_pcg.cuh"
#include "device_amg.cuh"

#include <memory>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>

using namespace brae;

// ------------------------------------------------------------------------------------- //

struct HostMatrix
{
    int nCells = 0;
    int nFaces = 0;
    std::vector<scalar> diag;
    std::vector<scalar> upper;
    std::vector<scalar> lower;
    std::vector<scalar> b;
    std::vector<label>  owner;
    std::vector<label>  neighbour;
};

// ------------------------------------------------------------------------------------- //
// matrix.csv: kind,index,owner,neighbour,value  with kind in {diag, upper, lower}.

static HostMatrix readMatrixCsv
(
    const std::string& matrixPath,
    const std::string& rhsPath
)
{
    HostMatrix M;
    std::ifstream in(matrixPath);

    if (!in)
    {
        std::fprintf(stderr, "cannot open %s\n", matrixPath.c_str());
        std::exit(1);
    }

    std::string line;
    std::getline(in, line);   // header

    while (std::getline(in, line))
    {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string kind, sIdx, sOwn, sNei, sVal;

        std::getline(ss, kind, ',');
        std::getline(ss, sIdx, ',');
        std::getline(ss, sOwn, ',');
        std::getline(ss, sNei, ',');
        std::getline(ss, sVal, ',');

        const int    idx = std::atoi(sIdx.c_str());
        const scalar val = std::atof(sVal.c_str());

        if (kind == "diag")
        {
            if (static_cast<int>(M.diag.size()) <= idx) M.diag.resize(idx + 1);
            M.diag[idx] = val;
        }
        else
        {
            const label own = std::atoi(sOwn.c_str());
            const label nei = std::atoi(sNei.c_str());

            if (static_cast<int>(M.upper.size()) <= idx)
            {
                M.upper.resize(idx + 1);
                M.lower.resize(idx + 1);
                M.owner.resize(idx + 1);
                M.neighbour.resize(idx + 1);
            }

            M.owner[idx]     = own;
            M.neighbour[idx] = nei;

            if (kind == "upper") M.upper[idx] = val;
            else                 M.lower[idx] = val;
        }
    }

    M.nCells = static_cast<int>(M.diag.size());
    M.nFaces = static_cast<int>(M.upper.size());

    std::ifstream rin(rhsPath);

    if (!rin)
    {
        std::fprintf(stderr, "cannot open %s\n", rhsPath.c_str());
        std::exit(1);
    }

    std::getline(rin, line);   // header
    M.b.assign(M.nCells, 0.0);

    while (std::getline(rin, line))
    {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string sCell, sB;

        std::getline(ss, sCell, ',');
        std::getline(ss, sB, ',');

        const int c = std::atoi(sCell.c_str());
        if (c >= 0 && c < M.nCells) M.b[c] = std::atof(sB.c_str());
    }

    return M;
}

// ------------------------------------------------------------------------------------- //

// ------------------------------------------------------------------------------------- //
// Fallback for meshes too large to round-trip through matrix.csv: assemble the same
// -laplacian(1, T) system with brae's own fvm. The boundary conditions are the demo case's
// (see case20/0/T), keyed by patch name, so the assembled system matches what the OpenFOAM
// app builds on the same mesh.

static HostMatrix assembleLaplacian
(
    const PrimitiveMesh& mesh,
    const FvGeometry& geom,
    const std::vector<FvPatch>& patches
)
{
    const label nC = mesh.nCells();
    const label nIf = mesh.nInternalFaces();

    GeometricField<scalar> T;
    T.internal.assign(nC, 0.0);

    for (const FvPatch& p : patches)
    {
        scalar value = 0.0;
        bool fixed = true;

        if (p.name == "left")        value = 1.0;
        else if (p.name == "right")  value = 0.0;
        else if (p.name == "top")    value = 0.5;
        else                         fixed = false;

        if (p.type == "empty")
        {
            T.boundary.push_back(std::make_unique<EmptyPatchField<scalar>>(p));
        }
        else if (!fixed)
        {
            T.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(p));
        }
        else
        {
            T.boundary.push_back
            (
                std::make_unique<FixedValuePatchField<scalar>>(p, true, value,
                                                              std::vector<scalar>{})
            );
        }
    }

    T.evaluateBoundary();

    // brae's fvm::laplacian carries the opposite sign to the OpenFOAM app's -fvm::laplacian,
    // so negate the assembled coefficients to land on the same positive-definite operator.
    FvScalarMatrix L = fvm::laplacian(T, 1.0, mesh, geom, patches);

    HostMatrix M;
    M.nCells = nC;
    M.nFaces = nIf;
    M.diag.assign(nC, 0.0);
    M.b.assign(nC, 0.0);
    M.upper.resize(nIf);
    M.lower.resize(nIf);

    for (label c = 0; c < nC; ++c)
    {
        M.diag[c] = -L.diag[c];
        M.b[c] = -L.source[c];
    }

    for (label f = 0; f < nIf; ++f)
    {
        M.upper[f] = -L.upper[f];
        M.lower[f] = -L.lower[f];
    }

    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        for (label i = 0; i < patches[pi].size; ++i)
        {
            const label c = patches[pi].faceCells[i];
            M.diag[c] += -L.internalCoeffs[pi][i];
            M.b[c] += -L.boundaryCoeffs[pi][i];
        }
    }

    M.owner.assign(mesh.owner().begin(), mesh.owner().begin() + nIf);
    M.neighbour = mesh.neighbour();

    return M;
}

// ------------------------------------------------------------------------------------- //
// Compact binary dump of the assembled system, so the AMGX tool can read a million-cell
// matrix without parsing a multi-hundred-megabyte CSV.

static void writeMatrixBinary
(
    const HostMatrix& M,
    const std::string& path
)
{
    std::ofstream os(path, std::ios::binary);

    os.write(reinterpret_cast<const char*>(&M.nCells), sizeof(int));
    os.write(reinterpret_cast<const char*>(&M.nFaces), sizeof(int));
    os.write(reinterpret_cast<const char*>(M.diag.data()), M.nCells * sizeof(scalar));
    os.write(reinterpret_cast<const char*>(M.b.data()), M.nCells * sizeof(scalar));
    os.write(reinterpret_cast<const char*>(M.upper.data()), M.nFaces * sizeof(scalar));
    os.write(reinterpret_cast<const char*>(M.lower.data()), M.nFaces * sizeof(scalar));
    os.write(reinterpret_cast<const char*>(M.owner.data()), M.nFaces * sizeof(label));
    os.write(reinterpret_cast<const char*>(M.neighbour.data()), M.nFaces * sizeof(label));

    std::printf("wrote %s (%d cells, %d faces)\n", path.c_str(), M.nCells, M.nFaces);
}

// ------------------------------------------------------------------------------------- //

static std::vector<scalar> residual
(
    const HostMatrix& M,
    const std::vector<scalar>& x
)
{
    std::vector<scalar> r(M.nCells);

    for (int c = 0; c < M.nCells; ++c)
    {
        r[c] = M.b[c] - M.diag[c] * x[c];
    }

    for (int f = 0; f < M.nFaces; ++f)
    {
        r[M.owner[f]]     -= M.upper[f] * x[M.neighbour[f]];
        r[M.neighbour[f]] -= M.lower[f] * x[M.owner[f]];
    }

    return r;
}

// ------------------------------------------------------------------------------------- //
// The real pairwise agglomeration. level[L].map[c] is the level-(L+1) cell that level-L
// cell c is merged into. We also chain the maps back to the finest grid so the animation
// can colour the original 20 cells by their aggregate at any level.

static void writeHierarchy
(
    const AMGData& amg,
    int nFine,
    const std::string& path
)
{
    std::ofstream os(path);
    os << "level,fine_cell,coarse_cell,finest_cell\n";

    std::vector<label> chain(nFine);

    for (int c = 0; c < nFine; ++c)
    {
        chain[c] = c;
    }

    for (int L = 0; L < amg.nLevels(); ++L)
    {
        const std::vector<label> map = amg.level[L].map.host();

        for (std::size_t c = 0; c < map.size(); ++c)
        {
            os << L << ',' << c << ',' << map[c] << ",-1\n";
        }

        for (int c0 = 0; c0 < nFine; ++c0)
        {
            chain[c0] = map[chain[c0]];
            os << L << ",-1," << chain[c0] << ',' << c0 << '\n';
        }

        std::printf
        (
            "  level %d: %d -> %d cells (%d coarse faces)\n",
            L,
            amg.level[L].nFine,
            amg.level[L].nCoarse,
            amg.level[L].nCoarseFaces
        );
    }

    std::printf("wrote %s (%d levels)\n", path.c_str(), amg.nLevels());
}

// ------------------------------------------------------------------------------------- //
// The Galerkin coarse operators, in the same long CSV form as the fine matrix, so the
// animation can draw A, A_1, A_2, ... with one code path.

static void writeLevelMatrices
(
    const AMGData& amg,
    const HostMatrix& M,
    const std::string& path
)
{
    std::ofstream os(path);
    os.precision(16);
    os << "grid,kind,index,owner,neighbour,value\n";

    for (int c = 0; c < M.nCells; ++c)
    {
        os << "0,diag," << c << ',' << c << ',' << c << ',' << M.diag[c] << '\n';
    }

    for (int f = 0; f < M.nFaces; ++f)
    {
        os << "0,upper," << f << ',' << M.owner[f] << ',' << M.neighbour[f] << ',' << M.upper[f] << '\n';
        os << "0,lower," << f << ',' << M.owner[f] << ',' << M.neighbour[f] << ',' << M.lower[f] << '\n';
    }

    for (int L = 0; L < amg.nLevels(); ++L)
    {
        const AMGLevel& lv = amg.level[L];

        const std::vector<scalar> cd = lv.cDiag.host();
        const std::vector<scalar> cu = lv.cUpper.host();
        const std::vector<scalar> cl = lv.cLower.host();
        const std::vector<label>  co = lv.cOwn.host();
        const std::vector<label>  cn = lv.cNei.host();

        const int g = L + 1;

        for (std::size_t c = 0; c < cd.size(); ++c)
        {
            os << g << ",diag," << c << ',' << c << ',' << c << ',' << cd[c] << '\n';
        }

        for (std::size_t f = 0; f < cu.size(); ++f)
        {
            os << g << ",upper," << f << ',' << co[f] << ',' << cn[f] << ',' << cu[f] << '\n';
            os << g << ",lower," << f << ',' << co[f] << ',' << cn[f] << ',' << cl[f] << '\n';
        }
    }

    std::printf("wrote %s\n", path.c_str());
}

// ------------------------------------------------------------------------------------- //

int main(int argc, char** argv)
{
    const std::string caseDir   = argc > 1 ? argv[1] : "demo/amgpcg/case20";
    const int         maxCycles = argc > 2 ? std::atoi(argv[2]) : 30;

    // The whole-loop conditional-graph PCG caches its graph exec keyed on psi.data() only, and
    // maxIter is a kernel parameter baked into the captured pcgSetCondK node. This tool calls the
    // solver repeatedly with a GROWING maxIter and a recycled psi buffer, which would replay the
    // first capture's iteration bound forever. The host-driven loop is documented as producing the
    // identical iteration count and psi (device_amg.cu:2549-2556), so trace on that path.
    setenv("BRAE_PCG_DEVICE", "0", 1);

    const std::string traceDir = caseDir + "/trace";

    // Face weights for the agglomeration are |Sf| on the internal faces, exactly what the
    // solver driver passes (device_simple_foam.cuh:216).
    PrimitiveMesh mesh;
    mesh.read(caseDir + "/constant/polyMesh");

    FvGeometry geom;
    geom.build(mesh);

    HostMatrix M;
    std::ifstream probe(traceDir + "/matrix.csv");

    if (probe)
    {
        probe.close();
        M = readMatrixCsv(traceDir + "/matrix.csv", traceDir + "/rhs.csv");
        std::printf("read matrix from CSV: %d cells, %d internal faces\n", M.nCells, M.nFaces);
    }
    else
    {
        const std::vector<FvPatch> patches = buildPatches(mesh, geom);
        M = assembleLaplacian(mesh, geom, patches);
        std::printf("assembled matrix: %d cells, %d internal faces\n", M.nCells, M.nFaces);

        writeMatrixBinary(M, traceDir + "/matrix.bin");
    }

    const std::vector<scalar> magSfInt(geom.magSf().begin(), geom.magSf().begin() + M.nFaces);

    scalar normFactor = 0;

    for (int c = 0; c < M.nCells; ++c)
    {
        normFactor += std::fabs(M.b[c]);
    }

    normFactor += 1e-20;

    DeviceLduMatrix dM = buildDeviceLdu(M.diag, M.upper, M.lower, M.owner, M.neighbour, M.nCells);
    DeviceBuffer<scalar> db(M.b);

    AMGData amg = buildAMG(M.owner, M.neighbour, magSfInt, M.nCells);
    amgGalerkin(amg, dM.diag, dM.upper, dM.lower);

    // Correction scaling is a deviceAMGPCG ARGUMENT, not an env read inside the library (only
    // gpuSimpleFoam wires BRAE_CORR_SCALING through). Read it here too so this harness can measure
    // it -- it forces the host-driven flexible-CG path (BRAE_PCG_DEVICE is bypassed when on).
    const bool corrScaling = std::getenv("BRAE_CORR_SCALING") != nullptr;

    // Benchmark mode (maxCycles < 0): time the PER-SOLVE cost, which is the number that decides
    // whether a config change is worth it. The agglomeration hierarchy is built ONCE outside the
    // timing loop -- that is how a real run behaves (static mesh, and -partition caches it to
    // disk across runs), so charging its setup to a single solve would be misleading.
    //
    // Two numbers are reported:
    //   solve        just deviceAMGPCG
    //   galerkin+solve  the honest per-SIMPLE-step cost: the coarse operators are re-evaluated
    //                   from the current fine matrix every step, and that re-evaluation is more
    //                   expensive under smoothed aggregation (general RAP scatter vs face scatter)
    if (maxCycles < 0)
    {
        const int nRepeat = -maxCycles;

        std::printf("  levels: %d\n", amg.nLevels());

        DeviceBuffer<scalar> dx(std::vector<scalar>(M.nCells, 0.0));
        DeviceSolverPerf perf;

        // warm up: first call pays graph capture, FP32 mirror allocation and autotuning
        for (int w = 0; w < 3; ++w)
        {
            dx.copyFrom(std::vector<scalar>(M.nCells, 0.0));
            perf = deviceAMGPCG(dM.view(), amg, db, dx, normFactor, 1e-8, 0.0, 5000, false, 1, corrScaling);
        }

        cudaDeviceSynchronize();

        double bestSolve = 1e30;
        double bestBoth = 1e30;

        for (int rep = 0; rep < nRepeat; ++rep)
        {
            dx.copyFrom(std::vector<scalar>(M.nCells, 0.0));
            cudaDeviceSynchronize();

            const auto t0 = std::chrono::high_resolution_clock::now();
            amgGalerkin(amg, dM.diag, dM.upper, dM.lower);
            cudaDeviceSynchronize();

            const auto t1 = std::chrono::high_resolution_clock::now();
            perf = deviceAMGPCG(dM.view(), amg, db, dx, normFactor, 1e-8, 0.0, 5000, false, 1, corrScaling);
            cudaDeviceSynchronize();

            const auto t2 = std::chrono::high_resolution_clock::now();

            const double solveMs =
                std::chrono::duration<double, std::milli>(t2 - t1).count();
            const double bothMs =
                std::chrono::duration<double, std::milli>(t2 - t0).count();

            bestSolve = std::fmin(bestSolve, solveMs);
            bestBoth = std::fmin(bestBoth, bothMs);
        }

        std::printf
        (
            "BENCH brae  cells=%d  iters=%d  solve=%.2fms  galerkin+solve=%.2fms  "
            "ms/cycle=%.3f  final=%.3e\n",
            M.nCells,
            perf.nIterations,
            bestSolve,
            bestBoth,
            bestSolve / std::max(1, perf.nIterations),
            perf.finalResidual
        );

        return 0;
    }

    // Measure-only mode (maxCycles = 0): one solve to a fixed tolerance, report the iteration
    // count and nothing else. The per-cycle dump below is O(cells x cycles) rows, which is fine
    // at 20 or 1024 cells and unusable at a million.
    if (maxCycles == 0)
    {
        std::printf("  levels: %d\n", amg.nLevels());

        for (int L = 0; L < amg.nLevels(); ++L)
        {
            std::printf("    %d -> %d\n", amg.level[L].nFine, amg.level[L].nCoarse);
        }

        DeviceBuffer<scalar> dx(std::vector<scalar>(M.nCells, 0.0));

        const DeviceSolverPerf perf =
            deviceAMGPCG(dM.view(), amg, db, dx, normFactor, 1e-8, 0.0, 5000, false, 1, corrScaling);

        const std::vector<scalar> x = dx.host();
        const std::vector<scalar> r = residual(M, x);

        scalar resSum = 0;
        scalar bSum = 0;

        for (int c = 0; c < M.nCells; ++c)
        {
            resSum += std::fabs(r[c]);
            bSum += std::fabs(M.b[c]);
        }

        std::printf
        (
            "MEASURE brae  cells=%d  iters=%d  init=%.4e  final=%.4e  |r|1/|b|1=%.4e\n",
            M.nCells,
            perf.nIterations,
            perf.initialResidual,
            perf.finalResidual,
            resSum / bSum
        );

        return 0;
    }

    writeHierarchy(amg, M.nCells, traceDir + "/brae_hierarchy.csv");
    writeLevelMatrices(amg, M, traceDir + "/brae_levels.csv");

    std::ofstream os(traceDir + "/brae_cycles.csv");
    os.precision(16);
    os << "cycle,cell,psi,residual,resnorm\n";

    for (int k = 0; k <= maxCycles; ++k)
    {
        std::vector<scalar> x(M.nCells, 0.0);
        scalar resNorm = 1;

        if (k > 0)
        {
            DeviceBuffer<scalar> dx(x);

            const DeviceSolverPerf perf =
                deviceAMGPCG(dM.view(), amg, db, dx, normFactor, 0.0, 0.0, k);

            x       = dx.host();
            resNorm = perf.finalResidual;
        }

        const std::vector<scalar> r = residual(M, x);

        for (int c = 0; c < M.nCells; ++c)
        {
            os << k << ',' << c << ',' << x[c] << ',' << r[c] << ',' << resNorm << '\n';
        }
    }

    std::printf("wrote %s (%d cycles)\n", (traceDir + "/brae_cycles.csv").c_str(), maxCycles);
    return 0;
}
