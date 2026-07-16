// Phase 4a EXIT GATE: the OOM proof -- a mesh too big for ONE GPU's memory runs SPLIT across two.
//
// This is the question that started the multi-GPU work: brae is device-resident, so the mesh it can solve is
// capped by one GPU's VRAM. The claim to prove is that N GPUs raise that cap ~N-fold, i.e. per-rank device
// memory scales as 1/nproc and a mesh that OOMs at np=1 completes at np=2.
//
// The mesh is generated IN MEMORY (a polyMesh big enough to OOM an 80GB H100 would be tens of GB of text):
// a structured Nx*Ny*Nz hex box, duct flow -- inlet (xmin, fixedValue U), outlet (xmax, fixedValue p, which
// keeps the pressure equation non-singular), four no-slip walls.
//
// Decomposition is a SLAB in x (contiguous cells per rank, one processor interface per neighbour) rather than
// Scotch: graph-partitioning tens of millions of cells would dominate the run and is not what is under test.
//
// Reports peak per-rank device memory, so np=1 vs np=2 shows the 1/nproc scaling directly.
//
// Run: mpirun -np {1,2} test_gpu_parallel_oom <Nx> <Ny> <Nz> [nIters]
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvc.cuh"
#include "parallel_simple.cuh"
#include "field_distribute.cuh"
#include "parallel_device_simple.cuh"
#include "cf_pstream.cuh"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

namespace {

// A structured Nx*Ny*Nz unit-spaced hex box. Faces are emitted in OpenFOAM order: internal faces first in
// upper-triangular order (owner ascending, neighbour ascending within an owner -- the lduAddressing contract),
// then one contiguous block per boundary patch. Face vertex winding is chosen so Sf points owner->neighbour
// for internal faces and outward for boundary faces (right-hand rule).
PrimitiveMesh boxMesh(label Nx, label Ny, label Nz)
{
    const label nC = Nx * Ny * Nz;
    auto cell  = [&](label i, label j, label k) { return i + Nx * (j + Ny * k); };
    auto point = [&](label i, label j, label k) { return i + (Nx + 1) * (j + (Ny + 1) * k); };

    std::vector<vector> pts(static_cast<std::size_t>(Nx + 1) * (Ny + 1) * (Nz + 1));
    for (label k = 0; k <= Nz; ++k)
        for (label j = 0; j <= Ny; ++j)
            for (label i = 0; i <= Nx; ++i)
                pts[point(i, j, k)] = vector{scalar(i), scalar(j), scalar(k)};

    // reserve: at ~1e8 cells these reach ~1e9 entries, and push_back's doubling would otherwise transiently
    // double an already multi-GB allocation. nFaces = 3*nC + the 6 boundary planes; nInternal = nFaces - nBnd.
    const std::size_t nBnd   = static_cast<std::size_t>(2) * (Ny * Nz + Nx * Nz + Nx * Ny);
    const std::size_t nFaces = static_cast<std::size_t>(3) * nC
                             - static_cast<std::size_t>(Ny * Nz + Nx * Nz + Nx * Ny) + nBnd;
    std::vector<label> fv, foff, own, nei;
    fv.reserve(4 * nFaces);
    foff.reserve(nFaces + 1);
    own.reserve(nFaces);
    nei.reserve(nFaces - nBnd);
    foff.push_back(0);
    auto quad = [&](label a, label b, label c, label d, label o, label n)
    {
        fv.push_back(a); fv.push_back(b); fv.push_back(c); fv.push_back(d);
        foff.push_back(static_cast<label>(fv.size()));
        own.push_back(o);
        if (n >= 0) nei.push_back(n);
    };
    // face planes with the winding that yields the +x / +y / +z normal
    auto faceX = [&](label i, label j, label k, label o, label n)   // plane x=i
    { quad(point(i,j,k), point(i,j+1,k), point(i,j+1,k+1), point(i,j,k+1), o, n); };
    auto faceY = [&](label i, label j, label k, label o, label n)   // plane y=j
    { quad(point(i,j,k), point(i,j,k+1), point(i+1,j,k+1), point(i+1,j,k), o, n); };
    auto faceZ = [&](label i, label j, label k, label o, label n)   // plane z=k
    { quad(point(i,j,k), point(i+1,j,k), point(i+1,j+1,k), point(i,j+1,k), o, n); };
    // reversed winding -> the opposite normal (for min-side boundary patches, whose outward normal is -)
    auto faceXr = [&](label i, label j, label k, label o)
    { quad(point(i,j,k+1), point(i,j+1,k+1), point(i,j+1,k), point(i,j,k), o, -1); };
    auto faceYr = [&](label i, label j, label k, label o)
    { quad(point(i+1,j,k), point(i+1,j,k+1), point(i,j,k+1), point(i,j,k), o, -1); };
    auto faceZr = [&](label i, label j, label k, label o)
    { quad(point(i,j+1,k), point(i+1,j+1,k), point(i+1,j,k), point(i,j,k), o, -1); };

    // internal faces, upper-triangular: for each cell ascending, neighbours +1 < +Nx < +Nx*Ny
    for (label k = 0; k < Nz; ++k)
        for (label j = 0; j < Ny; ++j)
            for (label i = 0; i < Nx; ++i)
            {
                const label c = cell(i, j, k);
                if (i + 1 < Nx) faceX(i + 1, j, k, c, cell(i + 1, j, k));
                if (j + 1 < Ny) faceY(i, j + 1, k, c, cell(i, j + 1, k));
                if (k + 1 < Nz) faceZ(i, j, k + 1, c, cell(i, j, k + 1));
            }

    std::vector<PatchInfo> patches;
    auto beginPatch = [&](const char* name, const char* type)
    {
        PatchInfo pi;
        pi.name  = name;
        pi.type  = type;
        pi.start = static_cast<label>(own.size());
        patches.push_back(pi);
    };
    auto endPatch = [&]() { patches.back().size = static_cast<label>(own.size()) - patches.back().start; };

    beginPatch("inlet", "patch");                                          // xmin, outward -x
    for (label k = 0; k < Nz; ++k) for (label j = 0; j < Ny; ++j) faceXr(0, j, k, cell(0, j, k));
    endPatch();
    beginPatch("outlet", "patch");                                         // xmax, outward +x
    for (label k = 0; k < Nz; ++k) for (label j = 0; j < Ny; ++j) faceX(Nx, j, k, cell(Nx - 1, j, k), -1);
    endPatch();
    beginPatch("wallYmin", "wall");                                        // outward -y
    for (label k = 0; k < Nz; ++k) for (label i = 0; i < Nx; ++i) faceYr(i, 0, k, cell(i, 0, k));
    endPatch();
    beginPatch("wallYmax", "wall");                                        // outward +y
    for (label k = 0; k < Nz; ++k) for (label i = 0; i < Nx; ++i) faceY(i, Ny, k, cell(i, Ny - 1, k), -1);
    endPatch();
    beginPatch("wallZmin", "wall");                                        // outward -z
    for (label j = 0; j < Ny; ++j) for (label i = 0; i < Nx; ++i) faceZr(i, j, 0, cell(i, j, 0));
    endPatch();
    beginPatch("wallZmax", "wall");                                        // outward +z
    for (label j = 0; j < Ny; ++j) for (label i = 0; i < Nx; ++i) faceZ(i, j, Nz, cell(i, j, Nz - 1), -1);
    endPatch();

    PrimitiveMesh m;
    m.assign(std::move(pts), std::move(fv), std::move(foff), std::move(own), std::move(nei),
             std::move(patches), nC);
    return m;
}

template <typename T>
PatchFieldData<T> pfd(const char* name, const char* type, T v, bool hasValue)
{
    PatchFieldData<T> d;
    d.name         = name;
    d.type         = type;
    d.hasValue     = hasValue;
    d.valueUniform = hasValue;
    d.uniformValue = v;
    return d;
}

}  // namespace

int main(int argc, char** argv)
{
    Pstream::init(argc, argv);
    const int rank = Pstream::myProcNo(), nproc = Pstream::nProcs();
    if (argc < 4)
    {
        if (Pstream::master()) std::printf("usage: %s <Nx> <Ny> <Nz> [nIters]\n", argv[0]);
        Pstream::finalize();
        return 2;
    }
    if (!Pstream::nvshmemActive())
    {
        if (Pstream::master()) std::printf("test_gpu_parallel_oom: NVSHMEM inactive -- SKIP\n");
        Pstream::finalize();
        return 0;
    }
    const label Nx = std::atol(argv[1]), Ny = std::atol(argv[2]), Nz = std::atol(argv[3]);
    const int   N  = (argc > 4) ? std::atoi(argv[4]) : 2;
    const scalar nu = 1e-3, relaxU = 0.7, relaxP = 0.3, tolU = 1e-6, tolP = 1e-6;
    const int maxIter = 50;   // memory is under test here, not convergence

    const label nC = Nx * Ny * Nz;
    if (Pstream::master())
        std::printf("test_gpu_parallel_oom np=%d: %ldx%ld%s%ld = %.3f M cells, %d iters\n",
                    nproc, (long)Nx, (long)Ny, "x", (long)Nz, nC / 1e6, N);

    std::size_t freeB = 0, totB = 0;
    cudaMemGetInfo(&freeB, &totB);
    const std::size_t before = totB - freeB;

    {
        const PrimitiveMesh gm = boxMesh(Nx, Ny, Nz);
        // slab decomposition in x: contiguous cells, one processor interface between adjacent slabs
        std::vector<label> cellToPart(nC);
        for (label k = 0; k < Nz; ++k)
            for (label j = 0; j < Ny; ++j)
                for (label i = 0; i < Nx; ++i)
                    cellToPart[i + Nx * (j + Ny * k)] = static_cast<label>((static_cast<long long>(i) * nproc) / Nx);
        const Partition P(gm, cellToPart, rank);

        FieldData<vector> Ufd;
        Ufd.internalUniform = true;
        Ufd.internalUniformValue = vector{0, 0, 0};
        Ufd.boundary.push_back(pfd<vector>("inlet",    "fixedValue",   vector{1, 0, 0}, true));
        Ufd.boundary.push_back(pfd<vector>("outlet",   "zeroGradient", vector{0, 0, 0}, false));
        Ufd.boundary.push_back(pfd<vector>("wallYmin", "fixedValue",   vector{0, 0, 0}, true));
        Ufd.boundary.push_back(pfd<vector>("wallYmax", "fixedValue",   vector{0, 0, 0}, true));
        Ufd.boundary.push_back(pfd<vector>("wallZmin", "fixedValue",   vector{0, 0, 0}, true));
        Ufd.boundary.push_back(pfd<vector>("wallZmax", "fixedValue",   vector{0, 0, 0}, true));

        FieldData<scalar> pfdta;
        pfdta.internalUniform = true;
        pfdta.internalUniformValue = 0.0;
        pfdta.boundary.push_back(pfd<scalar>("inlet",    "zeroGradient", 0.0, false));
        pfdta.boundary.push_back(pfd<scalar>("outlet",   "fixedValue",   0.0, true));   // pressure reference
        pfdta.boundary.push_back(pfd<scalar>("wallYmin", "zeroGradient", 0.0, false));
        pfdta.boundary.push_back(pfd<scalar>("wallYmax", "zeroGradient", 0.0, false));
        pfdta.boundary.push_back(pfd<scalar>("wallZmin", "zeroGradient", 0.0, false));
        pfdta.boundary.push_back(pfd<scalar>("wallZmax", "zeroGradient", 0.0, false));

        GeometricField<vector> U0 = distributeField<vector>(Ufd, gm.patches(), P.Lm, P.lp, P.procW, rank);
        U0.evaluateBoundary();
        GeometricField<scalar> p0 = distributeField<scalar>(pfdta, gm.patches(), P.Lm, P.lp, P.procW, rank);
        p0.evaluateBoundary();

        ParallelDeviceSimple solver(P, U0, p0, nu, relaxU, relaxP, tolU, tolP, maxIter);
        for (int it = 0; it < N; ++it) solver.step();
        cudaDeviceSynchronize();

        const std::vector<scalar> pl = solver.p();
        scalar s = 0;
        for (std::size_t c = 0; c < pl.size(); ++c) s += pl[c];
        const scalar tot = Pstream::allReduce(s, ReduceOp::Sum);

        cudaMemGetInfo(&freeB, &totB);
        const scalar usedGiB = (totB - freeB - before) / (1024.0 * 1024.0 * 1024.0);
        const scalar maxGiB  = Pstream::allReduce(usedGiB, ReduceOp::Max);
        const label  maxCells = Pstream::allReduce(static_cast<scalar>(P.nCells()), ReduceOp::Max);

        if (Pstream::master())
        {
            std::printf("  per-rank cells (max): %ld\n", (long)maxCells);
            std::printf("  per-rank device mem (max): %.2f GiB\n", maxGiB);
            std::printf("  sum(p) = %.6e %s\n", tot, std::isfinite(tot) ? "(finite)" : "(NON-FINITE!)");
            std::printf("%s\n", std::isfinite(tot) ? "PASS" : "FAIL");
        }
    }   // buffers must die before finalize

    Pstream::finalize();
    return 0;
}
