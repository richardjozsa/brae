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
#include "box_mesh.cuh"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;



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
        const PrimitiveMesh gm = boxtest::boxMesh(Nx, Ny, Nz);
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
        Ufd.boundary.push_back(boxtest::pfd<vector>("inlet",    "fixedValue",   vector{1, 0, 0}, true));
        Ufd.boundary.push_back(boxtest::pfd<vector>("outlet",   "zeroGradient", vector{0, 0, 0}, false));
        Ufd.boundary.push_back(boxtest::pfd<vector>("wallYmin", "fixedValue",   vector{0, 0, 0}, true));
        Ufd.boundary.push_back(boxtest::pfd<vector>("wallYmax", "fixedValue",   vector{0, 0, 0}, true));
        Ufd.boundary.push_back(boxtest::pfd<vector>("wallZmin", "fixedValue",   vector{0, 0, 0}, true));
        Ufd.boundary.push_back(boxtest::pfd<vector>("wallZmax", "fixedValue",   vector{0, 0, 0}, true));

        FieldData<scalar> pfdta;
        pfdta.internalUniform = true;
        pfdta.internalUniformValue = 0.0;
        pfdta.boundary.push_back(boxtest::pfd<scalar>("inlet",    "zeroGradient", 0.0, false));
        pfdta.boundary.push_back(boxtest::pfd<scalar>("outlet",   "fixedValue",   0.0, true));   // pressure reference
        pfdta.boundary.push_back(boxtest::pfd<scalar>("wallYmin", "zeroGradient", 0.0, false));
        pfdta.boundary.push_back(boxtest::pfd<scalar>("wallYmax", "zeroGradient", 0.0, false));
        pfdta.boundary.push_back(boxtest::pfd<scalar>("wallZmin", "zeroGradient", 0.0, false));
        pfdta.boundary.push_back(boxtest::pfd<scalar>("wallZmax", "zeroGradient", 0.0, false));

        GeometricField<vector> U0 = distributeField<vector>(Ufd, gm.patches(), P.Lm, P.lp, P.procW, rank);
        U0.evaluateBoundary();
        GeometricField<scalar> p0 = distributeField<scalar>(pfdta, gm.patches(), P.Lm, P.lp, P.procW, rank);
        p0.evaluateBoundary();

        ParallelDeviceSimple solver(P, U0, p0, nu, relaxU, relaxP, tolU, tolP, maxIter);
        // SPEEDUP: time the SOLVE only -- mesh generation, decomposition and field distribution are setup, and
        // at these sizes they dwarf a handful of iterations. One warm-up step first (first-touch allocation,
        // the halo's symmetric-heap registration and the initial exchanges are one-off costs), then a barrier
        // so every rank starts the timed region together, and the MAX across ranks (the slowest rank sets the
        // wall clock; an average would flatter an imbalanced decomposition).
        solver.step();
        cudaDeviceSynchronize();
        Pstream::barrier();
        const long k0 = solver.krylovIters();
        const auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < N; ++it) solver.step();
        cudaDeviceSynchronize();
        const scalar secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const scalar wall = Pstream::allReduce(secs, ReduceOp::Max);
        const long  kIt = solver.krylovIters() - k0;

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
            std::printf("  SOLVE: %.3f s for %d iters -> %.4f s/iter  (%.3f Mcell-iter/s)  krylov=%ld (%.1f/step)\n",
                        wall, N, wall / N, (nC / 1e6) * N / wall, kIt, double(kIt) / N);
            std::printf("  per-rank device mem (max): %.2f GiB\n", maxGiB);
            std::printf("  sum(p) = %.6e %s\n", tot, std::isfinite(tot) ? "(finite)" : "(NON-FINITE!)");
            std::printf("%s\n", std::isfinite(tot) ? "PASS" : "FAIL");
        }
    }   // buffers must die before finalize

    Pstream::finalize();
    return 0;
}
