// Phase 4a L8 (driver): the DEVICE distributed SIMPLE LOOP matches the host distributed loop.
//
// test_gpu_parallel_predictor validated ONE composed step stage-by-stage. This drives the packaged
// ParallelDeviceSimple for N iterations against the validated host parallelSimpleStepLaminar, which is the
// real test of the driver: phi is MAINTAINED on the device across iterations (the corrector's conservative
// flux feeds the next step's convection), so any interface leak compounds instead of cancelling.
//
// It also checks reconstructField: the per-rank device p gathered back to the global mesh must be
// independent of the partition count (that is what lets the multi-GPU run write a single result).
//
// Both sides run the same discretisation (div(phi,U) - laplacian(nuEff,U) + divDevReff); the host solves with
// PBiCGStab/DILU and the device with Jacobi-BiCGStab/PCG, so agreement is to the solver floor, and N
// iterations of that floor drift -- hence the same 1e-4 gate the host test_parallel_loop uses at np>1.
//
// Run: mpirun -np {1,2,4,8} test_gpu_parallel_loop <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvc.cuh"
#include "scotch_decomposition.cuh"
#include "parallel_simple.cuh"
#include "field_distribute.cuh"
#include "reconstruct.cuh"
#include "parallel_device_simple.cuh"
#include "cf_pstream.cuh"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

int main(int argc, char** argv)
{
    Pstream::init(argc, argv);
    const int rank = Pstream::myProcNo(), nproc = Pstream::nProcs();
    if (argc < 2)
    {
        if (Pstream::master()) std::printf("usage: %s <caseDir>\n", argv[0]);
        Pstream::finalize();
        return 2;
    }
    if (!Pstream::nvshmemActive())
    {
        if (Pstream::master()) std::printf("test_gpu_parallel_loop: NVSHMEM inactive -- SKIP\n");
        Pstream::finalize();
        return 0;
    }
    const std::string caseDir = argv[1];
    const scalar nu = 1e-3, relaxU = 0.7, relaxP = 0.3, tolU = 1e-9, tolP = 1e-9;
    const int maxIter = 3000, N = 5;

    scalar rU = 0, rP = 0, rRec = 0;
    {
        PrimitiveMesh gm;  gm.read(caseDir + "/constant/polyMesh");
        const label nC = gm.nCells();
        std::vector<label> cellToPart(nC, 0);
        if (Pstream::master()) cellToPart = scotchDecompose(gm, nproc);
        Pstream::broadcast(cellToPart.data(), nC, 0);
        const Partition P(gm, cellToPart, rank);
        const label lnC = P.nCells();

        const FieldData<vector> Ufd = readField<vector>(caseDir + "/282/U");
        const FieldData<scalar> pfd = readField<scalar>(caseDir + "/282/p");

        const std::vector<scalar> nuEffCell(lnC, nu);
        std::vector<std::vector<scalar>> nuEffBnd(P.lp.size());
        for (std::size_t pi = 0; pi < P.lp.size(); ++pi) nuEffBnd[pi].assign(P.lp[pi].size, nu);

        // ---- HOST distributed loop (the oracle) ----
        std::vector<vector> UhostF;
        std::vector<scalar> phostF;
        {
            GeometricField<vector> U = distributeField<vector>(Ufd, gm.patches(), P.Lm, P.lp, P.procW, rank);
            U.evaluateBoundary();
            GeometricField<scalar> p = distributeField<scalar>(pfd, gm.patches(), P.Lm, P.lp, P.procW, rank);
            p.evaluateBoundary();
            SurfaceScalarField phi = fvc::flux(U, P.Lm.mesh, P.lg, P.lp);
            for (int it = 0; it < N; ++it)
                parallelSimpleStepLaminar(P, U, p, phi, nuEffCell, nuEffBnd, Ufd, relaxU, relaxP, tolU, tolP);
            UhostF = U.internal;
            phostF = p.internal;
        }

        // ---- DEVICE distributed loop (the driver) ----
        std::vector<vector> UdevF;
        std::vector<scalar> pdevF, pRec;
        {
            GeometricField<vector> U0 = distributeField<vector>(Ufd, gm.patches(), P.Lm, P.lp, P.procW, rank);
            U0.evaluateBoundary();
            GeometricField<scalar> p0 = distributeField<scalar>(pfd, gm.patches(), P.Lm, P.lp, P.procW, rank);
            p0.evaluateBoundary();

            ParallelDeviceSimple solver(P, U0, p0, nu, relaxU, relaxP, tolU, tolP, maxIter);
            for (int it = 0; it < N; ++it) solver.step();
            cudaDeviceSynchronize();
            UdevF = solver.U();
            pdevF = solver.p();
            pRec  = solver.reconstructP();          // gathered to the global mesh on every rank
        }

        // device vs host, cell by cell
        scalar nU = 0, dU = 0, nP = 0, dP = 0;
        for (label c = 0; c < lnC; ++c)
        {
            nU = std::fmax(nU, std::fabs(UdevF[c].x - UhostF[c].x));
            nU = std::fmax(nU, std::fabs(UdevF[c].y - UhostF[c].y));
            nU = std::fmax(nU, std::fabs(UdevF[c].z - UhostF[c].z));
            dU = std::fmax(dU, std::fabs(UhostF[c].x));
            dU = std::fmax(dU, std::fabs(UhostF[c].y));
            nP = std::fmax(nP, std::fabs(pdevF[c] - phostF[c]));
            dP = std::fmax(dP, std::fabs(phostF[c]));
        }
        rU = Pstream::allReduce(nU, ReduceOp::Max) / std::fmax(Pstream::allReduce(dU, ReduceOp::Max), 1e-300);
        rP = Pstream::allReduce(nP, ReduceOp::Max) / std::fmax(Pstream::allReduce(dP, ReduceOp::Max), 1e-300);

        // reconstruction: the gathered global field must agree with the host's local values at every cell
        // this rank owns (reconstructField is an allReduce, so every rank holds the same global array).
        scalar nR = 0, dR = 0;
        for (label c = 0; c < lnC; ++c)
        {
            const label gc = P.Lm.cellProcAddr[c];
            nR = std::fmax(nR, std::fabs(pRec[gc] - pdevF[c]));
            dR = std::fmax(dR, std::fabs(pdevF[c]));
        }
        rRec = Pstream::allReduce(nR, ReduceOp::Max) / std::fmax(Pstream::allReduce(dR, ReduceOp::Max), 1e-300);
    }   // SymBuffer/DeviceHalo must die before Pstream::finalize (nvshmem_free after finalize crashes)

    if (Pstream::master())
    {
        const scalar gate = (nproc == 1) ? 1e-6 : 1e-4;   // N iterations of parallel-solver-floor drift
        std::printf("test_gpu_parallel_loop np=%d (%d iters): U rel=%.3e  p rel=%.3e  reconstruct=%.3e\n",
                    nproc, N, rU, rP, rRec);
        const bool ok = (rU < gate) && (rP < gate) && (rRec < 1e-12);
        std::printf("%s\n", ok ? "PASS" : "FAIL");
    }
    Pstream::finalize();
    return 0;
}
