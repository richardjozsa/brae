// Phase 4a: the linearUpwind/LUST DEFERRED CORRECTION at a processor face, on a cut that CARRIES FLUX.
//
// WHY THIS TEST EXISTS. Both interface corrections are multiplied by the cut-face flux:
//   linearUpwind:  corr[c] += phi * (grad(U)[upwind] . d_upwind)
//   LUST linear:   corr[c] += phi * (w - pos0) * (psi[local] - psi[remote])
// so on a cut where phi == 0 they are IDENTICALLY ZERO and any bug in them is invisible. That is exactly what
// the 3D cavity does: SCOTCH halves a 24^3 lid-driven cavity on a symmetry midplane, where the vortex lies in
// the x-y plane and the cut-normal velocity vanishes -- measured max|phi| there was 3.2e-14. Every "validated
// at np=2/4/8" result on the cavity was therefore blind to these terms: deleting them, or scaling them by
// 1000, changed the answer by nothing.
//
// This test removes that blind spot: a straight duct (inlet -> outlet along x) decomposed as SLABS IN X, so
// every cut plane is PERPENDICULAR to the flow and carries the full through-flux.
//
// The oracle is np=1 of the same code: with one partition there are no processor faces, so the interface terms
// are inactive and the answer is the pure internal discretisation. Any np>1 run must reproduce it. The test
// asserts max|phi| at the cut is actually non-trivial FIRST -- otherwise it would silently be a no-op again.
//
// Run: mpirun -np {1,2,4} test_gpu_parallel_duct <Nx> <Ny> <Nz> [nIters] [ref.txt]
//   np=1 writes ref.txt (the reference); np>1 reads it back and compares cell by cell.
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

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace brae;

int main(int argc, char** argv)
{
    Pstream::init(argc, argv);
    const int rank = Pstream::myProcNo(), nproc = Pstream::nProcs();
    if (argc < 4)
    {
        if (Pstream::master()) std::printf("usage: %s <Nx> <Ny> <Nz> [nIters] [ref.txt]\n", argv[0]);
        Pstream::finalize();
        return 2;
    }
    if (!Pstream::nvshmemActive())
    {
        if (Pstream::master()) std::printf("test_gpu_parallel_duct: NVSHMEM inactive -- SKIP\n");
        Pstream::finalize();
        return 0;
    }
    const label Nx = std::atol(argv[1]), Ny = std::atol(argv[2]), Nz = std::atol(argv[3]);
    const int   N  = (argc > 4) ? std::atoi(argv[4]) : 20;
    const std::string ref = (argc > 5) ? argv[5] : "duct_ref.txt";
    const bool  lust = (argc > 6) && (std::string(argv[6]) == "lust");
    const std::string dump = (argc > 7) ? argv[7] : "";   // stage-by-stage dump (see setStageDump)
    const scalar nu = 1e-2, relaxU = 0.7, relaxP = 0.3, tolU = 1e-9, tolP = 1e-9;
    const int maxIter = 500;

    const label nC = Nx * Ny * Nz;
    bool ok = true;
    scalar cutPhi = 0, relU = 0, relP = 0;
    {
        const PrimitiveMesh gm = boxtest::boxMesh(Nx, Ny, Nz);
        // SLABS IN X: cut planes normal to the flow, so every cut face carries the through-flux.
        std::vector<label> cellToPart(nC);
        for (label k = 0; k < Nz; ++k)
            for (label j = 0; j < Ny; ++j)
                for (label i = 0; i < Nx; ++i)
                    cellToPart[i + Nx * (j + Ny * k)] = static_cast<label>((static_cast<long long>(i) * nproc) / Nx);
        const Partition P(gm, cellToPart, rank);

        FieldData<vector> Ufd;
        Ufd.internalUniform = true;
        Ufd.internalUniformValue = vector{1, 0, 0};   // start with through-flow so the cut carries flux at once
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
        pfdta.boundary.push_back(boxtest::pfd<scalar>("outlet",   "fixedValue",   0.0, true));   // p reference
        pfdta.boundary.push_back(boxtest::pfd<scalar>("wallYmin", "zeroGradient", 0.0, false));
        pfdta.boundary.push_back(boxtest::pfd<scalar>("wallYmax", "zeroGradient", 0.0, false));
        pfdta.boundary.push_back(boxtest::pfd<scalar>("wallZmin", "zeroGradient", 0.0, false));
        pfdta.boundary.push_back(boxtest::pfd<scalar>("wallZmax", "zeroGradient", 0.0, false));

        GeometricField<vector> U0 = distributeField<vector>(Ufd, gm.patches(), P.Lm, P.lp, P.procW, rank);
        U0.evaluateBoundary();
        GeometricField<scalar> p0 = distributeField<scalar>(pfdta, gm.patches(), P.Lm, P.lp, P.procW, rank);
        p0.evaluateBoundary();

        // linearUpwind ON: this test exists to exercise its interface term.
        ParallelDeviceSimple solver(P, U0, p0, nu, relaxU, relaxP, tolU, tolP, maxIter,
                                    /*bounded*/ true, /*linearUpwind*/ !lust, /*lust*/ lust);
        if (!dump.empty()) solver.setStageDump(dump, 1);   // dump iteration 1: before anything diverges
        for (int it = 0; it < N; ++it) solver.step();
        cudaDeviceSynchronize();

        // TEETH CHECK: the cut must actually carry flux, else the interface terms are multiplied by zero and
        // this test proves nothing (which is precisely how the cavity fooled us).
        cutPhi = Pstream::allReduce(solver.maxProcFlux(), ReduceOp::Max);

        // gather to the global mesh
        std::vector<scalar> pg(nC, 0.0), ug(3 * nC, 0.0);
        const std::vector<scalar> pl = solver.p();
        const std::vector<vector> ul = solver.U();
        for (label c = 0; c < P.nCells(); ++c)
        {
            const label gc = P.Lm.cellProcAddr[c];
            pg[gc] = pl[c];
            ug[3*gc] = ul[c].x; ug[3*gc+1] = ul[c].y; ug[3*gc+2] = ul[c].z;
        }
        Pstream::allReduce(pg.data(), static_cast<int>(pg.size()), ReduceOp::Sum);
        Pstream::allReduce(ug.data(), static_cast<int>(ug.size()), ReduceOp::Sum);

        if (Pstream::master())
        {
            if (nproc == 1)   // write the reference: no processor faces -> pure internal discretisation
            {
                std::ofstream os(ref);
                os.precision(17);
                for (label c = 0; c < nC; ++c)
                    os << pg[c] << ' ' << ug[3*c] << ' ' << ug[3*c+1] << ' ' << ug[3*c+2] << '\n';
                std::printf("test_gpu_parallel_duct np=1: wrote reference %s (%ld cells)\n", ref.c_str(), (long)nC);
                std::printf("  max|phi| at cut = %.3e (no cut at np=1, as expected)\n", cutPhi);
                std::printf("PASS\n");
            }
            else
            {
                std::ifstream is(ref);
                if (!is)
                {
                    std::printf("test_gpu_parallel_duct: cannot read %s -- run np=1 first\n", ref.c_str());
                    ok = false;
                }
                else
                {
                    scalar nU = 0, dU = 0, nP = 0, dP = 0;
                    for (label c = 0; c < nC; ++c)
                    {
                        scalar rp, rx, ry, rz;
                        is >> rp >> rx >> ry >> rz;
                        nP = std::fmax(nP, std::fabs(pg[c] - rp));
                        dP = std::fmax(dP, std::fabs(rp));
                        nU = std::fmax(nU, std::fabs(ug[3*c] - rx));
                        nU = std::fmax(nU, std::fabs(ug[3*c+1] - ry));
                        nU = std::fmax(nU, std::fabs(ug[3*c+2] - rz));
                        dU = std::fmax(dU, std::fabs(rx));
                    }
                    relU = dU > 0 ? nU / dU : nU;
                    relP = dP > 0 ? nP / dP : nP;
                    // The cut must carry flux, or the interface terms were never exercised.
                    const bool teeth = cutPhi > 1e-6;
                    std::printf("test_gpu_parallel_duct np=%d (%d iters): U rel=%.3e  p rel=%.3e  max|phi| at cut=%.3e\n",
                                nproc, N, relU, relP, cutPhi);
                    if (!teeth)
                        std::printf("  NO TEETH: cut flux ~0, the interface terms were multiplied by zero\n");
                    ok = teeth && (relU < 1e-4) && (relP < 1e-4);
                    std::printf("%s\n", ok ? "PASS" : "FAIL");
                }
            }
        }
    }   // buffers must die before finalize

    Pstream::finalize();
    return ok ? 0 : 1;
}
