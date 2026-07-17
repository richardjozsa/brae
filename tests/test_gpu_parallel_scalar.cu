// Phase 4b foundation: the distributed DEVICE scalar transport solve reproduces np=1.
//
// deviceParallelScalarTransport is the reusable core of the turbulence step -- k, epsilon, omega and nuTilda
// all reduce to  div(phi,psi) - laplacian(gamma,psi) + Sp*psi == Su. This gates it the way 4a's interface
// terms were gated: np=1 (no processor faces -> the pure local solve, which composes only OF-validated
// primitives) is the oracle, and np>1 must reproduce it on a cut that CARRIES FLUX.
//
// TEETH: a passive scalar with inlet psi=1 and a volumetric SINK (Sp>0, decay) so psi decays ALONG the duct
// and therefore VARIES across every (flow-perpendicular) cut. Without that, psi would be uniform and the
// interface convection term (phi*(psi_local - psi_remote)) would be multiplied by ~0 -- the cavity's mistake.
// The test asserts both max|phi| at the cut AND the psi spread across the cut are non-trivial.
//
// Run: mpirun -np {1,2,4} test_gpu_parallel_scalar <Nx> <Ny> <Nz> [ref.txt] [shear]
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvc.cuh"
#include "parallel_simple.cuh"
#include "field_distribute.cuh"
#include "parallel_device_turbulence.cuh"
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
    if (argc < 4) { if (Pstream::master()) std::printf("usage: %s <Nx> <Ny> <Nz> [ref] [shear]\n", argv[0]); Pstream::finalize(); return 2; }
    if (!Pstream::nvshmemActive()) { if (Pstream::master()) std::printf("test_gpu_parallel_scalar: NVSHMEM inactive -- SKIP\n"); Pstream::finalize(); return 0; }

    const label Nx = std::atol(argv[1]), Ny = std::atol(argv[2]), Nz = std::atol(argv[3]);
    const std::string ref = (argc > 4) ? argv[4] : "scalar_ref.txt";
    const scalar shear = (argc > 5) ? std::atof(argv[5]) : 0.0;
    const scalar gamma = 1e-2, Spval = 0.03, tol = 1e-10;  // gentle sink: psi decays smoothly along the duct so it VARIES across every flow-perpendicular cut (Sp too large -> psi dies before the cut -> no teeth)
    const int maxIter = 2000;
    const label nC = Nx * Ny * Nz;
    bool ok = true;
    scalar cutPhi = 0, cutSpread = 0, rel = 0;
    {
        const PrimitiveMesh gm = boxtest::boxMesh(Nx, Ny, Nz, shear);
        std::vector<label> cellToPart(nC);
        for (label k = 0; k < Nz; ++k) for (label j = 0; j < Ny; ++j) for (label i = 0; i < Nx; ++i)
            cellToPart[i + Nx * (j + Ny * k)] = static_cast<label>((static_cast<long long>(i) * nproc) / Nx);
        const Partition P(gm, cellToPart, rank);
        const FvGeometry& lg = P.lg;
        const std::vector<FvPatch>& lp = P.lp;
        const label lnC = P.nCells();

        // through-flow phi (u=1 in x) so the cut carries flux
        FieldData<vector> Ufd;
        Ufd.internalUniform = true; Ufd.internalUniformValue = vector{1, 0, 0};
        Ufd.boundary.push_back(boxtest::pfd<vector>("inlet",    "fixedValue",   vector{1, 0, 0}, true));
        Ufd.boundary.push_back(boxtest::pfd<vector>("outlet",   "zeroGradient", vector{0, 0, 0}, false));
        for (const char* w : {"wallYmin","wallYmax","wallZmin","wallZmax"})
            Ufd.boundary.push_back(boxtest::pfd<vector>(w, "fixedValue", vector{0,0,0}, true));
        GeometricField<vector> U = distributeField<vector>(Ufd, gm.patches(), P.Lm, lp, P.procW, rank);
        U.evaluateBoundary();
        const SurfaceScalarField phi = fvc::flux(U, P.Lm.mesh, lg, lp);

        // passive scalar psi: inlet fixedValue 1, everything else zeroGradient
        FieldData<scalar> psifd;
        psifd.internalUniform = true; psifd.internalUniformValue = 0.0;
        psifd.boundary.push_back(boxtest::pfd<scalar>("inlet",  "fixedValue",   1.0, true));
        psifd.boundary.push_back(boxtest::pfd<scalar>("outlet", "zeroGradient", 0.0, false));
        for (const char* w : {"wallYmin","wallYmax","wallZmin","wallZmax"})
            psifd.boundary.push_back(boxtest::pfd<scalar>(w, "zeroGradient", 0.0, false));
        GeometricField<scalar> psiF = distributeField<scalar>(psifd, gm.patches(), P.Lm, lp, P.procW, rank);
        psiF.evaluateBoundary();
        const DeviceBoundary db = buildDeviceBoundary(psiF, lp, lg);

        const label nIf = P.Lm.mesh.nInternalFaces();
        DeviceMesh dm = buildDeviceMesh(P.Lm.mesh, P.lg, lp);

        // device fields
        DeviceBuffer<scalar> psi(psiF.internal), ones(std::vector<scalar>(lnC, 1.0));
        DeviceBuffer<scalar> Sp(std::vector<scalar>(lnC, 0.0)), Su(std::vector<scalar>(lnC, 0.0));
        { std::vector<scalar> sp(lnC); for (label c = 0; c < lnC; ++c) sp[c] = Spval * lg.V()[c]; Sp.copyFrom(sp); }  // V*Sp*psi

        DeviceBuffer<scalar> phiInt, phiBnd;
        { std::vector<scalar> pi(phi.internal.begin(), phi.internal.begin() + nIf); phiInt.copyFrom(pi); }
        { std::vector<scalar> pb; for (std::size_t p = 0; p < lp.size(); ++p) { if (lp[p].type=="cyclic"||lp[p].type=="cyclicAMI") continue; for (label i=0;i<lp[p].size;++i) pb.push_back(phi.boundary[p][i]); } phiBnd.copyFrom(pb); }

        // gamma at internal + boundary faces (constant)
        DeviceBuffer<scalar> gammaFint(std::vector<scalar>(nIf, gamma));
        label nBnd = 0; for (std::size_t p=0;p<lp.size();++p){ if(lp[p].type=="cyclic"||lp[p].type=="cyclicAMI")continue; nBnd += lp[p].size; }
        DeviceBuffer<scalar> gammaBnd(std::vector<scalar>(nBnd, gamma));

        // halo + processor addressing
        std::vector<int> nbrs; for (int q : P.Lm.procNbr) nbrs.push_back(q);
        DeviceHalo halo(rank, nbrs, P.Lm.procFaceCells);
        std::vector<DeviceBuffer<label>>  faceCellsD(P.Lm.procFaceCells.size());
        std::vector<DeviceBuffer<scalar>> weightsD(P.procW.size()), phiF, gammaCoeffGeo;
        for (std::size_t i=0;i<P.Lm.procFaceCells.size();++i) faceCellsD[i].copyFrom(P.Lm.procFaceCells[i]);
        for (std::size_t i=0;i<P.procW.size();++i) weightsD[i].copyFrom(P.procW[i]);
        std::vector<label> procStart; label bidx = 0;
        std::size_t pj = 0;
        for (std::size_t p = 0; p < lp.size(); ++p)
        {
            if (lp[p].type=="cyclic"||lp[p].type=="cyclicAMI") continue;
            if (lp[p].type=="processor")
            {
                procStart.push_back(bidx);
                DeviceBuffer<scalar> pf; std::vector<scalar> pfh(lp[p].size);
                for (label i=0;i<lp[p].size;++i) pfh[i] = phi.boundary[p][i];
                pf.copyFrom(pfh); phiF.push_back(std::move(pf));
                std::vector<scalar> cg(lp[p].size);
                for (label i=0;i<lp[p].size;++i) cg[i] = gamma * lg.magSf()[lp[p].start+i] * P.procDelta[pj][i];
                DeviceBuffer<scalar> cgd; cgd.copyFrom(cg); gammaCoeffGeo.push_back(std::move(cgd));
                ++pj;
            }
            bidx += lp[p].size;
        }
        // no wall constraint for a passive scalar
        DeviceBuffer<label> noWall; DeviceBuffer<scalar> noWallVal;   // passive scalar: no wall constraint

        // solve to convergence (iterate the transport a few times; it is linear so it converges fast)
        for (int it = 0; it < 200; ++it)
            deviceParallelScalarTransport(dm, halo, db, faceCellsD, procStart, weightsD, phiInt, phiBnd, phiF,
                                          gammaFint, gammaBnd, gammaCoeffGeo, Sp, Su, psi, /*relax*/1.0, tol,
                                          maxIter, P.globalNCells, ones, noWall, noWallVal);
        cudaDeviceSynchronize();

        // teeth diagnostics: cut flux, and the TRUE cross-cut jump |psi_local - psi_remote| (psi decays ALONG
        // the flow, so the coupling that matters is between the local owner and the remote cell across the cut,
        // NOT the spread among same-plane local cells). Exchange psi once through the halo to get psi_remote.
        {
            halo.exchange(psi.data());
            halo.waitExchange();
            const std::vector<scalar> ph = phiBnd.host(), ps = psi.host();
            for (std::size_t i = 0; i < procStart.size(); ++i)
            {
                const label n = static_cast<label>(halo.size(static_cast<int>(i)));
                const std::vector<scalar> nb = halo.neighbourField(static_cast<int>(i));
                for (label f = 0; f < n; ++f)
                {
                    cutPhi = std::fmax(cutPhi, std::fabs(ph[procStart[i]+f]));
                    cutSpread = std::fmax(cutSpread, std::fabs(ps[P.Lm.procFaceCells[i][f]] - nb[f]));
                }
            }
            cutPhi = Pstream::allReduce(cutPhi, ReduceOp::Max);
            cutSpread = Pstream::allReduce(cutSpread, ReduceOp::Max);
        }

        // gather to the global mesh
        std::vector<scalar> g(nC, 0.0); const std::vector<scalar> ps = psi.host();
        for (label c = 0; c < lnC; ++c) g[P.Lm.cellProcAddr[c]] = ps[c];
        Pstream::allReduce(g.data(), (int)g.size(), ReduceOp::Sum);

        if (Pstream::master())
        {
            if (nproc == 1)
            {
                std::ofstream os(ref); os.precision(17);
                for (label c = 0; c < nC; ++c) os << g[c] << '\n';
                std::printf("test_gpu_parallel_scalar np=1: wrote reference %s (%ld cells)\n", ref.c_str(), (long)nC);
                std::printf("PASS\n");
            }
            else
            {
                std::ifstream is(ref);
                if (!is) { std::printf("cannot read %s -- run np=1 first\n", ref.c_str()); ok = false; }
                else
                {
                    scalar num = 0, den = 0;
                    for (label c = 0; c < nC; ++c) { scalar r; is >> r; num = std::fmax(num, std::fabs(g[c]-r)); den = std::fmax(den, std::fabs(r)); }
                    rel = den > 0 ? num/den : num;
                    const bool teeth = (cutPhi > 1e-6) && (cutSpread > 1e-3);
                    std::printf("test_gpu_parallel_scalar np=%d: rel=%.3e  max|phi| at cut=%.3e  psi spread at cut=%.3e\n",
                                nproc, rel, cutPhi, cutSpread);
                    if (!teeth) std::printf("  NO TEETH: cut flux or psi spread ~0\n");
                    ok = teeth && (rel < 1e-6);
                    std::printf("%s\n", ok ? "PASS" : "FAIL");
                }
            }
        }
    }
    Pstream::finalize();
    return ok ? 0 : 1;
}
