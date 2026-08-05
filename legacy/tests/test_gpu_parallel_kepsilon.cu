// Phase 4b: the distributed DEVICE kEpsilon::correct() reproduces np=1.
//
// This composes the 4b scalar-transport core (validated in test_gpu_parallel_scalar) with the LOCAL single-GPU
// turbulence kernels (production, wall functions, nut) into the full parallelDeviceKEpsilonCorrect, and gates
// it the 4a way: np=1 (no processor faces -> the pure local correct(), which reuses the OF-validated single-GPU
// kernels + the validated scalar core) is the oracle; np>1 must reproduce it.
//
// The mesh is a duct decomposed as SLABS IN X. Its walls are the y/z boundaries, so a wall face is NEVER split
// from its owner cell (the cut is orthogonal to every wall), yet near-wall cells sit ADJACENT to the x-cuts --
// exactly the "near-wall cell next to a processor interface" case the plan flagged. So the test exercises the
// wall constraint AND the interface coupling together.
//
// TEETH: k/eps must vary across the cut (they develop along the duct from the turbulent inlet). The test
// asserts the cross-cut spread of k is non-trivial, else the interface terms were multiplied by ~0.
//
// Run: mpirun -np {1,2,4} test_gpu_parallel_kepsilon <Nx> <Ny> <Nz> [ref.txt]
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
    if (argc < 4) { if (Pstream::master()) std::printf("usage: %s <Nx> <Ny> <Nz> [ref]\n", argv[0]); Pstream::finalize(); return 2; }
    if (!Pstream::nvshmemActive()) { if (Pstream::master()) std::printf("test_gpu_parallel_kepsilon: NVSHMEM inactive -- SKIP\n"); Pstream::finalize(); return 0; }

    const label Nx = std::atol(argv[1]), Ny = std::atol(argv[2]), Nz = std::atol(argv[3]);
    const std::string ref = (argc > 4) ? argv[4] : "keps_ref.txt";
    const scalar nu = 1e-3, relaxEps = 0.7, relaxK = 0.7, tol = 1e-10;
    const scalar kIn = 0.375, epsIn = 0.09 * std::pow(kIn, 1.5) / 0.1;   // turbulent inlet
    const int maxIter = 2000, N = 8;
    KEpsilonCoeffs co;
    const label nC = Nx * Ny * Nz;
    bool ok = true;
    scalar cutK = 0, cutSpread = 0, relK = 0, relE = 0, relN = 0;
    {
        const PrimitiveMesh gm = boxtest::boxMesh(Nx, Ny, Nz);
        std::vector<label> cellToPart(nC);
        for (label k = 0; k < Nz; ++k) for (label j = 0; j < Ny; ++j) for (label i = 0; i < Nx; ++i)
            cellToPart[i + Nx * (j + Ny * k)] = static_cast<label>((static_cast<long long>(i) * nproc) / Nx);
        const Partition P(gm, cellToPart, rank);
        const FvGeometry& lg = P.lg;
        const std::vector<FvPatch>& lp = P.lp;
        const label lnC = P.nCells(), nIf = P.Lm.mesh.nInternalFaces();

        // U: inlet (1,0,0), outlet zeroGradient, no-slip walls
        FieldData<vector> Ufd; Ufd.internalUniform = true; Ufd.internalUniformValue = vector{1, 0, 0};
        Ufd.boundary.push_back(boxtest::pfd<vector>("inlet","fixedValue",vector{1,0,0},true));
        Ufd.boundary.push_back(boxtest::pfd<vector>("outlet","zeroGradient",vector{0,0,0},false));
        for (const char* w : {"wallYmin","wallYmax","wallZmin","wallZmax"})
            Ufd.boundary.push_back(boxtest::pfd<vector>(w,"fixedValue",vector{0,0,0},true));
        GeometricField<vector> U = distributeField<vector>(Ufd, gm.patches(), P.Lm, lp, P.procW, rank); U.evaluateBoundary();
        const SurfaceScalarField phi = fvc::flux(U, P.Lm.mesh, lg, lp);

        // k/eps: inlet fixedValue, outlet zeroGradient, walls zeroGradient (the eps wall value is applied by the
        // near-wall setValues constraint, not the boundary; k uses kqRWallFunction == zeroGradient)
        auto scalfd = [&](scalar in){ FieldData<scalar> f; f.internalUniform=true; f.internalUniformValue=in;
            f.boundary.push_back(boxtest::pfd<scalar>("inlet","fixedValue",in,true));
            f.boundary.push_back(boxtest::pfd<scalar>("outlet","zeroGradient",0.0,false));
            for (const char* w : {"wallYmin","wallYmax","wallZmin","wallZmax"}) f.boundary.push_back(boxtest::pfd<scalar>(w,"zeroGradient",0.0,false));
            return f; };
        GeometricField<scalar> kF   = distributeField<scalar>(scalfd(kIn),   gm.patches(), P.Lm, lp, P.procW, rank); kF.evaluateBoundary();
        GeometricField<scalar> epsF = distributeField<scalar>(scalfd(epsIn), gm.patches(), P.Lm, lp, P.procW, rank); epsF.evaluateBoundary();
        GeometricField<scalar> nutF = distributeField<scalar>(scalfd(co.Cmu*kIn*kIn/epsIn), gm.patches(), P.Lm, lp, P.procW, rank); nutF.evaluateBoundary();

        DeviceMesh dm = buildDeviceMesh(P.Lm.mesh, P.lg, lp);
        const DeviceBoundary dbK = buildDeviceBoundary(kF, lp, lg), dbEps = buildDeviceBoundary(epsF, lp, lg);
        const DeviceVectorBoundary dbU = buildDeviceVectorBoundary(U, lp, lg);
        const DeviceWallData wall = buildDeviceWallData(P.Lm.mesh, lg, lp, U);

        DeviceBuffer<scalar> Ux(lnC), Uy(lnC), Uz(lnC), dk(kF.internal), de(epsF.internal), dnut(nutF.internal), ones(std::vector<scalar>(lnC,1.0));
        { std::vector<scalar> ux(lnC),uy(lnC),uz(lnC); for(label c=0;c<lnC;++c){ux[c]=U.internal[c].x;uy[c]=U.internal[c].y;uz[c]=U.internal[c].z;} Ux.copyFrom(ux);Uy.copyFrom(uy);Uz.copyFrom(uz); }

        DeviceBuffer<scalar> phiInt, phiBnd;
        { std::vector<scalar> pi(phi.internal.begin(), phi.internal.begin()+nIf); phiInt.copyFrom(pi); }
        { std::vector<scalar> pb; for(std::size_t p=0;p<lp.size();++p){if(lp[p].type=="cyclic"||lp[p].type=="cyclicAMI")continue;for(label i=0;i<lp[p].size;++i)pb.push_back(phi.boundary[p][i]);} phiBnd.copyFrom(pb); }

        // isWallCell mask
        std::vector<label> isW(lnC, 0);
        for (std::size_t p=0;p<lp.size();++p) if (lp[p].type=="wall") for (label i=0;i<lp[p].size;++i) isW[lp[p].faceCells[i]]=1;
        DeviceBuffer<label> isWallCell(isW);

        // halo + processor addressing + geomD (magSf*procDelta per interface)
        std::vector<int> nbrs; for (int q : P.Lm.procNbr) nbrs.push_back(q);
        DeviceHalo halo(rank, nbrs, P.Lm.procFaceCells);
        std::vector<DeviceBuffer<label>>  faceCellsD(P.Lm.procFaceCells.size());
        std::vector<DeviceBuffer<scalar>> weightsD(P.procW.size()), phiF, geomD;
        for (std::size_t i=0;i<P.Lm.procFaceCells.size();++i) faceCellsD[i].copyFrom(P.Lm.procFaceCells[i]);
        for (std::size_t i=0;i<P.procW.size();++i) weightsD[i].copyFrom(P.procW[i]);
        std::vector<label> procStart; label bidx=0; std::size_t pj=0;
        for (std::size_t p=0;p<lp.size();++p){
            if(lp[p].type=="cyclic"||lp[p].type=="cyclicAMI")continue;
            if(lp[p].type=="processor"){
                procStart.push_back(bidx);
                std::vector<scalar> pfh(lp[p].size), gd(lp[p].size);
                for(label i=0;i<lp[p].size;++i){ pfh[i]=phi.boundary[p][i]; gd[i]=lg.magSf()[lp[p].start+i]*P.procDelta[pj][i]; }
                DeviceBuffer<scalar> pf,g; pf.copyFrom(pfh); g.copyFrom(gd); phiF.push_back(std::move(pf)); geomD.push_back(std::move(g));
                ++pj;
            }
            bidx += lp[p].size;
        }

        for (int it = 0; it < N; ++it)
            parallelDeviceKEpsilonCorrect(dm, halo, wall, dbU, dbK, dbEps, Ux, Uy, Uz, dk, de, dnut,
                phiInt, phiBnd, phiF, faceCellsD, procStart, weightsD, geomD, isWallCell,
                nu, relaxEps, relaxK, tol, maxIter, P.globalNCells, ones, /*bounded*/false, co);
        cudaDeviceSynchronize();

        // teeth: cross-cut spread of k
        {
            halo.exchange(dk.data()); halo.waitExchange();
            const std::vector<scalar> ph = phiBnd.host(), kk = dk.host();
            for (std::size_t i=0;i<procStart.size();++i){
                const label n = static_cast<label>(halo.size((int)i));
                const std::vector<scalar> nb = halo.neighbourField((int)i);
                for (label f=0;f<n;++f){ cutK=std::fmax(cutK,std::fabs(ph[procStart[i]+f])); cutSpread=std::fmax(cutSpread,std::fabs(kk[P.Lm.procFaceCells[i][f]]-nb[f])); }
            }
            cutK=Pstream::allReduce(cutK,ReduceOp::Max); cutSpread=Pstream::allReduce(cutSpread,ReduceOp::Max);
        }

        // gather k/eps/nut to the global mesh
        auto gather = [&](const DeviceBuffer<scalar>& d){ std::vector<scalar> g(nC,0.0); const std::vector<scalar> h=d.host();
            for(label c=0;c<lnC;++c) g[P.Lm.cellProcAddr[c]]=h[c]; Pstream::allReduce(g.data(),(int)g.size(),ReduceOp::Sum); return g; };
        const std::vector<scalar> gk=gather(dk), ge=gather(de), gn=gather(dnut);

        if (Pstream::master()){
            if (nproc==1){
                std::ofstream os(ref); os.precision(17);
                for(label c=0;c<nC;++c) os<<gk[c]<<' '<<ge[c]<<' '<<gn[c]<<'\n';
                std::printf("test_gpu_parallel_kepsilon np=1: wrote reference %s (%ld cells)\n", ref.c_str(),(long)nC);
                std::printf("PASS\n");
            } else {
                std::ifstream is(ref);
                if(!is){ std::printf("cannot read %s -- run np=1 first\n", ref.c_str()); ok=false; }
                else {
                    scalar nk=0,dkd=0,ne=0,dee=0,nn=0,dnn=0;
                    for(label c=0;c<nC;++c){ scalar rk,re,rn; is>>rk>>re>>rn;
                        nk=std::fmax(nk,std::fabs(gk[c]-rk)); dkd=std::fmax(dkd,std::fabs(rk));
                        ne=std::fmax(ne,std::fabs(ge[c]-re)); dee=std::fmax(dee,std::fabs(re));
                        nn=std::fmax(nn,std::fabs(gn[c]-rn)); dnn=std::fmax(dnn,std::fabs(rn)); }
                    relK=dkd>0?nk/dkd:nk; relE=dee>0?ne/dee:ne; relN=dnn>0?nn/dnn:nn;
                    const bool teeth = (cutK>1e-6) && (cutSpread>1e-6);
                    std::printf("test_gpu_parallel_kepsilon np=%d: k rel=%.3e  eps rel=%.3e  nut rel=%.3e  k spread at cut=%.3e\n",
                                nproc, relK, relE, relN, cutSpread);
                    if(!teeth) std::printf("  NO TEETH: cut flux or k spread ~0\n");
                    ok = teeth && relK<1e-6 && relE<1e-6 && relN<1e-6;
                    std::printf("%s\n", ok?"PASS":"FAIL");
                }
            }
        }
    }
    Pstream::finalize();
    return ok ? 0 : 1;
}
