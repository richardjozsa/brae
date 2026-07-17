// Phase 4b: the distributed DEVICE kOmegaSST::correct() reproduces np=1.
//
// SST is the second RAS model. It reuses the 4b scalar-transport core (k/omega solves) and the exposed LOCAL
// single-GPU SST kernels (S2, F1, F2, CDkOmega, blends, reactions, nutSST, wall omega); the processor-coupled
// inputs are gradU, grad(k), grad(omega) (halo-consistent) and the F1-blended diffusivities at cut faces.
// Gated np=1 (oracle) vs np=2/4 on the turbulent duct (walls y/z, cut x). TEETH: k varies across the cut.
//
// Run: mpirun -np {1,2,4} test_gpu_parallel_komegasst <Nx> <Ny> <Nz> [ref.txt]
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvc.cuh"
#include "cell_wall_dist.cuh"
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
    if (!Pstream::nvshmemActive()) { if (Pstream::master()) std::printf("test_gpu_parallel_komegasst: NVSHMEM inactive -- SKIP\n"); Pstream::finalize(); return 0; }

    const label Nx = std::atol(argv[1]), Ny = std::atol(argv[2]), Nz = std::atol(argv[3]);
    const std::string ref = (argc > 4) ? argv[4] : "sst_ref.txt";
    const scalar nu = 1e-3, relaxOmega = 0.7, relaxK = 0.7, tol = 1e-10;
    const scalar kIn = 0.375, omIn = std::sqrt(kIn) / (0.09 * 0.1);   // omega ~ sqrt(k)/(Cmu*L)
    const int maxIter = 2000, N = 6;
    KOmegaSSTCoeffs co;
    const label nC = Nx * Ny * Nz;
    bool ok = true;
    scalar cutK = 0, cutSpread = 0, relK = 0, relO = 0, relN = 0;
    {
        const PrimitiveMesh gm = boxtest::boxMesh(Nx, Ny, Nz);
        std::vector<label> cellToPart(nC);
        for (label k = 0; k < Nz; ++k) for (label j = 0; j < Ny; ++j) for (label i = 0; i < Nx; ++i)
            cellToPart[i + Nx * (j + Ny * k)] = static_cast<label>((static_cast<long long>(i) * nproc) / Nx);
        const Partition P(gm, cellToPart, rank);
        const FvGeometry& lg = P.lg;
        const std::vector<FvPatch>& lp = P.lp;
        const label lnC = P.nCells(), nIf = P.Lm.mesh.nInternalFaces();

        FieldData<vector> Ufd; Ufd.internalUniform = true; Ufd.internalUniformValue = vector{1, 0, 0};
        Ufd.boundary.push_back(boxtest::pfd<vector>("inlet","fixedValue",vector{1,0,0},true));
        Ufd.boundary.push_back(boxtest::pfd<vector>("outlet","zeroGradient",vector{0,0,0},false));
        for (const char* w : {"wallYmin","wallYmax","wallZmin","wallZmax"})
            Ufd.boundary.push_back(boxtest::pfd<vector>(w,"fixedValue",vector{0,0,0},true));
        GeometricField<vector> U = distributeField<vector>(Ufd, gm.patches(), P.Lm, lp, P.procW, rank); U.evaluateBoundary();
        const SurfaceScalarField phi = fvc::flux(U, P.Lm.mesh, lg, lp);

        auto scalfd = [&](scalar in){ FieldData<scalar> f; f.internalUniform=true; f.internalUniformValue=in;
            f.boundary.push_back(boxtest::pfd<scalar>("inlet","fixedValue",in,true));
            f.boundary.push_back(boxtest::pfd<scalar>("outlet","zeroGradient",0.0,false));
            for (const char* w : {"wallYmin","wallYmax","wallZmin","wallZmax"}) f.boundary.push_back(boxtest::pfd<scalar>(w,"zeroGradient",0.0,false));
            return f; };
        GeometricField<scalar> kF  = distributeField<scalar>(scalfd(kIn), gm.patches(), P.Lm, lp, P.procW, rank); kF.evaluateBoundary();
        GeometricField<scalar> omF = distributeField<scalar>(scalfd(omIn), gm.patches(), P.Lm, lp, P.procW, rank); omF.evaluateBoundary();
        GeometricField<scalar> nutF= distributeField<scalar>(scalfd(kIn/omIn), gm.patches(), P.Lm, lp, P.procW, rank); nutF.evaluateBoundary();

        DeviceMesh dm = buildDeviceMesh(P.Lm.mesh, P.lg, lp);
        const DeviceBoundary dbK = buildDeviceBoundary(kF, lp, lg), dbOm = buildDeviceBoundary(omF, lp, lg);
        const DeviceVectorBoundary dbU = buildDeviceVectorBoundary(U, lp, lg);
        const DeviceWallData wall = buildDeviceWallData(P.Lm.mesh, lg, lp, U);
        DeviceBuffer<scalar> y(cellWallDist(P.Lm.mesh, lg, lp));

        DeviceBuffer<scalar> Ux(lnC), Uy(lnC), Uz(lnC), dk(kF.internal), dom(omF.internal), dnut(nutF.internal), ones(std::vector<scalar>(lnC,1.0));
        { std::vector<scalar> ux(lnC),uy(lnC),uz(lnC); for(label c=0;c<lnC;++c){ux[c]=U.internal[c].x;uy[c]=U.internal[c].y;uz[c]=U.internal[c].z;} Ux.copyFrom(ux);Uy.copyFrom(uy);Uz.copyFrom(uz); }
        DeviceBuffer<scalar> phiInt, phiBnd;
        { std::vector<scalar> pi(phi.internal.begin(), phi.internal.begin()+nIf); phiInt.copyFrom(pi); }
        { std::vector<scalar> pb; for(std::size_t p=0;p<lp.size();++p){if(lp[p].type=="cyclic"||lp[p].type=="cyclicAMI")continue;for(label i=0;i<lp[p].size;++i)pb.push_back(phi.boundary[p][i]);} phiBnd.copyFrom(pb); }

        std::vector<label> isW(lnC, 0);
        for (std::size_t p=0;p<lp.size();++p) if (lp[p].type=="wall") for (label i=0;i<lp[p].size;++i) isW[lp[p].faceCells[i]]=1;
        DeviceBuffer<label> isWallCell(isW);

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
            parallelDeviceKOmegaSSTCorrect(dm, halo, wall, dbU, dbK, dbOm, Ux, Uy, Uz, dk, dom, dnut, y,
                phiInt, phiBnd, phiF, faceCellsD, procStart, weightsD, geomD, isWallCell,
                nu, relaxOmega, relaxK, tol, maxIter, P.globalNCells, ones, false, co);
        cudaDeviceSynchronize();

        { halo.exchange(dk.data()); halo.waitExchange();
          const std::vector<scalar> ph=phiBnd.host(), kk=dk.host();
          for(std::size_t i=0;i<procStart.size();++i){ const label n=static_cast<label>(halo.size((int)i)); const std::vector<scalar> nbf=halo.neighbourField((int)i);
            for(label f=0;f<n;++f){ cutK=std::fmax(cutK,std::fabs(ph[procStart[i]+f])); cutSpread=std::fmax(cutSpread,std::fabs(kk[P.Lm.procFaceCells[i][f]]-nbf[f])); } }
          cutK=Pstream::allReduce(cutK,ReduceOp::Max); cutSpread=Pstream::allReduce(cutSpread,ReduceOp::Max); }

        auto g=[&](const DeviceBuffer<scalar>& d){ std::vector<scalar> gg(nC,0.0); const std::vector<scalar> h=d.host();
            for(label c=0;c<lnC;++c) gg[P.Lm.cellProcAddr[c]]=h[c]; Pstream::allReduce(gg.data(),(int)gg.size(),ReduceOp::Sum); return gg; };
        const std::vector<scalar> gk=g(dk), go=g(dom), gn=g(dnut);

        if (Pstream::master()){
            if (nproc==1){ std::ofstream os(ref); os.precision(17);
                for(label c=0;c<nC;++c) os<<gk[c]<<' '<<go[c]<<' '<<gn[c]<<'\n';
                std::printf("test_gpu_parallel_komegasst np=1: wrote reference %s (%ld cells)\n", ref.c_str(),(long)nC);
                std::printf("PASS\n");
            } else { std::ifstream is(ref);
                if(!is){ std::printf("cannot read %s -- run np=1 first\n", ref.c_str()); ok=false; }
                else { scalar nk=0,dkd=0,no=0,doo=0,nn=0,dnn=0;
                    for(label c=0;c<nC;++c){ scalar rk,ro,rn; is>>rk>>ro>>rn;
                        nk=std::fmax(nk,std::fabs(gk[c]-rk)); dkd=std::fmax(dkd,std::fabs(rk));
                        no=std::fmax(no,std::fabs(go[c]-ro)); doo=std::fmax(doo,std::fabs(ro));
                        nn=std::fmax(nn,std::fabs(gn[c]-rn)); dnn=std::fmax(dnn,std::fabs(rn)); }
                    relK=dkd>0?nk/dkd:nk; relO=doo>0?no/doo:no; relN=dnn>0?nn/dnn:nn;
                    const bool teeth=(cutK>1e-6)&&(cutSpread>1e-6);
                    std::printf("test_gpu_parallel_komegasst np=%d: k rel=%.3e  omega rel=%.3e  nut rel=%.3e  k spread=%.3e\n",
                                nproc, relK, relO, relN, cutSpread);
                    if(!teeth) std::printf("  NO TEETH\n");
                    ok = teeth && relK<1e-6 && relO<1e-6 && relN<1e-6;
                    std::printf("%s\n", ok?"PASS":"FAIL");
                }
            }
        }
    }
    Pstream::finalize();
    return ok ? 0 : 1;
}
