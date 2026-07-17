// Phase 4b: the CLOSED distributed turbulent SIMPLE loop (momentum + pressure + kEpsilon) reproduces np=1.
//
// This is the turbulence counterpart of test_gpu_parallel_loop: it drives ParallelDeviceSimple with
// enableTurbulence() for N iterations, so each step runs the momentum/pressure with nuEff = nu + nut AND then
// parallelDeviceKEpsilonCorrect. U/p/phi/k/eps/nut all stay device-resident and are maintained across
// iterations -- the real test that the turbulent step composes correctly under a maintained flux, not just a
// single correct() (test_gpu_parallel_kepsilon) in isolation.
//
// np=1 (no processor faces -> the validated local turbulent step) is the oracle; np=2/4 must reproduce it.
// Duct walls y/z, cut x -> near-wall cells adjacent to the cut.
//
// Run: mpirun -np {1,2,4} test_gpu_parallel_turbloop <Nx> <Ny> <Nz> [ref.txt]
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
    if (argc < 4) { if (Pstream::master()) std::printf("usage: %s <Nx> <Ny> <Nz> [ref]\n", argv[0]); Pstream::finalize(); return 2; }
    if (!Pstream::nvshmemActive()) { if (Pstream::master()) std::printf("test_gpu_parallel_turbloop: NVSHMEM inactive -- SKIP\n"); Pstream::finalize(); return 0; }

    const label Nx = std::atol(argv[1]), Ny = std::atol(argv[2]), Nz = std::atol(argv[3]);
    const std::string ref = (argc > 4) ? argv[4] : "turbloop_ref.txt";
    const std::string dump = (argc > 5) ? argv[5] : "";
    const scalar nu = 1e-3, relaxU = 0.7, relaxP = 0.3, relaxK = 0.7, tolU = 1e-9, tolP = 1e-9;
    const scalar kIn = 0.375, epsIn = 0.09 * std::pow(kIn, 1.5) / 0.1;
    const int maxIter = 2000, N = 4;
    KEpsilonCoeffs co;
    const label nC = Nx * Ny * Nz;
    bool ok = true;
    scalar relU = 0, relP = 0, relK = 0, relN = 0;
    {
        const PrimitiveMesh gm = boxtest::boxMesh(Nx, Ny, Nz);
        std::vector<label> cellToPart(nC);
        for (label k = 0; k < Nz; ++k) for (label j = 0; j < Ny; ++j) for (label i = 0; i < Nx; ++i)
            cellToPart[i + Nx * (j + Ny * k)] = static_cast<label>((static_cast<long long>(i) * nproc) / Nx);
        const Partition P(gm, cellToPart, rank);
        const label lnC = P.nCells();

        FieldData<vector> Ufd; Ufd.internalUniform = true; Ufd.internalUniformValue = vector{1, 0, 0};
        Ufd.boundary.push_back(boxtest::pfd<vector>("inlet","fixedValue",vector{1,0,0},true));
        Ufd.boundary.push_back(boxtest::pfd<vector>("outlet","zeroGradient",vector{0,0,0},false));
        for (const char* w : {"wallYmin","wallYmax","wallZmin","wallZmax"})
            Ufd.boundary.push_back(boxtest::pfd<vector>(w,"fixedValue",vector{0,0,0},true));
        auto pfdta = [&](){ FieldData<scalar> f; f.internalUniform=true; f.internalUniformValue=0.0;
            f.boundary.push_back(boxtest::pfd<scalar>("inlet","zeroGradient",0.0,false));
            f.boundary.push_back(boxtest::pfd<scalar>("outlet","fixedValue",0.0,true));
            for (const char* w : {"wallYmin","wallYmax","wallZmin","wallZmax"}) f.boundary.push_back(boxtest::pfd<scalar>(w,"zeroGradient",0.0,false));
            return f; }();
        auto scalfd = [&](scalar in){ FieldData<scalar> f; f.internalUniform=true; f.internalUniformValue=in;
            f.boundary.push_back(boxtest::pfd<scalar>("inlet","fixedValue",in,true));
            f.boundary.push_back(boxtest::pfd<scalar>("outlet","zeroGradient",0.0,false));
            for (const char* w : {"wallYmin","wallYmax","wallZmin","wallZmax"}) f.boundary.push_back(boxtest::pfd<scalar>(w,"zeroGradient",0.0,false));
            return f; };

        GeometricField<vector> U0 = distributeField<vector>(Ufd, gm.patches(), P.Lm, P.lp, P.procW, rank); U0.evaluateBoundary();
        GeometricField<scalar> p0 = distributeField<scalar>(pfdta, gm.patches(), P.Lm, P.lp, P.procW, rank); p0.evaluateBoundary();
        GeometricField<scalar> k0   = distributeField<scalar>(scalfd(kIn),   gm.patches(), P.Lm, P.lp, P.procW, rank); k0.evaluateBoundary();
        GeometricField<scalar> eps0 = distributeField<scalar>(scalfd(epsIn), gm.patches(), P.Lm, P.lp, P.procW, rank); eps0.evaluateBoundary();
        GeometricField<scalar> nut0 = distributeField<scalar>(scalfd(co.Cmu*kIn*kIn/epsIn), gm.patches(), P.Lm, P.lp, P.procW, rank); nut0.evaluateBoundary();

        ParallelDeviceSimple solver(P, U0, p0, nu, relaxU, relaxP, tolU, tolP, maxIter);
        solver.enableTurbulence(U0, k0, eps0, nut0, co, relaxK, relaxK);
        if (!dump.empty()) solver.setStageDump(dump, (argc>6)?std::atoi(argv[6]):2);
        for (int it = 0; it < N; ++it) solver.step();
        cudaDeviceSynchronize();

        // gather to the global mesh
        auto gvec = [&](const std::vector<vector>& l){ std::vector<scalar> g(3*nC,0.0);
            for(label c=0;c<lnC;++c){const label gc=P.Lm.cellProcAddr[c]; g[3*gc]=l[c].x; g[3*gc+1]=l[c].y; g[3*gc+2]=l[c].z;}
            Pstream::allReduce(g.data(),(int)g.size(),ReduceOp::Sum); return g; };
        auto gsca = [&](const std::vector<scalar>& l){ std::vector<scalar> g(nC,0.0);
            for(label c=0;c<lnC;++c) g[P.Lm.cellProcAddr[c]]=l[c]; Pstream::allReduce(g.data(),(int)g.size(),ReduceOp::Sum); return g; };
        const std::vector<scalar> gU = gvec(solver.U()), gp = gsca(solver.p()), gk = gsca(solver.kLocal()), gn = gsca(solver.nutLocal());

        if (Pstream::master()){
            if (nproc==1){
                std::ofstream os(ref); os.precision(17);
                for(label c=0;c<nC;++c) os<<gU[3*c]<<' '<<gU[3*c+1]<<' '<<gU[3*c+2]<<' '<<gp[c]<<' '<<gk[c]<<' '<<gn[c]<<'\n';
                std::printf("test_gpu_parallel_turbloop np=1: wrote reference %s (%ld cells)\n", ref.c_str(),(long)nC);
                std::printf("PASS\n");
            } else {
                std::ifstream is(ref);
                if(!is){ std::printf("cannot read %s -- run np=1 first\n", ref.c_str()); ok=false; }
                else {
                    scalar nU=0,dU=0,nP=0,dP=0,nk=0,dkd=0,nn=0,dnn=0;
                    for(label c=0;c<nC;++c){ scalar ux,uy,uz,pp,kk,nn2; is>>ux>>uy>>uz>>pp>>kk>>nn2;
                        nU=std::fmax(nU,std::fabs(gU[3*c]-ux)); nU=std::fmax(nU,std::fabs(gU[3*c+1]-uy)); dU=std::fmax(dU,std::fabs(ux));
                        nP=std::fmax(nP,std::fabs(gp[c]-pp)); dP=std::fmax(dP,std::fabs(pp));
                        nk=std::fmax(nk,std::fabs(gk[c]-kk)); dkd=std::fmax(dkd,std::fabs(kk));
                        nn=std::fmax(nn,std::fabs(gn[c]-nn2)); dnn=std::fmax(dnn,std::fabs(nn2)); }
                    relU=dU>0?nU/dU:nU; relP=dP>0?nP/dP:nP; relK=dkd>0?nk/dkd:nk; relN=dnn>0?nn/dnn:nn;
                    const scalar gate = 1e-4;   // N turbulent iterations of parallel-solver-floor drift
                    std::printf("test_gpu_parallel_turbloop np=%d (%d iters): U rel=%.3e  p rel=%.3e  k rel=%.3e  nut rel=%.3e\n",
                                nproc, N, relU, relP, relK, relN);
                    ok = relU<gate && relP<gate && relK<gate && relN<gate;
                    std::printf("%s\n", ok?"PASS":"FAIL");
                }
            }
        }
    }
    Pstream::finalize();
    return ok ? 0 : 1;
}
