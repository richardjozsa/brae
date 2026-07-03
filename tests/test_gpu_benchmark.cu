// GPU offload: benchmark, the device-resident SIMPLE loop with the AMG-PCG pressure solve, timed against
// the CPU loop. The AMG hierarchy is built once (static geometry); the coarse matrix is re-Galerkined each
// iteration (rAU changes). Reports wall-clock + the AMG iteration count, and validates U/p still match CPU.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "fv_matrix_ops.cuh"
#include "solve_vector.cuh"
#include "pcg.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_ldu.cuh"
#include "device_blas.cuh"
#include "device_pcg.cuh"
#include "device_simple.cuh"
#include "device_boundary.cuh"
#include "device_amg.cuh"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>
#include <cuda_runtime.h>

using namespace brae;
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

int main(int argc, char** argv) {
    const std::string caseDir = argc > 1 ? argv[1] : "validation/pitzDaily";
    PrimitiveMesh m; m.read(caseDir + "/constant/polyMesh");
    FvGeometry g; g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    const scalar nu = 1e-3, tol = 1e-8, relaxU = 0.7, relaxP = 0.3; const int N = 20;

    auto mkU = [&]() { GeometricField<vector> U; U.internal.resize(nC);
        for (label c=0;c<nC;++c) U.internal[c]={1.0+0.2*std::sin(0.01*c),0.1*std::cos(0.013*c),0.0};
        for (const FvPatch& q:fvp){ if(q.type=="empty")U.boundary.push_back(std::make_unique<EmptyPatchField<vector>>(q));
            else if(q.name=="inlet")U.boundary.push_back(std::make_unique<FixedValuePatchField<vector>>(q,true,vector{10,0,0},std::vector<vector>{}));
            else if(q.type=="wall")U.boundary.push_back(std::make_unique<NoSlipPatchField<vector>>(q));
            else U.boundary.push_back(std::make_unique<ZeroGradientPatchField<vector>>(q)); } U.evaluateBoundary(); return U; };
    auto mkP = [&]() { GeometricField<scalar> p; p.internal.assign(nC,0.0);
        for (const FvPatch& q:fvp){ if(q.type=="empty")p.boundary.push_back(std::make_unique<EmptyPatchField<scalar>>(q));
            else if(q.name=="outlet")p.boundary.push_back(std::make_unique<FixedValuePatchField<scalar>>(q,true,0.0,std::vector<scalar>{}));
            else p.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q)); } p.evaluateBoundary(); return p; };

    // ---- CPU loop (timed) ----
    std::vector<scalar> Ucx, pc; double cpuMs;
    { GeometricField<vector> U=mkU(); GeometricField<scalar> p=mkP(); SurfaceScalarField phi=fvc::flux(U,m,g,fvp);
      const auto t0=Clock::now();
      for (int it=0;it<N;++it){
        const std::vector<vector> gP=fvc::gaussGrad(p,m,g,fvp);
        FvVectorMatrix UEqn=fvm::div(phi.internal,phi.boundary,U,m,fvp); addEqual(UEqn,fvm::laplacian(U,nu,m,g,fvp),-1.0);
        relaxMatrix(UEqn,U,m,fvp,relaxU); FvVectorMatrix Mp=UEqn; for(label c=0;c<nC;++c)Mp.source[c]+=(-g.V()[c])*gP[c]; solveVector(Mp,U,m,fvp,tol,0.0,2000);
        const std::vector<scalar> A=matrixA(UEqn,m,g,fvp); std::vector<scalar> rAU(nC); for(label c=0;c<nC;++c)rAU[c]=1.0/A[c];
        const std::vector<vector> H=matrixH(UEqn,U,m,g,fvp); GeometricField<vector> HbyA=mkU(); for(label c=0;c<nC;++c)HbyA.internal[c]=rAU[c]*H[c]; HbyA.evaluateBoundary();
        SurfaceScalarField phiHbyA=fvc::flux(HbyA,m,g,fvp); const SurfaceScalarField rAUf=fvc::interpolate(rAU,m,g,fvp);
        FvScalarMatrix pEqn=fvm::laplacian(rAUf,p,m,g,fvp); const std::vector<scalar> dphi=fvc::div(phiHbyA,m,g,fvp); for(label c=0;c<nC;++c)pEqn.source[c]+=g.V()[c]*dphi[c];
        const std::vector<scalar> pPrev=p.internal; pcg(pEqn,p.internal,m,fvp,tol,0.0,10000,0); const SurfaceScalarField pflux=matrixFlux(pEqn,p,m,fvp);
        for(label f=0;f<nIf;++f)phi.internal[f]=phiHbyA.internal[f]-pflux.internal[f];
        for(std::size_t pi=0;pi<fvp.size();++pi)for(label i=0;i<fvp[pi].size;++i)phi.boundary[pi][i]=phiHbyA.boundary[pi][i]-pflux.boundary[pi][i];
        for(label c=0;c<nC;++c)p.internal[c]=pPrev[c]+relaxP*(p.internal[c]-pPrev[c]); p.evaluateBoundary();
        const std::vector<vector> gPn=fvc::gaussGrad(p,m,g,fvp); for(label c=0;c<nC;++c)U.internal[c]=HbyA.internal[c]-rAU[c]*gPn[c]; U.evaluateBoundary(); }
      cpuMs=ms(t0,Clock::now()); Ucx.resize(nC); pc=p.internal; for(label c=0;c<nC;++c)Ucx[c]=U.internal[c].x; }

    // ---- GPU resident loop with AMG-PCG pressure (timed) ----
    const DeviceMesh dm=buildDeviceMesh(m,g,fvp);
    GeometricField<vector> U0=mkU(); GeometricField<scalar> p0=mkP();
    const DeviceVectorBoundary dbU=buildDeviceVectorBoundary(U0,fvp,g); const DeviceBoundary dbP=buildDeviceBoundary(p0,fvp,g);
    const SurfaceScalarField phi0=fvc::flux(U0,m,g,fvp);
    DeviceBuffer<scalar> Uk[3],dp(p0.internal),phiInt(phi0.internal);
    { std::vector<scalar> ux(nC),uy(nC),uz(nC); for(label c=0;c<nC;++c){ux[c]=U0.internal[c].x;uy[c]=U0.internal[c].y;uz[c]=U0.internal[c].z;} Uk[0].copyFrom(ux);Uk[1].copyFrom(uy);Uk[2].copyFrom(uz); }
    DeviceBuffer<scalar> phiBnd(([&]{std::vector<scalar> o;for(auto&a:phi0.boundary)o.insert(o.end(),a.begin(),a.end());return o;}()));
    DeviceBuffer<scalar> nuU(std::vector<scalar>(nIf,nu)),zeroSrc(std::vector<scalar>(nC,0.0)),zeroBndU(std::vector<scalar>(dbU.n,0.0)),nuCell(std::vector<scalar>(nC,nu));
    DeviceBuffer<scalar> lD,lU,lL; deviceLaplacianCoeffs(dm,nuU,lD,lU,lL);
    const std::vector<label> ownerInt(m.owner().begin(),m.owner().begin()+nIf);
    const std::vector<scalar> magSfInt(g.magSf().begin(),g.magSf().begin()+nIf);
    AMGData amg=buildAMG(ownerInt,m.neighbour(),magSfInt,nC);    // built once (static geometry)
    auto sm=[](DeviceBuffer<scalar>&b){return deviceSumMag(b)+1e-20;};
    int lastAmgIters=0;
    cudaDeviceSynchronize(); const auto t0=Clock::now();
    for (int it=0;it<N;++it){
        DeviceBuffer<scalar> pbv; deviceBCValue(dbP,dp,pbv); DeviceBuffer<scalar> gx,gy,gz; deviceGaussGrad(dm,dp,pbv,gx,gy,gz);
        DeviceBuffer<scalar> mDiag,mUp,mLo; deviceDivUpwindCoeffs(dm,phiInt,mDiag,mUp,mLo);
        deviceAxpy(-1.0,lD,mDiag); deviceAxpy(-1.0,lU,mUp); deviceAxpy(-1.0,lL,mLo);
        DeviceBuffer<scalar>* gg[3]={&gx,&gy,&gz};
        DeviceBuffer<scalar> r0IC,r0BC,r0lIC,r0lBC; deviceBCDivCoeffs(dbU.comp[0],phiBnd,r0IC,r0BC); deviceBCLaplacianCoeffs(dbU.comp[0],nuCell,r0lIC,r0lBC); deviceAxpy(-1.0,r0lIC,r0IC);
        DeviceBuffer<scalar> mDiagR,delta; deviceRelaxDiag(deviceLduView(dm,mDiag,mUp,mLo),dm,r0IC,relaxU,mDiagR,delta);
        const DeviceLduView Uview=deviceLduView(dm,mDiagR,mUp,mLo);
        DeviceBuffer<scalar> iC[3],bCb[3],relaxSrc[3];
        for(int kk=0;kk<3;++kk){ DeviceBuffer<scalar> dIC,dBC,lIC,lBC; deviceBCDivCoeffs(dbU.comp[kk],phiBnd,dIC,dBC); deviceBCLaplacianCoeffs(dbU.comp[kk],nuCell,lIC,lBC);
            deviceAxpy(-1.0,lIC,dIC); deviceAxpy(-1.0,lBC,dBC); iC[kk]=std::move(dIC); bCb[kk]=std::move(dBC);
            deviceHadamard(relaxSrc[kk],delta,Uk[kk]); DeviceBuffer<scalar> s; deviceHadamard(s,dm.V,*gg[kk]); deviceScale(s,-1.0); deviceAxpy(1.0,relaxSrc[kk],s);
            DeviceBuffer<scalar> diagC,b; deviceFold(dm,mDiagR,s,iC[kk],bCb[kk],diagC,b); deviceJacobiBiCGStab(deviceLduView(dm,diagC,mUp,mLo),b,Uk[kk],sm(b),tol,0.0,5000); }
        DeviceBuffer<scalar> diagA,dumb,rAU; deviceFold(dm,mDiagR,zeroSrc,iC[0],zeroBndU,diagA,dumb); deviceReciprocalV(dm,diagA,rAU);
        DeviceBuffer<scalar> HbyA[3]; for(int kk=0;kk<3;++kk){ DeviceBuffer<scalar> Hk; deviceMatrixH(Uview,dm,Uk[kk],relaxSrc[kk],zeroBndU,bCb[kk],Hk); deviceHadamard(HbyA[kk],rAU,Hk); }
        DeviceBuffer<scalar> phiHi; deviceVectorFlux(dm,HbyA[0],HbyA[1],HbyA[2],phiHi);
        DeviceBuffer<scalar> hxb,hyb,hzb; deviceBCValue(dbU.comp[0],HbyA[0],hxb); deviceBCValue(dbU.comp[1],HbyA[1],hyb); deviceBCValue(dbU.comp[2],HbyA[2],hzb);
        DeviceBuffer<scalar> phiHb; deviceBoundaryFlux(dm,hxb,hyb,hzb,phiHb);
        DeviceBuffer<scalar> rAUf; deviceInterpolate(dm,rAU,rAUf); DeviceBuffer<scalar> pD,pU,pL; deviceLaplacianCoeffs(dm,rAUf,pD,pU,pL);
        DeviceBuffer<scalar> pIC,pBC; deviceBCLaplacianCoeffs(dbP,rAU,pIC,pBC); DeviceBuffer<scalar> divPhiH; deviceDiv(dm,phiHi,phiHb,divPhiH);
        DeviceBuffer<scalar> diagCp,bp; deviceFoldPressure(dm,pD,divPhiH,pIC,pBC,diagCp,bp);
        DeviceBuffer<scalar> pPrev; deviceCopy(pPrev,dp);
        amgGalerkin(amg,diagCp,pU,pL);                                          // re-coarsen the pressure matrix
        const DeviceSolverPerf pp=deviceAMGPCG(deviceLduView(dm,diagCp,pU,pL),amg,bp,dp,sm(bp),tol,0.0,3000);
        lastAmgIters=pp.nIterations;
        const DeviceLduView pview=deviceLduView(dm,diagCp,pU,pL); DeviceBuffer<scalar> pfi; deviceMatrixFluxInternal(pview,dp,pfi); deviceCopy(phiInt,phiHi); deviceAxpy(-1.0,pfi,phiInt);
        DeviceBuffer<scalar> pfb; deviceMatrixFluxBoundary(dbP,pIC,pBC,dp,pfb); deviceCopy(phiBnd,phiHb); deviceAxpy(-1.0,pfb,phiBnd);
        deviceScale(dp,relaxP); deviceAxpy(1.0-relaxP,pPrev,dp);
        DeviceBuffer<scalar> pbv2; deviceBCValue(dbP,dp,pbv2); DeviceBuffer<scalar> gnx,gny,gnz; deviceGaussGrad(dm,dp,pbv2,gnx,gny,gnz);
        DeviceBuffer<scalar>* gn[3]={&gnx,&gny,&gnz}; for(int kk=0;kk<3;++kk){ DeviceBuffer<scalar> Un; deviceCorrector(HbyA[kk],rAU,*gn[kk],Un); Uk[kk]=std::move(Un); } }
    cudaDeviceSynchronize(); const double gpuMs=ms(t0,Clock::now());

    const std::vector<scalar> uxg=Uk[0].host(), pg=dp.host();
    scalar nU=0,dU=0,nP=0,dP=0; for(label c=0;c<nC;++c){nU=std::fmax(nU,std::fabs(uxg[c]-Ucx[c]));dU=std::fmax(dU,std::fabs(Ucx[c]));nP=std::fmax(nP,std::fabs(pg[c]-pc[c]));dP=std::fmax(dP,std::fabs(pc[c]));}

    std::printf("GPU benchmark, resident SIMPLE loop with AMG-PCG (%d steps, nCells=%d):\n", N, nC);
    std::printf("  CPU loop : %8.1f ms  (%.2f ms/iter)\n", cpuMs, cpuMs/N);
    std::printf("  GPU loop : %8.1f ms  (%.2f ms/iter)  [AMG pressure %d iters/solve]\n", gpuMs, gpuMs/N, lastAmgIters);
    std::printf("  speedup  : %.2fx  | U.x %.2e  p %.2e vs CPU\n", cpuMs/gpuMs, nU/dU, nP/dP);
    const bool pass = nU/dU < 1e-4 && nP/dP < 1e-4;     // correctness gate (timing is informational)
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
