// SpalartAllmaras (one-equation) turbulence model: the nuTilda transport + its coefficients (Stilda/fw/fv1/fv2),
// transcribed from SpalartAllmarasBase::correct(). Uses the shared scalar-transport scaffold (device_scalar_transport.cuh)
// and the public helpers deviceGradU / deviceBoundaryNutSpalding. Verbatim split of device_kepsilon.cu -- no logic change.
#include "device_kepsilon.cuh"          // deviceSpalartAllmaras* decls + deviceGradU / deviceBoundaryNutSpalding / ScalarSolveEntry
#include "device_scalar_transport.cuh"  // deviceSolveScalarTransport scaffold + nBlocks/TPB/turbStore
#include "spalart_coeffs.cuh"
#include "device_ldu.cuh"
#include "device_pcg.cuh"
#include "device_simple.cuh"
#include "device_blas.cuh"
#include "device_ami.cuh"
#include "device_cyclic.cuh"
#include "device_interface.cuh"
#include "device_amg.cuh"
#include "nut_wall_function.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

namespace brae {

// Spalart-Allmaras (one-equation)
// nuTilda transport, transcribed from SpalartAllmarasBase::correct() (incompressible, steady, ft2=off):
//   div(phi,nuTilda) - laplacian((nuTilda+nu)/sigma, nuTilda) + Sp(Cw1*fw*nuTilda/y^2)
//     == Cb1*Stilda*nuTilda + (Cb2/sigma)*|grad nuTilda|^2 ;   nut = nuTilda*fv1.
namespace {
// Stilda = max(Omega + fv2*nuTilda/(kappa*y)^2, Cs*Omega); Omega = sqrt(2)*mag(skew(gradU)) (vorticity magnitude).
__global__
void saStildaKernel(
    int nC,
    const scalar* __restrict__ gradU,
    const scalar* __restrict__ nt,
    const scalar* __restrict__ y,
    scalar nu,
    SpalartAllmarasCoeffs co,
    scalar* __restrict__ Stilda)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    const scalar chi = nt[c] / nu, chi3 = chi*chi*chi, Cv13 = co.Cv1*co.Cv1*co.Cv1;
    const scalar fv1 = chi3 / (chi3 + Cv13);
    const scalar fv2 = 1.0 - chi / (1.0 + chi*fv1);
    const scalar a = gradU[1*nC+c]-gradU[3*nC+c], b = gradU[2*nC+c]-gradU[6*nC+c], d = gradU[5*nC+c]-gradU[7*nC+c];
    const scalar Omega = sqrt(a*a + b*b + d*d);   // sqrt(2)*mag(skew(gradU))
    const scalar kd2 = fmax(co.kappa*y[c]*co.kappa*y[c], 1e-300);
    Stilda[c] = fmax(Omega + fv2*nt[c]/kd2, co.Cs*Omega);
}


// fw = g*((1+Cw3^6)/(g^6+Cw3^6))^(1/6),  g = r + Cw2*(r^6 - r),  r = min(nuTilda/(Stilda*(kappa*y)^2), 10).
__global__
void saFwKernel(
    int nC,
    const scalar* __restrict__ nt,
    const scalar* __restrict__ Stilda,
    const scalar* __restrict__ y,
    SpalartAllmarasCoeffs co,
    scalar* __restrict__ fw)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    const scalar kd2 = fmax(co.kappa*y[c]*co.kappa*y[c], 1e-300);
    scalar r = fmin(nt[c] / (fmax(Stilda[c], 1e-300) * kd2), 10.0);
    const scalar r6 = r*r*r*r*r*r;
    const scalar g = r + co.Cw2*(r6 - r), g6 = g*g*g*g*g*g, Cw36 = pow(co.Cw3, 6.0);
    fw[c] = g * pow((1.0 + Cw36) / (g6 + Cw36), 1.0/6.0);
}


__global__
void saReactionKernel(
    int nC,
    const scalar* __restrict__ V,
    const scalar* __restrict__ nt,
    const scalar* __restrict__ Stilda,
    const scalar* __restrict__ fw,
    const scalar* __restrict__ y,
    const scalar* __restrict__ gradNt2,
    SpalartAllmarasCoeffs co,
    scalar* __restrict__ diag,
    scalar* __restrict__ src)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    const scalar y2 = fmax(y[c]*y[c], 1e-300);
    diag[c] += V[c] * co.Cw1() * fw[c] * nt[c] / y2;                                          // destruction Sp
    src[c]  += V[c] * (co.Cb1 * Stilda[c] * nt[c] + (co.Cb2/co.sigmaNut) * gradNt2[c]);       // production + Cb2 grad^2
}


__global__
void saDEffKernel(int nC, const scalar* __restrict__ nt, scalar nu, scalar sigma, scalar* __restrict__ D)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < nC) D[c] = (nt[c] + nu) / sigma;
}


__global__
void saNutKernel(int nC, const scalar* __restrict__ nt, scalar nu, scalar Cv1, scalar* __restrict__ nut)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    const scalar chi = nt[c]/nu, chi3 = chi*chi*chi, Cv13 = Cv1*Cv1*Cv1;
    nut[c] = nt[c] * (chi3 / (chi3 + Cv13));   // nut = nuTilda*fv1
}


__global__
void saMagSqrKernel(
    int nC,
    const scalar* __restrict__ gx,
    const scalar* __restrict__ gy,
    const scalar* __restrict__ gz,
    scalar* __restrict__ out)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < nC) out[c] = gx[c]*gx[c]+gy[c]*gy[c]+gz[c]*gz[c];
}
} // namespace (SA kernels)

void deviceSpalartAllmarasCorrect(
    const DeviceMesh& dm,
    const DeviceVectorBoundary& dbU,
    const DeviceBoundary& dbNuTilda,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    DeviceBuffer<scalar>& nuTilda,
    DeviceBuffer<scalar>& nut,
    const DeviceBuffer<scalar>& y,
    const DeviceBuffer<scalar>& phiInt,
    const DeviceBuffer<scalar>& phiBnd,
    scalar nu,
    scalar relax,
    scalar tol,
    bool bounded,
    bool limited,
    scalar twoByk,
    const SpalartAllmarasCoeffs& co,
    scalar relTol,
    int checkEvery,
    bool linearUpwind,
    bool nonOrth,
    bool gsK,
    DeviceAMI* ami,
    DeviceCyclic* cyc,
    const ScalarDdt& ntDdt)
{
    const int nC = dm.nCells;
    DeviceBuffer<scalar> gradU;
    deviceGradU(dm, dbU, Ux, Uy, Uz, gradU, ami, cyc);   // interface-aware grad(U) for vorticity/production
    DeviceBuffer<scalar> Stilda(static_cast<std::size_t>(nC));
    saStildaKernel<<<nBlocks(nC), TPB>>>(nC, gradU.data(), nuTilda.data(), y.data(), nu, co, Stilda.data());
    DeviceBuffer<scalar> fw(static_cast<std::size_t>(nC));
    saFwKernel<<<nBlocks(nC), TPB>>>(nC, nuTilda.data(), Stilda.data(), y.data(), co, fw.data());
    DeviceBuffer<scalar> nbv;
    deviceBCValue(dbNuTilda, nuTilda, nbv);   // |grad nuTilda|^2
    DeviceBuffer<scalar> gnx, gny, gnz;
    deviceGaussGrad(dm, nuTilda, nbv, gnx, gny, gnz);
    DeviceBuffer<scalar> gradNt2(static_cast<std::size_t>(nC));
    saMagSqrKernel<<<nBlocks(nC), TPB>>>(nC, gnx.data(), gny.data(), gnz.data(), gradNt2.data());
    if (const char* sapath = std::getenv("BRAE_DUMP_SA"))   // cell-level SA-term dump (vs OF reference) -- first call only
    {
        static bool saDumped = false;
        if (!saDumped)
        {
            saDumped = true;
            const auto hgU = gradU.host(); const auto hnt = nuTilda.host(); const auto hy = y.host();
            const auto hSt = Stilda.host(); const auto hfw = fw.host(); const auto hg2 = gradNt2.host();
            std::FILE* f = std::fopen(sapath, "w");
            std::fprintf(f, "cell,nuTilda,y,Omega,Stilda,fw,gradNt2,P,D,nut\n");
            for (int c = 0; c < nC; ++c)
            {
                const scalar a = hgU[1*nC+c]-hgU[3*nC+c], b = hgU[2*nC+c]-hgU[6*nC+c], d = hgU[5*nC+c]-hgU[7*nC+c];
                const scalar Om = std::sqrt(a*a + b*b + d*d);
                const scalar chi = hnt[c]/nu, chi3 = chi*chi*chi, Cv13 = co.Cv1*co.Cv1*co.Cv1, fv1 = chi3/(chi3+Cv13);
                const scalar y2 = hy[c]*hy[c] > 1e-300 ? hy[c]*hy[c] : 1e-300;
                const scalar P = co.Cb1*hSt[c]*hnt[c] + (co.Cb2/co.sigmaNut)*hg2[c];
                const scalar D = co.Cw1()*hfw[c]*hnt[c]/y2;
                std::fprintf(f, "%d,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e\n",
                             c, hnt[c], hy[c], Om, hSt[c], hfw[c], hg2[c], P, D, hnt[c]*fv1);
            }
            std::fclose(f);
            std::fprintf(stderr, "[BRAE_DUMP_SA] wrote %d cells to %s\n", nC, sapath);
        }
    }
    DeviceBuffer<scalar> D(static_cast<std::size_t>(nC));
    saDEffKernel<<<nBlocks(nC), TPB>>>(nC, nuTilda.data(), nu, co.sigmaNut, D.data());
    DeviceBuffer<scalar> divU;
    deviceDiv(dm, phiInt, phiBnd, divU);   // for the bounded term
    if (ami && ami->n) interfaceAddDiv(*ami, dm.V, divU);
    if (cyc && cyc->n) interfaceAddDiv(*cyc, dm.V, divU);
    cudaCheck(cudaGetLastError(), "SA assemble");
    deviceSolveScalarTransport(dm, dbNuTilda, nuTilda, "nuTilda", D, phiInt, phiBnd, divU, bounded, limited, linearUpwind, nonOrth, twoByk,
                               relax, tol, relTol, checkEvery, gsK,
                               [&](DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& src){
                                   saReactionKernel<<<nBlocks(nC), TPB>>>(nC, dm.V.data(), nuTilda.data(), Stilda.data(),
                                       fw.data(), y.data(), gradNt2.data(), co, diag.data(), src.data()); },
                               nullptr, nullptr, ami, cyc, ntDdt);
    // deviceSolveScalarTransport already bounds to 1e-15 (~ bound(nuTilda, 0)). correctNut: nut = nuTilda*fv1(new).
    saNutKernel<<<nBlocks(nC), TPB>>>(nC, nuTilda.data(), nu, co.Cv1, nut.data());
    cudaCheck(cudaGetLastError(), "SA correctNut");
}


// Standalone SA correctNut (nut = nuTilda*fv1), used by the solver's startup validate() (OF
// eddyViscosity::validate()->correctNut()), so the FIRST momentum predictor sees a consistent nut.
void deviceNutSA(const DeviceBuffer<scalar>& nuTilda, scalar nu, scalar Cv1, DeviceBuffer<scalar>& nut)
{
    const int nC = static_cast<int>(nuTilda.size());
    nut.resize(nC);
    saNutKernel<<<nBlocks(nC), TPB>>>(nC, nuTilda.data(), nu, Cv1, nut.data());
    cudaCheck(cudaGetLastError(), "SA correctNut (validate)");
}

// ---- Exported SA (Spalart-Allmaras) source-prep wrappers for the DISTRIBUTED SA correct -----------------------
// Cell-local (gradU + grad(nuTilda) are the only halo-coupled inputs, supplied by the caller). Reuse the exact
// anon kernels + coeffs so the distributed nuTilda equation is byte-identical to the single-GPU physics.
void deviceSAStilda(const DeviceMesh& dm, const DeviceBuffer<scalar>& gradU, const DeviceBuffer<scalar>& nuTilda,
    const DeviceBuffer<scalar>& y, scalar nu, const SpalartAllmarasCoeffs& co, DeviceBuffer<scalar>& Stilda)
{
    const int nC = dm.nCells; Stilda.resize(nC);
    saStildaKernel<<<nBlocks(nC), TPB>>>(nC, gradU.data(), nuTilda.data(), y.data(), nu, co, Stilda.data());
    cudaCheck(cudaGetLastError(), "deviceSAStilda");
}
void deviceSAFw(const DeviceMesh& dm, const DeviceBuffer<scalar>& nuTilda, const DeviceBuffer<scalar>& Stilda,
    const DeviceBuffer<scalar>& y, const SpalartAllmarasCoeffs& co, DeviceBuffer<scalar>& fw)
{
    const int nC = dm.nCells; fw.resize(nC);
    saFwKernel<<<nBlocks(nC), TPB>>>(nC, nuTilda.data(), Stilda.data(), y.data(), co, fw.data());
    cudaCheck(cudaGetLastError(), "deviceSAFw");
}
void deviceSAMagSqr(const DeviceMesh& dm, const DeviceBuffer<scalar>& gx, const DeviceBuffer<scalar>& gy,
    const DeviceBuffer<scalar>& gz, DeviceBuffer<scalar>& out)
{
    const int nC = dm.nCells; out.resize(nC);
    saMagSqrKernel<<<nBlocks(nC), TPB>>>(nC, gx.data(), gy.data(), gz.data(), out.data());
    cudaCheck(cudaGetLastError(), "deviceSAMagSqr");
}
void deviceSADEff(const DeviceMesh& dm, const DeviceBuffer<scalar>& nuTilda, scalar nu, scalar sigmaNut, DeviceBuffer<scalar>& D)
{
    const int nC = dm.nCells; D.resize(nC);
    saDEffKernel<<<nBlocks(nC), TPB>>>(nC, nuTilda.data(), nu, sigmaNut, D.data());
    cudaCheck(cudaGetLastError(), "deviceSADEff");
}
void deviceSAReaction(const DeviceMesh& dm, const DeviceBuffer<scalar>& nuTilda, const DeviceBuffer<scalar>& Stilda,
    const DeviceBuffer<scalar>& fw, const DeviceBuffer<scalar>& y, const DeviceBuffer<scalar>& gradNt2,
    const SpalartAllmarasCoeffs& co, DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& src)
{
    saReactionKernel<<<nBlocks(dm.nCells), TPB>>>(dm.nCells, dm.V.data(), nuTilda.data(), Stilda.data(),
        fw.data(), y.data(), gradNt2.data(), co, diag.data(), src.data());
    cudaCheck(cudaGetLastError(), "deviceSAReaction");
}

} // namespace brae
