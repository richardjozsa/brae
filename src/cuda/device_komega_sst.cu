// cf GPU offload: k-omega SST blending (F1/F2), cross-diffusion (CDkOmega), strain (S2), and the coefficient
// blend. Pointwise per-cell kernels mirroring kOmegaSSTBase.C (v2412). See the header for the formulas.
#include "device_komega_sst.cuh"
#include "device_kepsilon.cuh"   // DeviceWallData (shared wall geometry)
#include "nut_wall_function.cuh" // nutkWallFunctionValue / yPlusWall (shared wall-nut physics, "G0 IDENTICAL to eps WF")
#include <cuda_runtime.h>
#include <cmath>

namespace brae {

namespace {
constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }
inline scalar yPlusLamHost(scalar kappa, scalar E) { scalar y = 11.0; for (int i = 0; i < 10; ++i) y = std::log(std::fmax(E*y, 1.0)) / kappa; return y; }


__global__
void s2Kernel(int nC, const scalar* __restrict__ gradU, scalar* __restrict__ S2)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    scalar t[9];
    for (int q = 0; q < 9; ++q)
        t[q] = gradU[q * nC + c];
    scalar s = 0.0;   // magSqr(symm(gradU)) = sum_ab symm_ab^2
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
        {
            const scalar sab = 0.5 * (t[a*3+b] + t[b*3+a]);
            s += sab * sab;
        }
    S2[c] = 2.0 * s;
}


__global__
void cdKernel(
    int nC,
    const scalar* __restrict__ gKx,
    const scalar* __restrict__ gKy,
    const scalar* __restrict__ gKz,
    const scalar* __restrict__ gOx,
    const scalar* __restrict__ gOy,
    const scalar* __restrict__ gOz,
    const scalar* __restrict__ om,
    scalar twoA2,
    scalar* __restrict__ CD)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    const scalar dot = gKx[c]*gOx[c] + gKy[c]*gOy[c] + gKz[c]*gOz[c];   // grad k . grad omega
    CD[c] = twoA2 * dot / om[c];
}


__global__
void f1Kernel(
    int nC,
    const scalar* __restrict__ k,
    const scalar* __restrict__ om,
    const scalar* __restrict__ y,
    const scalar* __restrict__ CD,
    scalar nu,
    scalar betaStar,
    scalar alphaOmega2,
    int lm,
    scalar* __restrict__ F1)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    const scalar kk = k[c], w = om[c], yy = y[c];
    const scalar CDplus = fmax(CD[c], (scalar)1.0e-10);
    const scalar a  = (1.0 / betaStar) * sqrt(kk) / (w * yy);
    const scalar b  = 500.0 * nu / (yy * yy * w);
    const scalar cc = (4.0 * alphaOmega2) * kk / (CDplus * yy * yy);
    const scalar arg1 = fmin(fmin(fmax(a, b), cc), (scalar)10.0);
    const scalar a2 = arg1 * arg1;   // pow4(arg1) = (arg1^2)^2
    scalar f1 = tanh(a2 * a2);
    // kOmegaSSTLM override (OF kOmegaSSTLM.C:42-52): F1 = max(F1, F3), F3 = exp(-(Ry/120)^8), Ry = y*sqrt(k)/nu.
    // Forces F1->1 in the near-wall transitional band (Ry<120) -> keeps the model in k-omega inner mode + suppresses
    // the (F1-1)*CDkOmega cross-diffusion source there. Only for the LM path (base kOmegaSST has no F3).
    if (lm)
    {
        const scalar Ry = yy * sqrt(kk) / nu;
        const scalar r = Ry / 120.0;
        const scalar r2 = r * r, r4 = r2 * r2;
        f1 = fmax(f1, exp(-(r4 * r4)));
    }
    F1[c] = f1;
}


__global__
void f2Kernel(
    int nC,
    const scalar* __restrict__ k,
    const scalar* __restrict__ om,
    const scalar* __restrict__ y,
    scalar nu,
    scalar betaStar,
    scalar* __restrict__ F2)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    const scalar kk = k[c], w = om[c], yy = y[c];
    const scalar a = (2.0 / betaStar) * sqrt(kk) / (w * yy);
    const scalar b = 500.0 * nu / (yy * yy * w);
    const scalar arg2 = fmin(fmax(a, b), (scalar)100.0);
    F2[c] = tanh(arg2 * arg2);   // sqr(arg2)
}


__global__
void blendKernel(int nC, const scalar* __restrict__ F1, scalar psi1, scalar psi2, scalar* __restrict__ out)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < nC) out[c] = F1[c] * (psi1 - psi2) + psi2;
}


// correctNut: nut = a1*k / max(a1*omega, b1*F2*sqrt(S2))
__global__
void nutSSTKernel(
    int nC,
    const scalar* __restrict__ k,
    const scalar* __restrict__ om,
    const scalar* __restrict__ F2,
    const scalar* __restrict__ S2,
    scalar a1,
    scalar b1,
    scalar* __restrict__ nut)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    const scalar denom = fmax(a1 * om[c], b1 * F2[c] * sqrt(S2[c]));
    nut[c] = a1 * k[c] / denom;
}


// Pk = min(G, c1*betaStar*k*omega)
__global__
void pkKernel(
    int nC,
    const scalar* __restrict__ G,
    const scalar* __restrict__ k,
    const scalar* __restrict__ om,
    scalar c1betaStar,
    scalar* __restrict__ Pk)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    Pk[c] = fmin(G[c], c1betaStar * k[c] * om[c]);
}


// GbyNu = min(GbyNu0, (c1/a1)*betaStar*omega*max(a1*omega, b1*F2*sqrt(S2)))
__global__
void gbyNuLimitKernel(
    int nC,
    const scalar* __restrict__ GbyNu0,
    const scalar* __restrict__ om,
    const scalar* __restrict__ F2,
    const scalar* __restrict__ S2,
    scalar a1,
    scalar b1,
    scalar c1,
    scalar betaStar,
    scalar* __restrict__ GbyNu)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    const scalar denom = fmax(a1 * om[c], b1 * F2[c] * sqrt(S2[c]));
    GbyNu[c] = fmin(GbyNu0[c], (c1 / a1) * betaStar * om[c] * denom);
}


// D = (F1*(alpha1-alpha2)+alpha2)*nut + nu
__global__
void dEffKernel(
    int nC,
    const scalar* __restrict__ F1,
    const scalar* __restrict__ nut,
    scalar alpha1,
    scalar alpha2,
    scalar nu,
    scalar* __restrict__ D)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    D[c] = (F1[c] * (alpha1 - alpha2) + alpha2) * nut[c] + nu;
}


// omega reaction: diag += V*(max(sp1,0) + beta*omega + max(sp2,0));
//                 source += V*gamma*GbyNu0lim - V*min(sp1,0)*omega - V*min(sp2,0)*omega
//   sp1 = (2/3)*gamma*divU   (SuSp),   beta*omega (Sp),   sp2 = (F1-1)*CDkOmega/omega   (SuSp)
__global__
void omegaReactionKernel(
    int nC,
    const scalar* __restrict__ V,
    const scalar* __restrict__ gamma,
    const scalar* __restrict__ beta,
    const scalar* __restrict__ GbyNu0,
    const scalar* __restrict__ F1,
    const scalar* __restrict__ CD,
    const scalar* __restrict__ om,
    const scalar* __restrict__ divU,
    scalar* __restrict__ diag,
    scalar* __restrict__ source)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    const scalar sp1 = (2.0/3.0) * gamma[c] * divU[c];
    const scalar sp2 = (F1[c] - 1.0) * CD[c] / om[c];
    diag[c]   += V[c] * (fmax(sp1, 0.0) + beta[c]*om[c] + fmax(sp2, 0.0));
    source[c] += V[c] * gamma[c]*GbyNu0[c] - V[c]*fmin(sp1, 0.0)*om[c] - V[c]*fmin(sp2, 0.0)*om[c];
}


// k reaction: diag += V*(betaStar*omega + max(sp,0)); source += V*Pk(G) - V*min(sp,0)*k;  sp=(2/3)divU;
//             Pk = min(G, c1*betaStar*k*omega).  (k-eps kReaction with eps/k->betaStar*omega, G->Pk(G).)
__global__
void kReactionSSTKernel(
    int nC,
    const scalar* __restrict__ V,
    const scalar* __restrict__ k,
    const scalar* __restrict__ om,
    const scalar* __restrict__ G,
    const scalar* __restrict__ divU,
    scalar betaStar,
    scalar c1betaStar,
    const scalar* __restrict__ gammaIntEff,
    scalar* __restrict__ diag,
    scalar* __restrict__ source)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    const scalar sp = (2.0/3.0) * divU[c];
    // kOmegaSSTLM transition: Pk *= gammaIntEff, epsilonByk (= betaStar*omega) *= clamp(gammaIntEff, 0.1, 1).
    // gammaIntEff == nullptr (plain kOmegaSST) -> geff = 1 -> bit-identical (clamp(1,0.1,1)=1).
    const scalar geff = gammaIntEff ? gammaIntEff[c] : 1.0;
    const scalar Pk = geff * fmin(G[c], c1betaStar * k[c] * om[c]);
    diag[c]   += V[c] * (fmin(fmax(geff, 0.1), 1.0) * betaStar * om[c] + fmax(sp, 0.0));
    source[c] += V[c] * Pk - V[c] * fmin(sp, 0.0) * k[c];
}


// omega wall function (BINOMIAL n=2 default): omega0 = sqrt(omegaVis^2 + omegaLog^2), scattered to wall cells
// with cornerWeight invNw. G0 IDENTICAL to the epsilon wall function. Clone of device_kepsilon wallFnKernel.
__global__
void wallOmegaG0Kernel(
    int nWF,
    const label* __restrict__ wfCell,
    const scalar* __restrict__ wfY,
    const scalar* __restrict__ wfDc,
    const scalar* __restrict__ wux,
    const scalar* __restrict__ wuy,
    const scalar* __restrict__ wuz,
    const scalar* __restrict__ invNw,
    const scalar* __restrict__ k,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    scalar nu,
    scalar yplLam,
    scalar Cmu25,
    scalar kappa,
    scalar E,
    scalar atmZ0,
    bool   atmBoundNut,
    scalar beta1,
    int nutWall,
    scalar* __restrict__ omega0,
    scalar* __restrict__ G0)
{
    const int wf = blockIdx.x * blockDim.x + threadIdx.x;
    if (wf >= nWF) return;

    const int c = wfCell[wf];
    const scalar y = wfY[wf], dc = wfDc[wf], kc = k[c];
    wallProductionG0(c, wf, y, dc, kc, invNw[c], wux, wuy, wuz, Ux, Uy, Uz, nu, yplLam, Cmu25, kappa, E, atmZ0, atmBoundNut, nutWall, G0);
    const scalar omegaVis = 6.0 * nu / (beta1 * y * y);
    const scalar omegaLog = sqrt(kc) / (Cmu25 * kappa * y);
    atomicAdd(&omega0[c], invNw[c] * sqrt(omegaVis*omegaVis + omegaLog*omegaLog));   // BINOMIAL n=2 (distinct omega wall value)
}
} // namespace


void deviceS2(const DeviceBuffer<scalar>& gradU, int nC, DeviceBuffer<scalar>& S2)
{
    S2.resize(nC);
    s2Kernel<<<nBlocks(nC), TPB>>>(nC, gradU.data(), S2.data());
    cudaCheck(cudaGetLastError(), "S2");
}


void deviceCDkOmega(
    const DeviceBuffer<scalar>& gKx,
    const DeviceBuffer<scalar>& gKy,
    const DeviceBuffer<scalar>& gKz,
    const DeviceBuffer<scalar>& gOx,
    const DeviceBuffer<scalar>& gOy,
    const DeviceBuffer<scalar>& gOz,
    const DeviceBuffer<scalar>& omega,
    scalar alphaOmega2,
    DeviceBuffer<scalar>& CD)
{
    const int nC = static_cast<int>(omega.size());
    CD.resize(nC);
    cdKernel<<<nBlocks(nC), TPB>>>(nC, gKx.data(), gKy.data(), gKz.data(), gOx.data(), gOy.data(), gOz.data(),
                                   omega.data(), 2.0 * alphaOmega2, CD.data());
    cudaCheck(cudaGetLastError(), "CDkOmega");
}


void deviceF1(
    const DeviceBuffer<scalar>& k,
    const DeviceBuffer<scalar>& omega,
    const DeviceBuffer<scalar>& y,
    const DeviceBuffer<scalar>& CD,
    scalar nu,
    const KOmegaSSTCoeffs& co,
    DeviceBuffer<scalar>& F1,
    bool lm)
{
    const int nC = static_cast<int>(k.size());
    F1.resize(nC);
    f1Kernel<<<nBlocks(nC), TPB>>>(nC, k.data(), omega.data(), y.data(), CD.data(), nu, co.betaStar, co.alphaOmega2, lm ? 1 : 0, F1.data());
    cudaCheck(cudaGetLastError(), "F1");
}


void deviceF2(
    const DeviceBuffer<scalar>& k,
    const DeviceBuffer<scalar>& omega,
    const DeviceBuffer<scalar>& y,
    scalar nu,
    const KOmegaSSTCoeffs& co,
    DeviceBuffer<scalar>& F2)
{
    const int nC = static_cast<int>(k.size());
    F2.resize(nC);
    f2Kernel<<<nBlocks(nC), TPB>>>(nC, k.data(), omega.data(), y.data(), nu, co.betaStar, F2.data());
    cudaCheck(cudaGetLastError(), "F2");
}


void deviceBlend(const DeviceBuffer<scalar>& F1, scalar psi1, scalar psi2, DeviceBuffer<scalar>& out)
{
    const int nC = static_cast<int>(F1.size());
    out.resize(nC);
    blendKernel<<<nBlocks(nC), TPB>>>(nC, F1.data(), psi1, psi2, out.data());
    cudaCheck(cudaGetLastError(), "blend");
}


void deviceNutSST(
    const DeviceBuffer<scalar>& k,
    const DeviceBuffer<scalar>& omega,
    const DeviceBuffer<scalar>& F2,
    const DeviceBuffer<scalar>& S2,
    const KOmegaSSTCoeffs& co,
    DeviceBuffer<scalar>& nut)
{
    const int nC = static_cast<int>(k.size());
    nut.resize(nC);
    nutSSTKernel<<<nBlocks(nC), TPB>>>(nC, k.data(), omega.data(), F2.data(), S2.data(), co.a1, co.b1, nut.data());
    cudaCheck(cudaGetLastError(), "nutSST");
}


void devicePk(
    const DeviceBuffer<scalar>& G,
    const DeviceBuffer<scalar>& k,
    const DeviceBuffer<scalar>& omega,
    const KOmegaSSTCoeffs& co,
    DeviceBuffer<scalar>& Pk)
{
    const int nC = static_cast<int>(k.size());
    Pk.resize(nC);
    pkKernel<<<nBlocks(nC), TPB>>>(nC, G.data(), k.data(), omega.data(), co.c1 * co.betaStar, Pk.data());
    cudaCheck(cudaGetLastError(), "Pk");
}


void deviceGbyNuLimit(
    const DeviceBuffer<scalar>& GbyNu0,
    const DeviceBuffer<scalar>& omega,
    const DeviceBuffer<scalar>& F2,
    const DeviceBuffer<scalar>& S2,
    const KOmegaSSTCoeffs& co,
    DeviceBuffer<scalar>& GbyNu)
{
    const int nC = static_cast<int>(omega.size());
    GbyNu.resize(nC);
    gbyNuLimitKernel<<<nBlocks(nC), TPB>>>(nC, GbyNu0.data(), omega.data(), F2.data(), S2.data(),
                                           co.a1, co.b1, co.c1, co.betaStar, GbyNu.data());
    cudaCheck(cudaGetLastError(), "GbyNuLimit");
}


void deviceDEff(
    const DeviceBuffer<scalar>& F1,
    const DeviceBuffer<scalar>& nut,
    scalar alpha1,
    scalar alpha2,
    scalar nu,
    DeviceBuffer<scalar>& D)
{
    const int nC = static_cast<int>(F1.size());
    D.resize(nC);
    dEffKernel<<<nBlocks(nC), TPB>>>(nC, F1.data(), nut.data(), alpha1, alpha2, nu, D.data());
    cudaCheck(cudaGetLastError(), "DEff");
}


void deviceOmegaReaction(
    const DeviceBuffer<scalar>& V,
    const DeviceBuffer<scalar>& gamma,
    const DeviceBuffer<scalar>& beta,
    const DeviceBuffer<scalar>& GbyNu0lim,
    const DeviceBuffer<scalar>& F1,
    const DeviceBuffer<scalar>& CD,
    const DeviceBuffer<scalar>& omega,
    const DeviceBuffer<scalar>& divU,
    DeviceBuffer<scalar>& diag,
    DeviceBuffer<scalar>& source)
{
    const int nC = static_cast<int>(V.size());
    omegaReactionKernel<<<nBlocks(nC), TPB>>>(nC, V.data(), gamma.data(), beta.data(), GbyNu0lim.data(), F1.data(),
                                              CD.data(), omega.data(), divU.data(), diag.data(), source.data());
    cudaCheck(cudaGetLastError(), "omegaReaction");
}


void deviceWallOmegaG0(
    const DeviceWallData& w,
    const DeviceBuffer<scalar>& k,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    scalar nu,
    DeviceBuffer<scalar>& omega0,
    DeviceBuffer<scalar>& G0,
    const KOmegaSSTCoeffs& co,
    int nutWall,
    scalar atmZ0,
    bool   atmBoundNut)
{
    const int nC = static_cast<int>(k.size());
    omega0.resize(nC); G0.resize(nC);
    cudaCheck(cudaMemsetAsync(omega0.data(), 0, nC*sizeof(scalar), cudaStreamPerThread), "omega0 zero");
    cudaCheck(cudaMemsetAsync(G0.data(),     0, nC*sizeof(scalar), cudaStreamPerThread), "G0 zero");
    const scalar Cmu25 = std::pow(co.betaStar, 0.25), yplLam = yPlusLamHost(co.kappa, co.E);
    if (w.nWF > 0)
        wallOmegaG0Kernel<<<nBlocks(w.nWF), TPB>>>(w.nWF, w.wfCell.data(), w.wfY.data(), w.wfDc.data(), w.wfUwx.data(),
                                                   w.wfUwy.data(), w.wfUwz.data(), w.invNw.data(), k.data(), Ux.data(),
                                                   Uy.data(), Uz.data(), nu, yplLam, Cmu25, co.kappa, co.E, atmZ0, atmBoundNut, co.beta1,
                                                   nutWall, omega0.data(), G0.data());
    cudaCheck(cudaGetLastError(), "wallOmegaG0");
}


void deviceKReactionSST(
    const DeviceBuffer<scalar>& V,
    const DeviceBuffer<scalar>& k,
    const DeviceBuffer<scalar>& omega,
    const DeviceBuffer<scalar>& G,
    const DeviceBuffer<scalar>& divU,
    const KOmegaSSTCoeffs& co,
    DeviceBuffer<scalar>& diag,
    DeviceBuffer<scalar>& source,
    const scalar* gammaIntEff)
{
    const int nC = static_cast<int>(V.size());
    kReactionSSTKernel<<<nBlocks(nC), TPB>>>(nC, V.data(), k.data(), omega.data(), G.data(), divU.data(),
                                             co.betaStar, co.c1 * co.betaStar, gammaIntEff, diag.data(), source.data());
    cudaCheck(cudaGetLastError(), "kReactionSST");
}

} // namespace brae
