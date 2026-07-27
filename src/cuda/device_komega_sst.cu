// cf GPU offload: k-omega SST blending (F1/F2), cross-diffusion (CDkOmega), strain (S2), and the coefficient
// blend. Pointwise per-cell kernels mirroring kOmegaSSTBase.C (v2412). See the header for the formulas.
#include "device_komega_sst.cuh"
#include "device_kepsilon.cuh"   // DeviceWallData (shared wall geometry)
#include "nut_wall_function.cuh" // nutkWallFunctionValue / yPlusWall (shared wall-nut physics, "G0 IDENTICAL to eps WF")
#include "device_scalar_transport.cuh"  // deviceSolveScalarTransport scaffold + depsKernel/overrideKernel + nBlocks/TPB
#include "device_ldu.cuh"
#include "device_pcg.cuh"
#include "device_simple.cuh"
#include "device_blas.cuh"
#include "device_ami.cuh"
#include "device_cyclic.cuh"
#include "device_interface.cuh"
#include "device_amg.cuh"
#include <cuda_runtime.h>
#include <cmath>

namespace brae {

namespace {
// TPB/nBlocks now come from device_scalar_transport.cuh (shared with device_kepsilon.cu / device_spalart.cu).
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
    scalar* __restrict__ source,
    const scalar* __restrict__ FDES)   // kOmegaSST-DDES: k-dissipation *= FDES (nullptr -> 1 -> plain RANS)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    const scalar sp = (2.0/3.0) * divU[c];
    // kOmegaSSTLM transition: Pk *= gammaIntEff, epsilonByk (= betaStar*omega) *= clamp(gammaIntEff, 0.1, 1).
    // gammaIntEff == nullptr (plain kOmegaSST) -> geff = 1 -> bit-identical (clamp(1,0.1,1)=1).
    const scalar geff = gammaIntEff ? gammaIntEff[c] : 1.0;
    // kOmegaSST-DDES: the DES limiter FDES>=1 enhances the k destruction (beta*k*omega -> beta*k*omega*FDES) so the
    // modelled length scale collapses to the LES scale in detached regions. FDES==nullptr (RANS) -> factor 1 -> unchanged.
    const scalar fdes = FDES ? FDES[c] : scalar(1);
    const scalar Pk = geff * fmin(G[c], c1betaStar * k[c] * om[c]);
    diag[c]   += V[c] * (fdes * fmin(fmax(geff, 0.1), 1.0) * betaStar * om[c] + fmax(sp, 0.0));
    source[c] += V[c] * Pk - V[c] * fmin(sp, 0.0) * k[c];
}

// kOmegaSST-DDES DES factor: FDES = max( (Lt/(CDES*Delta))*(1 - F2), 1 ), Lt = sqrt(k)/(betaStar*omega) (RANS length),
// Delta = cubeRootVol = V^(1/3), CDES = F1*CDES1 + (1-F1)*CDES2 (SST-blended), F2 = the DDES shielding (RANS in the
// boundary layer where F2->1 -> FDES=1; LES in free shear where F2->0). Matches OF kOmegaSSTDDES.
__global__
void kOmegaSSTDESfactorKernel(
    int nC, const scalar* __restrict__ k, const scalar* __restrict__ om, const scalar* __restrict__ V,
    const scalar* __restrict__ F1, const scalar* __restrict__ F2, scalar betaStar, scalar CDES1, scalar CDES2,
    scalar* __restrict__ FDES)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    const scalar delta = cbrt(V[c]);
    const scalar Lt = sqrt(fmax(k[c], scalar(0))) / fmax(betaStar * om[c], 1e-300);
    const scalar CDES = F1[c]*CDES1 + (scalar(1) - F1[c])*CDES2;
    FDES[c] = fmax((Lt / fmax(CDES*delta, 1e-300)) * (scalar(1) - F2[c]), scalar(1));
}

// kOmegaSST-IDDES DES factor (Gritskevich/Garbaruk/Schuetze/Menter 2012): the improved (WMLES) length scale replaces the
// k-dissipation scale. lRAS = sqrt(k)/(betaStar*omega); lLES = CDES*Delta, CDES = F1*CDES1 + (1-F1)*CDES2, Delta =
// min(max(Cw*y, Cw*hmax), hmax) (hwn omitted). Blending: rd_t/rd_l from nut/nu, fdt/fl/ft/fB/fe as in SA-IDDES (the SST
// rd denominator sqrt(0.5(S^2+Omega^2)) equals |gradU|). lIDDES = fdTilde*(1+fe)*lRAS + (1-fdTilde)*lLES, and the k
// destruction beta*k*omega is scaled by FDES = lRAS/lIDDES (== k^(3/2)/lIDDES). NOT clamped to 1: the fe elevated-stress
// branch makes lIDDES > lRAS -> FDES < 1 (less destruction), which is the intended IDDES behaviour.
__global__
void kOmegaSSTIDDESfactorKernel(
    int nC, const scalar* __restrict__ k, const scalar* __restrict__ om, const scalar* __restrict__ F1,
    const scalar* __restrict__ gradU, const scalar* __restrict__ nut, const scalar* __restrict__ y,
    const scalar* __restrict__ hmax, const scalar* __restrict__ hwn, scalar nu, KOmegaSSTCoeffs co, scalar* __restrict__ FDES)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    scalar g2 = 0;
    for (int q = 0; q < 9; ++q) { const scalar gk = gradU[q*nC + c]; g2 += gk*gk; }
    const scalar magGradU = fmax(sqrt(g2), scalar(1e-300));           // |gradU| = sqrt(0.5(S^2+Omega^2))
    const scalar lRAS = sqrt(fmax(k[c], scalar(0))) / fmax(co.betaStar*om[c], scalar(1e-300));   // RANS length (Lt)
    const scalar CDES = F1[c]*co.CDES1 + (scalar(1) - F1[c])*co.CDES2;
    const scalar hm = fmax(hmax[c], scalar(1e-300));
    const scalar delta = fmin(fmax(fmax(co.Cw*y[c], co.Cw*hm), hwn[c]), hm);   // IDDES delta = min(max(max(Cw*y,Cw*hmax),hwn), hmax)
    const scalar lLES = CDES*delta;
    const scalar kd2 = fmax(co.kappa*co.kappa*y[c]*y[c], scalar(1e-300));
    const scalar rdt = fmin(nut[c] / (magGradU*kd2), scalar(10));     // turbulent rd (SST nut)
    const scalar rdl = fmin(nu     / (magGradU*kd2), scalar(10));     // laminar rd (nu)
    const scalar adt = co.Cdt1*rdt;   const scalar fdt = scalar(1) - tanh(adt*adt*adt);
    const scalar al  = co.Cl*co.Cl*rdl; const scalar al2 = al*al, al4 = al2*al2, al8 = al4*al4;
    const scalar fl  = tanh(al8*al2);
    const scalar at  = co.Ct*co.Ct*rdt; const scalar ft = tanh(at*at*at);
    const scalar fe2 = scalar(1) - fmax(ft, fl);
    const scalar alpha = scalar(0.25) - y[c]/hm;
    const scalar fB  = fmin(scalar(2)*exp(scalar(-9)*alpha*alpha), scalar(1));
    const scalar fdTilde = fmax(scalar(1) - fdt, fB);
    const scalar fe1 = (alpha >= scalar(0)) ? scalar(2)*exp(scalar(-11.09)*alpha*alpha)
                                            : scalar(2)*exp(scalar(-9.0)*alpha*alpha);
    const scalar fe  = fmax(fe1 - scalar(1), scalar(0)) * fe2;
    const scalar lIDDES = fmax(fdTilde*(scalar(1) + fe)*lRAS + (scalar(1) - fdTilde)*lLES, scalar(1e-300));
    FDES[c] = lRAS / lIDDES;                                          // beta*k*omega scaling (== k^(3/2)/lIDDES)
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
    const scalar* gammaIntEff,
    const DeviceBuffer<scalar>* FDES)   // kOmegaSST-DDES DES factor per cell; nullptr -> plain RANS (unchanged)
{
    const int nC = static_cast<int>(V.size());
    kReactionSSTKernel<<<nBlocks(nC), TPB>>>(nC, V.data(), k.data(), omega.data(), G.data(), divU.data(),
                                             co.betaStar, co.c1 * co.betaStar, gammaIntEff, diag.data(), source.data(),
                                             FDES ? FDES->data() : nullptr);
    cudaCheck(cudaGetLastError(), "kReactionSST");
}

// Exported kOmegaSST-DDES DES-factor wrapper (single-GPU + unit-test hook): FDES from k/omega, cubeRootVol(V), F1, F2.
void deviceKOmegaSSTDESfactor(int nC, const DeviceBuffer<scalar>& k, const DeviceBuffer<scalar>& omega,
    const DeviceBuffer<scalar>& V, const DeviceBuffer<scalar>& F1, const DeviceBuffer<scalar>& F2,
    const KOmegaSSTCoeffs& co, DeviceBuffer<scalar>& FDES)
{
    FDES.resize(nC);
    kOmegaSSTDESfactorKernel<<<nBlocks(nC), TPB>>>(nC, k.data(), omega.data(), V.data(), F1.data(), F2.data(),
                                                   co.betaStar, co.CDES1, co.CDES2, FDES.data());
    cudaCheck(cudaGetLastError(), "kOmegaSSTDESfactor");
}

// kOmegaSST-IDDES factor FDES = lRAS/lIDDES (Gritskevich et al. 2012). Needs gradU (|gradU| for rd), the SST nut, the
// wall distance y and the maxDeltaxyz hmax; F1 blends CDES. (Unit-test/DES hook.)
void deviceKOmegaSSTIDDESfactor(int nC, const DeviceBuffer<scalar>& k, const DeviceBuffer<scalar>& omega,
    const DeviceBuffer<scalar>& F1, const DeviceBuffer<scalar>& gradU, const DeviceBuffer<scalar>& nut,
    const DeviceBuffer<scalar>& y, const DeviceBuffer<scalar>& hmax, const DeviceBuffer<scalar>& hwn, scalar nu,
    const KOmegaSSTCoeffs& co, DeviceBuffer<scalar>& FDES)
{
    FDES.resize(nC);
    kOmegaSSTIDDESfactorKernel<<<nBlocks(nC), TPB>>>(nC, k.data(), omega.data(), F1.data(), gradU.data(),
        nut.data(), y.data(), hmax.data(), hwn.data(), nu, co, FDES.data());
    cudaCheck(cudaGetLastError(), "kOmegaSSTIDDESfactor");
}


// ---- kOmegaSST + Langtry-Menter transition (moved from device_kepsilon.cu; uses the SST helpers above + the scaffold) ----

// S2 (nut) and the production GbyNu0. When grad(U) is `cellLimited Gauss linear <k>` (motorBike), that gradient is
// LIMITED, else on skewed cells the unlimited strain is huge and the SST nut limiter max(a1*omega, b1*F2*sqrt(S2))
// wrongly fires. Apply the per-component minmod limiter to the gradU TENSOR:
// limiter_j scales gradU[j],[3+j],[6+j] (= dU_j/dx_i, i=0..2). Reuses deviceCellLimitGrad (the momentum limiter).
void deviceCellLimitGradU(
    const DeviceMesh& dm,
    const DeviceVectorBoundary& dbU,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    DeviceBuffer<scalar>& gradU,
    scalar kc)
{
    const int nC = dm.nCells;
    const DeviceBuffer<scalar>* U[3] = {&Ux, &Uy, &Uz};
    for (int j = 0; j < 3; ++j)
    {
        DeviceBuffer<scalar> gx(nC), gy(nC), gz(nC);
        cudaMemcpyAsync(gx.data(), gradU.data()+(std::size_t)j*nC,     nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
        cudaMemcpyAsync(gy.data(), gradU.data()+(std::size_t)(3+j)*nC, nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
        cudaMemcpyAsync(gz.data(), gradU.data()+(std::size_t)(6+j)*nC, nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
        DeviceBuffer<scalar> ubv;
        deviceBCValue(dbU.comp[j], *U[j], ubv);
        deviceCellLimitGrad(dm, *U[j], ubv, gx, gy, gz, kc);
        cudaMemcpyAsync(gradU.data()+(std::size_t)j*nC,     gx.data(), nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
        cudaMemcpyAsync(gradU.data()+(std::size_t)(3+j)*nC, gy.data(), nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
        cudaMemcpyAsync(gradU.data()+(std::size_t)(6+j)*nC, gz.data(), nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    }
    cudaCheck(cudaGetLastError(), "cellLimitGradUTensor");
}


void deviceKOmegaSSTCorrect(
    const DeviceMesh& dm,
    const DeviceWallData& wall,
    const DeviceBoundary& dbOmega,
    const DeviceBoundary& dbK,
    const DeviceVectorBoundary& dbU,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    DeviceBuffer<scalar>& k,
    DeviceBuffer<scalar>& omega,
    DeviceBuffer<scalar>& nut,
    const DeviceBuffer<scalar>& y,
    const DeviceBuffer<scalar>& phiInt,
    const DeviceBuffer<scalar>& phiBnd,
    scalar nu,
    scalar relaxOmega,
    scalar relaxK,
    scalar tol,
    bool bounded,
    bool limitedK,
    bool limitedOmega,
    scalar twoBykK,
    scalar twoBykOmega,
    const KOmegaSSTCoeffs& co,
    scalar relTolKE,
    int keCheckEvery,
    bool linearUpwindK,
    bool linearUpwindOmega,
    bool nonOrth,
    scalar gradULimitK,
    bool gsK,
    bool gsEps,
    DeviceAMI* ami,
    DeviceCyclic* cyc,
    const scalar* gammaIntEff,
    int nutWall,
    scalar atmZ0,
    bool atmBoundNut,
    const ScalarDdt& kDdt,
    const ScalarDdt& sDdt,
    bool des,
    bool iddes,
    const DeviceBuffer<scalar>* hmax,
    const DeviceBuffer<scalar>* hwn)
{
    const int nC = dm.nCells;
    // production (raw GbyNu0) + G = nut*GbyNu0, divU, S2 (shared gradU = OF tgradU = grad(U) scheme).
    DeviceBuffer<scalar> gradU;
    deviceGradU(dm, dbU, Ux, Uy, Uz, gradU, ami, cyc);   // interface-aware grad(U)
    if (gradULimitK > 0.0) deviceCellLimitGradU(dm, dbU, Ux, Uy, Uz, gradU, gradULimitK);   // grad(U) cellLimited (OF)
    DeviceBuffer<scalar> GbyNu0;
    deviceGByNuFromGradU(gradU, nC, GbyNu0);
    DeviceBuffer<scalar> G;
    deviceHadamard(G, nut, GbyNu0);
    DeviceBuffer<scalar> divU;
    deviceDiv(dm, phiInt, phiBnd, divU);
    if (ami && ami->n) interfaceAddDiv(*ami, dm.V, divU);
    if (cyc && cyc->n) interfaceAddDiv(*cyc, dm.V, divU);
    DeviceBuffer<scalar> S2;
    deviceS2(gradU, nC, S2);

    // omega wall function FIRST (OF updateCoeffs before CDkOmega): omega0/G0, override omega & G at wall cells, so
    // grad(omega)/CDkOmega/F1/F2 and the reaction all see the wall-corrected omega (matches kOmegaSSTBase::correct).
    DeviceBuffer<scalar> omega0, G0;
    deviceWallOmegaG0(wall, k, Ux, Uy, Uz, nu, omega0, G0, co, nutWall, atmZ0, atmBoundNut);
    overrideKernel<<<nBlocks(nC), TPB>>>(nC, wall.isWallCell.data(), G0.data(), omega0.data(), G.data(), omega.data());

    // CDkOmega from grad(k), grad(omega); F1, F2.
    DeviceBuffer<scalar> kbv;
    deviceBCValue(dbK, k, kbv);
    DeviceBuffer<scalar> kgx, kgy, kgz;
    deviceGaussGrad(dm, k, kbv, kgx, kgy, kgz);
    DeviceBuffer<scalar> obv;
    deviceBCValue(dbOmega, omega, obv);
    DeviceBuffer<scalar> ogx, ogy, ogz;
    deviceGaussGrad(dm, omega, obv, ogx, ogy, ogz);
    DeviceBuffer<scalar> CD;
    deviceCDkOmega(kgx, kgy, kgz, ogx, ogy, ogz, omega, co.alphaOmega2, CD);
    DeviceBuffer<scalar> F1;
    deviceF1(k, omega, y, CD, nu, co, F1, gammaIntEff != nullptr);   // LM: F1=max(F1,F3) near-wall override
    DeviceBuffer<scalar> F2;
    deviceF2(k, omega, y, nu, co, F2);
    // kOmegaSST-DDES: the DES factor FDES>=1 (from the RANS length scale sqrt(k)/(betaStar*omega) vs C_DES*cubeRootVol,
    // shielded by 1-F2) multiplies the k destruction below. des==false -> FDES stays empty -> plain kOmegaSST(-RANS).
    DeviceBuffer<scalar> FDES;
    if (des)
    {
        if (iddes && hmax && hwn)   // kOmegaSST-IDDES: the improved (WMLES) length scale (needs the SST nut + hmax + hwn + gradU + y)
            deviceKOmegaSSTIDDESfactor(nC, k, omega, F1, gradU, nut, y, *hmax, *hwn, nu, co, FDES);
        else                 // kOmegaSST-DDES: the F2-shielded cubeRootVol DES factor
            deviceKOmegaSSTDESfactor(nC, k, omega, dm.V, F1, F2, co, FDES);
    }

    // blends, limited production-by-nu, DomegaEff.
    DeviceBuffer<scalar> gamma;
    deviceBlend(F1, co.gamma1, co.gamma2, gamma);
    DeviceBuffer<scalar> beta;
    deviceBlend(F1, co.beta1,  co.beta2,  beta);
    DeviceBuffer<scalar> GbyNu0lim;
    deviceGbyNuLimit(GbyNu0, omega, F2, S2, co, GbyNu0lim);
    DeviceBuffer<scalar> DomegaEff;
    deviceDEff(F1, nut, co.alphaOmega1, co.alphaOmega2, nu, DomegaEff);

    // omega equation (loose solve) with the near-wall setValues constraint (omega0)
    deviceSolveScalarTransport(dm, dbOmega, omega, "omega", DomegaEff, phiInt, phiBnd, divU, bounded, limitedOmega, linearUpwindOmega, nonOrth, twoBykOmega,
                               relaxOmega, tol, relTolKE, keCheckEvery, gsEps,
                               [&](DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& src){ deviceOmegaReaction(dm.V, gamma, beta, GbyNu0lim, F1, CD, omega, divU, diag, src); },
                               &wall, &omega0, ami, cyc, sDdt);
    deviceBoundField(dm, omega, 1e-15);   // OF bound(omega_, omegaMin_)

    // k equation (loose solve)
    DeviceBuffer<scalar> DkEff;
    deviceDEff(F1, nut, co.alphaK1, co.alphaK2, nu, DkEff);
    deviceSolveScalarTransport(dm, dbK, k, "k", DkEff, phiInt, phiBnd, divU, bounded, limitedK, linearUpwindK, nonOrth, twoBykK,
                               relaxK, tol, relTolKE, keCheckEvery, gsK,
                               [&](DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& src){ deviceKReactionSST(dm.V, k, omega, G, divU, co, diag, src, gammaIntEff, des ? &FDES : nullptr); },
                               nullptr, nullptr, ami, cyc, kDdt);
    deviceBoundField(dm, k, 1e-15);   // OF bound(k_, kMin_)

    // correctNut (Bradshaw): nut = a1*k / max(a1*omega, b1*F2*sqrt(S2)).
    deviceNutSST(k, omega, F2, S2, co, nut);
}


// kOmegaSSTLM (Langtry-Menter gamma-ReThetat transition)
// LM coeffs (OF defaults): ca1=2, ca2=0.06, ce1=1, ce2=50, cThetat=0.03, sigmaThetat=2; lambdaErr=1e-6, maxIter=10.
namespace { struct LMCoeffs { scalar ca1=2.0, ca2=0.06, ce1=1.0, ce2=50.0, cThetat=0.03, sigmaThetat=2.0, lambdaErr=1e-6; int maxIter=10; }; }


// DReThetatEff = sigmaThetat*(nut + nu)  (NOT nut/sigma + nu, depsKernel can't express this).
__global__
void lmReDiffKernel(int nC, const scalar* __restrict__ nut, scalar sigma, scalar nu, scalar* __restrict__ D)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < nC) D[c] = sigma*(nut[c] + nu);
}


// diag += V*sp ; source += V*su  (apply a precomputed semi-implicit reaction).
__global__
void lmAddReactionKernel(
    int nC,
    const scalar* __restrict__ V,
    const scalar* __restrict__ sp,
    const scalar* __restrict__ su,
    scalar* __restrict__ diag,
    scalar* __restrict__ source)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    diag[c] += V[c] * sp[c];
    source[c] += V[c] * su[c];
}


// per-cell strain helpers from the OF-convention gradU tensor (t[i*3+j] = dU_j/dx_i).
__device__ __forceinline__
void lmStrain(const scalar* t, scalar ux, scalar uy, scalar uz, scalar deltaU,
              scalar& S, scalar& Omega, scalar& Us, scalar& dUsds)
{
    scalar symSq = 0.0, skSq = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            const scalar sy = 0.5*(t[i*3+j]+t[j*3+i]), sk = 0.5*(t[i*3+j]-t[j*3+i]);
            symSq += sy*sy;
            skSq += sk*sk;
        }
    S = sqrt(2.0*symSq);
    Omega = sqrt(2.0*skSq);
    Us = fmax(sqrt(ux*ux+uy*uy+uz*uz), deltaU);
    const scalar Uv[3] = {ux, uy, uz};
    scalar num = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            num += Uv[i]*Uv[j]*t[i*3+j];
    dUsds = num/(Us*Us);
}


// Fthetat (OF kOmegaSSTLM::Fthetat), uses the lagged gammaInt; reused by both the ReThetat source and gammaSep.
__device__ __forceinline__
scalar lmFthetat(scalar S, scalar Omega, scalar Us, scalar nu, scalar y, scalar om,
                 scalar ReThetat, scalar gammaInt, scalar ce2)
{
    const scalar delta = fmax(375.0*Omega*nu*ReThetat*y/(Us*Us), 1e-37);
    const scalar ReOmega = y*y*om/nu;
    const scalar Fwake = exp(-(ReOmega/1e5)*(ReOmega/1e5));
    scalar ywd = y/delta; ywd *= ywd; ywd *= ywd;   // (y/delta)^4
    const scalar invCe2 = 1.0/ce2, b = (gammaInt - invCe2)/(1.0 - invCe2);
    return fmin(fmax(Fwake*exp(-ywd), 1.0 - b*b), 1.0);
}


// ReThetac(ReThetat) and Flength(ReThetat) empirical correlations (OF).
__device__ __forceinline__
scalar lmReThetac(scalar R)
{
    return (R <= 1870.0) ? R - 396.035e-2 + 120.656e-4*R - 868.230e-6*R*R + 696.506e-9*R*R*R - 174.105e-12*R*R*R*R
                         : R - 593.11 - 0.482*(R - 1870.0);
}


__device__ __forceinline__
scalar lmFlength(scalar R, scalar y, scalar om, scalar nu)
{
    scalar Fl;
    if (R < 400.0)       Fl = 398.189e-1 - 119.270e-4*R - 132.567e-6*R*R;
    else if (R < 596.0)  Fl = 263.404 - 123.939e-2*R + 194.548e-5*R*R - 101.695e-8*R*R*R;
    else if (R < 1200.0) Fl = 0.5 - 3e-4*(R - 596.0);
    else                 Fl = 0.3188;
    const scalar fs = y*y*om/(200.0*nu);
    const scalar Fsub = exp(-(fs*fs));
    return Fl*(1.0 - Fsub) + 40.0*Fsub;
}


// ReThetat reaction prep: ReThetat0 Newton loop + Pthetat; outputs sp=Pthetat, su=Pthetat*ReThetat0, and Fthetat.
__global__
void lmReThetatPrepKernel(
    int nC,
    const scalar* __restrict__ gradU,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    const scalar* __restrict__ k,
    const scalar* __restrict__ om,
    const scalar* __restrict__ y,
    const scalar* __restrict__ ReThetat,
    const scalar* __restrict__ gammaInt,
    scalar nu,
    scalar cThetat,
    scalar ce2,
    scalar deltaU,
    scalar lambdaErr,
    int maxIter,
    scalar* __restrict__ Fth,
    scalar* __restrict__ sp,
    scalar* __restrict__ su)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    scalar t[9];
    for (int q = 0; q < 9; ++q)
        t[q] = gradU[q*nC+c];
    scalar S, Omega, Us, dUsds;
    lmStrain(t, Ux[c], Uy[c], Uz[c], deltaU, S, Omega, Us, dUsds);
    const scalar kc = k[c], nuc = nu, yc = y[c], omc = om[c], ret = ReThetat[c], gi = gammaInt[c];
    const scalar Tu = fmax(100.0*sqrt((2.0/3.0)*kc)/Us, 0.027);
    scalar lambda = 0.0, thetat = 0.0;
    int iter = 0;
    for (;;)
    {
        const scalar lam0 = lambda;
        scalar Fl;
        if (Tu <= 1.3)
        {
            Fl = (dUsds <= 0.0) ? 1.0 - (-12.986*lambda - 123.66*lambda*lambda - 405.689*lambda*lambda*lambda)*exp(-pow(Tu/1.5, 1.5))
                                : 1.0 + 0.275*(1.0 - exp(-35.0*lambda))*exp(-Tu/0.5);
            thetat = (1173.51 - 589.428*Tu + 0.2196/(Tu*Tu))*Fl*nuc/Us;
        }
        else
        {
            Fl = (dUsds <= 0.0) ? 1.0 - (-12.986*lambda - 123.66*lambda*lambda - 405.689*lambda*lambda*lambda)*exp(-pow(Tu/1.5, 1.5))
                                : 1.0 + 0.275*(1.0 - exp(-35.0*lambda))*exp(-2.0*Tu);
            thetat = 331.50*pow(Tu - 0.5658, -0.671)*Fl*nuc/Us;
        }
        lambda = fmin(fmax(thetat*thetat/nuc*dUsds, -0.1), 0.1);
        if (fabs(lambda - lam0) <= lambdaErr || ++iter >= maxIter) break;
    }
    const scalar ReThetat0 = fmax(thetat*Us/nuc, 20.0);
    const scalar Fthetat = lmFthetat(S, Omega, Us, nuc, yc, omc, ret, gi, ce2);
    Fth[c] = Fthetat;
    const scalar Pthetat = (cThetat/(500.0*nuc/(Us*Us)))*(1.0 - Fthetat);   // cThetat/t, t=500*nu/Us^2
    sp[c] = Pthetat;
    su[c] = Pthetat*ReThetat0;
}


// gammaInt reaction prep: Pgamma, Egamma -> sp=ce1*Pgamma+ce2*Egamma, su=Pgamma+Egamma.
__global__
void lmGammaPrepKernel(
    int nC,
    const scalar* __restrict__ gradU,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    const scalar* __restrict__ k,
    const scalar* __restrict__ om,
    const scalar* __restrict__ y,
    const scalar* __restrict__ ReThetat,
    const scalar* __restrict__ gammaInt,
    scalar nu,
    scalar ca1,
    scalar ca2,
    scalar ce1,
    scalar ce2,
    scalar deltaU,
    scalar* __restrict__ sp,
    scalar* __restrict__ su)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    scalar t[9];
    for (int q = 0; q < 9; ++q)
        t[q] = gradU[q*nC+c];
    scalar S, Omega, Us, dUsds;
    lmStrain(t, Ux[c], Uy[c], Uz[c], deltaU, S, Omega, Us, dUsds);
    const scalar kc = k[c], omc = om[c], yc = y[c], gi = gammaInt[c];
    const scalar ReThetac = lmReThetac(ReThetat[c]);
    const scalar Rev = yc*yc*S/nu, RT = kc/(nu*omc);
    const scalar Fonset1 = Rev/(2.193*ReThetac);
    const scalar Fonset2 = fmin(fmax(Fonset1, Fonset1*Fonset1*Fonset1*Fonset1), 2.0);
    const scalar Fonset3 = fmax(1.0 - (RT/2.5)*(RT/2.5)*(RT/2.5), 0.0);
    const scalar Fonset = fmax(Fonset2 - Fonset3, 0.0);
    const scalar Fturb = exp(-(0.25*RT)*(0.25*RT)*(0.25*RT)*(0.25*RT));
    const scalar Pgamma = ca1*lmFlength(ReThetat[c], yc, omc, nu)*S*sqrt(gi*Fonset);
    const scalar Egamma = ca2*Omega*Fturb*gi;
    sp[c] = ce1*Pgamma + ce2*Egamma;
    su[c] = Pgamma + Egamma;
}


// gammaIntEff = max(gammaInt, gammaSep); gammaSep = min(2*max(Rev/(3.235*ReThetac)-1,0)*Freattach, 2)*Fthetat.
__global__
void lmGammaEffKernel(
    int nC,
    const scalar* __restrict__ gradU,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    const scalar* __restrict__ k,
    const scalar* __restrict__ om,
    const scalar* __restrict__ y,
    const scalar* __restrict__ ReThetat,
    const scalar* __restrict__ gammaInt,
    const scalar* __restrict__ Fth,
    scalar nu,
    scalar deltaU,
    scalar* __restrict__ gammaIntEff)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    scalar t[9];
    for (int q = 0; q < 9; ++q)
        t[q] = gradU[q*nC+c];
    scalar S, Omega, Us, dUsds;
    lmStrain(t, Ux[c], Uy[c], Uz[c], deltaU, S, Omega, Us, dUsds);
    const scalar ReThetac = lmReThetac(ReThetat[c]);
    const scalar Rev = y[c]*y[c]*S/nu, RT = k[c]/(nu*om[c]);
    const scalar Freattach = exp(-(RT/20.0)*(RT/20.0)*(RT/20.0)*(RT/20.0));
    const scalar gammaSep = fmin(2.0*fmax(Rev/(3.235*ReThetac) - 1.0, 0.0)*Freattach, 2.0)*Fth[c];
    gammaIntEff[c] = fmax(gammaInt[c], gammaSep);
}


void deviceKOmegaSSTLMCorrect(
    const DeviceMesh& dm,
    const DeviceVectorBoundary& dbU,
    const DeviceBoundary& dbReThetat,
    const DeviceBoundary& dbGammaInt,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    const DeviceBuffer<scalar>& k,
    const DeviceBuffer<scalar>& omega,
    const DeviceBuffer<scalar>& nut,
    const DeviceBuffer<scalar>& y,
    DeviceBuffer<scalar>& ReThetat,
    DeviceBuffer<scalar>& gammaInt,
    DeviceBuffer<scalar>& gammaIntEff,
    const DeviceBuffer<scalar>& phiInt,
    const DeviceBuffer<scalar>& phiBnd,
    scalar nu,
    scalar relax,
    scalar tol,
    scalar relTolKE,
    int keCheckEvery,
    bool bounded,
    bool nonOrth,
    bool gsEps,
    DeviceAMI* ami,
    DeviceCyclic* cyc,
    const ScalarDdt& reDdt,
    const ScalarDdt& giDdt)
{
    const int nC = dm.nCells;
    const LMCoeffs lm;
    DeviceBuffer<scalar> gradU;
    deviceGradU(dm, dbU, Ux, Uy, Uz, gradU, ami, cyc);
    DeviceBuffer<scalar> divU;
    deviceDiv(dm, phiInt, phiBnd, divU);
    if (ami && ami->n) interfaceAddDiv(*ami, dm.V, divU);
    if (cyc && cyc->n) interfaceAddDiv(*cyc, dm.V, divU);

    // ReThetat: DReThetatEff = sigmaThetat*(nut+nu); reaction = Pthetat*ReThetat0 - Sp(Pthetat). Fthetat stored for gammaSep.
    DeviceBuffer<scalar> Fth(nC), spR(nC), suR(nC);
    lmReThetatPrepKernel<<<nBlocks(nC), TPB>>>(nC, gradU.data(), Ux.data(), Uy.data(), Uz.data(), k.data(), omega.data(),
        y.data(), ReThetat.data(), gammaInt.data(), nu, lm.cThetat, lm.ce2, 1e-37, lm.lambdaErr, lm.maxIter,
        Fth.data(), spR.data(), suR.data());
    cudaCheck(cudaGetLastError(), "lmReThetatPrep");
    DeviceBuffer<scalar> DRe(nC);
    lmReDiffKernel<<<nBlocks(nC), TPB>>>(nC, nut.data(), lm.sigmaThetat, nu, DRe.data());   // sigmaThetat*(nut+nu)
    deviceSolveScalarTransport(dm, dbReThetat, ReThetat, "ReThetat", DRe, phiInt, phiBnd, divU, bounded, false, false, nonOrth, 2.0,
                               relax, tol, relTolKE, keCheckEvery, gsEps,
                               [&](DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& src){ lmAddReactionKernel<<<nBlocks(nC), TPB>>>(nC, dm.V.data(), spR.data(), suR.data(), diag.data(), src.data()); },
                               nullptr, nullptr, ami, cyc, reDdt);
    deviceBoundField(dm, ReThetat, 0.0);

    // gammaInt: DgammaIntEff = nut+nu; reaction = Pgamma+Egamma - Sp(ce1*Pgamma+ce2*Egamma).
    DeviceBuffer<scalar> spG(nC), suG(nC);
    lmGammaPrepKernel<<<nBlocks(nC), TPB>>>(nC, gradU.data(), Ux.data(), Uy.data(), Uz.data(), k.data(), omega.data(),
        y.data(), ReThetat.data(), gammaInt.data(), nu, lm.ca1, lm.ca2, lm.ce1, lm.ce2, 1e-37, spG.data(), suG.data());
    cudaCheck(cudaGetLastError(), "lmGammaPrep");
    DeviceBuffer<scalar> DgI(nC);
    depsKernel<<<nBlocks(nC), TPB>>>(nC, nut.data(), 1.0, nu, DgI.data());   // nut/1 + nu
    deviceSolveScalarTransport(dm, dbGammaInt, gammaInt, "gammaInt", DgI, phiInt, phiBnd, divU, bounded, false, false, nonOrth, 2.0,
                               relax, tol, relTolKE, keCheckEvery, gsEps,
                               [&](DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& src){ lmAddReactionKernel<<<nBlocks(nC), TPB>>>(nC, dm.V.data(), spG.data(), suG.data(), diag.data(), src.data()); },
                               nullptr, nullptr, ami, cyc, giDdt);
    deviceBoundField(dm, gammaInt, 0.0);
    gammaIntEff.resize(nC);
    lmGammaEffKernel<<<nBlocks(nC), TPB>>>(nC, gradU.data(), Ux.data(), Uy.data(), Uz.data(), k.data(), omega.data(),
        y.data(), ReThetat.data(), gammaInt.data(), Fth.data(), nu, 1e-37, gammaIntEff.data());
    cudaCheck(cudaGetLastError(), "lmGammaEff");
}

// ---- Exported LM (kOmegaSSTLM transition) source-prep wrappers ----------------------------------------------
// The transition kernels + LMCoeffs are cell-local (only gradU needs the halo, which the DISTRIBUTED SST correct
// already builds). These thin wrappers expose them so parallelDeviceKOmegaSSTLMCorrect can reuse the exact same
// physics + coefficients, feeding the distributed scalar-transport core -- the realizableKE pattern, two equations.
void deviceLMReDiff(const DeviceBuffer<scalar>& nut, scalar nu, DeviceBuffer<scalar>& D)
{
    const int nC = static_cast<int>(nut.size()); const LMCoeffs lm; D.resize(nC);
    lmReDiffKernel<<<nBlocks(nC), TPB>>>(nC, nut.data(), lm.sigmaThetat, nu, D.data());   // sigmaThetat*(nut+nu)
    cudaCheck(cudaGetLastError(), "deviceLMReDiff");
}
void deviceLMReThetatPrep(const DeviceMesh& dm, const DeviceBuffer<scalar>& gradU,
    const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
    const DeviceBuffer<scalar>& k, const DeviceBuffer<scalar>& omega, const DeviceBuffer<scalar>& y,
    const DeviceBuffer<scalar>& ReThetat, const DeviceBuffer<scalar>& gammaInt, scalar nu,
    DeviceBuffer<scalar>& Fth, DeviceBuffer<scalar>& spR, DeviceBuffer<scalar>& suR)
{
    const int nC = dm.nCells; const LMCoeffs lm; Fth.resize(nC); spR.resize(nC); suR.resize(nC);
    lmReThetatPrepKernel<<<nBlocks(nC), TPB>>>(nC, gradU.data(), Ux.data(), Uy.data(), Uz.data(), k.data(), omega.data(),
        y.data(), ReThetat.data(), gammaInt.data(), nu, lm.cThetat, lm.ce2, 1e-37, lm.lambdaErr, lm.maxIter,
        Fth.data(), spR.data(), suR.data());
    cudaCheck(cudaGetLastError(), "deviceLMReThetatPrep");
}
void deviceLMGammaPrep(const DeviceMesh& dm, const DeviceBuffer<scalar>& gradU,
    const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
    const DeviceBuffer<scalar>& k, const DeviceBuffer<scalar>& omega, const DeviceBuffer<scalar>& y,
    const DeviceBuffer<scalar>& ReThetat, const DeviceBuffer<scalar>& gammaInt, scalar nu,
    DeviceBuffer<scalar>& spG, DeviceBuffer<scalar>& suG)
{
    const int nC = dm.nCells; const LMCoeffs lm; spG.resize(nC); suG.resize(nC);
    lmGammaPrepKernel<<<nBlocks(nC), TPB>>>(nC, gradU.data(), Ux.data(), Uy.data(), Uz.data(), k.data(), omega.data(),
        y.data(), ReThetat.data(), gammaInt.data(), nu, lm.ca1, lm.ca2, lm.ce1, lm.ce2, 1e-37, spG.data(), suG.data());
    cudaCheck(cudaGetLastError(), "deviceLMGammaPrep");
}
void deviceLMGammaEff(const DeviceMesh& dm, const DeviceBuffer<scalar>& gradU,
    const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
    const DeviceBuffer<scalar>& k, const DeviceBuffer<scalar>& omega, const DeviceBuffer<scalar>& y,
    const DeviceBuffer<scalar>& ReThetat, const DeviceBuffer<scalar>& gammaInt, const DeviceBuffer<scalar>& Fth,
    scalar nu, DeviceBuffer<scalar>& gammaIntEff)
{
    const int nC = dm.nCells; gammaIntEff.resize(nC);
    lmGammaEffKernel<<<nBlocks(nC), TPB>>>(nC, gradU.data(), Ux.data(), Uy.data(), Uz.data(), k.data(), omega.data(),
        y.data(), ReThetat.data(), gammaInt.data(), Fth.data(), nu, 1e-37, gammaIntEff.data());
    cudaCheck(cudaGetLastError(), "deviceLMGammaEff");
}
void deviceLMAddReaction(const DeviceMesh& dm, const DeviceBuffer<scalar>& sp, const DeviceBuffer<scalar>& su,
    DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& source)
{
    lmAddReactionKernel<<<nBlocks(dm.nCells), TPB>>>(dm.nCells, dm.V.data(), sp.data(), su.data(), diag.data(), source.data());
    cudaCheck(cudaGetLastError(), "deviceLMAddReaction");
}

} // namespace brae
