// cf GPU offload: k-epsilon production + eddy viscosity. gradU is the OF-convention tensor (column i =
// gaussGrad(U_i), as in divDevReff); GbyNu = sum_ij g_ij*(g_ij + g_ji - (2/3)tr d_ij).
#include "device_kepsilon.cuh"
#include "device_komega_sst.cuh"
#include "spalart_coeffs.cuh"
#include "device_ldu.cuh"
#include "device_pcg.cuh"
#include "device_simple.cuh"
#include "device_blas.cuh"
#include "device_ami.cuh"        // cyclicAMI scalar-transport interface coupling
#include "device_cyclic.cuh"     // cyclic scalar-transport interface coupling
#include "device_amg.cuh"        // deviceSymGaussSeidel (scalar smoothSolver, for stiff low-Re k/omega)
#include "nut_wall_function.cuh" // nutkWallFunctionValue / yPlusWall (shared wall-nut physics, BRAE_HD)
#include <cuda_runtime.h>
#include <vector>

namespace brae {

// OF-style turbulence residual report (see device_kepsilon.cuh). Single-threaded per solve; the SIMPLE driver
// clears it before turbulence->correct() and reads it after, to print the "Solving for k/omega/..." lines.
static std::vector<ScalarSolveEntry>& turbStore() { static std::vector<ScalarSolveEntry> s; return s; }
void clearTurbulenceReport() { turbStore().clear(); }
const std::vector<ScalarSolveEntry>& turbulenceReport() { return turbStore(); }

namespace {
constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }
inline scalar yPlusLamHost(scalar kappa, scalar E) { scalar y = 11.0; for (int i = 0; i < 10; ++i) y = std::log(std::fmax(E * y, 1.0)) / kappa; return y; }


__global__
void gByNuKernel(int nC, const scalar* __restrict__ gradU, scalar* __restrict__ gByNu)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    scalar t[9];
    for (int q = 0; q < 9; ++q)
        t[q] = gradU[q * nC + c];

    const scalar t23 = (2.0 / 3.0) * (t[0] + t[4] + t[8]);
    scalar gg = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            const scalar dts = t[i*3+j] + t[j*3+i] - ((i == j) ? t23 : 0.0);   // devTwoSymm
            gg += t[i*3+j] * dts;                                              // doubleDot
        }
    gByNu[c] = gg;
}


__global__
void nutKernel(int nC, const scalar* __restrict__ k, const scalar* __restrict__ eps, scalar Cmu, scalar* __restrict__ nut)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < nC) nut[c] = Cmu * k[c] * k[c] / eps[c];
}


// realizableKE rCmu (variable Cmu) + magS from the gradU tensor. S=devSymm(gradU); S2=2*magSqr(S); magS=sqrt(S2);
// W=2sqrt2*tr(S^3)/(magS*S2); phis=acos(clamp(sqrt6*W,[-1,1]))/3; As=sqrt6*cos(phis); Us=sqrt(S2/2+magSqr(skew));
// rCmu=1/(A0 + As*Us*k/eps).  (OF realizableKE::rCmu, gradU_ij = t[i*3+j] = dU_j/dx_i.)
__global__
void rkeStrainKernel(
    int nC,
    const scalar* __restrict__ gradU,
    const scalar* __restrict__ k,
    const scalar* __restrict__ eps,
    scalar A0,
    scalar* __restrict__ rCmu,
    scalar* __restrict__ magS)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    scalar t[9];
    for (int q = 0; q < 9; ++q)
        t[q] = gradU[q * nC + c];

    const scalar third = (t[0] + t[4] + t[8]) / 3.0;
    scalar S[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            S[i*3+j] = 0.5*(t[i*3+j] + t[j*3+i]) - ((i==j) ? third : 0.0);

    scalar magSqrS = 0.0, skSq = 0.0, tr3 = 0.0;
    for (int q = 0; q < 9; ++q)
        magSqrS += S[q]*S[q];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            const scalar sk = 0.5*(t[i*3+j]-t[j*3+i]);
            skSq += sk*sk;
        }
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int m = 0; m < 3; ++m)
                tr3 += S[i*3+j]*S[j*3+m]*S[m*3+i];

    const scalar S2 = 2.0*magSqrS, mS = sqrt(S2);
    scalar arg = sqrt(6.0) * (2.0*sqrt(2.0)*tr3 / (mS*S2 + 1e-37));
    arg = fmin(fmax(arg, -1.0), 1.0);
    const scalar As = sqrt(6.0)*cos((1.0/3.0)*acos(arg));
    const scalar Us = sqrt(0.5*S2 + skSq);
    rCmu[c] = 1.0/(A0 + As*Us*k[c]/eps[c]);
    magS[c] = mS;
}


__global__
void rkeNutKernel(
    int nC,
    const scalar* __restrict__ rCmu,
    const scalar* __restrict__ k,
    const scalar* __restrict__ eps,
    scalar* __restrict__ nut)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < nC) nut[c] = rCmu[c] * k[c] * k[c] / eps[c];
}


// realizableKE eps reaction: production C1*magS*eps (explicit), destruction fvm::Sp(C2*eps/(k+sqrt(nu*eps)),eps).
// C1 = max(eta/(5+eta), 0.43), eta = magS*k/eps. No divU term (unlike standard kEpsilon).
__global__
void rkeEpsReactionKernel(
    int nC,
    const scalar* __restrict__ V,
    const scalar* __restrict__ eps,
    const scalar* __restrict__ k,
    const scalar* __restrict__ magS,
    scalar nu,
    scalar C2,
    scalar* __restrict__ diag,
    scalar* __restrict__ source)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    const scalar e = eps[c], kc = k[c], mS = magS[c];
    const scalar eta = mS * kc / e;
    const scalar C1 = fmax(eta/(5.0 + eta), 0.43);
    diag[c]   += V[c] * (C2 * e / (kc + sqrt(fmax(nu*e, 0.0))));   // destruction Sp
    source[c] += V[c] * (C1 * mS * e);                            // production
}


__global__
void wallFnKernel(
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
    scalar Cmu75,
    scalar kappa,
    scalar E,
    scalar atmZ0,
    bool   atmBoundNut,
    int nutWall,
    scalar* __restrict__ eps0,
    scalar* __restrict__ G0)
{
    const int wf = blockIdx.x * blockDim.x + threadIdx.x;
    if (wf >= nWF) return;

    const int c = wfCell[wf];
    const scalar y = wfY[wf], dc = wfDc[wf], kc = k[c];
    wallProductionG0(c, wf, y, dc, kc, invNw[c], wux, wuy, wuz, Ux, Uy, Uz, nu, yplLam, Cmu25, kappa, E, atmZ0, atmBoundNut, nutWall, G0);
    atomicAdd(&eps0[c], invNw[c] * Cmu75 * pow(kc, 1.5) / (kappa * y));   // kEpsilon: distinct eps wall value
}


// boundary nut per face: wall -> nutkWallFunction(k[cell], y, nu); else -> nut[cell] (calculated/extrapolated).
__global__
void boundaryNutKernel(
    int n,
    const label* __restrict__ fc,
    const label* __restrict__ isWall,
    const scalar* __restrict__ y,
    const scalar* __restrict__ k,
    const scalar* __restrict__ nut,
    scalar nu,
    scalar yplLam,
    scalar Cmu25,
    scalar kappa,
    scalar E,
    scalar atmZ0,        // >0 -> atmNutkWallFunction (rough); 0 -> nutkWallFunction (smooth)
    bool   atmBoundNut,
    scalar* __restrict__ nutBnd)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int c = fc[i];
    if (isWall[i])
    {
        nutBnd[i] = kBasedWallNut(yPlusWall(Cmu25, y[i], k[c], nu), y[i], atmZ0, atmBoundNut, nu, yplLam, kappa, E);
    }
    else
    {
        nutBnd[i] = nut[c];
    }
}


__global__
void epsReactionKernel(
    int nC,
    const scalar* __restrict__ V,
    const scalar* __restrict__ eps,
    const scalar* __restrict__ k,
    const scalar* __restrict__ gByNu,
    const scalar* __restrict__ divU,
    scalar C1,
    scalar C2,
    scalar C3,
    scalar Cmu,
    scalar* __restrict__ diag,
    scalar* __restrict__ source)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    const scalar sp = ((2.0/3.0) * C1 - C3) * divU[c];   // OF SuSp(((2/3)C1 - C3)*divU, eps)
    diag[c]   += V[c] * (C2 * eps[c] / k[c] + fmax(sp, 0.0));
    source[c] += V[c] * (C1 * Cmu * k[c] * gByNu[c]) - V[c] * fmin(sp, 0.0) * eps[c];
}


__global__
void kReactionKernel(
    int nC,
    const scalar* __restrict__ V,
    const scalar* __restrict__ k,
    const scalar* __restrict__ eps,
    const scalar* __restrict__ G,
    const scalar* __restrict__ divU,
    scalar* __restrict__ diag,
    scalar* __restrict__ source)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    const scalar sp = (2.0/3.0) * divU[c];
    diag[c]   += V[c] * (eps[c] / k[c] + fmax(sp, 0.0));
    source[c] += V[c] * G[c] - V[c] * fmin(sp, 0.0) * k[c];
}


__global__
void depsKernel(int nC, const scalar* __restrict__ nut, scalar sigma, scalar nu, scalar* __restrict__ D)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < nC) D[c] = nut[c] / sigma + nu;
}


__global__
void overrideKernel(
    int nC,
    const label* __restrict__ isW,
    const scalar* __restrict__ G0,
    const scalar* __restrict__ eps0,
    scalar* __restrict__ G,
    scalar* __restrict__ eps)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < nC && isW[c]) { G[c] = G0[c]; eps[c] = eps0[c]; }
}


// OF Foam::bound(): negative cells -> fvc::average(max(field,floor)) (local face-neighbour avg), positive cells ->
// max(field,floor). Prevents nut=Cmu k^2/eps blowing up when limitedLinear overshoots eps<0 (vs clamping to floor).
__global__
void boundClampKernel(int nC, scalar floor, const scalar* __restrict__ x, scalar* __restrict__ cl)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < nC) cl[c] = fmax(x[c], floor);
}


__global__
void boundAvgGatherKernel(
    int nC,
    const label* __restrict__ ownerStart,
    const label* __restrict__ losort,
    const label* __restrict__ losortStart,
    const label* __restrict__ bndCellStart,
    const scalar* __restrict__ faceInterp,
    const scalar* __restrict__ cl,
    scalar* __restrict__ avg)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    scalar s = 0.0;
    int nf = 0;
    for (int f = ownerStart[c]; f < ownerStart[c + 1]; ++f) { s += faceInterp[f]; ++nf; }             // c is owner
    for (int k = losortStart[c]; k < losortStart[c + 1]; ++k) { s += faceInterp[losort[k]]; ++nf; }    // c is neighbour
    const int nb = bndCellStart[c + 1] - bndCellStart[c];                                              // boundary faces (zeroGradient = cell value)
    s += cl[c] * nb;
    nf += nb;
    avg[c] = (nf > 0) ? s / nf : cl[c];
}


__global__
void boundApplyKernel(int nC, scalar floor, const scalar* __restrict__ avg, scalar* __restrict__ x)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    x[c] = (x[c] <= 0.0) ? avg[c] : fmax(x[c], floor);   // OF: max(max(vsf, avg*pos0(-vsf)), floor)
}


// setValues (eps wall constraint): zero wall-cell off-diagonals + move the known eps0 to the neighbour RHS.
__global__
void svFaceKernel(
    int nIf,
    const label* __restrict__ own,
    const label* __restrict__ nei,
    const label* __restrict__ isW,
    const scalar* __restrict__ eps0,
    scalar* __restrict__ upper,
    scalar* __restrict__ lower,
    scalar* __restrict__ source)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIf) return;

    const int o = own[f], n = nei[f];
    const bool ow = isW[o], nw = isW[n];
    if (ow) atomicAdd(&source[n], -lower[f] * eps0[o]);
    if (nw) atomicAdd(&source[o], -upper[f] * eps0[n]);
    if (ow || nw) { upper[f] = 0.0; lower[f] = 0.0; }
}


__global__
void svBndKernel(
    int nB,
    const label* __restrict__ faceCell,
    const label* __restrict__ isW,
    scalar* __restrict__ iC,
    scalar* __restrict__ bC)
{
    const int bi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bi < nB && isW[faceCell[bi]]) { iC[bi] = 0.0; bC[bi] = 0.0; }
}


__global__
void svCellKernel(
    int nC,
    const label* __restrict__ isW,
    const scalar* __restrict__ relaxedDiag,
    const scalar* __restrict__ eps0,
    scalar* __restrict__ source)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < nC && isW[c]) source[c] = relaxedDiag[c] * eps0[c];
}
} // namespace


// Build the OF-convention gradU tensor (9*nC, column i = gaussGrad(U_i)). Shared by GbyNu (k-eps) and S2 (SST).
void deviceGradU(
    const DeviceMesh& dm,
    const DeviceVectorBoundary& dbU,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    DeviceBuffer<scalar>& gradU,
    DeviceAMI* ami,
    DeviceCyclic* cyc)
{
    const int nC = dm.nCells;
    const DeviceBuffer<scalar>* Uc[3] = { &Ux, &Uy, &Uz };
    gradU.resize(static_cast<std::size_t>(9) * nC);

    // interface (cyclic/cyclicAMI) face contribution to grad(U): without it the gradient at interface cells is
    // ONE-SIDED -> wrong turbulence production G=nut*(gradU&&devTwoSymm(gradU)) there (the divDevReff x-invariance
    // bug, in the production term). For a ROTATIONAL interface the neighbour vector rotates (forwardT).
    DeviceBuffer<scalar> amiURot[3];
    if (ami && ami->n && ami->rotational) deviceAmiInterpolateVec(*ami, Ux, Uy, Uz, amiURot[0], amiURot[1], amiURot[2]);
    for (int i = 0; i < 3; ++i)
    {
        DeviceBuffer<scalar> bval;
        deviceBCValue(dbU.comp[i], *Uc[i], bval);
        DeviceBuffer<scalar> gx, gy, gz;
        deviceGaussGrad(dm, *Uc[i], bval, gx, gy, gz);
        if (ami && ami->n) { if (ami->rotational) deviceAmiAddGradRot(*ami, *Uc[i], amiURot[i], dm.V, gx, gy, gz);
                             else deviceAmiAddGrad(*ami, *Uc[i], dm.V, gx, gy, gz); }
        if (cyc && cyc->n) { if (cyc->rotational) deviceCyclicAddGradRot(*cyc, Ux, Uy, Uz, i, dm.V, gx, gy, gz);
                             else deviceCyclicAddGrad(*cyc, *Uc[i], dm.V, gx, gy, gz); }
        // async D2D on the per-thread stream (ordered before the consumer kernel): a plain cudaMemcpy here would
        // drain the GPU pipeline every turbulence iteration.
        cudaCheck(cudaMemcpyAsync(gradU.data() + (0*3+i)*nC, gx.data(), nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread), "gradU g");
        cudaCheck(cudaMemcpyAsync(gradU.data() + (1*3+i)*nC, gy.data(), nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread), "gradU g");
        cudaCheck(cudaMemcpyAsync(gradU.data() + (2*3+i)*nC, gz.data(), nC*sizeof(scalar), cudaMemcpyDeviceToDevice, cudaStreamPerThread), "gradU g");
    }
}


void deviceGByNuFromGradU(const DeviceBuffer<scalar>& gradU, int nC, DeviceBuffer<scalar>& gByNu)
{
    gByNu.resize(nC);
    gByNuKernel<<<nBlocks(nC), TPB>>>(nC, gradU.data(), gByNu.data());
    cudaCheck(cudaGetLastError(), "gByNu");
}


void deviceGbyNu(
    const DeviceMesh& dm,
    const DeviceVectorBoundary& dbU,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    DeviceBuffer<scalar>& gByNu,
    DeviceAMI* ami,
    DeviceCyclic* cyc)
{
    DeviceBuffer<scalar> gradU;
    deviceGradU(dm, dbU, Ux, Uy, Uz, gradU, ami, cyc);
    deviceGByNuFromGradU(gradU, dm.nCells, gByNu);
}


void deviceNut(
    const DeviceBuffer<scalar>& k,
    const DeviceBuffer<scalar>& eps,
    DeviceBuffer<scalar>& nut,
    const KEpsilonCoeffs& co)
{
    const int nC = static_cast<int>(k.size());
    nut.resize(nC);
    nutKernel<<<nBlocks(nC), TPB>>>(nC, k.data(), eps.data(), co.Cmu, nut.data());
    cudaCheck(cudaGetLastError(), "nut");
}


// realizableKE: rCmu + magS from gradU, nut = rCmu*k^2/eps, eps reaction (strain production + k+sqrt(nu*eps) destruction).
void deviceRealizableStrain(
    const DeviceBuffer<scalar>& gradU,
    const DeviceBuffer<scalar>& k,
    const DeviceBuffer<scalar>& eps,
    scalar A0,
    int nC,
    DeviceBuffer<scalar>& rCmu,
    DeviceBuffer<scalar>& magS)
{
    rCmu.resize(nC);
    magS.resize(nC);
    rkeStrainKernel<<<nBlocks(nC), TPB>>>(nC, gradU.data(), k.data(), eps.data(), A0, rCmu.data(), magS.data());
    cudaCheck(cudaGetLastError(), "rkeStrain");
}


void deviceRealizableNut(
    const DeviceBuffer<scalar>& rCmu,
    const DeviceBuffer<scalar>& k,
    const DeviceBuffer<scalar>& eps,
    DeviceBuffer<scalar>& nut)
{
    const int nC = static_cast<int>(k.size());
    nut.resize(nC);
    rkeNutKernel<<<nBlocks(nC), TPB>>>(nC, rCmu.data(), k.data(), eps.data(), nut.data());
    cudaCheck(cudaGetLastError(), "rkeNut");
}


void deviceEpsReactionRealizable(
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& eps,
    const DeviceBuffer<scalar>& k,
    const DeviceBuffer<scalar>& magS,
    scalar nu,
    scalar C2,
    DeviceBuffer<scalar>& diag,
    DeviceBuffer<scalar>& source)
{
    rkeEpsReactionKernel<<<nBlocks(dm.nCells), TPB>>>(dm.nCells, dm.V.data(), eps.data(), k.data(), magS.data(),
                                                      nu, C2, diag.data(), source.data());
    cudaCheck(cudaGetLastError(), "rkeEpsReaction");
}


// OF Foam::bound(field, floor): negative cells -> fvc::average(max(field,floor)); positive -> max(field,floor).
void deviceBoundField(const DeviceMesh& dm, DeviceBuffer<scalar>& x, scalar floor)
{
    const int nC = dm.nCells;
    DeviceBuffer<scalar> cl(static_cast<std::size_t>(nC));
    boundClampKernel<<<nBlocks(nC), TPB>>>(nC, floor, x.data(), cl.data());
    DeviceBuffer<scalar> fi;
    deviceInterpolate(dm, cl, fi);   // linearInterpolate(max(field,floor))
    DeviceBuffer<scalar> avg(static_cast<std::size_t>(nC));
    boundAvgGatherKernel<<<nBlocks(nC), TPB>>>(nC, dm.ownerStart.data(), dm.losort.data(), dm.losortStart.data(),
                                               dm.bndCellStart.data(), fi.data(), cl.data(), avg.data());
    boundApplyKernel<<<nBlocks(nC), TPB>>>(nC, floor, avg.data(), x.data());
    cudaCheck(cudaGetLastError(), "boundField");
}


void deviceWallEpsG0(
    const DeviceWallData& w,
    const DeviceBuffer<scalar>& k,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    scalar nu,
    DeviceBuffer<scalar>& eps0,
    DeviceBuffer<scalar>& G0,
    const KEpsilonCoeffs& co,
    int nutWall,
    scalar atmZ0,
    bool   atmBoundNut)
{
    const int nC = static_cast<int>(k.size());
    eps0.resize(nC);
    G0.resize(nC);   // zero on-device (memsetAsync) instead of a host-vector alloc + H2D every iter
    cudaCheck(cudaMemsetAsync(eps0.data(), 0, nC*sizeof(scalar), cudaStreamPerThread), "eps0 zero");
    cudaCheck(cudaMemsetAsync(G0.data(),   0, nC*sizeof(scalar), cudaStreamPerThread), "G0 zero");
    const scalar Cmu25 = std::pow(co.Cmu, 0.25), Cmu75 = std::pow(co.Cmu, 0.75), yplLam = yPlusLamHost(co.kappa, co.E);
    if (w.nWF > 0)
        wallFnKernel<<<nBlocks(w.nWF), TPB>>>(w.nWF, w.wfCell.data(), w.wfY.data(), w.wfDc.data(), w.wfUwx.data(),
                                              w.wfUwy.data(), w.wfUwz.data(), w.invNw.data(), k.data(), Ux.data(), Uy.data(),
                                              Uz.data(), nu, yplLam, Cmu25, Cmu75, co.kappa, co.E, atmZ0, atmBoundNut, nutWall, eps0.data(), G0.data());
    cudaCheck(cudaGetLastError(), "wallFn");
}


void deviceEpsReaction(
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& eps,
    const DeviceBuffer<scalar>& k,
    const DeviceBuffer<scalar>& gByNu,
    const DeviceBuffer<scalar>& divU,
    DeviceBuffer<scalar>& diag,
    DeviceBuffer<scalar>& source,
    const KEpsilonCoeffs& co)
{
    epsReactionKernel<<<nBlocks(dm.nCells), TPB>>>(dm.nCells, dm.V.data(), eps.data(), k.data(), gByNu.data(), divU.data(),
                                                   co.C1, co.C2, co.C3, co.Cmu, diag.data(), source.data());
    cudaCheck(cudaGetLastError(), "epsReaction");
}


void deviceKReaction(
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& k,
    const DeviceBuffer<scalar>& eps,
    const DeviceBuffer<scalar>& G,
    const DeviceBuffer<scalar>& divU,
    DeviceBuffer<scalar>& diag,
    DeviceBuffer<scalar>& source)
{
    kReactionKernel<<<nBlocks(dm.nCells), TPB>>>(dm.nCells, dm.V.data(), k.data(), eps.data(), G.data(), divU.data(), diag.data(), source.data());
    cudaCheck(cudaGetLastError(), "kReaction");
}


void deviceBoundaryNut(
    const DeviceBoundary& db,
    const DeviceBuffer<label>& isWall,
    const DeviceBuffer<scalar>& y,
    const DeviceBuffer<scalar>& k,
    const DeviceBuffer<scalar>& nut,
    scalar nu,
    DeviceBuffer<scalar>& nutBnd,
    const KEpsilonCoeffs& co,
    scalar atmZ0,
    bool   atmBoundNut)
{
    nutBnd.resize(db.n);
    const scalar Cmu25 = std::pow(co.Cmu, 0.25), yplLam = yPlusLamHost(co.kappa, co.E);
    boundaryNutKernel<<<nBlocks(db.n), TPB>>>(db.n, db.faceCell.data(), isWall.data(), y.data(), k.data(), nut.data(),
                                              nu, yplLam, Cmu25, co.kappa, co.E, atmZ0, atmBoundNut, nutBnd.data());
    cudaCheck(cudaGetLastError(), "boundaryNut");
}


// One scalar-transport sub-step of a two-equation RAS correct(): assemble div(phi,f) - laplacian(D,f)
// [- Sp(div(phi),f) if bounded], add the model reaction (diag+source via `reaction`), relax, optionally
// apply the eps near-wall setValues constraint, fold, BiCGStab (loose relTol), bound. Reused by k-omega SST.
template <class Reaction>
static void deviceSolveScalarTransport(
    const DeviceMesh& dm,
    const DeviceBoundary& db,
    DeviceBuffer<scalar>& field,
    const char* fieldName,
    const DeviceBuffer<scalar>& D,
    const DeviceBuffer<scalar>& phiInt,
    const DeviceBuffer<scalar>& phiBnd,
    const DeviceBuffer<scalar>& divU,
    bool bounded,
    bool limited,
    bool linearUpwind,
    bool nonOrth,
    scalar twoByk,
    scalar relax,
    scalar tol,
    scalar relTolKE,
    int keCheckEvery,
    bool useGS,
    Reaction&& reaction,
    const DeviceWallData* wall = nullptr,
    const DeviceBuffer<scalar>* eps0 = nullptr,
    DeviceAMI* ami = nullptr,
    DeviceCyclic* cyc = nullptr)
{
    const int nC = dm.nCells;
    DeviceBuffer<scalar> Df;
    deviceInterpolate(dm, D, Df);
    DeviceBuffer<scalar> aD, aU, aL, lD, lU, lL, luCorr, lapCorr;   // luCorr/lapCorr = linearUpwind / non-orth deferred sources (empty otherwise)
    DeviceBuffer<scalar> gx, gy, gz;                               // grad(field): shared by limitedLinear / linearUpwind / non-orth
    if (limited || linearUpwind || nonOrth) { DeviceBuffer<scalar> bv; deviceBCValue(db, field, bv); deviceGaussGrad(dm, field, bv, gx, gy, gz); }
    if (limited) deviceDivLimitedCoeffs(dm, phiInt, field, gx, gy, gz, twoByk, aD, aU, aL);   // Gauss limitedLinear: implicit limited weight
    else
    {
        deviceDivUpwindCoeffs(dm, phiInt, aD, aU, aL);
        if (linearUpwind) deviceLinearUpwindCorr(dm, phiInt, gx, gy, gz, luCorr);             // Gauss linearUpwind: upwind matrix + deferred corr
    }
    // laplacian "corrected": nonOrthDeltaCoeffs implicit (in deviceLaplacianCoeffs) + corrVec.grad(field) explicit (deviceLaplacianCorr).
    deviceLaplacianCoeffs(dm, Df, lD, lU, lL, nonOrth);
    deviceAxpy(-1.0, lD, aD); deviceAxpy(-1.0, lU, aU); deviceAxpy(-1.0, lL, aL);
    if (nonOrth) deviceLaplacianCorr(dm, Df, gx, gy, gz, lapCorr);
    if (bounded) { DeviceBuffer<scalar> bt; deviceHadamard(bt, divU, dm.V); deviceAxpy(-1.0, bt, aD); }   // -Sp(div(phi),f)
    DeviceBuffer<scalar> src(static_cast<std::size_t>(nC));
    cudaCheck(cudaMemsetAsync(src.data(), 0, nC*sizeof(scalar), cudaStreamPerThread), "src zero");
    reaction(aD, src);                                            // model reaction: adds to diag + source
    if (luCorr.size())  deviceAxpy(-1.0, luCorr, src);           // linearUpwind deferred correction (explicit RHS)
    if (lapCorr.size()) deviceAxpy(-1.0, lapCorr, src);          // non-orth laplacian correction (explicit RHS, mirrors momentum)
    DeviceBuffer<scalar> aIC, aBC, lIC, lBC; deviceBCDivCoeffs(db, phiBnd, aIC, aBC); deviceBCLaplacianCoeffs(db, D, lIC, lBC);
    deviceAxpy(-1.0, lIC, aIC); deviceAxpy(-1.0, lBC, aBC);
    // interface (cyclic/cyclicAMI) coupling: fold div(phi,f) - laplacian(D,f) at the interface into the diagonal and
    // set the off-diagonal ifCoeff. A scalar is invariant under the cyclic transform (no rotation of the value), so the
    // translational momentum assembly + a plain weighted off-diagonal apply even for a ROTATIONAL interface.
    DeviceBuffer<scalar> ifSumOff;
    if (ami && ami->n) { deviceAmiAssembleMomentum(*ami, D, aD);
        ifSumOff.copyFrom(std::vector<scalar>(nC, 0.0)); deviceAmiOffDiagSum(*ami, ifSumOff); }
    else if (cyc && cyc->n) { deviceCyclicAssembleMomentum(*cyc, D, aD);
        ifSumOff.copyFrom(std::vector<scalar>(nC, 0.0)); deviceCyclicOffDiagSum(*cyc, ifSumOff); }
    DeviceBuffer<scalar> aRD, aDelta; deviceRelaxDiag(deviceLduView(dm, aD, aU, aL), dm, aIC, relax, aRD, aDelta,
                                                      ifSumOff.size() ? ifSumOff.data() : nullptr);
    { DeviceBuffer<scalar> t; deviceHadamard(t, aDelta, field); deviceAxpy(1.0, t, src); }
    if (wall && eps0)   // eps near-wall setValues constraint (k has none)
    {
        svFaceKernel<<<nBlocks(dm.nInternalFaces), TPB>>>(dm.nInternalFaces, dm.owner.data(), dm.nei.data(), wall->isWallCell.data(), eps0->data(), aU.data(), aL.data(), src.data());
        svBndKernel<<<nBlocks(dm.nBndFaces), TPB>>>(dm.nBndFaces, dm.bndCell.data(), wall->isWallCell.data(), aIC.data(), aBC.data());
        svCellKernel<<<nBlocks(nC), TPB>>>(nC, wall->isWallCell.data(), aRD.data(), eps0->data(), src.data());
        if (ami && ami->n) deviceAmiZeroWallIfCoeff(*ami, wall->isWallCell);   // wall/interface cells: don't perturb the fixed eps
        if (cyc && cyc->n) deviceCyclicZeroWallIfCoeff(*cyc, wall->isWallCell);
        cudaCheck(cudaGetLastError(), "setValues");
    }
    DeviceBuffer<scalar> diagC, B; deviceFold(dm, aRD, src, aIC, aBC, diagC, B);
    const DeviceLduView sv = (ami && ami->n)
        ? deviceLduViewAmi(dm, diagC, aU, aL, ami->n, ami->ownCell.data(), ami->off.data(), ami->nbrCell.data(), ami->weight.data(), ami->ifCoeff.data())
        : (cyc && cyc->n)
          ? deviceLduViewCyclic(dm, diagC, aU, aL, cyc->n, cyc->ownCell.data(), cyc->nbrCell.data(), cyc->ifCoeff.data())
          : deviceLduView(dm, diagC, aU, aL);
    // Linear solver SELECTED FROM fvSolution like OF (solvers.<field>: solver smoothSolver; smoother symGaussSeidel
    // -> useGS, set by the caller from the dict). cf's multicolor GS does NOT carry the interface (cyclic/AMI)
    // off-diagonal, so interface LDUs fall back to BiCGStab (which folds the interface into Amul), a documented GPU
    // limitation, NOT a heuristic. BRAE_SCALAR_GS overrides for debugging (=0 force off, !=0 force on).
    bool wantGS = useGS;
    if (const char* e = std::getenv("BRAE_SCALAR_GS")) wantGS = (std::atoi(e) != 0);
    const bool gs = wantGS && !(ami && ami->n) && !(cyc && cyc->n);
    DeviceSolverPerf perf;                                        // OF-style report: init/final/nIter for this scalar
    if (gs) deviceSymGaussSeidel(sv, B, field, deviceSumMag(B) + 1e-20, tol, relTolKE, 3000, &perf);
    else    perf = deviceJacobiBiCGStab(sv, B, field, deviceSumMag(B) + 1e-20, tol, relTolKE, 3000, keCheckEvery);  // loose (relTol); interface in the Amul
    turbStore().push_back({fieldName, perf});                    // record for the "Solving for <field>" line
    deviceBoundField(dm, field, 1e-15);                           // OF bound(field): neg -> local avg, not floor
}


void deviceKEpsilonCorrect(
    const DeviceMesh& dm,
    const DeviceWallData& wall,
    const DeviceBoundary& dbEps,
    const DeviceBoundary& dbK,
    const DeviceVectorBoundary& dbU,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    DeviceBuffer<scalar>& k,
    DeviceBuffer<scalar>& eps,
    DeviceBuffer<scalar>& nut,
    const DeviceBuffer<scalar>& phiInt,
    const DeviceBuffer<scalar>& phiBnd,
    scalar nu,
    scalar relaxEps,
    scalar relaxK,
    scalar tol,
    bool bounded,
    bool limitedK,
    bool limitedEps,
    scalar twoBykK,
    scalar twoBykEps,
    const KEpsilonCoeffs& co,
    scalar relTolKE,
    int keCheckEvery,
    bool linearUpwindK,
    bool linearUpwindEps,
    bool nonOrth,
    bool gsK,
    bool gsEps,
    DeviceAMI* ami,
    DeviceCyclic* cyc,
    int nutWall,
    scalar atmZ0,
    bool atmBoundNut)
{
    const int nC = dm.nCells;
    // production + divU, wall functions + near-wall override.
    DeviceBuffer<scalar> gByNu, rCmu, magS;
    if (co.realizable)   // realizableKE: needs the full gradU tensor for rCmu/magS (deviceGbyNu only returns gByNu)
    {
        DeviceBuffer<scalar> gradU;
        deviceGradU(dm, dbU, Ux, Uy, Uz, gradU, ami, cyc);
        deviceGByNuFromGradU(gradU, nC, gByNu);
        deviceRealizableStrain(gradU, k, eps, co.A0, nC, rCmu, magS);
    }
    else deviceGbyNu(dm, dbU, Ux, Uy, Uz, gByNu, ami, cyc);   // interface-aware grad(U) for production
    DeviceBuffer<scalar> G;
    deviceHadamard(G, nut, gByNu);
    DeviceBuffer<scalar> divU;
    deviceDiv(dm, phiInt, phiBnd, divU);
    if (ami && ami->n) deviceAmiAddDiv(*ami, dm.V, divU);   // interface flux into div(phi) (bounded term + reaction Sp)
    if (cyc && cyc->n) deviceCyclicAddDiv(*cyc, dm.V, divU);
    DeviceBuffer<scalar> eps0, G0;
    deviceWallEpsG0(wall, k, Ux, Uy, Uz, nu, eps0, G0, co, nutWall, atmZ0, atmBoundNut);
    overrideKernel<<<nBlocks(nC), TPB>>>(nC, wall.isWallCell.data(), G0.data(), eps0.data(), G.data(), eps.data());

    // epsilon equation (loose solve) with the near-wall setValues constraint
    DeviceBuffer<scalar> Deps(static_cast<std::size_t>(nC));
    depsKernel<<<nBlocks(nC), TPB>>>(nC, nut.data(), co.sigmaEps, nu, Deps.data());
    deviceSolveScalarTransport(dm, dbEps, eps, "epsilon", Deps, phiInt, phiBnd, divU, bounded, limitedEps, linearUpwindEps, nonOrth, twoBykEps,
                               relaxEps, tol, relTolKE, keCheckEvery, gsEps,
                               [&](DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& src){
                                   if (co.realizable) deviceEpsReactionRealizable(dm, eps, k, magS, nu, co.C2, diag, src);
                                   else               deviceEpsReaction(dm, eps, k, gByNu, divU, diag, src, co); },
                               &wall, &eps0, ami, cyc);

    // k equation (loose solve)
    DeviceBuffer<scalar> Dk(static_cast<std::size_t>(nC));
    depsKernel<<<nBlocks(nC), TPB>>>(nC, nut.data(), co.sigmaK, nu, Dk.data());
    deviceSolveScalarTransport(dm, dbK, k, "k", Dk, phiInt, phiBnd, divU, bounded, limitedK, linearUpwindK, nonOrth, twoBykK,
                               relaxK, tol, relTolKE, keCheckEvery, gsK,
                               [&](DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& src){ deviceKReaction(dm, k, eps, G, divU, diag, src); },
                               nullptr, nullptr, ami, cyc);

    // correctNut (cell): nut = Cmu k^2 / eps (realizableKE: rCmu k^2 / eps with the variable Cmu).
    if (co.realizable) deviceRealizableNut(rCmu, k, eps, nut);
    else               deviceNut(k, eps, nut, co);
}


// OF kOmegaSSTBase::correct/correctNut use tgradU = fvc::grad(U) = the named grad(U) scheme for BOTH the strain
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
    bool atmBoundNut)
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
    if (ami && ami->n) deviceAmiAddDiv(*ami, dm.V, divU);
    if (cyc && cyc->n) deviceCyclicAddDiv(*cyc, dm.V, divU);
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
                               &wall, &omega0, ami, cyc);
    deviceBoundField(dm, omega, 1e-15);   // OF bound(omega_, omegaMin_)

    // k equation (loose solve)
    DeviceBuffer<scalar> DkEff;
    deviceDEff(F1, nut, co.alphaK1, co.alphaK2, nu, DkEff);
    deviceSolveScalarTransport(dm, dbK, k, "k", DkEff, phiInt, phiBnd, divU, bounded, limitedK, linearUpwindK, nonOrth, twoBykK,
                               relaxK, tol, relTolKE, keCheckEvery, gsK,
                               [&](DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& src){ deviceKReactionSST(dm.V, k, omega, G, divU, co, diag, src, gammaIntEff); },
                               nullptr, nullptr, ami, cyc);
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
    DeviceCyclic* cyc)
{
    const int nC = dm.nCells;
    const LMCoeffs lm;
    DeviceBuffer<scalar> gradU;
    deviceGradU(dm, dbU, Ux, Uy, Uz, gradU, ami, cyc);
    DeviceBuffer<scalar> divU;
    deviceDiv(dm, phiInt, phiBnd, divU);
    if (ami && ami->n) deviceAmiAddDiv(*ami, dm.V, divU);
    if (cyc && cyc->n) deviceCyclicAddDiv(*cyc, dm.V, divU);

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
                               nullptr, nullptr, ami, cyc);
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
                               nullptr, nullptr, ami, cyc);
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


// nutUSpaldingWallFunction: wall faces -> Newton uTau from Spalding's law, nut = max(0, uTau^2/magGradU - nu).
// magGradU = |snGrad U| = |U_cell|*deltaCoeffs (noSlip U_wall=0); magUp = |U_cell|; y = near-wall distance.
// Warm-started from the previous wall nut (nutBnd in/out), 10 Newton iters with the OF tol=0.01 early-out.
__global__
void spaldingNutKernel(
    int n,
    const label* __restrict__ fc,
    const label* __restrict__ isWall,
    const scalar* __restrict__ y,
    const scalar* __restrict__ dc,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    const scalar* __restrict__ nutCell,
    scalar nu,
    scalar kappa,
    scalar E,
    scalar* __restrict__ nutBnd)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int c = fc[i];
    if (!isWall[i]) { nutBnd[i] = nutCell[c]; return; }
    const scalar magUp = sqrt(Ux[c]*Ux[c] + Uy[c]*Uy[c] + Uz[c]*Uz[c]);
    const scalar magGradU = magUp * dc[i];
    nutBnd[i] = spaldingNutValue(magUp, magGradU, y[i], nu, kappa, E, nutBnd[i]);   // shared with G0 (nut_wall_function.cuh)
}


// nutUBlendedWallFunction (OpenFOAM v2412): wall faces -> uTau from the binomial blend of the viscous and
// log velocity scales, nut = max(0, uTau^2/magGradU - nu). uTau = (uTauVis^n + uTauLog^n)^(1/n), n=4:
//   yPlus = y*uTau/nu ; uTauVis = magUp/yPlus ; uTauLog = kappa*magUp/log(max(E*yPlus, 1+1e-4)).
// 10 iters, tol 1e-3, under-relaxed uTau update (ut = 0.5*(ut+utNew)), warm-started like the Spalding kernel.
// magUp/magGradU use the same convention as spaldingNutKernel (|U_cell|, |snGrad U|=|U_cell|*deltaCoeffs).
__global__
void blendedNutKernel(
    int n,
    const label* __restrict__ fc,
    const label* __restrict__ isWall,
    const scalar* __restrict__ y,
    const scalar* __restrict__ dc,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    const scalar* __restrict__ nutCell,
    scalar nu,
    scalar kappa,
    scalar E,
    scalar* __restrict__ nutBnd)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int c = fc[i];
    if (!isWall[i]) { nutBnd[i] = nutCell[c]; return; }
    const scalar magUp = sqrt(Ux[c]*Ux[c] + Uy[c]*Uy[c] + Uz[c]*Uz[c]);
    const scalar magGradU = magUp * dc[i];
    nutBnd[i] = blendedNutValue(magUp, magGradU, y[i], nu, kappa, E, nutBnd[i]);   // shared with G0 (nut_wall_function.cuh)
}
} // namespace


void deviceBoundaryNutSpalding(
    const DeviceVectorBoundary& dbU,
    const DeviceBuffer<label>& isWall,
    const DeviceBuffer<scalar>& y,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    const DeviceBuffer<scalar>& nutCell,
    scalar nu,
    const SpalartAllmarasCoeffs& co,
    DeviceBuffer<scalar>& nutBnd)
{
    const int n = dbU.comp[0].n;
    if (static_cast<int>(nutBnd.size()) != n)   // first call: zero seed (warm-starts thereafter)
    {
        nutBnd.resize(n);
        cudaCheck(cudaMemsetAsync(nutBnd.data(), 0, n*sizeof(scalar), cudaStreamPerThread), "spalding init");
    }
    spaldingNutKernel<<<nBlocks(n), TPB>>>(n, dbU.comp[0].faceCell.data(), isWall.data(), y.data(),
                                           dbU.comp[0].deltaCoeffs.data(), Ux.data(), Uy.data(), Uz.data(),
                                           nutCell.data(), nu, co.kappa, co.E, nutBnd.data());
    cudaCheck(cudaGetLastError(), "spaldingNut");
}


// nutUBlendedWallFunction wall nut (velocity-based binomial blend). kappa/E passed explicitly so it works
// on ANY RAS model (kEpsilon/kOmegaSST/SA), honouring the 0/nut BC type rather than the model.
void deviceBoundaryNutBlended(
    const DeviceVectorBoundary& dbU,
    const DeviceBuffer<label>& isWall,
    const DeviceBuffer<scalar>& y,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    const DeviceBuffer<scalar>& nutCell,
    scalar nu,
    scalar kappa,
    scalar E,
    DeviceBuffer<scalar>& nutBnd)
{
    const int n = dbU.comp[0].n;
    if (static_cast<int>(nutBnd.size()) != n)   // first call: zero seed (warm-starts thereafter)
    {
        nutBnd.resize(n);
        cudaCheck(cudaMemsetAsync(nutBnd.data(), 0, n*sizeof(scalar), cudaStreamPerThread), "blended init");
    }
    blendedNutKernel<<<nBlocks(n), TPB>>>(n, dbU.comp[0].faceCell.data(), isWall.data(), y.data(),
                                          dbU.comp[0].deltaCoeffs.data(), Ux.data(), Uy.data(), Uz.data(),
                                          nutCell.data(), nu, kappa, E, nutBnd.data());
    cudaCheck(cudaGetLastError(), "blendedNut");
}


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
    DeviceCyclic* cyc)
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
    if (ami && ami->n) deviceAmiAddDiv(*ami, dm.V, divU);
    if (cyc && cyc->n) deviceCyclicAddDiv(*cyc, dm.V, divU);
    cudaCheck(cudaGetLastError(), "SA assemble");
    deviceSolveScalarTransport(dm, dbNuTilda, nuTilda, "nuTilda", D, phiInt, phiBnd, divU, bounded, limited, linearUpwind, nonOrth, twoByk,
                               relax, tol, relTol, checkEvery, gsK,
                               [&](DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& src){
                                   saReactionKernel<<<nBlocks(nC), TPB>>>(nC, dm.V.data(), nuTilda.data(), Stilda.data(),
                                       fw.data(), y.data(), gradNt2.data(), co, diag.data(), src.data()); },
                               nullptr, nullptr, ami, cyc);
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
