// cf GPU offload: k-epsilon production + eddy viscosity. gradU is the OF-convention tensor (column i =
// gaussGrad(U_i), as in divDevReff); GbyNu = sum_ij g_ij*(g_ij + g_ji - (2/3)tr d_ij).
#include "device_kepsilon.cuh"
#include "device_komega_sst.cuh"
#include "device_scalar_transport.cuh"  // generic scalar-transport scaffold (deviceSolveScalarTransport)
#include "spalart_coeffs.cuh"
#include "device_ldu.cuh"
#include "device_pcg.cuh"
#include "device_simple.cuh"
#include "device_blas.cuh"
#include "device_ami.cuh"        // cyclicAMI scalar-transport interface coupling
#include "device_cyclic.cuh"     // cyclic scalar-transport interface coupling
#include "device_interface.cuh"  // interface<Op>() overloads dispatching to the cyclic/AMI backends
#include "device_amg.cuh"        // deviceSymGaussSeidel (scalar smoothSolver, for stiff low-Re k/omega)
#include "nut_wall_function.cuh" // nutkWallFunctionValue / yPlusWall (shared wall-nut physics, BRAE_HD)
#include <cuda_runtime.h>
#include <vector>

namespace brae {

// OF-style turbulence residual report (see device_kepsilon.cuh). Single-threaded per solve; the SIMPLE driver
// clears it before turbulence->correct() and reads it after, to print the "Solving for k/omega/..." lines.
void clearTurbulenceReport() { turbStore().clear(); }
const std::vector<ScalarSolveEntry>& turbulenceReport() { return turbStore(); }

namespace {
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
    scalar* __restrict__ nutBnd,
    const scalar* __restrict__ nuFace)   // compressible: per-face nu = mu_b/rho_b, null -> the scalar nu
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int c = fc[i];
    const scalar nuw = nuFace ? nuFace[i] : nu;
    if (isWall[i])
    {
        nutBnd[i] = kBasedWallNut(yPlusWall(Cmu25, y[i], k[c], nuw), y[i], atmZ0, atmBoundNut, nuw, yplLam, kappa, E);
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
                             else interfaceAddGrad(*ami, *Uc[i], dm.V, gx, gy, gz); }
        if (cyc && cyc->n) { if (cyc->rotational) deviceCyclicAddGradRot(*cyc, Ux, Uy, Uz, i, dm.V, gx, gy, gz);
                             else interfaceAddGrad(*cyc, *Uc[i], dm.V, gx, gy, gz); }
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
    bool   atmBoundNut,
    const DeviceBuffer<scalar>* nuFace)
{
    // OF turbulenceModel::nu(patchi) = mu(patchi)/rho.boundaryField()[patchi] -- a per-FACE kinematic
    // viscosity. Constant-property incompressible flow makes that a single number, which is why the scalar
    // argument exists; compressible flow does not, so nuFace overrides it when supplied.
    nutBnd.resize(db.n);
    const scalar Cmu25 = std::pow(co.Cmu, 0.25), yplLam = yPlusLamHost(co.kappa, co.E);
    boundaryNutKernel<<<nBlocks(db.n), TPB>>>(db.n, db.faceCell.data(), isWall.data(), y.data(), k.data(), nut.data(),
                                              nu, yplLam, Cmu25, co.kappa, co.E, atmZ0, atmBoundNut, nutBnd.data(),
                                              nuFace ? nuFace->data() : nullptr);
    cudaCheck(cudaGetLastError(), "boundaryNut");
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
    bool atmBoundNut,
    const ScalarDdt& kDdt,
    const ScalarDdt& eDdt)
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
    if (ami && ami->n) interfaceAddDiv(*ami, dm.V, divU);   // interface flux into div(phi) (bounded term + reaction Sp)
    if (cyc && cyc->n) interfaceAddDiv(*cyc, dm.V, divU);
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
                               &wall, &eps0, ami, cyc, eDdt);

    // k equation (loose solve)
    DeviceBuffer<scalar> Dk(static_cast<std::size_t>(nC));
    depsKernel<<<nBlocks(nC), TPB>>>(nC, nut.data(), co.sigmaK, nu, Dk.data());
    deviceSolveScalarTransport(dm, dbK, k, "k", Dk, phiInt, phiBnd, divU, bounded, limitedK, linearUpwindK, nonOrth, twoBykK,
                               relaxK, tol, relTolKE, keCheckEvery, gsK,
                               [&](DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& src){ deviceKReaction(dm, k, eps, G, divU, diag, src); },
                               nullptr, nullptr, ami, cyc, kDdt);

    // correctNut (cell): nut = Cmu k^2 / eps (realizableKE: rCmu k^2 / eps with the variable Cmu).
    if (co.realizable) deviceRealizableNut(rCmu, k, eps, nut);
    else               deviceNut(k, eps, nut, co);
}


// OF kOmegaSSTBase::correct/correctNut use tgradU = fvc::grad(U) = the named grad(U) scheme for BOTH the strain


namespace {  // wall-function nut kernels (spaldingNut/blendedNut), used by deviceBoundaryNut* below
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
    scalar* __restrict__ nutBnd,
    const scalar* __restrict__ nuFace)   // compressible: per-face nu = mu_b/rho_b, null -> the scalar nu
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int c = fc[i];
    if (!isWall[i]) { nutBnd[i] = nutCell[c]; return; }
    const scalar magUp = sqrt(Ux[c]*Ux[c] + Uy[c]*Uy[c] + Uz[c]*Uz[c]);
    const scalar magGradU = magUp * dc[i];
    const scalar nuw = nuFace ? nuFace[i] : nu;
    nutBnd[i] = spaldingNutValue(magUp, magGradU, y[i], nuw, kappa, E, nutBnd[i]);   // shared with G0 (nut_wall_function.cuh)
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
    scalar* __restrict__ nutBnd,
    const scalar* __restrict__ nuFace)   // compressible: per-face nu = mu_b/rho_b, null -> the scalar nu
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int c = fc[i];
    if (!isWall[i]) { nutBnd[i] = nutCell[c]; return; }
    const scalar magUp = sqrt(Ux[c]*Ux[c] + Uy[c]*Uy[c] + Uz[c]*Uz[c]);
    const scalar magGradU = magUp * dc[i];
    const scalar nuw = nuFace ? nuFace[i] : nu;
    nutBnd[i] = blendedNutValue(magUp, magGradU, y[i], nuw, kappa, E, nutBnd[i]);   // shared with G0 (nut_wall_function.cuh)
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
    DeviceBuffer<scalar>& nutBnd,
    const DeviceBuffer<scalar>* nuFace)
{
    // OF turbulenceModel::nu(patchi) = mu(patchi)/rho.boundaryField()[patchi] -- a per-FACE kinematic
    // viscosity. Constant-property incompressible flow makes that a single number, which is why the scalar
    // argument exists; compressible flow does not, so nuFace overrides it when supplied.

    const int n = dbU.comp[0].n;
    if (static_cast<int>(nutBnd.size()) != n)   // first call: zero seed (warm-starts thereafter)
    {
        nutBnd.resize(n);
        cudaCheck(cudaMemsetAsync(nutBnd.data(), 0, n*sizeof(scalar), cudaStreamPerThread), "spalding init");
    }
    spaldingNutKernel<<<nBlocks(n), TPB>>>(n, dbU.comp[0].faceCell.data(), isWall.data(), y.data(),
                                           dbU.comp[0].deltaCoeffs.data(), Ux.data(), Uy.data(), Uz.data(),
                                           nutCell.data(), nu, co.kappa, co.E, nutBnd.data(),
                                           nuFace ? nuFace->data() : nullptr);
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
    DeviceBuffer<scalar>& nutBnd,
    const DeviceBuffer<scalar>* nuFace)
{
    // OF turbulenceModel::nu(patchi) = mu(patchi)/rho.boundaryField()[patchi] -- a per-FACE kinematic
    // viscosity. Constant-property incompressible flow makes that a single number, which is why the scalar
    // argument exists; compressible flow does not, so nuFace overrides it when supplied.

    const int n = dbU.comp[0].n;
    if (static_cast<int>(nutBnd.size()) != n)   // first call: zero seed (warm-starts thereafter)
    {
        nutBnd.resize(n);
        cudaCheck(cudaMemsetAsync(nutBnd.data(), 0, n*sizeof(scalar), cudaStreamPerThread), "blended init");
    }
    blendedNutKernel<<<nBlocks(n), TPB>>>(n, dbU.comp[0].faceCell.data(), isWall.data(), y.data(),
                                          dbU.comp[0].deltaCoeffs.data(), Ux.data(), Uy.data(), Uz.data(),
                                          nutCell.data(), nu, kappa, E, nutBnd.data(),
                                          nuFace ? nuFace->data() : nullptr);
    cudaCheck(cudaGetLastError(), "blendedNut");
}


} // namespace brae
