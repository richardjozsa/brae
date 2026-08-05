// device_energy.cu -- alphaEff assembly and the he solve.

#include "device_energy.cuh"
#include "device_scalar_transport.cuh"
#include "thermo_model.cuh"
#include <stdexcept>

namespace brae {

// Boundary he from boundary T. Linear for hConst, so bcType is untouched and only the value moves.
__global__
void heBndFromTK(
    int n,
    ThermoCoeffs c,
    const scalar* __restrict__ refT,
    scalar* __restrict__ refHe)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    refHe[i] = hConstTToHe(refT[i], c);
}

void deviceEnergyBoundaryFromT(
    const DeviceBoundary& dbT,
    const ThermoCoeffs& c,
    DeviceBoundary& dbHe)
{
    if (dbT.n == 0) return;
    if (dbHe.n != dbT.n)
    {
        throw std::runtime_error(
            "brae: deviceEnergyBoundaryFromT needs the he boundary to share the T boundary's structure.");
    }
    dbHe.refValue.resize(dbT.n);
    heBndFromTK<<<nBlocks(dbT.n), TPB>>>(
        dbT.n,
        c,
        dbT.refValue.data(),
        dbHe.refValue.data());
    cudaCheck(cudaGetLastError(), "heBndFromT");
}

// alphaEff = alpha + alphat. Separate kernel rather than folded into thermoUpdateK because alphat is
// owned by the turbulence model, which runs after the thermo update, not before it.
__global__
void alphaEffK(
    int n,
    scalar CpByCpv,
    const scalar* __restrict__ alpha,
    const scalar* __restrict__ alphat,
    scalar* __restrict__ alphaEff)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // OF heThermo::alphaEff = CpByCpv*(alpha + alphat). CpByCpv = 1 for sensibleEnthalpy, gamma = Cp/Cv
    // for sensibleInternalEnergy -- so an "e" case diffuses ~1.4x faster than an "h" case with the same
    // alpha, and dropping the factor still converges.
    alphaEff[i] = CpByCpv * (alpha[i] + alphat[i]);
}

void deviceAlphaEff(
    const DeviceThermo& th,
    const ThermoCoeffs& c,
    DeviceBuffer<scalar>& alphaEff)
{
    if (th.n == 0) return;
    alphaEff.resize(th.n);
    alphaEffK<<<nBlocks(th.n), TPB>>>(
        th.n,
        thermoCpByCpv(c),
        th.alpha.data(),
        th.alphat.data(),
        alphaEff.data());
    cudaCheck(cudaGetLastError(), "alphaEff");
}

// The quantity OF's EEqn convects alongside he:
//
//   sensibleEnthalpy       ->  K   = 0.5|U|^2
//   sensibleInternalEnergy ->  Ekp = 0.5|U|^2 + p/rho
//
// (OF EEqn.H: he.name() == "e" ? fvc::div(phi, Ekp) : fvc::div(phi, K).) The p/rho term is the flow work
// that enthalpy already carries internally and internal energy does not, so leaving it out of an "e" case
// loses the pressure work entirely -- and still converges.
__global__
void kineticEnergyK(
    int n,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    const scalar* __restrict__ p,     // null -> K (sensibleEnthalpy); else Ekp adds p/rho
    const scalar* __restrict__ rho,
    scalar* __restrict__ K)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    const scalar k = 0.5 * (Ux[c]*Ux[c] + Uy[c]*Uy[c] + Uz[c]*Uz[c]);
    K[c] = p ? k + p[c] / rho[c] : k;
}

// Face value of K/Ekp times the mass flux: ffc_f = phi_f * K_f.
//
// The face value follows the case's div(phi,K|Ekp) scheme, exactly as the implicit div(phi,he) does.
// Upwind is W = pos0(phi); limitedLinear is OF's NVD/TVD blend W = limiter*cdw + (1-limiter)*pos0 with
// limiter = clamp(twoByk*r, 0, 1) -- the SAME formula divLimitedFaceKernel uses for the implicit term, so
// the explicit and implicit halves of the energy equation cannot end up on different schemes.
// Hardcoding upwind here is worth ~6.6e-4 in T on a limitedLinear case whose implicit term is exact.
__global__
void kineticFaceFluxK(
    int nF,
    const label* __restrict__ owner,
    const label* __restrict__ neighbour,
    const scalar* __restrict__ phiInt,
    const scalar* __restrict__ K,
    const scalar* __restrict__ cdw,     // null -> upwind; else limitedLinear with the gradient below
    const scalar* __restrict__ gx,
    const scalar* __restrict__ gy,
    const scalar* __restrict__ gz,
    const scalar* __restrict__ dOwnX,
    const scalar* __restrict__ dOwnY,
    const scalar* __restrict__ dOwnZ,
    const scalar* __restrict__ dNeiX,
    const scalar* __restrict__ dNeiY,
    const scalar* __restrict__ dNeiZ,
    scalar twoByk,
    scalar* __restrict__ ffc)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nF) return;
    const int P = owner[f], N = neighbour[f];
    const scalar ph = phiInt[f];
    const scalar pos0 = (ph >= scalar(0)) ? scalar(1) : scalar(0);
    scalar W = pos0;
    if (cdw)
    {
        const scalar dx = dOwnX[f] - dNeiX[f], dy = dOwnY[f] - dNeiY[f], dz = dOwnZ[f] - dNeiZ[f];
        const int U = (ph > scalar(0)) ? P : N;
        const scalar gradcf = dx*gx[U] + dy*gy[U] + dz*gz[U];
        const scalar gradf  = K[N] - K[P];
        scalar r;
        if (fabs(gradcf) >= scalar(1000) * fabs(gradf))
            r = scalar(2)*scalar(1000)*((gradcf >= 0) ? scalar(1) : scalar(-1))*((gradf >= 0) ? scalar(1) : scalar(-1)) - scalar(1);
        else
            r = scalar(2)*(gradcf/gradf) - scalar(1);
        scalar limiter = twoByk * r;
        limiter = (limiter < scalar(0)) ? scalar(0) : (limiter > scalar(1) ? scalar(1) : limiter);
        W = limiter * cdw[f] + (scalar(1) - limiter) * pos0;
    }
    ffc[f] = ph * (W * K[P] + (scalar(1) - W) * K[N]);
}

// Boundary half: += phi_b*K_b, and the "bounded" correction -K_c*(sum_f phi_f) that OF's
// boundedConvectionScheme subtracts. Both scattered with atomics onto the adjacent cell.
__global__
void kineticBoundaryK(
    int nB,
    const label* __restrict__ faceCell,
    const scalar* __restrict__ phiBnd,
    const scalar* __restrict__ Kb,
    scalar* __restrict__ src,
    scalar* __restrict__ sumPhi)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nB) return;
    const int c = faceCell[i];
    atomicAdd(&src[c], phiBnd[i] * Kb[i]);
    atomicAdd(&sumPhi[c], phiBnd[i]);
}

__global__
void kineticBoundedK(
    int n,
    const scalar* __restrict__ K,
    const scalar* __restrict__ sumPhi,
    scalar* __restrict__ src)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    src[c] -= K[c] * sumPhi[c];   // OF: boundedConvectionScheme subtracts fvc::surfaceIntegrate(phi)*vf
}

void deviceEnergyKineticSource(
    const DeviceMesh& dm,
    const DeviceVectorBoundary& dbU,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    const DeviceBuffer<scalar>& phiInt,
    const DeviceBuffer<scalar>& phiBnd,
    DeviceBuffer<scalar>& src,
    const DeviceBuffer<scalar>* p,        // sensibleInternalEnergy: cell p and rho -> Ekp = K + p/rho
    const DeviceBuffer<scalar>* rho,
    const DeviceBuffer<scalar>* pBndIn,   // and the same at boundary faces
    const DeviceBuffer<scalar>* rhoBndIn,
    const DeviceBoundary* dbHe,           // for grad(K) when the scheme is limitedLinear
    bool limited,
    scalar twoByk)
{
    const bool ekp = (p && rho && p->size() && rho->size());
    const int nC = dm.nCells;
    const int nF = dm.nInternalFaces;
    DeviceBuffer<scalar> K;
    K.resize(nC);
    kineticEnergyK<<<nBlocks(nC), TPB>>>(nC, Ux.data(), Uy.data(), Uz.data(),
                                         ekp ? p->data() : nullptr,
                                         ekp ? rho->data() : nullptr,
                                         K.data());
    cudaCheck(cudaGetLastError(), "kineticEnergy");

    // grad(K) for the limiter. K is built from U and p, so it has no BC of its own -- the adjacent-cell
    // (zeroGradient) boundary value is the consistent choice and is what OF's fvc::grad of a constructed
    // volScalarField("Ekp", ...) sees, since that field is created with calculated/extrapolated patches.
    DeviceBuffer<scalar> gx, gy, gz;
    if (limited)
    {
        DeviceBuffer<scalar> Kb2;
        if (dbHe) deviceBCValue(*dbHe, K, Kb2);
        deviceGaussGrad(dm, K, Kb2, gx, gy, gz);
    }
    DeviceBuffer<scalar> ffc;
    ffc.resize(nF);
    kineticFaceFluxK<<<nBlocks(nF), TPB>>>(nF, dm.owner.data(), dm.nei.data(), phiInt.data(), K.data(),
                                           limited ? dm.w.data() : nullptr,
                                           limited ? gx.data() : nullptr,
                                           limited ? gy.data() : nullptr,
                                           limited ? gz.data() : nullptr,
                                           dm.dOwnX.data(), dm.dOwnY.data(), dm.dOwnZ.data(),
                                           dm.dNeiX.data(), dm.dNeiY.data(), dm.dNeiZ.data(),
                                           twoByk, ffc.data());
    cudaCheck(cudaGetLastError(), "kineticFaceFlux");
    // deviceFaceDivSource returns MINUS V*div (it exists for the laplacian non-orth correction, which
    // wants that sign). Negate so src is +V*div(phi,K), matching the boundary half added below and the
    // sign OF's fvc::div has. Getting this wrong is invisible in an "h" case, where K is ~0.003% of he,
    // and dominant in an "e" case, where Ekp carries p/rho ~ 86 kJ/kg against e ~ 215 kJ/kg.
    deviceFaceDivSource(dm, ffc, src);
    deviceScale(src, -1.0);

    // K at boundary faces, from the boundary velocity (noSlip walls give K_b = 0, which is the point:
    // the flow gives up its kinetic energy there and OF puts that into the enthalpy equation).
    const int nB = dbU.comp[0].n;
    if (nB > 0)
    {
        DeviceBuffer<scalar> ubx, uby, ubz;
        deviceBCValue(dbU.comp[0], Ux, ubx);
        deviceBCValue(dbU.comp[1], Uy, uby);
        deviceBCValue(dbU.comp[2], Uz, ubz);
        DeviceBuffer<scalar> Kb;
        Kb.resize(nB);
        const bool ekpB = ekp && pBndIn && rhoBndIn && pBndIn->size() == static_cast<std::size_t>(nB)
                       && rhoBndIn->size() == static_cast<std::size_t>(nB);
        kineticEnergyK<<<nBlocks(nB), TPB>>>(nB, ubx.data(), uby.data(), ubz.data(),
                                             ekpB ? pBndIn->data() : nullptr,
                                             ekpB ? rhoBndIn->data() : nullptr,
                                             Kb.data());
        cudaCheck(cudaGetLastError(), "kineticEnergyBnd");

        DeviceBuffer<scalar> sumPhi;
        sumPhi.resize(nC);
        cudaCheck(cudaMemsetAsync(sumPhi.data(), 0, nC*sizeof(scalar), cudaStreamPerThread), "sumPhi zero");
        kineticBoundaryK<<<nBlocks(nB), TPB>>>(nB, dbU.comp[0].faceCell.data(), phiBnd.data(), Kb.data(),
                                               src.data(), sumPhi.data());
        cudaCheck(cudaGetLastError(), "kineticBoundary");
        // internal faces contribute to sum_f phi_f too -- same gather, same negation as above so the
        // internal and boundary halves of sumPhi carry the SAME sign.
        DeviceBuffer<scalar> divPhiInt;
        deviceFaceDivSource(dm, phiInt, divPhiInt);
        deviceAxpy(-1.0, divPhiInt, sumPhi);
        kineticBoundedK<<<nBlocks(nC), TPB>>>(nC, K.data(), sumPhi.data(), src.data());
        cudaCheck(cudaGetLastError(), "kineticBounded");
    }
}

void deviceSolveEnergy(
    const DeviceMesh& dm,
    const DeviceBoundary& dbHe,
    DeviceThermo& th,
    const ThermoCoeffs& c,
    const DeviceBuffer<scalar>& phiInt,
    const DeviceBuffer<scalar>& phiBnd,
    const DeviceBuffer<scalar>& divU,
    bool limited,
    bool linearUpwind,
    bool nonOrth,
    scalar twoByk,
    scalar relax,
    scalar tol,
    scalar relTol,
    int checkEvery,
    bool useGS,
    DeviceAMI* ami,
    DeviceCyclic* cyc,
    const DeviceBuffer<scalar>* kineticSrc,
    const DeviceBuffer<scalar>* alphaEffBnd)
{
    if (th.n == 0) return;

    DeviceBuffer<scalar> alphaEff;
    deviceAlphaEff(th, c, alphaEff);

    // bounded = false: he is not a positive-definite turbulence quantity, so the -Sp(div(phi),he)
    // correction the k/epsilon models use does not belong here.
    //
    // The only source is OF's kinetic-energy transport: EEqn carries + fvc::div(phi, K) on the LHS with
    // K = 0.5|U|^2, so it enters the RHS with a minus. Negligible in a slow laminar duct, which is why
    // phase 1 left it out; at 50 m/s against a noSlip wall it is not, and it shifts T by percent.
    deviceSolveScalarTransport(
        dm,
        dbHe,
        th.he,
        "he",
        alphaEff,
        phiInt,
        phiBnd,
        divU,
        false,
        limited,
        linearUpwind,
        nonOrth,
        twoByk,
        relax,
        tol,
        relTol,
        checkEvery,
        useGS,
        [&](DeviceBuffer<scalar>&, DeviceBuffer<scalar>& source)
        {
            if (kineticSrc && kineticSrc->size()) deviceAxpy(-1.0, *kineticSrc, source);
        },
        nullptr,
        nullptr,
        ami,
        cyc,
        ScalarDdt{},
        alphaEffBnd);
}

} // namespace brae
