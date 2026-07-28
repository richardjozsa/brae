// Smagorinsky LES sub-grid model: the ALGEBRAIC eddy viscosity nut = Ck*delta*sqrt(k_sgs), transcribed from OF's
// Smagorinsky<BasicTurbulenceModel>::k(gradU) + correctNut(). Pure LES -> no scalar transport (no k/epsilon/omega/
// nuTilda equation); nut is a closed-form function of the local strain and cell size, recomputed each step from U.
//   D = symm(gradU);  a = Ce/delta;  b = (2/3)tr(D);  c = 2*Ck*delta*(dev(D) && D);
//   sqrt(k_sgs) = (-b + sqrt(b^2 + 4ac))/(2a);  nut = Ck*delta*sqrt(k_sgs),  delta = cbrt(V) (cubeRootVol).
// dev(D) && D = D:D - (1/3)tr(D)^2 = |dev(D)|^2 >= 0, so the discriminant b^2+4ac >= 0 and sqrt(k_sgs) >= 0.
// Incompressible (tr(D)=div(U)->0): nut = Ck*sqrt(2*Ck/Ce)*delta^2*sqrt(S:S), the classic Cs~0.168 Smagorinsky.
#include "device_smagorinsky.cuh"
#include "device_scalar_transport.cuh"  // nBlocks/TPB (shared launch geometry)
#include <cuda_runtime.h>

namespace brae {

namespace {

__global__
void smagorinskyNutKernel(
    int nC, const scalar* __restrict__ gradU, const scalar* __restrict__ V,
    SmagorinskyCoeffs co, scalar* __restrict__ nut)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    scalar t[9];
    for (int q = 0; q < 9; ++q)
        t[q] = gradU[q * nC + c];
    const scalar delta = cbrt(V[c]);                              // LES filter width (cubeRootVol = V^(1/3))
    const scalar trD = t[0] + t[4] + t[8];                        // tr(symm gradU) = div(U)
    scalar DD = 0;                                                // D:D = magSqr(symm(gradU))
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            const scalar sab = scalar(0.5) * (t[i*3+j] + t[j*3+i]);
            DD += sab * sab;
        }
    const scalar devDD = fmax(DD - trD*trD/scalar(3), scalar(0));  // dev(D) && D = |dev(D)|^2 (>= 0)
    // OF Smagorinsky<>::k(): sqrt(k_sgs) = (-b + sqrt(b^2 + 4ac))/(2a)
    const scalar a = co.Ce / fmax(delta, scalar(1e-300));
    const scalar b = (scalar(2)/scalar(3)) * trD;
    const scalar cc = scalar(2) * co.Ck * delta * devDD;
    const scalar sqrtk = (-b + sqrt(fmax(b*b + scalar(4)*a*cc, scalar(0)))) / (scalar(2)*a);
    nut[c] = co.Ck * delta * fmax(sqrtk, scalar(0));              // nut = Ck*delta*sqrt(k_sgs)
}

} // namespace

void deviceSmagorinskyNut(int nC, const DeviceBuffer<scalar>& gradU, const DeviceBuffer<scalar>& V,
                          const SmagorinskyCoeffs& co, DeviceBuffer<scalar>& nut)
{
    nut.resize(nC);
    if (nC == 0) return;
    smagorinskyNutKernel<<<nBlocks(nC), TPB>>>(nC, gradU.data(), V.data(), co, nut.data());
    cudaCheck(cudaGetLastError(), "deviceSmagorinskyNut");
}

} // namespace brae
