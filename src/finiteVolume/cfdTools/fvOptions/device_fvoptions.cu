// cf device DarcyForchheimer porosity. See device_fvoptions.cuh. Mirrors OF porosityModels::DarcyForchheimer::apply
// (incompressible: mu=nu, rho=1), implicit isotropic resistance into the diagonal + explicit anisotropic remainder.
#include "device_fvoptions.cuh"
#include <cuda_runtime.h>

namespace brae {
namespace {
constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }


__device__ __forceinline__
void cdComponents(
    scalar nu,
    scalar magU,
    scalar dx,
    scalar dy,
    scalar dz,
    scalar fx,
    scalar fy,
    scalar fz,
    scalar& cx,
    scalar& cy,
    scalar& cz)
{
    cx = nu*dx + magU*0.5*fx;
    cy = nu*dy + magU*0.5*fy;
    cz = nu*dz + magU*0.5*fz;
}


__global__
void porDiagKernel(
    int n,
    const label* __restrict__ cells,
    scalar nu,
    scalar dx,
    scalar dy,
    scalar dz,
    scalar fx,
    scalar fy,
    scalar fz,
    const scalar* __restrict__ V,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    scalar* __restrict__ diag)
{
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;
    const int c = cells[i];
    const scalar magU = sqrt(Ux[c]*Ux[c] + Uy[c]*Uy[c] + Uz[c]*Uz[c]);
    scalar cx, cy, cz;
    cdComponents(nu, magU, dx,dy,dz, fx,fy,fz, cx,cy,cz);
    diag[c] += V[c]*(cx + cy + cz);                                    // += V*isoCd  (cellZone cells unique -> no atomic)
}


__global__
void porSrcKernel(
    int n,
    const label* __restrict__ cells,
    int comp,
    scalar nu,
    scalar dx,
    scalar dy,
    scalar dz,
    scalar fx,
    scalar fy,
    scalar fz,
    const scalar* __restrict__ V,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    scalar* __restrict__ src)
{
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;
    const int c = cells[i];
    const scalar magU = sqrt(Ux[c]*Ux[c] + Uy[c]*Uy[c] + Uz[c]*Uz[c]);
    scalar cx, cy, cz;
    cdComponents(nu, magU, dx,dy,dz, fx,fy,fz, cx,cy,cz);
    const scalar iso = cx + cy + cz;
    const scalar ccomp = (comp==0)?cx:(comp==1)?cy:cz;
    const scalar Uc    = ((comp==0)?Ux:(comp==1)?Uy:Uz)[c];
    src[c] += V[c]*(iso - ccomp)*Uc;                                   // -= V*((Cd-I*iso).U)[comp]
}
} // namespace


void deviceFvoPorosityDiag(
    const DevicePorosity& por,
    scalar nu,
    const DeviceBuffer<scalar>& V,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    DeviceBuffer<scalar>& diag)
{
    const int n = static_cast<int>(por.cells.size());
    if (!por.active || !n) return;
    porDiagKernel<<<nBlocks(n), TPB>>>(n, por.cells.data(), nu, por.d.x,por.d.y,por.d.z, por.f.x,por.f.y,por.f.z,
                                       V.data(), Ux.data(), Uy.data(), Uz.data(), diag.data());
    cudaCheck(cudaGetLastError(), "porDiag");
}


void deviceFvoPorositySource(
    const DevicePorosity& por,
    int comp,
    scalar nu,
    const DeviceBuffer<scalar>& V,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    DeviceBuffer<scalar>& src)
{
    const int n = static_cast<int>(por.cells.size());
    if (!por.active || !n) return;
    porSrcKernel<<<nBlocks(n), TPB>>>(n, por.cells.data(), comp, nu, por.d.x,por.d.y,por.d.z, por.f.x,por.f.y,por.f.z,
                                      V.data(), Ux.data(), Uy.data(), Uz.data(), src.data());
    cudaCheck(cudaGetLastError(), "porSrc");
}


namespace {
__global__
void limitUKernel(
    int n,
    const label* __restrict__ cells,
    scalar maxSqrU,
    scalar* __restrict__ Ux,
    scalar* __restrict__ Uy,
    scalar* __restrict__ Uz)
{
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;
    const int c = cells[i];
    const scalar magSqr = Ux[c]*Ux[c] + Uy[c]*Uy[c] + Uz[c]*Uz[c];
    if (magSqr > maxSqrU)
    {
        const scalar s = sqrt(maxSqrU/magSqr);
        Ux[c]*=s;
        Uy[c]*=s;
        Uz[c]*=s;
    }
}
} // namespace


// he clamp on a cell selection. Same shape as limitUKernel: gather through the selection list.
__global__
void limitEnergyKernel(int n, const label* __restrict__ cells, scalar heMin, scalar heMax,
                       scalar* __restrict__ he)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const label c = cells[i];
    if      (he[c] < heMin) he[c] = heMin;
    else if (he[c] > heMax) he[c] = heMax;
}

void deviceFvoLimitEnergy(
    const DeviceBuffer<label>& cells,
    scalar heMin,
    scalar heMax,
    DeviceBuffer<scalar>& he)
{
    const int n = static_cast<int>(cells.size());
    if (!n) return;
    limitEnergyKernel<<<nBlocks(n), TPB>>>(n, cells.data(), heMin, heMax, he.data());
    cudaCheck(cudaGetLastError(), "limitTemperature");
}

// Boundary half: every face whose patch does not fix a value. OF's test is fvPatchField::fixesValue(); on
// the device that is bcType == 1 (brae: 0 extrapolated, 1 fixedValue, 2 calculated), and an inletOutlet
// face has already been resolved to 0|1 for this iteration, so the same test covers it.
__global__
void limitEnergyBndKernel(int n, const label* __restrict__ bcType, scalar heMin, scalar heMax,
                          scalar* __restrict__ heBnd)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (bcType[i] == 1) return;                     // fixesValue() -> OF leaves it alone
    if      (heBnd[i] < heMin) heBnd[i] = heMin;
    else if (heBnd[i] > heMax) heBnd[i] = heMax;
}

void deviceFvoLimitEnergyBoundary(
    const DeviceBoundary& dbHe,
    scalar heMin,
    scalar heMax,
    DeviceBuffer<scalar>& heBnd)
{
    const int n = static_cast<int>(heBnd.size());
    if (!n || dbHe.n != n) return;
    limitEnergyBndKernel<<<nBlocks(n), TPB>>>(n, dbHe.bcType.data(), heMin, heMax, heBnd.data());
    cudaCheck(cudaGetLastError(), "limitTemperatureBnd");
}

void deviceFvoLimitVelocity(
    const DeviceBuffer<label>& cells,
    scalar maxU,
    DeviceBuffer<scalar>& Ux,
    DeviceBuffer<scalar>& Uy,
    DeviceBuffer<scalar>& Uz)
{
    const int n = static_cast<int>(cells.size());
    if (!n) return;
    limitUKernel<<<nBlocks(n), TPB>>>(n, cells.data(), maxU*maxU, Ux.data(), Uy.data(), Uz.data());
    cudaCheck(cudaGetLastError(), "limitVelocity");
}


// velocityDampingConstraint: diag[c] += C*V[c]^(2/3)*(|U|-UMax) where |U| > UMax.
__global__
void velDampKernel(
    int n,
    const label* __restrict__ cells,
    scalar UMax,
    scalar C,
    const scalar* __restrict__ V,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    scalar* __restrict__ diag)
{
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;
    const int c = cells[i];
    const scalar magU = sqrt(Ux[c]*Ux[c] + Uy[c]*Uy[c] + Uz[c]*Uz[c]);
    if (magU > UMax)
    {
        const scalar s = cbrt(V[c]);
        diag[c] += C*s*s*(magU - UMax);   // s*s = V^(2/3)
    }
}


void deviceFvoVelocityDamping(
    const DeviceBuffer<label>& cells,
    scalar UMax,
    scalar C,
    const DeviceBuffer<scalar>& V,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    DeviceBuffer<scalar>& diag)
{
    const int n = static_cast<int>(cells.size());
    if (!n) return;
    velDampKernel<<<nBlocks(n), TPB>>>(n, cells.data(), UMax, C, V.data(), Ux.data(), Uy.data(), Uz.data(), diag.data());
    cudaCheck(cudaGetLastError(), "velocityDampingConstraint");
}

} // namespace brae
