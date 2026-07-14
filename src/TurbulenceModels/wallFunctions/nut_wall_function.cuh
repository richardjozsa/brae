#pragma once
// nutkWallFunction (OpenFOAM v2412, STEPWISE blender), turbulent viscosity nut at a wall.
//   yPlus = Cmu^0.25 * y * sqrt(k_nearWall) / nu       (y = near-wall distance)
//   nut   = (yPlus > yPlusLam) ? nu*yPlus*kappa/log(max(E*yPlus, 1+1e-4)) - nu : 0
// y is the near-wall distance (OF turbModel.y() = wallDist field), passed in by the caller.
#include "cf_types.cuh"
#include "fv_patch.cuh"
#include <vector>

namespace brae {

// Single source of truth for the OF nutkWallFunction value (the log-law wall viscosity, 0 in the viscous sublayer).
// Shared by the host nutkWallFunction below AND the device wall kernels (kEpsilon wallFnKernel/boundaryNutKernel,
// kOmegaSST wallOmegaG0Kernel) so the wall-nut physics has ONE definition, not four copies. BRAE_HD (__host__
// __device__) so it compiles identically on both; log/fmax resolve to the device intrinsics on the GPU and libm on
// the host (same result for double). yPlus = Cmu^0.25 * y * sqrt(k_nearWall) / nu (see yPlusWall).
BRAE_HD inline scalar nutkWallFunctionValue(scalar yPlus, scalar nu, scalar yplLam, scalar kappa, scalar E)
{
    return (yPlus > yplLam) ? (nu * yPlus * kappa / log(fmax(E * yPlus, scalar(1.0 + 1e-4))) - nu) : scalar(0.0);
}
BRAE_HD inline scalar yPlusWall(scalar Cmu25, scalar y, scalar kNearWall, scalar nu) { return Cmu25 * y * sqrt(kNearWall) / nu; }

// Shared near-wall production for the kEpsilon and kOmegaSST wall functions: compute nutw + |grad(U)|_wall and scatter
// the turbulence production into G0[c]. The G0 term is IDENTICAL for both models (this enforces that by construction,
// the komega wall kernel used to carry a hand-copy); each caller then adds only its distinct eps0 / omega0 term.
// __device__ (uses atomicAdd), called from the wall kernels only.
__device__ inline void wallProductionG0(
    int c,
    int wf,
    scalar y,
    scalar dc,
    scalar kc,
    scalar invNwC,
    const scalar* wux,
    const scalar* wuy,
    const scalar* wuz,
    const scalar* Ux,
    const scalar* Uy,
    const scalar* Uz,
    scalar nu,
    scalar yplLam,
    scalar Cmu25,
    scalar kappa,
    scalar E,
    scalar* G0)
{
    const scalar nutw = nutkWallFunctionValue(yPlusWall(Cmu25, y, kc, nu), nu, yplLam, kappa, E);
    const scalar dux = (wux[wf]-Ux[c])*dc, duy = (wuy[wf]-Uy[c])*dc, duz = (wuz[wf]-Uz[c])*dc;
    const scalar magG = sqrt(dux*dux + duy*duy + duz*duz);
    atomicAdd(&G0[c], invNwC * (nutw + nu) * magG * Cmu25 * sqrt(kc) / (kappa * y));
}

std::vector<scalar> nutkWallFunction(
    const FvPatch& wall,
    const std::vector<scalar>& y,
    const std::vector<scalar>& kInternal,
    scalar nu,
    scalar Cmu = 0.09,
    scalar kappa = 0.41,
    scalar E = 9.8);

} // namespace brae
