#pragma once
// cf GPU offload (#3): k-epsilon turbulence physics on device. The production term GbyNu = gradU &&
// devTwoSymm(gradU) and the eddy viscosity nut = Cmu k^2/eps. The transport equations (eps/k) reuse the
// existing device assembly (div + laplacian, G4) + reaction terms + the scalar BiCGStab (G6); the wall
// functions + setValues constraint are the remaining wiring.
#include "cf_types.cuh"
#include "kepsilon_coeffs.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "near_wall_dist.cuh"
#include "komega_sst_coeffs.cuh"
#include "spalart_coeffs.cuh"
#include "device_pcg.cuh"     // DeviceSolverPerf (turbulence solve report)
#include <vector>
#include <string>

namespace brae {

struct DeviceAMI;     // cyclicAMI interface (scalar-transport coupling, optional)
struct DeviceCyclic;  // cyclic interface     (scalar-transport coupling, optional)

// OF-style turbulence residual reporting. Every turbulence scalar solve (k / epsilon / omega / nuTilda / ReThetat /
// gammaInt) appends its field name + solve perf here, in solve order, so the SIMPLE driver can print an
// "smoothSolver:  Solving for <field>, Initial residual = ..." block exactly like OpenFOAM. Cleared once per SIMPLE step.
struct ScalarSolveEntry
{
    std::string field;
    DeviceSolverPerf perf;
};
void clearTurbulenceReport();
const std::vector<ScalarSolveEntry>& turbulenceReport();

// k-epsilon model coefficients (shared CPU/device definition).
// GbyNu = gradU && devTwoSymm(gradU), devTwoSymm(g) = g + g^T - (2/3)tr(g) I.  G = nut*GbyNu.
void deviceGbyNu(const DeviceMesh& dm, const DeviceVectorBoundary& dbU,
                 const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
                 DeviceBuffer<scalar>& gByNu, DeviceAMI* ami = nullptr, DeviceCyclic* cyc = nullptr);

// gradU tensor (9*nC, OF convention column i = gaussGrad(U_i)) + GbyNu from a prebuilt gradU. Shared by k-eps
// (GbyNu) and kOmegaSST (which also needs gradU for S2 = 2 magSqr(symm(gradU))).
void deviceGradU(const DeviceMesh& dm, const DeviceVectorBoundary& dbU,
                 const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
                 DeviceBuffer<scalar>& gradU, DeviceAMI* ami = nullptr, DeviceCyclic* cyc = nullptr);
void deviceGByNuFromGradU(const DeviceBuffer<scalar>& gradU, int nC, DeviceBuffer<scalar>& gByNu);
// OF grad(U) cellLimited Gauss linear <k> applied to a grad(U) TENSOR (per U-component minmod limiter). For the
// turbulence strain S2 + production, which OF computes from fvc::grad(U) = the (cellLimited) grad(U) scheme.
void deviceCellLimitGradU(const DeviceMesh& dm, const DeviceVectorBoundary& dbU,
                          const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
                          DeviceBuffer<scalar>& gradU, scalar kc);

// nut = Cmu k^2 / eps.
void deviceNut(const DeviceBuffer<scalar>& k, const DeviceBuffer<scalar>& eps, DeviceBuffer<scalar>& nut,
               const KEpsilonCoeffs& co = {});

// OF Foam::bound(vsf, lowerBound): where a cell falls below the floor, replace it with the bounded
// neighbour-average (fvc::average(max(vsf,floor))*pos0(-vsf)), NOT a hard clamp; floor elsewhere.
void deviceBoundField(const DeviceMesh& dm, DeviceBuffer<scalar>& x, scalar floor);

// Static wall geometry for the wall functions (nearWallDist y, wall-face cell/deltaCoeffs/velocity, 1/nWallFaces).
struct DeviceWallData
{
    int nWF = 0;
    DeviceBuffer<label>  wfCell, isWallCell;
    DeviceBuffer<scalar> wfY, wfDc, wfUwx, wfUwy, wfUwz, invNw;
};
inline DeviceWallData buildDeviceWallData(
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& fvp,
    const GeometricField<vector>& U)
{
    const std::vector<std::vector<scalar>> yW = nearWallDist(m, g, fvp);
    std::vector<label> wfCell;
    std::vector<scalar> wfY, wfDc, wux, wuy, wuz;
    std::vector<label> nw(m.nCells(), 0);
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        if (fvp[pi].type == "wall")
        {
            const std::vector<vector>& uv = U.boundary[pi]->value();
            for (label i = 0; i < fvp[pi].size; ++i)
            {
                const label c = fvp[pi].faceCells[i];
                wfCell.push_back(c);
                wfY.push_back(yW[pi][i]);
                wfDc.push_back(fvp[pi].deltaCoeffs[i]);
                wux.push_back(uv[i].x);
                wuy.push_back(uv[i].y);
                wuz.push_back(uv[i].z);
                ++nw[c];
            }
        }
    std::vector<scalar> invNw(m.nCells(), 0.0);
    std::vector<label> isW(m.nCells(), 0);
    for (label c = 0; c < m.nCells(); ++c)
        if (nw[c] > 0)
        {
            invNw[c] = 1.0 / nw[c];
            isW[c] = 1;
        }
    DeviceWallData w;
    w.nWF = static_cast<int>(wfCell.size());
    w.wfCell.copyFrom(wfCell);
    w.wfY.copyFrom(wfY);
    w.wfDc.copyFrom(wfDc);
    w.wfUwx.copyFrom(wux);
    w.wfUwy.copyFrom(wuy);
    w.wfUwz.copyFrom(wuz);
    w.invNw.copyFrom(invNw);
    w.isWallCell.copyFrom(isW);
    return w;
}

// epsilonWallFunction near-wall values: eps0 = (1/nWall) Cmu^.75 k^1.5/(kappa y); G0 = (1/nWall)(nutw+nu)*
// |snGrad U|*Cmu^.25 sqrt(k)/(kappa y); nutw = nutkWallFunction. (eps0/G0 zeroed then scattered with atomics.)
void deviceWallEpsG0(const DeviceWallData& w, const DeviceBuffer<scalar>& k, const DeviceBuffer<scalar>& Ux,
                     const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz, scalar nu,
                     DeviceBuffer<scalar>& eps0, DeviceBuffer<scalar>& G0, const KEpsilonCoeffs& co = {});

// add the eps / k reaction (Sp/Su) + SuSp(divU) terms to a matrix's diag/source (in place).
void deviceEpsReaction(const DeviceMesh& dm, const DeviceBuffer<scalar>& eps, const DeviceBuffer<scalar>& k,
                       const DeviceBuffer<scalar>& gByNu, const DeviceBuffer<scalar>& divU,
                       DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& source, const KEpsilonCoeffs& co = {});
void deviceKReaction(const DeviceMesh& dm, const DeviceBuffer<scalar>& k, const DeviceBuffer<scalar>& eps,
                     const DeviceBuffer<scalar>& G, const DeviceBuffer<scalar>& divU,
                     DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& source);
// realizableKE (OF RAS/realizableKE): variable Cmu (rCmu) + strain magnitude magS from the gradU tensor; nut =
// rCmu*k^2/eps; eps reaction = strain production C1*magS*eps - destruction C2*eps^2/(k+sqrt(nu*eps)). Cell-local
// (no halo), so the distributed kEps correct reuses them on its already-halo-consistent gradU tensor.
void deviceRealizableStrain(const DeviceBuffer<scalar>& gradU, const DeviceBuffer<scalar>& k,
                            const DeviceBuffer<scalar>& eps, scalar A0, int nC,
                            DeviceBuffer<scalar>& rCmu, DeviceBuffer<scalar>& magS);
void deviceRealizableNut(const DeviceBuffer<scalar>& rCmu, const DeviceBuffer<scalar>& k,
                         const DeviceBuffer<scalar>& eps, DeviceBuffer<scalar>& nut);
void deviceEpsReactionRealizable(const DeviceMesh& dm, const DeviceBuffer<scalar>& eps, const DeviceBuffer<scalar>& k,
                                 const DeviceBuffer<scalar>& magS, scalar nu, scalar C2,
                                 DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& source);

// Boundary nut per face: wall faces -> nutkWallFunction(k[cell], y, nu); other faces -> nut[cell]. Gives the
// true wall eddy viscosity for the momentum boundary nuEff (vs the cell-value approximation).
void deviceBoundaryNut(const DeviceBoundary& db, const DeviceBuffer<label>& isWall, const DeviceBuffer<scalar>& y,
                       const DeviceBuffer<scalar>& k, const DeviceBuffer<scalar>& nut, scalar nu, DeviceBuffer<scalar>& nutBnd,
                       const KEpsilonCoeffs& co = {});

// The full device kEpsilon::correct(): production -> wall functions/override -> eps eqn (with the wall
// setValues constraint) -> k eqn -> correctNut. Updates k/eps/nut in place. Mirrors the CPU correct().
// div(phi,k)/div(phi,epsilon) scheme: upwind by default; limitedLinear when limitedK/limitedEps set (twoByk* =
// 2/max(k_,SMALL) from the scheme coefficient). limitedLinear reduces to upwind at limiter=0 (bit-identical off).
void deviceKEpsilonCorrect(const DeviceMesh& dm, const DeviceWallData& wall, const DeviceBoundary& dbEps,
                           const DeviceBoundary& dbK, const DeviceVectorBoundary& dbU,
                           const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
                           DeviceBuffer<scalar>& k, DeviceBuffer<scalar>& eps, DeviceBuffer<scalar>& nut,
                           const DeviceBuffer<scalar>& phiInt, const DeviceBuffer<scalar>& phiBnd,
                           scalar nu, scalar relaxEps, scalar relaxK, scalar tol, bool bounded = false,
                           bool limitedK = false, bool limitedEps = false, scalar twoBykK = 2.0, scalar twoBykEps = 2.0,
                           const KEpsilonCoeffs& co = {}, scalar relTolKE = 0.0, int keCheckEvery = 1,
                           bool linearUpwindK = false, bool linearUpwindEps = false, bool nonOrth = false,
                           bool gsK = false, bool gsEps = false, DeviceAMI* ami = nullptr, DeviceCyclic* cyc = nullptr);

// Closed device kOmegaSST::correct(): production (raw GbyNu0 + omega-wall G0 override) -> F1/F2/CDkOmega/S2 ->
// omega eqn (loose solve, omega-wall setValues) -> bound -> k eqn (loose solve) -> bound -> correctNut (Bradshaw
// limiter). y = precomputed cell wall distance (cellWallDist). All blocks reuse the C1-6 validated kernels +
// the shared deviceSolveScalarTransport scaffold. Updates k/omega/nut in place. Mirrors kOmegaSSTBase::correct().
void deviceKOmegaSSTCorrect(const DeviceMesh& dm, const DeviceWallData& wall, const DeviceBoundary& dbOmega,
                            const DeviceBoundary& dbK, const DeviceVectorBoundary& dbU,
                            const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
                            DeviceBuffer<scalar>& k, DeviceBuffer<scalar>& omega, DeviceBuffer<scalar>& nut,
                            const DeviceBuffer<scalar>& y, const DeviceBuffer<scalar>& phiInt, const DeviceBuffer<scalar>& phiBnd,
                            scalar nu, scalar relaxOmega, scalar relaxK, scalar tol, bool bounded = false,
                            bool limitedK = false, bool limitedOmega = false, scalar twoBykK = 2.0, scalar twoBykOmega = 2.0,
                            const KOmegaSSTCoeffs& co = {}, scalar relTolKE = 0.0, int keCheckEvery = 1,
                            bool linearUpwindK = false, bool linearUpwindOmega = false, bool nonOrth = false, scalar gradULimitK = 0.0,
                            bool gsK = false, bool gsEps = false, DeviceAMI* ami = nullptr, DeviceCyclic* cyc = nullptr,
                            const scalar* gammaIntEff = nullptr);   // kOmegaSSTLM transition: scales k Pk/epsilonByk

// kOmegaSSTLM (Langtry-Menter gamma-ReThetat transition) extra step: after the SST k/omega solve, transport ReThetat
// and gammaInt and update gammaIntEff (fed back into the SST k equation next iteration). See PORTING_KOMEGASSTLM.md.
void deviceKOmegaSSTLMCorrect(const DeviceMesh& dm, const DeviceVectorBoundary& dbU, const DeviceBoundary& dbReThetat,
                              const DeviceBoundary& dbGammaInt, const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy,
                              const DeviceBuffer<scalar>& Uz, const DeviceBuffer<scalar>& k, const DeviceBuffer<scalar>& omega,
                              const DeviceBuffer<scalar>& nut, const DeviceBuffer<scalar>& y,
                              DeviceBuffer<scalar>& ReThetat, DeviceBuffer<scalar>& gammaInt, DeviceBuffer<scalar>& gammaIntEff,
                              const DeviceBuffer<scalar>& phiInt, const DeviceBuffer<scalar>& phiBnd, scalar nu,
                              scalar relax, scalar tol, scalar relTolKE, int keCheckEvery, bool bounded, bool nonOrth,
                              bool gsEps = false, DeviceAMI* ami = nullptr, DeviceCyclic* cyc = nullptr);

// Spalart-Allmaras (one-equation): solve the nuTilda transport (div - laplacian(DnuTildaEff) + Sp(destruction) ==
// production + Cb2 grad^2) via the shared scaffold, then nut = nuTilda*fv1. Mirrors SpalartAllmarasBase::correct()
// (incompressible, steady, ft2 off). dbNuTilda = the nuTilda boundary (freestream + fixedValue-0 wall). y = cell
// wall distance. nuTilda/nut updated in place.
void deviceSpalartAllmarasCorrect(const DeviceMesh& dm, const DeviceVectorBoundary& dbU, const DeviceBoundary& dbNuTilda,
                                  const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
                                  DeviceBuffer<scalar>& nuTilda, DeviceBuffer<scalar>& nut, const DeviceBuffer<scalar>& y,
                                  const DeviceBuffer<scalar>& phiInt, const DeviceBuffer<scalar>& phiBnd,
                                  scalar nu, scalar relax, scalar tol, bool bounded = false, bool limited = false,
                                  scalar twoByk = 2.0, const SpalartAllmarasCoeffs& co = {}, scalar relTol = 0.0, int checkEvery = 1,
                                  bool linearUpwind = false, bool nonOrth = false,
                                  bool gsK = false, DeviceAMI* ami = nullptr, DeviceCyclic* cyc = nullptr);

// Standalone SA correctNut (nut = nuTilda*fv1(nuTilda)) for the solver startup validate().
void deviceNutSA(const DeviceBuffer<scalar>& nuTilda, scalar nu, scalar Cv1, DeviceBuffer<scalar>& nut);

// nutUSpaldingWallFunction boundary nut: wall faces -> Newton uTau from Spalding; other faces -> adjacent cell nut.
// y / isWall are per boundary face (nearWallDist y, wall mask); nutBnd is in/out (warm-started seed). nu+nutBnd at
// walls gives the SA wall shear (deep wall-function meshes, y+ >> 1).
void deviceBoundaryNutSpalding(const DeviceVectorBoundary& dbU, const DeviceBuffer<label>& isWall,
                               const DeviceBuffer<scalar>& y, const DeviceBuffer<scalar>& Ux,
                               const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
                               const DeviceBuffer<scalar>& nutCell, scalar nu, const SpalartAllmarasCoeffs& co,
                               DeviceBuffer<scalar>& nutBnd);

} // namespace brae
