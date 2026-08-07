#pragma once
// device_scalar_transport.cuh -- the GENERIC scalar-transport solve scaffold, templated on the model Reaction
// functor. Assembles div(phi,f) [+ -Sp(div(phi),f) if bounded] + laplacian(D,f) with the deferred limited/
// linearUpwind/non-orth corrections, adds the model reaction (diag+source), relaxes, applies the eps near-wall
// setValues, couples cyclic/AMI interfaces, solves (BiCGStab or symGaussSeidel), and bounds. Shared by k/epsilon/
// omega/nuTilda today and by energy/species/compressible transport later. Extracted verbatim from device_kepsilon.cu.
#include "device_kepsilon.cuh"    // deviceBoundField + DeviceMesh/DeviceBoundary/DeviceWallData
#include "device_ldu.cuh"
#include "device_pcg.cuh"         // deviceJacobiBiCGStab
#include "device_simple.cuh"      // deviceFold/deviceRelaxDiag/deviceDiv*Coeffs/deviceLinearUpwindCorr/...
#include "device_blas.cuh"
#include "device_ami.cuh"
#include "device_cyclic.cuh"
#include "device_interface.cuh"   // interfaceAssembleMomentum/OffDiagSum/ZeroWallIfCoeff
#include "device_amg.cuh"         // deviceSymGaussSeidel
#include "stage_dump.cuh"      // Phase 0 stage harness
#include "device_ddt.cuh"         // ScalarDdt + deviceFvmDdtDiag/Source (transient turbulence)
#include <cuda_runtime.h>
#include <vector>

namespace brae {

// --- scaffold-local helpers (moved from device_kepsilon.cu so the template above sees them) ---
constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }
// OF-style turbulence residual report store; clearTurbulenceReport/turbulenceReport (device_kepsilon.cu) wrap this.
inline std::vector<ScalarSolveEntry>& turbStore() { static std::vector<ScalarSolveEntry> s; return s; }

namespace {
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
// shared turbulence-common kernels (effective diffusivity D + OF bound override); used by k/eps + k-omega.
static __global__
void depsKernel(int nC, const scalar* __restrict__ nut, scalar sigma, scalar nu, scalar* __restrict__ D)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < nC) D[c] = nut[c] / sigma + nu;
}


static __global__
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
} // anon (scaffold setValues kernels)


// One scalar-transport sub-step of a two-equation RAS correct(): assemble div(phi,f) - laplacian(D,f)
// [- Sp(div(phi),f) if bounded], add the model reaction (diag+source via `reaction`), relax, optionally
// apply the eps near-wall setValues constraint, fold, BiCGStab (loose relTol), bound. Reused by k-omega SST.
template <class Reaction>
void deviceSolveScalarTransport(
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
    DeviceCyclic* cyc = nullptr,
    const ScalarDdt& ddt = ScalarDdt{},   // transient fvm::ddt(f); default (steady) -> no-op
    const DeviceBuffer<scalar>* DBnd = nullptr,   // per-FACE boundary diffusivity; null -> adjacent-cell value
    // OF `grad(<field>) cellLimited Gauss linear <k>`. 0 = unlimited. The gradient feeds the limitedLinear
    // weight, the linearUpwind deferred correction and the non-orthogonal laplacian correction, so an
    // unlimited gradient on a field the case asked to limit is a different discretisation, not a detail.
    // Only grad(U) was ever honoured; grad(k)/grad(omega)/grad(e) lines were parsed by nothing.
    scalar gradLimitK = 0.0,
    // OF's bound(f, fMin) exists for POSITIVE-DEFINITE quantities: k, epsilon, omega, nuTilda. Every
    // turbulence caller here is one of those, so the default is true. The ENERGY is not: EEqn.H never
    // bounds he, and OF's sensible energy is legitimately NEGATIVE below the reference temperature
    // (air at 298 K has he = -8.59e4 J/kg). Bounding it there replaces the whole field with ~0 and
    // pins T at Cp*Tref/Cv = 417.7 K, which is exactly what NACA0012 did once he was given OF's
    // reference point. Harmless while brae used he = Cv*T > 0 -- which is why it survived this long.
    bool boundPositive = true,
    // fvOptions scalarFixedValueConstraint: OF FixedValueConstraint::constrain -> eqn.setValues(cells, value).
    // Applied BEFORE the wall block, because kEpsilon.C runs fvOptions.constrain(epsEqn) at line 266 and
    // boundaryManipulate (the epsilon wall function's own setValues) at 267 -- so on a cell claimed by both,
    // the WALL FUNCTION wins. Same three kernels; only the mask and the values differ.
    const DeviceBuffer<label>* fvoSetMask = nullptr,
    const DeviceBuffer<scalar>* fvoSetVal = nullptr)
{
    const int nC = dm.nCells;
    DeviceBuffer<scalar> Df;
    deviceInterpolate(dm, D, Df);
    DeviceBuffer<scalar> aD, aU, aL, lD, lU, lL, luCorr, lapCorr;   // luCorr/lapCorr = linearUpwind / non-orth deferred sources (empty otherwise)
    DeviceBuffer<scalar> gx, gy, gz;                               // grad(field): shared by limitedLinear / linearUpwind / non-orth
    if (limited || linearUpwind || nonOrth)
    {
        DeviceBuffer<scalar> bv;
        deviceBCValue(db, field, bv);
        deviceGaussGrad(dm, field, bv, gx, gy, gz);
        if (gradLimitK > 0.0) deviceCellLimitGrad(dm, field, bv, gx, gy, gz, gradLimitK);
    }
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
    DeviceBuffer<scalar> aIC, aBC, lIC, lBC; deviceBCDivCoeffs(db, phiBnd, aIC, aBC);
    // OF evaluates the laplacian coefficient with the PATCH diffusivity (gamma.boundaryField()), not the
    // adjacent cell's. They coincide for a zeroGradient wall (no flux anyway) but not at a fixedValue one,
    // which is where a wall function puts its whole effect -- so the energy equation supplies DBnd.
    if (DBnd && DBnd->size()) deviceBCLaplacianCoeffsFace(db, *DBnd, lIC, lBC);
    else                      deviceBCLaplacianCoeffs(db, D, lIC, lBC);
    deviceAxpy(-1.0, lIC, aIC); deviceAxpy(-1.0, lBC, aBC);
    // interface (cyclic/cyclicAMI) coupling: fold div(phi,f) - laplacian(D,f) at the interface into the diagonal and
    // set the off-diagonal ifCoeff. A scalar is invariant under the cyclic transform (no rotation of the value), so the
    // translational momentum assembly + a plain weighted off-diagonal apply even for a ROTATIONAL interface.
    DeviceBuffer<scalar> ifSumOff;
    if (ami && ami->n) { interfaceAssembleMomentum(*ami, D, aD);
        ifSumOff.copyFrom(std::vector<scalar>(nC, 0.0)); interfaceOffDiagSum(*ami, ifSumOff); }
    else if (cyc && cyc->n) { interfaceAssembleMomentum(*cyc, D, aD);
        ifSumOff.copyFrom(std::vector<scalar>(nC, 0.0)); interfaceOffDiagSum(*cyc, ifSumOff); }
    // implicit fvm::ddt(f) (URANS transient turbulence): the diagonal into the assembled aD (BEFORE relax = OF assembles
    // ddt into the eqn then relaxes), the source (old-time) into src. steady (ddt.c.active==false) -> exact no-op, so this
    // stays byte-for-byte the steady scalar transport. rho=1 (incompressible). Matches the momentum ddt wiring.
    deviceFvmDdtDiag(dm.V, ddt.c, 1.0, aD);
    if (ddt.old) { DeviceBuffer<scalar> e2; deviceFvmDdtSource(dm.V, ddt.c, 1.0, *ddt.old, ddt.old2 ? *ddt.old2 : e2, src, ddt.ddt0); }
    DeviceBuffer<scalar> aRD, aDelta; deviceRelaxDiag(deviceLduView(dm, aD, aU, aL), dm, aIC, relax, aRD, aDelta,
                                                      ifSumOff.size() ? ifSumOff.data() : nullptr);
    { DeviceBuffer<scalar> t; deviceHadamard(t, aDelta, field); deviceAxpy(1.0, t, src); }
    if (fvoSetMask && fvoSetVal)   // fvOptions scalarFixedValueConstraint (OF: before boundaryManipulate)
    {
        svFaceKernel<<<nBlocks(dm.nInternalFaces), TPB>>>(dm.nInternalFaces, dm.owner.data(), dm.nei.data(), fvoSetMask->data(), fvoSetVal->data(), aU.data(), aL.data(), src.data());
        svBndKernel<<<nBlocks(dm.nBndFaces), TPB>>>(dm.nBndFaces, dm.bndCell.data(), fvoSetMask->data(), aIC.data(), aBC.data());
        svCellKernel<<<nBlocks(nC), TPB>>>(nC, fvoSetMask->data(), aRD.data(), fvoSetVal->data(), src.data());
        cudaCheck(cudaGetLastError(), "fvOptionsSetValues");
    }
        if (wall && eps0)   // eps near-wall setValues constraint (k has none)
    {
        svFaceKernel<<<nBlocks(dm.nInternalFaces), TPB>>>(dm.nInternalFaces, dm.owner.data(), dm.nei.data(), wall->isWallCell.data(), eps0->data(), aU.data(), aL.data(), src.data());
        svBndKernel<<<nBlocks(dm.nBndFaces), TPB>>>(dm.nBndFaces, dm.bndCell.data(), wall->isWallCell.data(), aIC.data(), aBC.data());
        svCellKernel<<<nBlocks(nC), TPB>>>(nC, wall->isWallCell.data(), aRD.data(), eps0->data(), src.data());
        if (ami && ami->n) interfaceZeroWallIfCoeff(*ami, wall->isWallCell);   // wall/interface cells: don't perturb the fixed eps
        if (cyc && cyc->n) interfaceZeroWallIfCoeff(*cyc, wall->isWallCell);
        cudaCheck(cudaGetLastError(), "setValues");
    }
    DeviceBuffer<scalar> diagC, B; deviceFold(dm, aRD, src, aIC, aBC, diagC, B);
    // Stage harness: the assembled system for this transported scalar, at its first assembly only.
    if (stageDumpActive() && stageDumpFirstOnly((std::string("xport-") + fieldName).c_str()))
    {
        stageDump(std::string("stage_") + fieldName + "D",   diagC);
        stageDump(std::string("stage_") + fieldName + "Src", B);
        stageDump(std::string("stage_") + fieldName + "Diff", D);
        stageDump(std::string("stage_") + fieldName + "Psi",  field);
    }
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
    if (boundPositive) deviceBoundField(dm, field, 1e-15);        // OF bound(field): neg -> local avg, not floor
}

} // namespace brae
