#pragma once
// cf GPU offload (G2): device-resident Jacobi-preconditioned CG. Composes the G1 SpMV (deviceAmul) with
// the G0 BLAS-1 kernels; the whole Krylov loop runs on the GPU, only the scalar reductions (wArA, wApA,
// residual) return to the host each iteration. Same CG recurrence as cf's CPU pcg, with a Jacobi (diagonal)
// preconditioner in place of DIC (DIC's sweeps are sequential, a later phase). normFactor is the OF
// normalisation, passed from the host. Solves to the same converged solution as the CPU solver.
#include "cf_types.cuh"
#include "device_ldu.cuh"
#include "device_buffer.cuh"

namespace brae {

struct DeviceSolverPerf { scalar initialResidual = 0, finalResidual = 0; int nIterations = 0; };

// OpenFOAM's lduMatrix::solver::normFactor: sum(|A*psi - sumA*avg(psi)| + |b - sumA*avg(psi)|) + small,
// where sumA = rowSum(A) = A*ones. Pass this as the solver's normFactor so the reported initial residual
// (and the tolerance test) match OF's residualControl convention. ones is a cached unit vector (length nCells).
scalar deviceNormFactor(const DeviceLduView& A, const DeviceBuffer<scalar>& psi,
                        const DeviceBuffer<scalar>& b, const DeviceBuffer<scalar>& ones);

DeviceSolverPerf deviceJacobiPCG(const DeviceLduView& A, const DeviceBuffer<scalar>& b,
                                 DeviceBuffer<scalar>& psi, scalar normFactor,
                                 scalar tol, scalar relTol, int maxIter);

// Jacobi-preconditioned BiCGStab for the NON-symmetric momentum matrix (upwind convection -> upper!=lower).
// Same recurrence as brae::pbicgstab, Jacobi in place of DILU; device-resident.
// checkEvery: read the |s|/|r| convergence norms (the 2 of 4 D2H reads/iter that aren't breakdown guards) only every
// K iters -> batched convergence, like deviceAMGPCG's checkEvery. Breakdown guards (rA0rA, omega) stay per-iter for
// safety. K=1 = exact (bit-identical); K>1 overshoots convergence by < K iters. Default 1.
DeviceSolverPerf deviceJacobiBiCGStab(const DeviceLduView& A, const DeviceBuffer<scalar>& b,
                                      DeviceBuffer<scalar>& psi, scalar normFactor,
                                      scalar tol, scalar relTol, int maxIter, int checkEvery = 1);

} // namespace brae
