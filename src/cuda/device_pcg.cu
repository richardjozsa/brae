// cf GPU offload (G2): device-resident Jacobi-PCG. Mirrors brae::pcg's CG recurrence exactly, but the
// preconditioner is Jacobi (wA = rA/diag) and every vector op is a device kernel.
#include "device_pcg.cuh"
#include "device_blas.cuh"
#include <cuda_runtime.h>
#include <cmath>


namespace brae {

namespace {
// BiCGStab device-scalar recurrence kernels (1 thread): keep omega/beta off the host (like the AMG-PCG scalars).
// Breakdown is OF PBiCGStab's SolverPerformance::checkSingularity: |rho(rA0rA)| or |omega| < VSMALL. We detect it
// ON-DEVICE into a flag `bd` (read only on the K-cadence convergence check, not every iter -> removes the 2 per-iter
// breakdown D2H reads). The guarded recurrence stays NaN-safe between checks; for a NON-breakdown solve nothing trips
// (rho/omega stay >> VSMALL until convergence breaks first) so the iteration is bit-identical to the host-guarded form.
constexpr scalar BICG_VSMALL = 1e-300;   // = OF solveScalar VSMALL (SolverPerformance.C checkSingularity: mag(x) < vsmall_)
__global__ void omegaK(const scalar* __restrict__ ts, const scalar* __restrict__ tt,
                       scalar* __restrict__ om, scalar* __restrict__ negOm, scalar* __restrict__ bd) {
    if (threadIdx.x==0 && blockIdx.x==0) {
        const scalar t=*tt; const scalar o = (t > BICG_VSMALL) ? (*ts)/t : 0.0; *om=o; *negOm=-o;
        if (fabs(o) < BICG_VSMALL) *bd = 1.0;            // OF: checkSingularity(mag(omega)) (guards next-iter beta)
    }
}
__global__ void bicgBetaK(const scalar* __restrict__ rr, const scalar* __restrict__ rrOld,
                          const scalar* __restrict__ al, const scalar* __restrict__ om, scalar* __restrict__ beta,
                          scalar* __restrict__ bd) {
    if (threadIdx.x==0 && blockIdx.x==0) {
        const scalar ro=*rrOld, o=*om; const bool ok = fabs(ro) >= BICG_VSMALL && fabs(o) >= BICG_VSMALL;
        *beta = ok ? (*rr / ro) * (*al / o) : 0.0;       // beta = (rA0rA/rA0rAold)*(alpha/omega), guarded
        if (!ok) *bd = 1.0;
    }
}
__global__ void bicgRhoSingK(const scalar* __restrict__ rr, scalar* __restrict__ bd) {
    if (threadIdx.x==0 && blockIdx.x==0 && fabs(*rr) < BICG_VSMALL) *bd = 1.0;   // OF: checkSingularity(mag(rA0rA))
}
__global__ void bicgAlphaK(const scalar* __restrict__ rr, const scalar* __restrict__ r0Ay,
                           scalar* __restrict__ al, scalar* __restrict__ negAl, scalar* __restrict__ bd) {
    if (threadIdx.x==0 && blockIdx.x==0) {
        const scalar d=*r0Ay; const bool ok = fabs(d) >= BICG_VSMALL; const scalar q = ok ? (*rr)/d : 0.0;
        *al=q; *negAl=-q;                                // OF: alpha = rA0rA/rA0AyA (guarded vs 0/0 breakdown)
        if (!ok) *bd = 1.0;
    }
}
// Persistent BiCGStab device scalars (one-time alloc; single solve at a time, sequential across the 3 U components).
struct BiCGScalars {
    DeviceBuffer<scalar> rr, rrOld, alpha, negAlpha, omega, negOmega, beta, r0Ay, tt, ts, sNorm, rNorm, bd;
    BiCGScalars() { for (auto* p : {&rr,&rrOld,&alpha,&negAlpha,&omega,&negOmega,&beta,&r0Ay,&tt,&ts,&sNorm,&rNorm,&bd}) p->resize(1); }
};
inline BiCGScalars& bicgScalars() { static BiCGScalars s; return s; }
} // namespace

DeviceSolverPerf deviceJacobiPCG(const DeviceLduView& A, const DeviceBuffer<scalar>& b,
                                 DeviceBuffer<scalar>& psi, scalar normFactor,
                                 scalar tol, scalar relTol, int maxIter) {
    const int nC = A.nCells;
    DeviceBuffer<scalar> wA(nC), rA(nC), pA(nC), Ax(nC);

    deviceAmul(A, psi, Ax);                                  // rA = b - A*psi
    deviceCopy(rA, b); deviceAxpy(-1.0, Ax, rA);

    DeviceSolverPerf perf;
    perf.initialResidual = deviceSumMag(rA) / normFactor;
    perf.finalResidual   = perf.initialResidual;
    auto converged = [&](scalar fr) { return (fr < tol) || (relTol > 0.0 && fr < relTol * perf.initialResidual); };

    scalar wArA = 1e300, wArAold;
    int nIter = 0;
    if (!converged(perf.finalResidual)) {
        do {
            wArAold = wArA;
            deviceJacobi(wA, rA, A.diag);                   // wA = M^-1 rA  (Jacobi)
            wArA = deviceDot(wA, rA);
            if (nIter == 0) deviceCopy(pA, wA);             // pA = wA
            else { const scalar beta = wArA / wArAold; deviceScale(pA, beta); deviceAxpy(1.0, wA, pA); }  // pA = wA + beta*pA
            deviceAmul(A, pA, wA);                          // wA = A*pA
            const scalar wApA  = deviceDot(wA, pA);
            const scalar alpha = wArA / wApA;
            deviceAxpy(alpha, pA, psi);                     // psi += alpha*pA
            deviceAxpy(-alpha, wA, rA);                     // rA  -= alpha*wA
            perf.finalResidual = deviceSumMag(rA) / normFactor;
            ++nIter;
        } while (nIter < maxIter && !converged(perf.finalResidual));
    }
    perf.nIterations = nIter;
    return perf;
}

scalar deviceNormFactor(const DeviceLduView& A, const DeviceBuffer<scalar>& psi,
                        const DeviceBuffer<scalar>& b, const DeviceBuffer<scalar>& ones) {
    const int nC = A.nCells;
    DeviceBuffer<scalar> Apsi(nC), sumA(nC), tmp(nC), t(nC);
    // Device-resident: the 3 reductions (avgPsi, n1, n2) stay on the device; only the final normFactor is read to
    // the host (it scales the residual for the convergence check). Same kernels + same IEEE ops (the divide by nC,
    // the avg-multiply, and the (n1+n2)+1e-20 add are reproduced exactly) -> bit-identical, 3 D2H syncs -> 1.
    DeviceBuffer<scalar> dAvg(1), dN1(1), dN2(1), dNorm(1);
    deviceAmul(A, psi, Apsi);                                // A*psi
    deviceAmul(A, ones, sumA);                               // sumA = rowSum(A) = A*1
    deviceDotInto(psi, ones, dAvg.data());                  // psi.ones
    deviceScalarDivConst(dAvg.data(), (scalar)nC, dAvg.data());   // avgPsi = gAverage(psi) = (psi.ones)/nC
    deviceCopy(tmp, sumA); deviceScaleDev(dAvg.data(), tmp);      // tmp = sumA*avg(psi)
    deviceCopy(t, Apsi); deviceAxpy(-1.0, tmp, t); deviceSumMagInto(t, dN1.data());   // n1 = |A*psi - tmp|
    deviceCopy(t, b);    deviceAxpy(-1.0, tmp, t); deviceSumMagInto(t, dN2.data());   // n2 = |b - tmp|
    deviceScalarAdd2(dN1.data(), dN2.data(), 1e-20, dNorm.data());                    // n1 + n2 + 1e-20
    return deviceReadScalar(dNorm.data());                                            // the only host sync
}

DeviceSolverPerf deviceJacobiBiCGStab(const DeviceLduView& A, const DeviceBuffer<scalar>& b,
                                      DeviceBuffer<scalar>& psi, scalar normFactor,
                                      scalar tol, scalar relTol, int maxIter, int checkEvery) {
    const int nC = A.nCells;
    const int K = (checkEvery > 1) ? checkEvery : 1;             // convergence-read cadence (1 = exact per-iter)
    DeviceBuffer<scalar> rA(nC), rA0(nC), pA(nC), yA(nC), AyA(nC), sA(nC), zA(nC), tA(nC), Ax(nC);

    deviceAmul(A, psi, Ax);                                  // rA = b - A*psi
    deviceCopy(rA, b); deviceAxpy(-1.0, Ax, rA);
    deviceCopy(rA0, rA);

    DeviceSolverPerf perf;
    perf.initialResidual = deviceSumMag(rA) / normFactor;
    perf.finalResidual   = perf.initialResidual;
    auto converged = [&](scalar fr) { return (fr < tol) || (relTol > 0.0 && fr < relTol * perf.initialResidual); };

    // DEVICE-RESIDENT BiCGStab: alpha/omega/beta and all dots live on the device (fed to the *Dev kernels by pointer).
    // Breakdown (OF checkSingularity: |rA0rA| or |omega| < VSMALL) is detected ON-DEVICE into `bd` and read only on the
    // K-cadence check (with |s|/|r|), so the host reads NOTHING per off-check iter (was 2 per-iter breakdown D2H reads).
    // For any non-breakdown solve nothing trips (rho/omega stay >> VSMALL until convergence breaks first) -> same kernels
    // + IEEE ops + branches -> bit-identical. On a true breakdown the guarded recurrence damps (no NaN) and the batched
    // bd read stops it within K iters (vs OF's immediate break -- equivalent result; breakdown is pathological & rare).
    BiCGScalars& s = bicgScalars();
    cudaCheck(cudaMemsetAsync(s.bd.data(), 0, sizeof(scalar), cudaStreamPerThread), "bicg bd zero");
    int nIter = 0;
    if (!converged(perf.finalResidual)) {
        do {
            if (nIter > 0) deviceScalarCopy(s.rr.data(), s.rrOld.data());   // rA0rAold = rA0rA
            deviceDotInto(rA0, rA, s.rr.data());                  // rA0rA = rA0 . rA
            bicgRhoSingK<<<1,1>>>(s.rr.data(), s.bd.data());      // OF checkSingularity(mag(rA0rA)) -> flag (no host read)
            if (nIter == 0) deviceCopy(pA, rA);
            else {
                bicgBetaK<<<1,1>>>(s.rr.data(), s.rrOld.data(), s.alpha.data(), s.omega.data(), s.beta.data(), s.bd.data());
                deviceFusedBicgP(rA, pA, AyA, s.beta.data(), s.negOmega.data());            // pA = rA + beta*(pA - omega*AyA)  [fused 3->1]
            }
            deviceJacobi(yA, pA, A.diag); deviceAmul(A, yA, AyA);          // yA = M^-1 pA; AyA = A yA
            deviceDotInto(rA0, AyA, s.r0Ay.data());
            bicgAlphaK<<<1,1>>>(s.rr.data(), s.r0Ay.data(), s.alpha.data(), s.negAlpha.data(), s.bd.data());   // alpha = rA0rA/(rA0.AyA), guarded
            deviceFusedSxpy(sA, rA, s.negAlpha.data(), AyA);                            // sA = rA - alpha*AyA  [fused 2->1]
            const bool check = ((nIter + 1) % K == 0) || (nIter + 1 >= maxIter);        // read |s|/|r|/bd only on check iters
            if (check) {                                          // mid-iter early-exit only when we read |s|
                deviceSumMagInto(sA, s.sNorm.data());
                perf.finalResidual = deviceReadScalar(s.sNorm.data()) / normFactor;
                if (converged(perf.finalResidual)) { deviceAxpyDev(s.alpha.data(), yA, psi); ++nIter; break; }
            }
            deviceJacobi(zA, sA, A.diag); deviceAmul(A, zA, tA);           // zA = M^-1 sA; tA = A zA
            deviceDotInto(tA, tA, s.tt.data()); deviceDotInto(tA, sA, s.ts.data());
            omegaK<<<1,1>>>(s.ts.data(), s.tt.data(), s.omega.data(), s.negOmega.data(), s.bd.data());     // omega = tt>tiny ? ts/tt : 0 (+singularity flag)
            deviceFusedAxpy2(psi, s.alpha.data(), yA, s.omega.data(), zA);               // psi += alpha*yA + omega*zA  [fused 2->1]
            deviceFusedSxpy(rA, sA, s.negOmega.data(), tA);                             // rA = sA - omega*tA  [fused 2->1]
            if (check) {
                deviceSumMagInto(rA, s.rNorm.data());
                perf.finalResidual = deviceReadScalar(s.rNorm.data()) / normFactor;
                if (deviceReadScalar(s.bd.data()) != 0.0) { ++nIter; break; }   // OF break on singularity, batched to K
            }
            ++nIter;
        } while (nIter < maxIter && !converged(perf.finalResidual));
    }
    perf.nIterations = nIter;
    return perf;
}

} // namespace brae
