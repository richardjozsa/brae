#pragma once
// cf parallel PCG (DIC preconditioner) for a distributed symmetric lduMatrix. Same algorithm as
// the serial pcg (lduMatrix PCG + DICPreconditioner + normFactor) with three changes for the
// decomposed case, mirroring OpenFOAM:
//   - A*x via parallelAmul (processor-interface coupling + one halo exchange per product);
//   - every inner product / sum is a GLOBAL reduction (Pstream::allReduce);
//   - the DIC preconditioner is LOCAL per rank (ignores interfaces), exactly as OpenFOAM's
//     parallel DIC, the off-diagonal interface terms are dropped from the preconditioner only.
#include "parallel_amul.cuh"
#include "pcg.cuh"   // SolverPerformance
#include <cmath>
#include <vector>

namespace brae {

inline SolverPerformance parallelPCG(const DistributedMatrix& L, std::vector<scalar>& psi,
                                     std::vector<ProcessorInterface>& interfaces,
                                     label globalNCells,
                                     scalar tol, scalar relTol, int maxIter, int minIter = 0) {
    const label nC  = static_cast<label>(L.diagC.size());
    const label nIf = static_cast<label>(L.upperAddr.size());
    const std::vector<label>&  own   = L.lowerAddr;
    const std::vector<label>&  nei   = L.upperAddr;
    const std::vector<scalar>& upper = L.upper;
    const std::vector<scalar>& diagC = L.diagC;
    const std::vector<scalar>& b     = L.b;

    DistributedMatrix Lf = L; Lf.diag = L.diagC;                 // Amul uses the boundary-folded diag
    auto Amul = [&](const std::vector<scalar>& x) { return parallelAmul(Lf, x, interfaces); };
    auto gSum = [&](scalar s) { return Pstream::allReduce(s, ReduceOp::Sum); };
    auto gSumProd = [&](const std::vector<scalar>& a, const std::vector<scalar>& c) {
        scalar s = 0; for (label i = 0; i < nC; ++i) s += a[i] * c[i]; return gSum(s);
    };
    auto gSumMag = [&](const std::vector<scalar>& x) {
        scalar s = 0; for (label i = 0; i < nC; ++i) s += std::fabs(x[i]); return gSum(s);
    };

    SolverPerformance perf;
    std::vector<scalar> wA = Amul(psi), rA(nC), pA(nC, 0.0);
    for (label c = 0; c < nC; ++c) rA[c] = b[c] - wA[c];

    // normFactor (global): xRef = global mean psi; sumA = full row sum (incl. interface off-diagonals,
    // which are -interfaceCoeff). Mirrors lduMatrix::solver::normFactor.
    const scalar xRef = gSum([&]{ scalar s = 0; for (label c = 0; c < nC; ++c) s += psi[c]; return s; }())
                      / static_cast<scalar>(globalNCells);
    std::vector<scalar> sumA = diagC;
    for (label f = 0; f < nIf; ++f) { sumA[nei[f]] += L.lower[f]; sumA[own[f]] += upper[f]; }
    for (std::size_t i = 0; i < interfaces.size(); ++i)
        for (std::size_t f = 0; f < L.interfaceCoeffs[i].size(); ++f)
            sumA[interfaces[i].faceCells()[f]] += -L.interfaceCoeffs[i][f];
    scalar nf = 0;
    for (label c = 0; c < nC; ++c) { const scalar t = sumA[c] * xRef; nf += std::fabs(wA[c] - t) + std::fabs(b[c] - t); }
    const scalar normFactor = gSum(nf) + 1e-20;

    perf.initialResidual = gSumMag(rA) / normFactor;
    perf.finalResidual   = perf.initialResidual;

    // Local DIC reciprocal diagonal (interfaces dropped).
    std::vector<scalar> rD = diagC;
    for (label f = 0; f < nIf; ++f) rD[nei[f]] -= upper[f] * upper[f] / rD[own[f]];
    for (label c = 0; c < nC; ++c) rD[c] = 1.0 / rD[c];

    auto converged = [&](scalar res) { return res < tol || (relTol > 0 && res < relTol * perf.initialResidual); };

    scalar wArAold = 1.0;
    int nIter = 0;
    if (minIter > 0 || !converged(perf.initialResidual)) {
        do {
            std::vector<scalar> wAp(nC);                       // DIC precondition: wAp = M^-1 rA
            for (label c = 0; c < nC; ++c) wAp[c] = rD[c] * rA[c];
            for (label f = 0; f < nIf; ++f)      wAp[nei[f]] -= rD[nei[f]] * upper[f] * wAp[own[f]];
            for (label f = nIf - 1; f >= 0; --f) wAp[own[f]] -= rD[own[f]] * upper[f] * wAp[nei[f]];

            const scalar wArA = gSumProd(wAp, rA);
            if (nIter == 0) pA = wAp;
            else { const scalar beta = wArA / wArAold; for (label c = 0; c < nC; ++c) pA[c] = wAp[c] + beta * pA[c]; }
            wArAold = wArA;

            wA = Amul(pA);
            const scalar alpha = wArA / gSumProd(wA, pA);
            for (label c = 0; c < nC; ++c) { psi[c] += alpha * pA[c]; rA[c] -= alpha * wA[c]; }
            perf.finalResidual = gSumMag(rA) / normFactor;
            ++nIter;
        } while ((nIter < maxIter && !converged(perf.finalResidual)) || nIter < minIter);
    }
    perf.nIterations = nIter;
    return perf;
}

} // namespace brae
