#pragma once
// cf parallel PBiCGStab (DILU preconditioner) for a distributed asymmetric lduMatrix. Same algorithm
// as the serial pbicgstab with the decomposed-case changes (mirroring OpenFOAM): A*x via
// parallelAmul (interface coupling), all sums GLOBAL (Pstream::allReduce), DILU preconditioner LOCAL
// per rank (interfaces dropped from the preconditioner only).
#include "parallel_amul.cuh"
#include "pcg.cuh"   // SolverPerformance
#include <cmath>
#include <vector>

namespace brae {

inline SolverPerformance parallelPBiCGStab(const DistributedMatrix& L, std::vector<scalar>& psi,
                                           std::vector<ProcessorInterface>& interfaces,
                                           label globalNCells,
                                           scalar tol, scalar relTol, int maxIter, int minIter = 0) {
    const label nC  = static_cast<label>(L.diagC.size());
    const label nIf = static_cast<label>(L.upperAddr.size());
    const std::vector<label>&  own   = L.lowerAddr;
    const std::vector<label>&  nei   = L.upperAddr;
    const std::vector<scalar>& upper = L.upper;
    const std::vector<scalar>& lower = L.lower;
    const std::vector<scalar>& diagC = L.diagC;
    const std::vector<scalar>& b     = L.b;

    DistributedMatrix Lf = L; Lf.diag = L.diagC;
    auto Amul = [&](const std::vector<scalar>& x, std::vector<scalar>& Ax) { Ax = parallelAmul(Lf, x, interfaces); };
    auto gSum = [&](scalar s) { return Pstream::allReduce(s, ReduceOp::Sum); };
    auto sumMag  = [&](const std::vector<scalar>& x) { scalar s = 0; for (label c = 0; c < nC; ++c) s += std::fabs(x[c]); return gSum(s); };
    auto sumProd = [&](const std::vector<scalar>& a, const std::vector<scalar>& c) { scalar s = 0; for (label i = 0; i < nC; ++i) s += a[i] * c[i]; return gSum(s); };
    auto sumSqr  = [&](const std::vector<scalar>& a) { scalar s = 0; for (label i = 0; i < nC; ++i) s += a[i] * a[i]; return gSum(s); };

    // Local DILU reciprocal diagonal (interfaces dropped).
    std::vector<scalar> rD = diagC;
    for (label f = 0; f < nIf; ++f) rD[nei[f]] -= upper[f] * lower[f] / rD[own[f]];
    for (label c = 0; c < nC; ++c) rD[c] = 1.0 / rD[c];
    auto precondition = [&](std::vector<scalar>& wA, const std::vector<scalar>& rA) {
        for (label c = 0; c < nC; ++c) wA[c] = rD[c] * rA[c];
        for (label f = 0; f < nIf; ++f)      wA[nei[f]] -= rD[nei[f]] * lower[f] * wA[own[f]];
        for (label f = nIf - 1; f >= 0; --f) wA[own[f]] -= rD[own[f]] * upper[f] * wA[nei[f]];
    };

    std::vector<scalar> yA(nC, 0.0), rA(nC);
    Amul(psi, yA);
    for (label c = 0; c < nC; ++c) rA[c] = b[c] - yA[c];

    // normFactor (global): sumA row sum incl. interface off-diagonals (= -interfaceCoeff).
    std::vector<scalar> sumA = diagC;
    for (label f = 0; f < nIf; ++f) { sumA[nei[f]] += lower[f]; sumA[own[f]] += upper[f]; }
    for (std::size_t i = 0; i < interfaces.size(); ++i)
        for (std::size_t f = 0; f < L.interfaceCoeffs[i].size(); ++f)
            sumA[interfaces[i].faceCells()[f]] += -L.interfaceCoeffs[i][f];
    const scalar xRef = gSum([&]{ scalar s = 0; for (label c = 0; c < nC; ++c) s += psi[c]; return s; }())
                      / static_cast<scalar>(globalNCells);
    scalar nf = 0;
    for (label c = 0; c < nC; ++c) { const scalar t = sumA[c] * xRef; nf += std::fabs(yA[c] - t) + std::fabs(b[c] - t); }
    const scalar normFactor = gSum(nf) + 1e-20;

    SolverPerformance perf;
    perf.initialResidual = sumMag(rA) / normFactor;
    perf.finalResidual   = perf.initialResidual;
    auto converged = [&](scalar fr) { return (fr < tol) || (relTol > 0.0 && fr < relTol * perf.initialResidual); };

    if (minIter > 0 || !converged(perf.finalResidual)) {
        std::vector<scalar> pA(nC, 0.0), AyA(nC, 0.0), sA(nC), zA(nC), tA(nC);
        const std::vector<scalar> rA0 = rA;
        scalar rA0rA = 0, alpha = 0, omega = 0;
        int nIter = 0;
        do {
            const scalar rA0rAold = rA0rA;
            rA0rA = sumProd(rA0, rA);
            if (std::fabs(rA0rA) < 1e-300) break;

            if (nIter == 0) pA = rA;
            else {
                if (std::fabs(omega) < 1e-300) break;
                const scalar beta = (rA0rA / rA0rAold) * (alpha / omega);
                for (label c = 0; c < nC; ++c) pA[c] = rA[c] + beta * (pA[c] - omega * AyA[c]);
            }

            precondition(yA, pA);
            Amul(yA, AyA);
            alpha = rA0rA / sumProd(rA0, AyA);
            for (label c = 0; c < nC; ++c) sA[c] = rA[c] - alpha * AyA[c];

            perf.finalResidual = sumMag(sA) / normFactor;
            if (nIter >= minIter && converged(perf.finalResidual)) {
                for (label c = 0; c < nC; ++c) psi[c] += alpha * yA[c];
                perf.nIterations = ++nIter;
                return perf;
            }

            precondition(zA, sA);
            Amul(zA, tA);
            omega = sumProd(tA, sA) / sumSqr(tA);
            for (label c = 0; c < nC; ++c) { psi[c] += alpha * yA[c] + omega * zA[c]; rA[c] = sA[c] - omega * tA[c]; }
            perf.finalResidual = sumMag(rA) / normFactor;
            ++nIter;
        } while ((nIter < maxIter && !converged(perf.finalResidual)) || nIter < minIter);
        perf.nIterations = nIter;
    }
    return perf;
}

} // namespace brae
