#pragma once
// Courant-limited time-step control -- OF src/finiteVolume/cfdTools/general/include/
// {readTimeControls,CourantNo,setInitialDeltaT,setDeltaT}.H
//
// WHY IT LIVES HERE, and not inside gpuPimpleFoam. In OpenFOAM these are #include files that EVERY
// transient solver pulls in unchanged -- pimpleFoam, rhoPimpleFoam, interFoam, sonicFoam all use the
// same four. None of them reimplements the formula. brae had none of it: gpuPimpleFoam refused
// `adjustTimeStep yes` outright, which is 11 of OpenFOAM's 35 pimpleFoam tutorials and the default for
// transient cases. Putting it here means the compressible transient solver gets it by including one
// header, exactly as rhoPimpleFoam.C does.
//
// EXACT OF SEMANTICS, transcribed rather than approximated:
//
//   CourantNo.H       sumPhi = surfaceSum(mag(phi))            [per cell]
//                     CoNum     = 0.5*gMax(sumPhi/V)*deltaT
//                     meanCoNum = 0.5*(gSum(sumPhi)/gSum(V))*deltaT
//
//   setInitialDeltaT  if (timeIndex == 0 && CoNum > SMALL)
//                         deltaT = min(maxCo*deltaT/CoNum, min(deltaT, maxDeltaT))
//
//   setDeltaT         maxDeltaTFact = maxCo/(CoNum + SMALL)
//                     deltaTFact    = min(min(maxDeltaTFact, 1 + 0.1*maxDeltaTFact), 1.2)
//                     deltaT        = min(deltaTFact*deltaT, maxDeltaT)
//
// The 1.2 cap and the `1 + 0.1*maxDeltaTFact` damping are not arbitrary: OF's own header says
// "Reduction of time-step is immediate, but increase is damped to avoid unstable oscillations". A
// reimplementation that merely scaled by maxCo/CoNum would ring. Both are reproduced.

#include "cf_types.cuh"
#include "foam_dict.cuh"
#include <algorithm>
#include <cmath>
#include <vector>

namespace brae {

// OF readTimeControls.H. maxDeltaT defaults to GREAT (i.e. no cap) exactly as OF does.
struct TimeControls
{
    bool   adjustTimeStep = false;
    scalar maxCo          = 1.0;
    scalar maxDeltaT      = 1.0e300;   // OF: GREAT

    static TimeControls read(const FoamDict& controlDict)
    {
        TimeControls tc;
        const std::string a = controlDict.wordOr("adjustTimeStep", "no");
        tc.adjustTimeStep = (a == "yes" || a == "true" || a == "on" || a == "1");
        tc.maxCo     = controlDict.scalarOr("maxCo", 1.0);
        tc.maxDeltaT = controlDict.scalarOr("maxDeltaT", 1.0e300);
        return tc;
    }
};

struct CourantNumbers
{
    scalar CoNum     = 0;
    scalar meanCoNum = 0;
};

// surfaceSum(mag(phi)) per cell -- OF fvc::surfaceSum, which adds |phi| to BOTH the owner and the
// neighbour of every internal face, and to the owner of every boundary face. Missing the boundary half
// understates Co near inlets, which is exactly where the limiting cell usually is.
inline std::vector<scalar> surfaceSumMagPhi(
    const std::vector<label>& owner,
    const std::vector<label>& neighbour,
    const std::vector<scalar>& phiInternal,
    const std::vector<scalar>& phiBoundary,
    label nCells,
    label nInternalFaces)
{
    std::vector<scalar> sumPhi(static_cast<std::size_t>(nCells), 0.0);
    for (label f = 0; f < nInternalFaces && f < (label)phiInternal.size(); ++f)
    {
        const scalar a = std::fabs(phiInternal[f]);
        if (owner[f] >= 0 && owner[f] < nCells)         sumPhi[owner[f]]     += a;
        if (f < (label)neighbour.size() && neighbour[f] >= 0 && neighbour[f] < nCells)
            sumPhi[neighbour[f]] += a;
    }
    for (std::size_t b = 0; b < phiBoundary.size(); ++b)
    {
        const label f = nInternalFaces + static_cast<label>(b);
        if (f < (label)owner.size() && owner[f] >= 0 && owner[f] < nCells)
            sumPhi[owner[f]] += std::fabs(phiBoundary[b]);
    }
    return sumPhi;
}

// OF CourantNo.H. `sumPhi` is surfaceSum(mag(phi)) per cell -- the caller supplies it because the flux
// lives on the device and summing it there is the solver's job, not this header's.
inline CourantNumbers courantNo(
    const std::vector<scalar>& sumPhi,
    const std::vector<scalar>& V,
    scalar deltaT)
{
    CourantNumbers c;
    if (sumPhi.empty() || V.empty()) return c;
    scalar maxRatio = 0, sPhi = 0, sV = 0;
    for (std::size_t i = 0; i < sumPhi.size() && i < V.size(); ++i)
    {
        if (V[i] > 0) maxRatio = std::max(maxRatio, sumPhi[i]/V[i]);
        sPhi += sumPhi[i];
        sV   += V[i];
    }
    c.CoNum     = 0.5*maxRatio*deltaT;
    c.meanCoNum = (sV > 0) ? 0.5*(sPhi/sV)*deltaT : scalar(0);
    return c;
}

// OF setInitialDeltaT.H -- applied ONCE, before the first step, so the run starts at the requested
// Courant number instead of spending steps ramping to it.
inline scalar setInitialDeltaT(scalar deltaT, scalar CoNum, const TimeControls& tc)
{
    if (!tc.adjustTimeStep) return deltaT;
    const scalar kSmall = 1.0e-37;                       // OF SMALL
    if (CoNum <= kSmall) return deltaT;
    return std::min(tc.maxCo*deltaT/CoNum, std::min(deltaT, tc.maxDeltaT));
}

// OF setDeltaT.H -- applied every step. Reduction is immediate; growth is damped and capped at 1.2x.
inline scalar setDeltaT(scalar deltaT, scalar CoNum, const TimeControls& tc)
{
    if (!tc.adjustTimeStep) return deltaT;
    const scalar kSmall = 1.0e-37;
    const scalar maxDeltaTFact = tc.maxCo/(CoNum + kSmall);
    const scalar deltaTFact = std::min(std::min(maxDeltaTFact, scalar(1) + scalar(0.1)*maxDeltaTFact), scalar(1.2));
    return std::min(deltaTFact*deltaT, tc.maxDeltaT);
}

}   // namespace brae
