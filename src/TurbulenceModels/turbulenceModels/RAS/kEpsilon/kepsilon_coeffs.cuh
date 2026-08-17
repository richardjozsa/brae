#pragma once
// k-epsilon model coefficients, shared by the device (device_kepsilon) and CPU (k_epsilon/parallel_kepsilon)
// paths. Defaults = OpenFOAM v2412 kEpsilon defaults; read from turbulenceProperties RAS.kEpsilonCoeffs
// (Cmu/C1/C2/C3/sigmak/sigmaEps). kappa/E are the wall-function coeffs (OF reads them from the wall-function
// BC dicts; default 0.41/9.8). A default-constructed struct reproduces the previous hardcoded constants exactly,
// so unchanged callers stay bit-identical.
#include "cf_types.cuh"

namespace brae {

struct KEpsilonCoeffs
{
    scalar Cmu = 0.09, C1 = 1.44, C2 = 1.92, C3 = 0.0;
    scalar sigmaK = 1.0, sigmaEps = 1.3;
    scalar kappa = 0.41, E = 9.8;
    // realizableKE (OF RAS/realizableKE): variable Cmu (rCmu from strain invariants), strain-based eps production
    // C1=max(eta/(5+eta),0.43)*magS*eps, destruction C2*eps^2/(k+sqrt(nu*eps)). Defaults A0=4, C2=1.9, sigmaEps=1.2.
    bool   realizable = false;
    scalar A0 = 4.0;
    // RNGkEpsilon (OF RAS/RNGkEpsilon): standard k-epsilon with renormalisation-group coefficients and ONE
    // extra term -- the epsilon production coefficient becomes (C1 - R) instead of C1, with
    //     eta = sqrt(|S2|)*k/epsilon,   R = eta*(1 - eta/eta0)/(1 + beta*eta^3)
    // R is a strain-rate-dependent SINK at high strain (eta > eta0) and a source below it, which is what
    // lets RNG handle separated and swirling flow that the standard model over-predicts. The SuSp divU
    // term keeps the plain C1 -- OF applies (C1 - R) to the G production alone.
    bool   rng  = false;
    scalar eta0 = 4.38;
    scalar beta = 0.012;
};

} // namespace brae
