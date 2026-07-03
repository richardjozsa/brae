#pragma once
// k-epsilon model coefficients, shared by the device (device_kepsilon) and CPU (k_epsilon/parallel_kepsilon)
// paths. Defaults = OpenFOAM v2412 kEpsilon defaults; read from turbulenceProperties RAS.kEpsilonCoeffs
// (Cmu/C1/C2/C3/sigmak/sigmaEps). kappa/E are the wall-function coeffs (OF reads them from the wall-function
// BC dicts; default 0.41/9.8). A default-constructed struct reproduces the previous hardcoded constants exactly,
// so unchanged callers stay bit-identical.
#include "cf_types.cuh"

namespace brae {

struct KEpsilonCoeffs {
    scalar Cmu = 0.09, C1 = 1.44, C2 = 1.92, C3 = 0.0;
    scalar sigmaK = 1.0, sigmaEps = 1.3;
    scalar kappa = 0.41, E = 9.8;
    // realizableKE (OF RAS/realizableKE): variable Cmu (rCmu from strain invariants), strain-based eps production
    // C1=max(eta/(5+eta),0.43)*magS*eps, destruction C2*eps^2/(k+sqrt(nu*eps)). Defaults A0=4, C2=1.9, sigmaEps=1.2.
    bool   realizable = false;
    scalar A0 = 4.0;
};

} // namespace brae
