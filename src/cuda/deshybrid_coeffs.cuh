#pragma once
// DEShybrid convection scheme coefficients (OF TurbulenceModels/schemes/DEShybrid).
//
// DEShybrid blends a LOW-DISSIPATION scheme with an UPWIND-BIASED one face by face, using a DES-style
// sensor that detects where the flow is resolved-turbulent (blend toward the low-dissipation scheme) and
// where it is not (blend toward upwind). fvSchemes gives it as
//
//     div(phi,U)  Gauss DEShybrid
//         linear                  // scheme 1 (low dissipation)
//         linearUpwind grad(U)    // scheme 2 (upwind-biased)
//         delta                   // name of the LES delta field
//         0.65                    // CDES
//         1.0                     // U0, reference velocity scale
//         1.0                     // L0, reference length scale
//         0.0                     // sigmaMin
//         1.0                     // sigmaMax
//         1.0                     // OmegaLim
//         10.0;                   // nutLim (optional, default 1)
//
// and the sensor is (DEShybrid.H::calcBlendingFactor), with CH1=3, CH2=1, CH3=2, Cs=0.18 fixed in OF:
//
//     S     = sqrt(2)*mag(symm(gradU)),  Omega = sqrt(2)*mag(skew(gradU)),  tau0 = L0/U0
//     B     = CH3*Omega*max(S,Omega) / max(0.5*(S^2 + Omega^2), (OmegaLim/tau0)^2)
//     g     = tanh(B^4)
//     K     = max(sqrt(0.5*(S^2 + Omega^2)), 0.1/tau0)
//     lTurb = sqrt(max( (max(nut, min((Cs*delta)^2*S, nutLim*nut)) + nu) / (0.09^1.5 * K), 0 ))
//     A     = CH2*max(0, CDES*delta/max(lTurb*g, SMALL*L0) - 0.5)
//     sigma = max(sigmaMax*tanh(A^CH1), sigmaMin)          [per CELL]
//     bf    = fvc::interpolate(sigma)                      [per FACE]
//
// bf = 0 selects scheme 1 outright and bf = 1 scheme 2, so the face value is
//     (1-bf)*linear_face + bf*linearUpwind_face
// which, against brae's upwind matrix, is a deferred correction of
//     (1-bf)*(linear - upwind) + bf*(linearUpwind - upwind)
// i.e. exactly the LUST blend with a per-face factor instead of the fixed 0.75/0.25.
#include "cf_types.cuh"

namespace brae {

struct DesHybridCoeffs
{
    scalar CDES     = 0.65;
    scalar U0       = 1.0;    // reference velocity scale [m/s], must be > 0
    scalar L0       = 1.0;    // reference length scale [m], must be > 0
    scalar sigmaMin = 0.0;    // in [0,1]
    scalar sigmaMax = 1.0;    // in [0,1]
    scalar OmegaLim = 1.0;
    scalar nutLim   = 1.0;    // > 1 activates OF's GAM extension
    // Fixed in OF's constructor; not readable from the scheme entry.
    scalar CH1 = 3.0, CH2 = 1.0, CH3 = 2.0, Cs = 0.18;
};

} // namespace brae
