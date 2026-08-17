#pragma once
// Smagorinsky LES sub-grid model coefficients (OF Smagorinsky defaults). POD: passed to kernels by value.
// OF's Smagorinsky is the algebraic k-equilibrium form nut = Ck*delta*sqrt(k_sgs), with k_sgs from the local
// production=dissipation balance (Smagorinsky<>::k(gradU)):
//     a = Ce/delta,  b = (2/3)tr(D),  c = 2*Ck*delta*(dev(D) && D),  sqrt(k) = (-b + sqrt(b^2 + 4ac))/(2a)
// (D = symm(gradU)). For an incompressible field (tr(D)=div(U)->0) this reduces to the classic Smagorinsky
//     nut = Ck*sqrt(2*Ck/Ce)*delta^2*sqrt(S:S) = (Cs*delta)^2*|S|,  |S| = sqrt(2*S:S),  Cs = sqrt(Ck*sqrt(Ck/Ce)).
// With the OF defaults Ck=0.094, Ce=1.048 this gives Cs ~ 0.168 (the standard Smagorinsky constant).
#include "cf_types.cuh"
#include <cmath>

namespace brae {

struct SmagorinskyCoeffs {
    scalar Ck = 0.094;   // OF Smagorinsky default (sub-grid kinetic-energy coefficient)
    scalar Ce = 1.048;   // OF sub-grid dissipation coefficient
    // Equivalent classic Smagorinsky constant Cs (nut = (Cs*delta)^2*|S|); ~0.168 for the OF defaults. Host-only
    // (diagnostics/printf); the kernel works directly in Ck/Ce to stay byte-identical to OF's Smagorinsky::k().
    scalar Cs() const { return std::sqrt(Ck * std::sqrt(Ck / Ce)); }
};

// WALE (Nicoud & Ducros) LES sub-grid model coefficients, OF LESModels::WALE defaults. Same family as
// Smagorinsky -- an ALGEBRAIC nut with no transport equation -- but a different velocity scale, built from
// the traceless symmetric part of the SQUARE of the velocity gradient rather than from the strain alone:
//     Sd = devSymm(gradU & gradU)
//     k  = (Cw^2*delta/Ck)^2 * |Sd|^6 / ( (|symm(gradU)|^5 + |Sd|^(5/2))^2 + SMALL )
//     nut = Ck*delta*sqrt(k)
// which collapses to the textbook form nut = (Cw*delta)^2 * (Sd:Sd)^(3/2) / ((S:S)^(5/2) + (Sd:Sd)^(5/4)).
// Its point is the near-wall limit: Sd:Sd vanishes as y^6 in a laminar shear layer where S:S does not, so
// nut ~ y^3 without any damping function -- which is why channel LES cases reach for it over Smagorinsky.
struct WaleCoeffs {
    scalar Ck = 0.094;   // OF WALE default
    scalar Ce = 1.048;   // sub-grid dissipation coefficient (epsilon only; nut does not use it)
    scalar Cw = 0.325;   // OF WALE default
};

} // namespace brae
