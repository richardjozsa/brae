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

} // namespace brae
