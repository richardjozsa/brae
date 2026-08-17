#pragma once
// Spalart-Allmaras model coefficients (OF SpalartAllmarasBase defaults). Cw1 is derived. The wall E is the
// nutUSpaldingWallFunction coefficient (wallCoeffs default E=9.8, kappa shared). POD: passed to kernels by value.
#include "cf_types.cuh"

namespace brae {

struct SpalartAllmarasCoeffs {
    scalar sigmaNut = 0.66666;   // 2/3
    scalar kappa    = 0.41;
    scalar Cb1      = 0.1355;
    scalar Cb2      = 0.622;
    scalar Cw2      = 0.3;
    scalar Cw3      = 2.0;
    scalar Cv1      = 7.1;
    scalar Cs       = 0.3;
    scalar E        = 9.8;       // nutUSpaldingWallFunction wall E (Spalding law)
    scalar CDES     = 0.65;      // SA-DES/DDES/IDDES model constant (OF SpalartAllmarasDES default)
    // SA-IDDES (SpalartAllmarasIDDES, Shur/Spalart/Strelets/Travin 2008) blending constants. Exponents are fixed per
    // the reference (f_dt cube, f_l ^10, f_t cube); only the multipliers are carried here.
    scalar Cdt1     = 20.0;      // f_dt = 1 - tanh((Cdt1*rd_t)^3)
    scalar Cl       = 5.0;       // f_l  = tanh((Cl^2*rd_l)^10)
    scalar Ct       = 1.87;      // f_t  = tanh((Ct^2*rd_t)^3)
    scalar Cw       = 0.15;      // IDDES length scale delta = min(max(Cw*y, Cw*hmax), hmax)
    scalar fwStar   = 0.424;     // low-Re DES correction Psi: f_w in the log layer (Spalart et al. 2006)
    // DDES shielding function fd = 1 - tanh((Cd1*r)^Cd2). OF exposes Cd1/Cd2 as dictionary entries
    // (SpalartAllmarasDDES.C); the defaults are the values the standard formulation has always used.
    scalar Cd1      = 8.0;
    scalar Cd2      = 3.0;
    // ZDES2020 (Deck & Renard 2020, OF `shielding ZDES2020`): a SECOND shielding built from the
    // wall-normal derivatives of nuTilda and of |curl U|, which protects the boundary layer where the
    // standard fd would already have released it. Off unless the case asks for it.
    bool   zdes     = false;
    scalar Cd3      = 25.0;      // GnuTilda = Cd3*max(grad(nuTilda).n, 0)/(|gradU| kappa y)
    scalar Cd4      = 0.03;      // the GOmega thresholds of fR(GOmega)
    scalar betaZDES = 2.5;       // only with usefP2
    bool   usefP2   = false;     // OF Switch, default false: the more conservative fP2 form
    BRAE_HD scalar Cw1() const { return Cb1 / (kappa * kappa) + (scalar(1) + Cb2) / sigmaNut; }
};

} // namespace brae
