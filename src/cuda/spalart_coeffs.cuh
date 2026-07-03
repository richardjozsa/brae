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
    BRAE_HD scalar Cw1() const { return Cb1 / (kappa * kappa) + (scalar(1) + Cb2) / sigmaNut; }
};

} // namespace brae
