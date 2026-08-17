// WALE's sub-grid viscosity, and the property that makes it worth having.
//
// OF LESModels::WALE::correctNut():
//     Sd  = devSymm(gradU & gradU)
//     k   = sqr(sqr(Cw)*delta/Ck) * pow3(magSqr(Sd))
//           / ( sqr( pow(magSqr(symm(gradU)), 5/2) + pow(magSqr(Sd), 5/4) ) + SMALL )
//     nut = Ck*delta*sqrt(k)
// which collapses to  nut = (Cw*delta)^2 * (Sd:Sd)^(3/2) / ( (S:S)^(5/2) + (Sd:Sd)^(5/4) ).
//
// LEG 1 IS THE WHOLE POINT OF THE MODEL. In PURE SHEAR, gradU & gradU is identically zero -- the only
// non-zero entry of the gradient cannot pair with itself in the matrix product -- so Sd = 0 and nut = 0.
// Smagorinsky gives a large nut on the same field. That is why WALE needs no van Driest damping: near a
// wall the flow is shear-dominated, Sd:Sd dies as y^6 where S:S does not, and nut ~ y^3 falls out. A
// transcription that fumbled the tensor product would almost certainly still be positive-definite and
// plausible, and would fail here and nowhere else.
//
// Leg 3 re-derives the closed form on the host for a strain the model DOES see, so leg 1 cannot pass
// by the kernel simply returning zero.
#include "device_smagorinsky.cuh"
#include "device_buffer.cuh"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace brae;

namespace {
int failures = 0;

// gradU packed as brae/OF do: q = 3i + j = d(U_j)/d(x_i), component-major over cells.
std::vector<scalar> pack(const std::vector<std::vector<scalar>>& perCell)
{
    const int nC = (int)perCell.size();
    std::vector<scalar> g(9*nC);
    for (int c = 0; c < nC; ++c)
        for (int q = 0; q < 9; ++q) g[q*nC + c] = perCell[c][q];
    return g;
}

scalar waleHost(const std::vector<scalar>& t, scalar V, scalar Ck, scalar Cw)
{
    const scalar delta = std::cbrt(V);
    scalar gg[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            scalar s = 0;
            for (int k = 0; k < 3; ++k) s += t[i*3+k]*t[k*3+j];
            gg[i*3+j] = s;
        }
    const scalar trGG = gg[0] + gg[4] + gg[8];
    scalar sdsd = 0, ss = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            scalar sd = 0.5*(gg[i*3+j] + gg[j*3+i]);
            if (i == j) sd -= trGG/3.0;
            sdsd += sd*sd;
            const scalar s = 0.5*(t[i*3+j] + t[j*3+i]);
            ss += s*s;
        }
    const scalar A = Cw*Cw*delta/Ck;
    const scalar den = std::pow(ss, 2.5) + std::pow(sdsd, 1.25);
    const scalar k = A*A*(sdsd*sdsd*sdsd)/(den*den + 1e-15);
    return Ck*delta*std::sqrt(std::fmax(k, 0.0));
}
}   // namespace

int main()
{
    const WaleCoeffs co;
    const SmagorinskyCoeffs sco;
    const scalar V = 8.0;   // delta = 2

    // cell 0: PURE SHEAR   d(Ux)/dy = 10   -> t[3] (i=1, j=0)
    // cell 1: a diagonal, traceless strain (incompressible) that WALE does see
    // cell 2: zero gradient -> nut must be exactly 0, no NaN out of the 0/0
    std::vector<std::vector<scalar>> cells(3, std::vector<scalar>(9, 0.0));
    cells[0][3] = 10.0;
    cells[1][0] = 3.0; cells[1][4] = -1.0; cells[1][8] = -2.0;
    const int nC = (int)cells.size();

    DeviceBuffer<scalar> gradU, vol, nut;
    gradU.copyFrom(pack(cells));
    vol.copyFrom(std::vector<scalar>(nC, V));
    deviceWaleNut(nC, gradU, vol, co, nut);
    const std::vector<scalar> got = nut.host();

    // ---- 1. pure shear -> exactly zero ----
    std::printf("  pure shear (dUx/dy = 10): WALE nut = %.6e\n", (double)got[0]);
    if (std::fabs(got[0]) > 1e-30)
    {
        std::printf("  FAIL WALE must give nut = 0 in pure shear -- gradU & gradU vanishes there, so\n"
                    "       Sd = devSymm(gradU & gradU) = 0. A non-zero value means the tensor product is\n"
                    "       not the matrix product (a transposed or element-wise one is still plausible-looking)\n");
        ++failures;
    }

    // ---- 2. ...and Smagorinsky is NOT zero on the same field, so leg 1 is a property, not a dead kernel
    {
        DeviceBuffer<scalar> snut;
        deviceSmagorinskyNut(nC, gradU, vol, sco, snut);
        const std::vector<scalar> s = snut.host();
        std::printf("  same field, Smagorinsky nut = %.6e (must be > 0)\n", (double)s[0]);
        if (!(s[0] > 1e-6))
        {
            std::printf("  FAIL vacuous: Smagorinsky is zero here too, so leg 1 says nothing about WALE --\n"
                        "       the input gradient is not actually exercising either model\n");
            ++failures;
        }
    }

    // ---- 3. a strain WALE does see: match the closed form, computed independently ----
    {
        const scalar want = waleHost(cells[1], V, co.Ck, co.Cw);
        std::printf("  diagonal strain (3,-1,-2): device %.9e  host %.9e\n", (double)got[1], (double)want);
        if (!(want > 1e-6))
        { std::printf("  FAIL vacuous: the reference value is ~0, so this leg compares nothing\n"); ++failures; }
        if (std::fabs(got[1] - want) > 1e-12*std::fmax(want, scalar(1)))
        { std::printf("  FAIL the kernel does not reproduce OF's closed form\n"); ++failures; }
    }

    // ---- 4. a zero gradient must give 0, not NaN (the SMALL in the denominator is what protects it) ----
    std::printf("  zero gradient: nut = %.6e\n", (double)got[2]);
    if (!std::isfinite((double)got[2]) || std::fabs(got[2]) > 1e-30)
    { std::printf("  FAIL a zero velocity gradient must give exactly zero, finite, nut\n"); ++failures; }

    std::printf("wale_nut: %d failures\n", failures);
    return failures ? 1 : 0;
}
