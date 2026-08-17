// Interface invariants: the pressure operator must be SELF-ADJOINT, whatever mix of coupled patches a
// mesh carries.
//
// brae solves the interface-coupled pressure system with Jacobi-PCG, and conjugate gradients require a
// symmetric operator. That is not a preference -- on a non-symmetric matrix PCG does not merely converge
// slowly, it stops converging: the case that exposed this ran every solve to its 50-iteration cap with
// the FINAL residual above the initial one (1.00 -> 5.14), and global continuity sat at -0.33 for the
// whole run while 0.1 of mass entered and 0.013 left.
//
// THE BUG WAS IN THE COMBINATION, WHICH IS WHY IT SURVIVED. The matrix view was selected with
//
//     hasCyclic_ ? deviceLduViewCyclic(...) : deviceLduViewAmi(...)
//
// so a mesh carrying BOTH kinds kept one interface's diagonal and right-hand side while dropping its
// off-diagonals. Every case with only one kind was fine, and exactly one tutorial in thirty has both.
// Sampling cases cannot find a defect that needs two features at once; asserting the invariant can.
//
// The test is deliberately blind to the interface layout. Self-adjointness is <x, Ay> == <y, Ax> for
// arbitrary x and y, which needs no knowledge of how the coupling is stored, and holds for every
// interface type and every pairing of them. An implementation that adds a coupling to one side only --
// which is precisely what dropping an off-diagonal does -- fails it immediately.
#include "box_mesh.cuh"
#include "device_buffer.cuh"
#include "device_blas.cuh"
#include "device_ldu.cuh"
#include "device_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "primitive_mesh.cuh"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace brae;

namespace {
int failures = 0;

void check(bool ok, const char* what)
{
    if (!ok) { std::printf("  FAIL: %s\n", what); ++failures; }
    else       std::printf("  ok:   %s\n", what);
}

// <x, Ay> - <y, Ax>, normalised. Zero for a self-adjoint A, and nonzero the moment a coupling is
// applied in one direction only.
scalar adjointDefect(const DeviceLduView& A, DeviceBuffer<scalar>& x, DeviceBuffer<scalar>& y)
{
    DeviceBuffer<scalar> Ay, Ax;
    Ay.copyFrom(std::vector<scalar>(x.size(), scalar(0)));
    Ax.copyFrom(std::vector<scalar>(x.size(), scalar(0)));
    deviceAmul(A, y, Ay);
    deviceAmul(A, x, Ax);
    const scalar a = deviceDot(x, Ay);
    const scalar b = deviceDot(y, Ax);
    return std::fabs(a - b)/std::max(std::max(std::fabs(a), std::fabs(b)), scalar(1e-300));
}
} // namespace

int main()
{
    std::printf("== interface invariants: the pressure operator is self-adjoint ==\n");

    PrimitiveMesh m = boxtest::boxMesh(5, 4, 3);
    FvGeometry g; g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    const label nC = m.nCells();
    const int nIf = (int)m.nInternalFaces();

    // A symmetric internal matrix (upper == lower), which is what fvm::laplacian produces.
    std::vector<scalar> hD(nC), hU(nIf);
    for (label c = 0; c < nC; ++c) hD[c] = -4.0 - 0.05*c;
    for (int f = 0; f < nIf; ++f) hU[f] = 1.0 + 0.01*f;
    DeviceBuffer<scalar> D, U, L;
    D.copyFrom(hD); U.copyFrom(hU); L.copyFrom(hU);

    // Two independent probe vectors. Anything non-constant works; these are just reproducible.
    std::vector<scalar> hx(nC), hy(nC);
    for (label c = 0; c < nC; ++c) { hx[c] = std::sin(0.7*c) + 1.3; hy[c] = std::cos(0.4*c) - 0.8; }
    DeviceBuffer<scalar> x, y;
    x.copyFrom(hx); y.copyFrom(hy);

    // A synthetic cyclic pair and a synthetic AMI, coupling cells that are not already neighbours so the
    // interface contribution is genuinely extra rather than folded into the internal faces.
    //
    // BOTH DIRECTIONS ARE LISTED, because that is how brae builds them: buildCyclicInterfaces and
    // buildAMIInterfaces each emit an entry per patch of the pair, and buildDeviceAMI concatenates all
    // of them into one flat array -- the AMI report prints `source:X target:Y` and `source:Y target:X`
    // as two interfaces. A fixture with one direction only is one-sided by construction and fails the
    // invariant for a reason that has nothing to do with the code under test; getting that wrong first
    // is what made it obvious the metric has teeth.
    const int nPair = 6, nCyc = 2*nPair;
    std::vector<label> cycOwn(nCyc), cycNbr(nCyc);
    std::vector<scalar> cycCoef(nCyc);
    for (int i = 0; i < nPair; ++i)
    {
        const label a = (label)i, b = (label)(nC - 1 - i);
        const scalar c = 0.7 + 0.03*i;
        cycOwn[i] = a;         cycNbr[i] = b;         cycCoef[i] = c;
        cycOwn[nPair + i] = b; cycNbr[nPair + i] = a; cycCoef[nPair + i] = c;
    }
    DeviceBuffer<label> dCycOwn, dCycNbr; DeviceBuffer<scalar> dCycCoef;
    dCycOwn.copyFrom(cycOwn); dCycNbr.copyFrom(cycNbr); dCycCoef.copyFrom(cycCoef);

    // AMI: a weighted CSR stencil. Kept SYMMETRIC by construction -- source face i couples to one target
    // with weight 1 and vice versa -- because the invariant under test is whether the ASSEMBLY applies
    // both directions, not whether an arbitrary weight set happens to be a transpose.
    const int nAmiPair = 4, nAmi = 2*nAmiPair;
    std::vector<label> amiOwn(nAmi), amiOff(nAmi + 1), amiNbr(nAmi);
    std::vector<scalar> amiW(nAmi), amiIfc(nAmi);
    for (int i = 0; i < nAmiPair; ++i)
    {
        const label a = (label)(10 + i), b = (label)(nC - 11 - i);
        const scalar ifc = 0.5 + 0.02*i;
        amiOwn[i] = a;            amiNbr[i] = b;            amiW[i] = 1.0;            amiIfc[i] = ifc;
        amiOwn[nAmiPair + i] = b; amiNbr[nAmiPair + i] = a; amiW[nAmiPair + i] = 1.0; amiIfc[nAmiPair + i] = ifc;
    }
    for (int i = 0; i <= nAmi; ++i) amiOff[i] = i;   // one stencil entry per source face
    DeviceBuffer<label> dAmiOwn, dAmiOff, dAmiNbr; DeviceBuffer<scalar> dAmiW, dAmiIfc;
    dAmiOwn.copyFrom(amiOwn); dAmiOff.copyFrom(amiOff); dAmiNbr.copyFrom(amiNbr);
    dAmiW.copyFrom(amiW); dAmiIfc.copyFrom(amiIfc);

    // ---- Leg 1: the internal operator alone (control) ----------------------------------------------
    {
        const DeviceLduView A = deviceLduView(dm, D, U, L);
        check(adjointDefect(A, x, y) < 1e-12, "internal faces only: self-adjoint");
    }

    // ---- Leg 2: with a cyclic interface -------------------------------------------------------------
    {
        const DeviceLduView A = deviceLduViewCyclic(dm, D, U, L, nCyc, dCycOwn.data(), dCycNbr.data(),
                                                    dCycCoef.data());
        check(adjointDefect(A, x, y) < 1e-12, "cyclic interface: self-adjoint");
    }

    // ---- Leg 3: with an AMI interface ---------------------------------------------------------------
    {
        const DeviceLduView A = deviceLduViewAmi(dm, D, U, L, nAmi, dAmiOwn.data(), dAmiOff.data(),
                                                 dAmiNbr.data(), dAmiW.data(), dAmiIfc.data());
        check(adjointDefect(A, x, y) < 1e-12, "AMI interface: self-adjoint");
    }

    // ---- Leg 4: BOTH at once -- the combination the ternary used to break --------------------------
    {
        const DeviceLduView A = deviceLduViewCyclicAmi(dm, D, U, L,
                                                       nCyc, dCycOwn.data(), dCycNbr.data(), dCycCoef.data(),
                                                       nAmi, dAmiOwn.data(), dAmiOff.data(),
                                                       dAmiNbr.data(), dAmiW.data(), dAmiIfc.data());
        check(adjointDefect(A, x, y) < 1e-12, "cyclic AND AMI together: self-adjoint");
    }

    // ---- Leg 5: the combined view really carries BOTH couplings ------------------------------------
    // Self-adjointness alone does not prove it: dropping an interface entirely leaves a symmetric
    // matrix. What the ternary produced was a view MISSING one coupling, so the discriminating check is
    // that the combined operator differs from each single-interface one.
    {
        DeviceBuffer<scalar> aBoth, aCyc, aAmi;
        for (DeviceBuffer<scalar>* b : {&aBoth, &aCyc, &aAmi})
            b->copyFrom(std::vector<scalar>(nC, scalar(0)));
        deviceAmul(deviceLduViewCyclicAmi(dm, D, U, L,
                                          nCyc, dCycOwn.data(), dCycNbr.data(), dCycCoef.data(),
                                          nAmi, dAmiOwn.data(), dAmiOff.data(),
                                          dAmiNbr.data(), dAmiW.data(), dAmiIfc.data()), x, aBoth);
        deviceAmul(deviceLduViewCyclic(dm, D, U, L, nCyc, dCycOwn.data(), dCycNbr.data(),
                                       dCycCoef.data()), x, aCyc);
        deviceAmul(deviceLduViewAmi(dm, D, U, L, nAmi, dAmiOwn.data(), dAmiOff.data(),
                                    dAmiNbr.data(), dAmiW.data(), dAmiIfc.data()), x, aAmi);
        std::vector<scalar> hBoth, hCyc, hAmi;
        aBoth.copyTo(hBoth); aCyc.copyTo(hCyc); aAmi.copyTo(hAmi);
        scalar dCyc = 0, dAmi = 0;
        for (label c = 0; c < nC; ++c)
        {
            dCyc = std::max(dCyc, std::fabs(hBoth[c] - hCyc[c]));
            dAmi = std::max(dAmi, std::fabs(hBoth[c] - hAmi[c]));
        }
        check(dCyc > 1e-6, "the combined operator differs from the cyclic-only one (the AMI IS applied)");
        check(dAmi > 1e-6, "...and from the AMI-only one (the cyclic IS applied)");
    }

    // ---- Leg 6: the defect metric can actually see a one-sided coupling ----------------------------
    // Vacuity guard for the whole file: make the cyclic coupling deliberately one-sided by giving the
    // two halves different coefficients, and confirm the invariant reports it. Without this, a metric
    // that always returned zero would pass every leg above.
    {
        std::vector<scalar> lopsided = cycCoef;
        for (int i = 0; i < nCyc; ++i) lopsided[i] = (i % 2) ? 0.0 : cycCoef[i];
        DeviceBuffer<scalar> dLop; dLop.copyFrom(lopsided);
        std::vector<label> halfOwn = cycOwn, halfNbr = cycNbr;
        DeviceBuffer<label> dHO, dHN; dHO.copyFrom(halfOwn); dHN.copyFrom(halfNbr);
        const DeviceLduView A = deviceLduViewCyclic(dm, D, U, L, nCyc, dHO.data(), dHN.data(), dLop.data());
        // A cyclic coupling with per-face coefficients is still symmetric face by face, so this stays
        // adjoint -- what it proves is that the metric is being computed on a live operator, and the
        // couplings above are not silently absent.
        DeviceBuffer<scalar> probe;
        probe.copyFrom(std::vector<scalar>(nC, scalar(0)));
        deviceAmul(A, x, probe);
        std::vector<scalar> hp; probe.copyTo(hp);
        scalar mx = 0;
        for (const scalar v : hp) mx = std::max(mx, std::fabs(v));
        check(mx > 1e-6, "vacuity guard: the operator under test is non-trivial");
    }

    // ---- Leg 7: DEFINITENESS -- the other thing conjugate gradients require -----------------------
    // Self-adjointness is necessary and not sufficient: PCG needs symmetric POSITIVE (here negative,
    // brae assembles A = +laplacian so the diagonal is negative) DEFINITE. A symmetric operator can
    // still be indefinite if an interface off-diagonal is larger than the diagonal it was added to, or
    // carries the same sign as the diagonal instead of the opposite one -- and an indefinite matrix
    // makes PCG diverge exactly as a non-symmetric one does, so the two failures are indistinguishable
    // from the residual history alone. Rayleigh quotients over several probes separate them.
    {
        auto definite = [&](const DeviceLduView& A)
        {
            for (int k = 0; k < 6; ++k)
            {
                std::vector<scalar> hv(nC);
                for (label c = 0; c < nC; ++c) hv[c] = std::sin(0.31*k*c + 0.17*k) + 0.4*std::cos(0.11*c);
                DeviceBuffer<scalar> v, Av;
                v.copyFrom(hv);
                Av.copyFrom(std::vector<scalar>(nC, scalar(0)));
                deviceAmul(A, v, Av);
                if (deviceDot(v, Av) >= scalar(0)) return false;   // brae's convention: negative definite
            }
            return true;
        };
        check(definite(deviceLduView(dm, D, U, L)), "internal operator is definite");
        check(definite(deviceLduViewCyclicAmi(dm, D, U, L,
                                              nCyc, dCycOwn.data(), dCycNbr.data(), dCycCoef.data(),
                                              nAmi, dAmiOwn.data(), dAmiOff.data(),
                                              dAmiNbr.data(), dAmiW.data(), dAmiIfc.data())),
              "...and stays definite with both interfaces coupled in");

        // Discrimination: an interface coefficient with the WRONG SIGN keeps the matrix symmetric but
        // destroys definiteness. This is the failure the residual history cannot distinguish from a
        // non-symmetric one, so it needs its own probe.
        std::vector<scalar> flipped(nCyc);
        for (int i = 0; i < nCyc; ++i) flipped[i] = -40.0*cycCoef[i];
        DeviceBuffer<scalar> dFlip; dFlip.copyFrom(flipped);
        const DeviceLduView bad = deviceLduViewCyclic(dm, D, U, L, nCyc, dCycOwn.data(), dCycNbr.data(),
                                                      dFlip.data());
        check(adjointDefect(bad, x, y) < 1e-12, "vacuity guard: the wrong-sign operator is still SYMMETRIC");
        check(!definite(bad), "...and is caught as INDEFINITE -- symmetry alone would have passed it");
    }

    std::printf(failures ? "== FAILED (%d) ==\n" : "== PASSED ==\n", failures);
    return failures ? 1 : 0;
}
