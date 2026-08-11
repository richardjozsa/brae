// Gate 1 for the rhoSimpleFoam work: the sensible-enthalpy equation on a frozen velocity field.
//
// The plan called for comparing against OpenFOAM's scalarTransportFoam. This checks against CLOSED-FORM
// solutions instead, which is a stronger oracle: another code carries its own discretisation error, an
// analytic solution carries none. It also makes the gate self-contained -- no OF install, no reference
// data to regenerate, runs anywhere ctest runs.
//
// Three cases, chosen so the first two are exact and the third is physics:
//
//   A  pure diffusion, he fixed at both ends      -> profile is exactly linear      (machine precision)
//   B  pure advection, he fixed in / zeroGradient -> profile is exactly uniform     (machine precision)
//   C  advection + diffusion                      -> exponential, vs the analytic solution
//
// A pins the laplacian assembly, the boundary handling and the he<->T conversion. B pins the divergence
// assembly: if div(phi,he) is wrong, convecting a constant stops giving a constant. C only then has to
// confirm the two combine, which is why its tolerance can be loose without weakening the gate.

#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvc.cuh"
#include "parallel_simple.cuh"
#include "field_distribute.cuh"
#include "box_mesh.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "device_energy.cuh"
#include "thermo_types.cuh"
#include "thermo_model.cuh"
#include "device_thermo.cuh"
#include "foam_dict.cuh"
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>

using namespace brae;

namespace {

int failures = 0;

void report(
    const char* what,
    scalar worst,
    scalar tol)
{
    const bool ok = (worst <= tol);
    std::printf("  %-38s worst %.3e  tol %.1e  %s\n", what, worst, tol, ok ? "OK" : "FAIL");
    if (!ok) failures++;
}

// Solve he to convergence on a frozen phi, then hand back T.
std::vector<scalar> solveT(
    const std::string& outletType,
    scalar u,
    scalar alphaLam,
    scalar TinK,
    scalar ToutK,
    label Nx,
    label Ny,
    label Nz,
    bool linearUpwind)
{
    ThermoCoeffs c;
    c.Cp = 1005.0;
    c.Hf = 0.0;

    const PrimitiveMesh gm = boxtest::boxMesh(Nx, Ny, Nz);
    const label nC = Nx * Ny * Nz;
    const std::vector<label> cellToPart(nC, 0);
    const Partition P(gm, cellToPart, 0);
    const FvGeometry& lg = P.lg;
    const std::vector<FvPatch>& lp = P.lp;

    // frozen velocity: uniform u in x, no-slip on the side walls
    FieldData<vector> Ufd;
    Ufd.internalUniform = true;
    Ufd.internalUniformValue = vector{u, 0, 0};
    Ufd.boundary.push_back(boxtest::pfd<vector>("inlet", "fixedValue", vector{u, 0, 0}, true));
    Ufd.boundary.push_back(boxtest::pfd<vector>("outlet", "zeroGradient", vector{0, 0, 0}, false));
    for (const char* w : {"wallYmin", "wallYmax", "wallZmin", "wallZmax"})
    {
        Ufd.boundary.push_back(boxtest::pfd<vector>(w, "fixedValue", vector{u, 0, 0}, true));
    }
    GeometricField<vector> U = distributeField<vector>(Ufd, gm.patches(), P.Lm, lp, P.procW, 0);
    U.evaluateBoundary();
    const SurfaceScalarField phi = fvc::flux(U, P.Lm.mesh, lg, lp);

    // temperature: hot inlet, side walls adiabatic so the problem stays 1-D in x
    FieldData<scalar> Tfd;
    Tfd.internalUniform = true;
    Tfd.internalUniformValue = TinK;
    Tfd.boundary.push_back(boxtest::pfd<scalar>("inlet", "fixedValue", TinK, true));
    Tfd.boundary.push_back(boxtest::pfd<scalar>("outlet", outletType.c_str(), ToutK, outletType == "fixedValue"));
    for (const char* w : {"wallYmin", "wallYmax", "wallZmin", "wallZmax"})
    {
        Tfd.boundary.push_back(boxtest::pfd<scalar>(w, "zeroGradient", 0.0, false));
    }
    GeometricField<scalar> Tf = distributeField<scalar>(Tfd, gm.patches(), P.Lm, lp, P.procW, 0);
    Tf.evaluateBoundary();

    const DeviceBoundary dbT = buildDeviceBoundary(Tf, lp, lg);
    DeviceMesh dm = buildDeviceMesh(P.Lm.mesh, P.lg, lp);

    // he boundary shares the T boundary's structure; only the values are converted
    DeviceBoundary dbHe = buildDeviceBoundary(Tf, lp, lg);
    deviceEnergyBoundaryFromT(dbT, c, dbHe);

    DeviceThermo th;
    th.allocate(static_cast<int>(nC));
    th.T.copyFrom(std::vector<scalar>(nC, TinK));
    deviceThermoHeFromT(th, c);
    th.alpha.copyFrom(std::vector<scalar>(nC, alphaLam));
    th.alphat.copyFrom(std::vector<scalar>(nC, 0.0));

    const label nIf = P.Lm.mesh.nInternalFaces();
    DeviceBuffer<scalar> phiInt;
    DeviceBuffer<scalar> phiBnd;
    {
        std::vector<scalar> pi(phi.internal.begin(), phi.internal.begin() + nIf);
        phiInt.copyFrom(pi);
    }
    {
        std::vector<scalar> pb;
        for (std::size_t p = 0; p < lp.size(); ++p)
        {
            if (lp[p].type == "cyclic" || lp[p].type == "cyclicAMI") continue;
            for (label i = 0; i < lp[p].size; ++i) pb.push_back(phi.boundary[p][i]);
        }
        phiBnd.copyFrom(pb);
    }
    DeviceBuffer<scalar> divU(std::vector<scalar>(nC, 0.0));

    for (int it = 0; it < 400; ++it)
    {
        deviceSolveEnergy(
            dm,
            dbHe,
            th,
            c,
            phiInt,
            phiBnd,
            divU,
            false,   // bounded
            false,
            linearUpwind,
            false,
            0.0,
            1.0,
            1e-14,
            0.0,
            1,
            false);
    }

    std::vector<scalar> he(nC);
    th.he.copyTo(he);
    std::vector<scalar> T(nC);
    for (label i = 0; i < nC; ++i) T[i] = hConstHeToT(he[i], c);
    return T;
}

// Worst relative deviation of the x-profile from f(x), sampled at cell centres x = i + 0.5.
scalar worstVs(
    const std::vector<scalar>& T,
    label Nx,
    label Ny,
    label Nz,
    scalar (*f)(scalar))
{
    scalar worst = 0.0;
    for (label k = 0; k < Nz; ++k)
    {
        for (label j = 0; j < Ny; ++j)
        {
            for (label i = 0; i < Nx; ++i)
            {
                const scalar want = f(static_cast<scalar>(i) + 0.5);
                const scalar got = T[i + Nx * (j + Ny * k)];
                const scalar rel = std::abs(got - want) / std::abs(want);
                if (rel > worst) worst = rel;
            }
        }
    }
    return worst;
}

scalar gT0 = 300.0;
scalar gT1 = 400.0;
scalar gL = 40.0;
scalar gPe = 0.0;

scalar linearProfile(scalar x)
{
    return gT0 + (gT1 - gT0) * (x / gL);
}

scalar uniformProfile(scalar)
{
    return gT0;
}

scalar exponentialProfile(scalar x)
{
    return gT0 + (gT1 - gT0) * (std::exp(gPe * x / gL) - 1.0) / (std::exp(gPe) - 1.0);
}

}   // namespace

// OpenFOAM comparison. scalarTransportFoam solves div(phi,T) - laplacian(DT,T) = 0; brae solves
// div(phi,he) - laplacian(alphaEff,he) = 0. With hConst, he = Cp*T + Hf is linear, so dividing the
// second by Cp gives the first exactly when alphaEff == DT. That equivalence is what makes OF a valid
// oracle for the energy equation before any compressible coupling exists -- and it is why the case sets
// alpha = DT and alphat = 0.
int compareAgainstOpenFoam(const std::string& caseDir, const std::string& ofTimeDir)
{
    ThermoCoeffs c;
    c.Cp = 1005.0;
    c.Hf = 0.0;

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();

    const FoamDict transport = readDict(caseDir + "/constant/transportProperties");
    const scalar DT = transport.scalarOr("DT", 0.0);
    if (DT <= 0.0)
    {
        std::printf("  FAIL could not read DT from transportProperties\n");
        return 1;
    }

    GeometricField<vector> U = buildField<vector>(readField<vector>(caseDir + "/0/U"), fvp, nC);
    U.evaluateBoundary();
    const SurfaceScalarField phi = fvc::flux(U, m, g, fvp);

    GeometricField<scalar> Tf = buildField<scalar>(readField<scalar>(caseDir + "/0/T"), fvp, nC);
    Tf.evaluateBoundary();

    const DeviceBoundary dbT = buildDeviceBoundary(Tf, fvp, g);
    DeviceBoundary dbHe = buildDeviceBoundary(Tf, fvp, g);
    deviceEnergyBoundaryFromT(dbT, c, dbHe);
    DeviceMesh dm = buildDeviceMesh(m, g, fvp);

    DeviceThermo th;
    th.allocate(static_cast<int>(nC));
    th.T.copyFrom(Tf.internal);
    deviceThermoHeFromT(th, c);
    th.alpha.copyFrom(std::vector<scalar>(nC, DT));
    th.alphat.copyFrom(std::vector<scalar>(nC, 0.0));

    const label nIf = m.nInternalFaces();
    DeviceBuffer<scalar> phiInt;
    DeviceBuffer<scalar> phiBnd;
    {
        std::vector<scalar> pi(phi.internal.begin(), phi.internal.begin() + nIf);
        phiInt.copyFrom(pi);
    }
    {
        std::vector<scalar> pb;
        for (std::size_t p = 0; p < fvp.size(); ++p)
        {
            if (fvp[p].type == "cyclic" || fvp[p].type == "cyclicAMI") continue;
            for (label i = 0; i < fvp[p].size; ++i) pb.push_back(phi.boundary[p][i]);
        }
        phiBnd.copyFrom(pb);
    }
    DeviceBuffer<scalar> divU(std::vector<scalar>(nC, 0.0));

    for (int it = 0; it < 600; ++it)
    {
        deviceSolveEnergy(
            dm, dbHe, th, c, phiInt, phiBnd, divU,
            false, false, true, false, 0.0, 1.0, 1e-14, 0.0, 1, false);
    }

    std::vector<scalar> he(nC);
    th.he.copyTo(he);

    GeometricField<scalar> ofT = buildField<scalar>(readField<scalar>(ofTimeDir + "/T"), fvp, nC);
    if (static_cast<label>(ofT.internal.size()) != nC)
    {
        std::printf("  FAIL OF field has %zu cells, mesh has %ld\n", ofT.internal.size(), (long)nC);
        return 1;
    }

    scalar num = 0.0;
    scalar den = 0.0;
    for (label i = 0; i < nC; ++i)
    {
        const scalar mine = hConstHeToT(he[i], c);
        const scalar theirs = ofT.internal[i];
        num += (mine - theirs) * (mine - theirs);
        den += theirs * theirs;
    }
    const scalar l2 = std::sqrt(num / den);
    report("OF scalarTransportFoam, L2 rel", l2, 1e-6);
    return 0;
}

int main(int argc, char** argv)
{
    // Primary gate: against OpenFOAM. The analytic cases below are supporting checks, not the oracle.
    if (argc > 2)
    {
        std::printf("energy_frozen: comparing against OpenFOAM\n");
        compareAgainstOpenFoam(argv[1], argv[2]);
        std::printf("energy_frozen: %d failures\n", failures);
        return failures == 0 ? 0 : 1;
    }

    const label Nx = 40;
    const label Ny = 2;
    const label Nz = 2;
    gL = static_cast<scalar>(Nx);

    std::printf("energy_frozen: %ldx%ldx%ld box, frozen phi\n",
                (long)Nx, (long)Ny, (long)Nz);

    // A: pure diffusion. u = 0, fixed at both ends -> exactly linear.
    {
        const std::vector<scalar> T = solveT("fixedValue", 0.0, 1.0, gT0, gT1, Nx, Ny, Nz, false);
        report("A pure diffusion, linear profile", worstVs(T, Nx, Ny, Nz, &linearProfile), 1e-10);
    }

    // B: pure advection. Convecting a constant must give back that constant.
    {
        const std::vector<scalar> T = solveT("zeroGradient", 1.0, 1e-12, gT0, gT0, Nx, Ny, Nz, false);
        report("B pure advection, uniform profile", worstVs(T, Nx, Ny, Nz, &uniformProfile), 1e-10);
    }

    // C: advection + diffusion against the analytic exponential. Upwind is first order, so the
    // tolerance is a discretisation bound, not a correctness one -- A and B already pin correctness.
    {
        const scalar u = 1.0;
        const scalar alphaLam = 20.0;
        gPe = u * gL / alphaLam;
        const std::vector<scalar> T = solveT("fixedValue", u, alphaLam, gT0, gT1, Nx, Ny, Nz, true);
        // linearUpwind lands at ~9e-5 on this mesh. 5e-4 leaves headroom for the scheme without
        // being so loose that a real regression slips through -- a gate that cannot fail is not a gate.
        report("C advection-diffusion vs analytic", worstVs(T, Nx, Ny, Nz, &exponentialProfile), 5e-4);
    }

    std::printf("energy_frozen: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
