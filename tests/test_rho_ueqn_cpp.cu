// rhoSimpleFoam's UEqn.H against REAL OpenFOAM's own assembled momentum matrix.
//
// THE ORACLE is tools/dumpPEqn -- OpenFOAM's rhoSimpleFoam with a stage harness that writes, at a chosen
// SIMPLE iteration:
//
//   stage_rAU    1/UEqn.A()                       the matrix diagonal, AFTER relax()
//   stage_UIC    UEqn.internalCoeffs(), per patch  the boundary contribution to the diagonal
//   stage_UBC    UEqn.boundaryCoeffs(),  per patch the boundary contribution to the source
//   stage_muEff  turbulence->muEff()               the DYNAMIC viscosity the assembly actually used
//
// WHY stage_muEff MATTERS TO THE DESIGN OF THIS TEST. brae has no ported compressible turbulence model
// yet -- that is a separate manifest component. Injecting OpenFOAM's own muEff makes this a test of the
// ASSEMBLY and of nothing else: if it fails, the momentum equation is wrong, not the closure. Mixing the
// two would produce a number that cannot be attributed, which is the failure mode the whole manifest
// exists to avoid.
//
// WHAT IS ACTUALLY UNDER TEST. linearViscousStress.C defines one operator:
//
//     divDevRhoReff(U) = -fvc::div((alpha*rho*nuEff)*dev2(T(grad U))) - fvm::laplacian(alpha*rho*nuEff, U)
//
// and the compressible solver reaches it with the DYNAMIC viscosity mu_eff = rho*nu_eff where the
// incompressible one reaches it with the kinematic nu_eff. That single factor of rho is the entire
// difference between this file and simpleFoam's UEqn, it is order 1 for air at ambient conditions, and it
// varies with the solution -- so a port that reuses the incompressible form is wrong by an amount that
// looks plausible in every field plot. THE CONTROL below assembles the incompressible way on purpose and
// requires it to disagree with OpenFOAM. Without that, a passing bound would not distinguish the two.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_dict.cuh"
#include "fv_matrix_ops.cuh"
#include "createFields_cpp.cuh"
#include "UEqn_cpp.cuh"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

// The staged fields are OpenFOAM's own diagnostic output and carry whatever boundary type OpenFOAM gave
// them -- muEff comes out as `extrapolatedCalculated`. brae's patch-field factory has no such type, and
// teaching it one for the sake of reading a test fixture would be changing shared code to suit a test.
// These read the parsed file directly instead: the staged values are what is wanted, not a re-evaluated
// boundary condition.
static std::vector<scalar> rawInternal(
    const FieldData<scalar>& fd,
    label nC)
{
    if (fd.internalUniform) return std::vector<scalar>(nC, fd.internalUniformValue);
    return fd.internalField;
}

template <typename T>
static std::vector<std::vector<T>> rawBoundary(
    const FieldData<T>&         fd,
    const std::vector<FvPatch>& patches)
{
    std::vector<std::vector<T>> out(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        out[pi].assign(patches[pi].size, T{});
        for (const auto& b : fd.boundary)
        {
            if (b.name != patches[pi].name) continue;
            if (b.valueUniform)
            {
                out[pi].assign(patches[pi].size, b.uniformValue);
            }
            else if (static_cast<label>(b.values.size()) == patches[pi].size)
            {
                out[pi] = b.values;
            }
            break;
        }
    }
    return out;
}

static int failures = 0;

static void report(
    const std::string& what,
    double got,
    double bound)
{
    const bool ok = got < bound;
    if (!ok) ++failures;
    std::printf("     %-44s %.6e   %s\n", what.c_str(), got, ok ? "ok" : "FAIL");
}

static void check(
    const std::string& what,
    bool ok)
{
    if (!ok) ++failures;
    std::printf("     %-44s %s\n", what.c_str(), ok ? "ok" : "FAIL");
}

static double relL2(
    const std::vector<scalar>& a,
    const std::vector<scalar>& b)
{
    double num = 0.0, den = 0.0;
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i)
    {
        const double d = (double)a[i] - (double)b[i];
        num += d * d;
        den += (double)b[i] * (double)b[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

// Boundary coefficients, flattened patch by patch, so one number covers the whole boundary.
static double relL2Boundary(
    const std::vector<std::vector<vector>>& a,
    const std::vector<std::vector<vector>>& b,
    const std::vector<FvPatch>&             patches)
{
    std::vector<scalar> fa, fb;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const std::vector<vector>& bv = b[pi];
        for (label i = 0; i < patches[pi].size; ++i)
        {
            fa.push_back(a[pi][i].x); fa.push_back(a[pi][i].y); fa.push_back(a[pi][i].z);
            fb.push_back(bv[i].x);    fb.push_back(bv[i].y);    fb.push_back(bv[i].z);
        }
    }
    return relL2(fa, fb);
}

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::printf("usage: %s <caseDir> <startTime> <dumpTime>\n", argv[0]);
        return 2;
    }
    const std::string caseDir = argv[1];
    const std::string startT  = argv[2];
    const std::string dumpT   = argv[3];

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nC = m.nCells();

    const FoamDict fvSolution = readDict(caseDir + "/system/fvSolution");
    const FoamDict* simpleDict = fvSolution.subDict("SIMPLE");

    std::printf("rhoSimpleFoam UEqn vs OpenFOAM (%d cells)\n", (int)nC);

    // The state OpenFOAM assembled from: the start-time fields, and its own initial phi.
    cpu::rhoSimple::RhoSimpleFields f =
        cpu::rhoSimple::createFields(caseDir + "/" + startT, caseDir, simpleDict, &fvSolution,
                                     m, g, patches);

    // OpenFOAM's own muEff, so this measures the assembly and not an unported turbulence closure.
    const FieldData<scalar> muEffFd = readField<scalar>(caseDir + "/" + dumpT + "/stage_muEff");
    const std::vector<scalar> muEffInt = rawInternal(muEffFd, nC);
    const std::vector<std::vector<scalar>> muEffBnd = rawBoundary<scalar>(muEffFd, patches);

    // U EXACTLY as OpenFOAM assembled from. sbMatched's inlet is flowRateInletVelocity, whose value
    // depends on which rho it is fed -- a separate component with its own history in this repo. Taking
    // OpenFOAM's evaluated U here keeps this gate measuring the momentum ASSEMBLY; the BC gets its own.
    const FieldData<vector> uAssFd = readField<vector>(caseDir + "/" + dumpT + "/stage_Uass");
    const std::vector<std::vector<vector>> uAssBnd = rawBoundary<vector>(uAssFd, patches);
    {
        // Report how far brae's OWN boundary evaluation is from OpenFOAM's BEFORE overwriting it, so the
        // substitution is visible rather than silent. A measurement, not an assertion: the inlet BC is a
        // separate manifest component and is not what this gate is for.
        double d = 0.0, n = 0.0;
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            const std::vector<vector>& bv = f.U.boundary[pi]->value();
            for (label i = 0; i < patches[pi].size; ++i)
            {
                const vector& a = bv[i];
                const vector& b = uAssBnd[pi][i];
                d += (double)(a.x-b.x)*(a.x-b.x) + (double)(a.y-b.y)*(a.y-b.y) + (double)(a.z-b.z)*(a.z-b.z);
                n += (double)b.x*b.x + (double)b.y*b.y + (double)b.z*b.z;
            }
        }
        // With the gate's neutralised inlet this must be zero: both codes evaluate the same plain
        // fixedValue. It is asserted rather than printed, because if brae's boundary state differs from
        // OpenFOAM's then everything below is comparing two different problems.
        const double ub = n > 0.0 ? std::sqrt(d/n) : std::sqrt(d);
        report("brae's U boundary == OpenFOAM's", ub, 1e-14);
    }
    // Adopt OpenFOAM's evaluated U in place; GeometricField owns unique_ptr patch fields and cannot be
    // copied, and a copy is not wanted anyway -- the point is to assemble from OpenFOAM's exact input.
    if (!uAssFd.internalUniform && static_cast<label>(uAssFd.internalField.size()) == nC)
        f.U.internal = uAssFd.internalField;
    for (std::size_t pi = 0; pi < patches.size(); ++pi) f.U.boundary[pi]->setValue(uAssBnd[pi]);

    std::vector<std::vector<scalar>> rhoBnd(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi) rhoBnd[pi] = f.rho.boundary[pi]->value();

    // The case: `div(phi,U) bounded Gauss upwind`, `laplacianSchemes default Gauss linear corrected`,
    // relaxation U 0.9. Read rather than assumed where it is cheap to read.
    const FoamDict* eqnRelax = simpleDict ? nullptr : nullptr;
    const FoamDict* rf = fvSolution.subDict("relaxationFactors");
    const FoamDict* re = rf ? rf->subDict("equations") : nullptr;
    const scalar relaxU = re ? re->scalarOr("U", 1.0) : 1.0;
    (void)eqnRelax;
    std::printf("  relaxation U = %g\n", (double)relaxU);

    cpu::rhoSimple::RhoMomentumInput in;
    in.phi                = &f.phi.internal;
    in.phiBnd             = &f.phi.boundary;
    in.rho                = &f.rho.internal;
    in.rhoBnd             = &rhoBnd;
    in.muEff              = &muEffInt;
    in.muEffBnd           = &muEffBnd;
    in.relaxU             = relaxU;
    in.bounded            = true;
    in.scheme             = cpu::rhoSimple::DivScheme::upwind;
    in.correctedLaplacian = true;

    // ---- 1. The assembled momentum matrix, against OpenFOAM's. ----
    std::printf("  1. UEqn assembled with the COMPRESSIBLE divDevRhoReff (mu_eff = rho*nu_eff)\n");
    FvVectorMatrix M = cpu::rhoSimple::assembleUEqn(f.U, in, m, g, patches);

    // rAU = 1/A(). Compared as rAU rather than A so the number is the one pEqn.H actually consumes.
    const std::vector<scalar> A = matrixA<vector>(M, m, g, patches);
    std::vector<scalar> rAU(nC);
    for (label c = 0; c < nC; ++c) rAU[c] = 1.0 / A[c];
    const std::vector<scalar> ofRAU = rawInternal(readField<scalar>(caseDir + "/" + dumpT + "/stage_rAU"), nC);
    const double rauErr = relL2(rAU, ofRAU);
    report("rAU = 1/UEqn.A() vs OpenFOAM", rauErr, 1e-10);

    const std::vector<std::vector<vector>> ofUIC =
        rawBoundary<vector>(readField<vector>(caseDir + "/" + dumpT + "/stage_UIC"), patches);
    const std::vector<std::vector<vector>> ofUBC =
        rawBoundary<vector>(readField<vector>(caseDir + "/" + dumpT + "/stage_UBC"), patches);
    report("UEqn.internalCoeffs() vs OpenFOAM",
           relL2Boundary(M.internalCoeffs, ofUIC, patches), 1e-10);
    report("UEqn.boundaryCoeffs() vs OpenFOAM",
           relL2Boundary(M.boundaryCoeffs, ofUBC, patches), 1e-10);
    // Per-patch, so a failure names the patch and its BC rather than one number over the whole boundary.
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (!patches[pi].size) continue;
        std::vector<std::vector<vector>> a1(patches.size()), b1(patches.size());
        std::vector<std::vector<vector>> a2(patches.size()), b2(patches.size());
        std::vector<FvPatch> one(patches.size());
        for (std::size_t q = 0; q < patches.size(); ++q)
        {
            one[q] = patches[q];
            if (q != pi) one[q].size = 0;
        }
        std::printf("       %-18s iC %.3e   bC %.3e   (%d faces)\n",
                    patches[pi].name.c_str(),
                    relL2Boundary(M.internalCoeffs, ofUIC, one),
                    relL2Boundary(M.boundaryCoeffs, ofUBC, one),
                    (int)patches[pi].size);
    }

    // ---- 2. THE CONTROL. Assemble the INCOMPRESSIBLE way -- kinematic nu_eff in place of the dynamic
    //         mu_eff -- and require it to disagree. This is the one thing that distinguishes this
    //         component from simpleFoam's, so a gate that cannot see the difference is not gating it.
    std::printf("  2. control -- the INCOMPRESSIBLE form must NOT reproduce it\n");
    std::vector<scalar> nuEff(nC);
    for (label c = 0; c < nC; ++c) nuEff[c] = muEffInt[c] / f.rho.internal[c];
    std::vector<std::vector<scalar>> nuEffBnd(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        nuEffBnd[pi].resize(patches[pi].size);
        for (label i = 0; i < patches[pi].size; ++i)
            nuEffBnd[pi][i] = muEffBnd[pi][i] / rhoBnd[pi][i];
    }
    cpu::rhoSimple::RhoMomentumInput wrong = in;
    wrong.muEff    = &nuEff;        // kinematic, as the incompressible solver would pass
    wrong.muEffBnd = &nuEffBnd;
    FvVectorMatrix W = cpu::rhoSimple::assembleUEqn(f.U, wrong, m, g, patches);
    const std::vector<scalar> WA = matrixA<vector>(W, m, g, patches);
    std::vector<scalar> wrAU(nC);
    for (label c = 0; c < nC; ++c) wrAU[c] = 1.0 / WA[c];
    const double wrongErr = relL2(wrAU, ofRAU);
    check("kinematic nu_eff disagrees with OpenFOAM", wrongErr > 1e-3);
    std::printf("     %-44s %.6e\n", "  (its rAU error, for the record)", wrongErr);
    check("and it is worse than the dynamic form", wrongErr > rauErr);

    // ---- 3. dynamicViscosity is exactly rho*nuEff, cells and boundary. ----
    std::printf("  3. mu_eff = rho*nu_eff\n");
    const std::vector<scalar> mu = cpu::rhoSimple::dynamicViscosity(f.rho.internal, nuEff);
    double muErr = 0.0;
    for (label c = 0; c < nC; ++c)
        muErr = std::max(muErr, std::fabs((double)mu[c] - (double)muEffInt[c])
                                / std::max(1e-300, std::fabs((double)muEffInt[c])));
    report("dynamicViscosity round-trips (max rel)", muErr, 1e-14);

    // ---- 4. REFUSALS. ----
    std::printf("  4. refusals\n");
    {
        cpu::rhoSimple::RhoMomentumInput bad = in;
        bad.hasMRF = true;
        bad.mrf    = nullptr;
        bool threw = false;
        try { (void)cpu::rhoSimple::assembleUEqn(f.U, bad, m, g, patches); }
        catch (const std::exception&) { threw = true; }
        check("a declared-but-unsupplied MRF is refused", threw);
    }
    {
        cpu::rhoSimple::RhoMomentumInput bad = in;
        bad.hasFvOptions = true;
        bool threw = false;
        try { (void)cpu::rhoSimple::assembleUEqn(f.U, bad, m, g, patches); }
        catch (const std::exception&) { threw = true; }
        check("a declared fvOptions is refused", threw);
    }
    {
        cpu::rhoSimple::RhoMomentumInput bad = in;
        bad.muEff = nullptr; bad.muEffBnd = nullptr;
        bad.rho   = nullptr; bad.rhoBnd   = nullptr;
        bool threw = false;
        try { (void)cpu::rhoSimple::assembleUEqn(f.U, bad, m, g, patches); }
        catch (const std::exception&) { threw = true; }
        check("no viscosity at all is refused", threw);
    }

    if (failures == 0) std::printf("PASS\n");
    else               std::printf("FAIL (%d)\n", failures);
    return failures == 0 ? 0 : 1;
}
