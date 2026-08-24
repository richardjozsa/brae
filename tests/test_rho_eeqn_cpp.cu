// rhoSimpleFoam's EEqn.H against REAL OpenFOAM's own assembled energy equation.
//
// THE ORACLE is tools/dumpPEqn's EEqn stage harness, which writes at SIMPLE iteration
// BRAE_DUMP_STAGE_ITER, AFTER EEqn.relax():
//
//   stage_Ekp       the kinetic-energy field OpenFOAM's OWN branch produced
//   stage_he        he as assembled
//   stage_eD        EEqn.D()                      diag + the boundary internalCoeffs
//   stage_eSrc      EEqn.source() + sum(boundaryCoeffs)   the full right-hand side
//   stage_alphaEff  turbulence->alphaEff()
//
// alphaEff is INJECTED, for the same reason muEff is injected into the momentum gate: the compressible
// turbulence closure is a separate manifest component, and a number covering both cannot be attributed
// to either. What is measured here is the ENERGY ASSEMBLY.
//
// WHAT IS ACTUALLY UNDER TEST. EEqn.H branches on he.name():
//
//     e  ->  Ekp = 0.5|U|^2 + p/rho
//     h  ->  K   = 0.5|U|^2
//
// On this fixture (sensibleInternalEnergy, so the `e` arm) p ~ 1.1e5 and rho ~ 0.38, so p/rho ~ 2.9e5
// against 0.5|U|^2 of order 10 -- four orders of magnitude. A solver using the wrong arm converges to a
// smooth, plausible, wrong temperature field, which is why THE CONTROL below builds the `h` arm on
// purpose and requires both the kinetic field and the assembled matrix to disagree with OpenFOAM.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_dict.cuh"
#include "fv_matrix_ops.cuh"
#include "createFields_cpp.cuh"
#include "EEqn_cpp.cuh"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

// The staged fields carry OpenFOAM's own boundary types (extrapolatedCalculated among them), which brae's
// patch-field factory has no entry for. These read the parsed file directly: the staged VALUES are what
// is wanted, not a re-evaluated boundary condition.
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
            if (b.valueUniform) out[pi].assign(patches[pi].size, b.uniformValue);
            else if (static_cast<label>(b.values.size()) == patches[pi].size) out[pi] = b.values;
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

// D() = diag + the boundary internalCoeffs folded into their face cells (fvMatrix::D()).
static std::vector<scalar> matrixD(
    const FvScalarMatrix&       M,
    const std::vector<FvPatch>& patches)
{
    std::vector<scalar> D = M.diag;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        for (label i = 0; i < patches[pi].size; ++i)
            D[patches[pi].faceCells[i]] += M.internalCoeffs[pi][i];
    return D;
}

// source() + sum(boundaryCoeffs), which is what the harness writes as stage_eSrc.
static std::vector<scalar> matrixRhs(
    const FvScalarMatrix&       M,
    const std::vector<FvPatch>& patches)
{
    std::vector<scalar> r = M.source;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        for (label i = 0; i < patches[pi].size; ++i)
            r[patches[pi].faceCells[i]] += M.boundaryCoeffs[pi][i];
    return r;
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

    std::printf("rhoSimpleFoam EEqn vs OpenFOAM (%d cells)\n", (int)nC);

    cpu::rhoSimple::RhoSimpleFields f =
        cpu::rhoSimple::createFields(caseDir + "/" + startT, caseDir, simpleDict, &fvSolution,
                                     m, g, patches);
    std::printf("  energy variable: '%s'\n", f.heName.c_str());

    // OpenFOAM's own alphaEff and he, so this measures the assembly alone.
    const FieldData<scalar> aFd  = readField<scalar>(caseDir + "/" + dumpT + "/stage_alphaEff");
    const std::vector<scalar>              alphaEff    = rawInternal(aFd, nC);
    const std::vector<std::vector<scalar>> alphaEffBnd = rawBoundary<scalar>(aFd, patches);

    // U AFTER the momentum predictor. rhoSimpleFoam runs UEqn before EEqn, so the kinetic-energy source
    // is built from the just-solved velocity, NOT the field the iteration started with. The harness
    // writes it as stage_Upred; using the initial U instead makes 0.5|U|^2 wrong by the whole of the
    // momentum solve, which on this fixture is small next to p/rho and therefore easy to miss.
    const FieldData<vector> upFd = readField<vector>(caseDir + "/" + dumpT + "/stage_Upred");
    const std::vector<std::vector<vector>> upBnd = rawBoundary<vector>(upFd, patches);
    if (!upFd.internalUniform && static_cast<label>(upFd.internalField.size()) == nC)
        f.U.internal = upFd.internalField;
    for (std::size_t pi = 0; pi < patches.size(); ++pi) f.U.boundary[pi]->setValue(upBnd[pi]);

    // he is built from the staged file itself, TYPES INCLUDED. Borrowing T's field and overwriting only
    // its values is not enough: a fixedValue patch's boundaryCoeffs come from its refValue, so a he built
    // that way carries T's 1000 K where the coefficients want he's 4.19e5 J/kg -- which reads as a source
    // error on every inlet-adjacent cell and nowhere else. The gate rewrites OpenFOAM's energy BC types
    // into brae-known equivalents first, and asserts that rewrite is exact.
    GeometricField<scalar> he =
        buildField<scalar>(readField<scalar>(caseDir + "/" + dumpT + "/stage_he"), patches, nC);
    he.evaluateBoundary();

    // ---- 1. THE BRANCH, on its own. ----
    std::printf("  1. the kinetic-energy source (EEqn.H branches on he.name())\n");
    const std::vector<scalar> ke = cpu::rhoSimple::kineticEnergy(f.heName, f.U, f.p, f.rho);
    const std::vector<scalar> ofEkp =
        rawInternal(readField<scalar>(caseDir + "/" + dumpT + "/stage_Ekp"), nC);
    report("Ekp = 0.5|U|^2 + p/rho vs OpenFOAM", relL2(ke, ofEkp), 1e-12);
    {
        // The BOUNDARY values matter as much as the internal ones: EEqn's explicit convection term picks
        // up phi_b*Ekp_b on every boundary face, so an inlet Ekp that is right in the cell and wrong on
        // the face still poisons the source there.
        const FieldData<scalar> ekpFd = readField<scalar>(caseDir + "/" + dumpT + "/stage_Ekp");
        const std::vector<std::vector<scalar>> ofEkpB = rawBoundary<scalar>(ekpFd, patches);
        const std::vector<std::vector<scalar>> keB =
            cpu::rhoSimple::kineticEnergyBoundary(f.heName, f.U, f.p, f.rho, patches);
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            if (!patches[pi].size) continue;
            std::vector<scalar> a, b;
            for (label i = 0; i < patches[pi].size; ++i) { a.push_back(keB[pi][i]); b.push_back(ofEkpB[pi][i]); }
            std::printf("       Ekp on %-14s %.4e\n", patches[pi].name.c_str(), relL2(a, b));
        }
    }

    // The control for the branch: the OTHER arm must not reproduce it.
    const std::vector<scalar> keH = cpu::rhoSimple::kineticEnergy("h", f.U, f.p, f.rho);
    const double keHErr = relL2(keH, ofEkp);
    check("the 'h' arm (K = 0.5|U|^2) does NOT match", keHErr > 1e-3);
    std::printf("     %-44s %.6e\n", "  (its error, for the record)", keHErr);

    // ---- 2. The assembled energy matrix. ----
    std::printf("  2. EEqn assembled\n");
    cpu::rhoSimple::EnergyInput in;
    in.phi                = &f.phi.internal;
    in.phiBnd             = &f.phi.boundary;
    in.alphaEff           = &alphaEff;
    in.alphaEffBnd        = &alphaEffBnd;
    in.heName             = f.heName;
    in.boundedHe          = true;
    in.boundedKE          = true;
    in.schemeHe           = cpu::rhoSimple::DivScheme::upwind;
    in.schemeKE           = cpu::rhoSimple::DivScheme::upwind;
    in.correctedLaplacian = true;
    const FoamDict* rf = fvSolution.subDict("relaxationFactors");
    const FoamDict* re = rf ? rf->subDict("equations") : nullptr;
    in.relaxHe = re ? re->scalarOr(f.heName, 1.0) : 1.0;
    std::printf("     relaxation %s = %g\n", f.heName.c_str(), (double)in.relaxHe);

    FvScalarMatrix E = cpu::rhoSimple::assembleEEqn(he, f.U, f.p, f.rho, in, m, g, patches);
    const std::vector<scalar> ofD   = rawInternal(readField<scalar>(caseDir + "/" + dumpT + "/stage_eD"), nC);
    const std::vector<scalar> ofSrc = rawInternal(readField<scalar>(caseDir + "/" + dumpT + "/stage_eSrc"), nC);
    const double dErr = relL2(matrixD(E, patches), ofD);
    const double sErr = relL2(matrixRhs(E, patches), ofSrc);
    report("EEqn.D() vs OpenFOAM", dErr, 1e-10);
    {
        // Decomposition, so a source gap can be attributed to a term instead of guessed at.
        const std::vector<scalar> mine = matrixRhs(E, patches);
        double nm = 0.0, no = 0.0, nd = 0.0, nke = 0.0, nrel = 0.0;
        const std::vector<scalar> keD = cpu::rhoSimple::kineticEnergy(f.heName, f.U, f.p, f.rho);
        for (label c = 0; c < nC; ++c)
        {
            nm += (double)mine[c]*mine[c];
            no += (double)ofSrc[c]*ofSrc[c];
            const double d = (double)mine[c] - (double)ofSrc[c];
            nd += d*d;
            nrel += (double)ofD[c]*(double)he.internal[c]*(double)ofD[c]*(double)he.internal[c];
            nke += (double)keD[c]*keD[c];
        }
        std::printf("     |brae src| %.4e   |OF src| %.4e   |diff| %.4e   |D*he| %.4e\n",
                    std::sqrt(nm), std::sqrt(no), std::sqrt(nd), std::sqrt(nrel));
        const std::vector<scalar> kd = cpu::rhoSimple::kineticEnergyDivergence(f.U, f.p, f.rho, in, m, g, patches);
        cpu::rhoSimple::EnergyInput ub = in; ub.boundedKE = false;
        const std::vector<scalar> kdu = cpu::rhoSimple::kineticEnergyDivergence(f.U, f.p, f.rho, ub, m, g, patches);
        double nk = 0.0, nku = 0.0;
        for (label c = 0; c < nC; ++c) { nk += (double)kd[c]*kd[c]; nku += (double)kdu[c]*kdu[c]; }
        std::printf("     |KE div| bounded %.4e   unbounded %.4e\n", std::sqrt(nk), std::sqrt(nku));
        // WHERE does the difference live? If it sits on boundary-adjacent cells it is a boundary
        // treatment; if it is spread through the interior it is a volumetric term.
        std::vector<char> isBnd(nC, 0);
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
            for (label i = 0; i < patches[pi].size; ++i) isBnd[patches[pi].faceCells[i]] = 1;
        double db = 0.0, di = 0.0; long nb = 0;
        for (label c = 0; c < nC; ++c)
        {
            const double d = (double)mine[c] - (double)ofSrc[c];
            if (isBnd[c]) { db += d*d; ++nb; } else di += d*d;
        }
        std::printf("     diff on %ld boundary cells %.4e   on %ld interior cells %.4e\n",
                    nb, std::sqrt(db), (long)nC - nb, std::sqrt(di));
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            if (!patches[pi].size) continue;
            double dp = 0.0;
            for (label i = 0; i < patches[pi].size; ++i)
            {
                const label c = patches[pi].faceCells[i];
                const double d = (double)mine[c] - (double)ofSrc[c];
                dp += d*d;
            }
            std::printf("       %-18s diff %.4e  (he BC in brae: %s)\n", patches[pi].name.c_str(),
                        std::sqrt(dp), he.boundary[pi]->fixesValue() ? "fixesValue" : "not-fixesValue");
        }
    }
    report("EEqn source + boundaryCoeffs vs OpenFOAM", sErr, 1e-10);

    // ---- 3. THE CONTROL. ----
    //
    // NOT on the assembled source, and the reason is worth stating: `div(phi,Ekp)` is `bounded`, and
    // boundedConvectionScheme::fvcDiv subtracts surfaceIntegrate(phi)*vf. At iteration 1 continuity is
    // nearly satisfied, so that subtraction very nearly cancels the divergence itself -- measured here,
    // |KE div| falls from 1.4e+01 unbounded to 1.3e-04 bounded. A control asserting that the wrong ARM
    // changes the assembled source would therefore be asserting something this state cannot show, and
    // would pass or fail on rounding. The branch is instead gated where it is actually observable: the
    // kinetic-energy FIELD against OpenFOAM's own stage_Ekp (check 1), and the divergence the two arms
    // produce.
    std::printf("  3. control -- the two arms must produce different equations\n");
    cpu::rhoSimple::EnergyInput wrong = in;
    wrong.heName = (f.heName == "e") ? "h" : "e";
    // Compared UNBOUNDED. The bounded correction removes the near-uniform part of the field, and at this
    // state p/rho -- the entire difference between the arms -- IS near-uniform, so the bounded term is
    // insensitive to the branch here (measured: |KE div| 1.3e-04 bounded against 1.4e+01 unbounded, and
    // the two arms' bounded divergences agree to better than 1e-3 of that). That insensitivity is a
    // property of iteration 1 on this fixture, not of the discretisation, so the control is taken on the
    // convection term the scheme actually builds, before the bounded subtraction.
    cpu::rhoSimple::EnergyInput ue = in;    ue.boundedKE    = false;
    cpu::rhoSimple::EnergyInput uh = wrong; uh.boundedKE    = false;
    const std::vector<scalar> kdE = cpu::rhoSimple::kineticEnergyDivergence(f.U, f.p, f.rho, ue, m, g, patches);
    const std::vector<scalar> kdH = cpu::rhoSimple::kineticEnergyDivergence(f.U, f.p, f.rho, uh, m, g, patches);
    double na = 0.0, nb2 = 0.0;
    for (label c = 0; c < nC; ++c) { na += (double)(kdE[c]-kdH[c])*(kdE[c]-kdH[c]); nb2 += (double)kdE[c]*kdE[c]; }
    check("the arms build different convection terms", std::sqrt(na) > 1e-3 * std::sqrt(nb2));
    std::printf("     %-44s %.4e vs %.4e\n", "  |KE div| unbounded vs |arm difference|",
                std::sqrt(nb2), std::sqrt(na));
    // And the wrong arm's FIELD must not reproduce OpenFOAM's -- asserted in check 1, restated here as
    // the thing that actually distinguishes the two implementations.
    check("wrong arm's field is far from OpenFOAM's", keHErr > 1e-3);

    // ---- 4. REFUSALS. ----
    std::printf("  4. refusals\n");
    {
        cpu::rhoSimple::EnergyInput bad = in;
        bad.heName = "absoluteEnthalpy";
        bool threw = false;
        std::string msg;
        try { (void)cpu::rhoSimple::assembleEEqn(he, f.U, f.p, f.rho, bad, m, g, patches); }
        catch (const std::exception& e) { threw = true; msg = e.what(); }
        check("an energy variable that is neither e nor h is refused", threw);
        check("and the refusal names it", msg.find("absoluteEnthalpy") != std::string::npos);
    }
    {
        cpu::rhoSimple::EnergyInput bad = in;
        bad.hasMRF = true;
        bool threw = false;
        try { (void)cpu::rhoSimple::assembleEEqn(he, f.U, f.p, f.rho, bad, m, g, patches); }
        catch (const std::exception&) { threw = true; }
        check("a declared MRF is refused", threw);
    }
    {
        cpu::rhoSimple::EnergyInput bad = in;
        bad.schemeKE = cpu::rhoSimple::DivScheme::LUST;
        bool threw = false;
        try { (void)cpu::rhoSimple::assembleEEqn(he, f.U, f.p, f.rho, bad, m, g, patches); }
        catch (const std::exception&) { threw = true; }
        check("an unported convection scheme is refused", threw);
    }
    {
        cpu::rhoSimple::EnergyInput bad = in;
        bad.alphaEff = nullptr; bad.alphaEffBnd = nullptr;
        bool threw = false;
        try { (void)cpu::rhoSimple::assembleEEqn(he, f.U, f.p, f.rho, bad, m, g, patches); }
        catch (const std::exception&) { threw = true; }
        check("a missing alphaEff is refused", threw);
    }

    if (failures == 0) std::printf("PASS\n");
    else               std::printf("FAIL (%d)\n", failures);
    return failures == 0 ? 0 : 1;
}
