// DISPATCH for the rebuilt simpleFoam -- see simpleFoamV2.cuh for why the guard exists.
#include "simpleFoamV2.cuh"
#include "simpleFoam.cuh"
#include "createFields_cpp.cuh"
#include "simpleControl_cpp.cuh"
#include "linearViscousStress_cpp.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_field_writer.cuh"
#include "foam_dict.cuh"
#include "fvc.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "device_kepsilon.cuh"
#include "device_komega_sst.cuh"    // deviceKOmegaSSTCorrect building blocks + KOmegaSSTCoeffs
#include "komega_sst_coeffs.cuh"    // readKOmegaSSTCoeffs (RAS.kOmegaSSTCoeffs, OF defaults when absent)
#include "cell_wall_dist.cuh"       // cellWallDist: kOmegaSST's F1/F2 need y at every CELL, not just walls
#include "near_wall_dist.cuh"
#include "solver_dispatch.cuh"   // readDdtSchemeWord: the same steady/transient test the dispatcher uses

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <regex>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace brae {
namespace gpu {

namespace {

bool fileExists(const std::string& p)
{
    return std::filesystem::exists(p) || std::filesystem::exists(p + ".gz");
}

// The div SCHEME the case asks for on div(phi,U). Returned as the selectable keyword, i.e. the last word
// that is not `Gauss`, `bounded` or a numeric coefficient -- the same rule ofscan's case layer applies.
// Does div(phi,U) carry the `bounded` prefix? OpenFOAM's `bounded Gauss <scheme>` adds
//     - fvm::Sp(fvc::div(phi), U)
// to the momentum equation (boundedConvectionScheme). It vanishes at convergence, where div(phi) -> 0,
// which is exactly why dropping it is invisible in a converged comparison and very visible in the
// approach to it: it is what keeps the equation diagonally dominant while the flux is not yet
// conservative. The rebuilt UEqn does not implement it.
bool divUBounded(const std::string& caseDir);

std::string divUScheme(const std::string& caseDir)
{
    std::string text;
    try { text = readFileExpanded(caseDir + "/system/fvSchemes"); } catch (...) { return ""; }
    const std::size_t blk = text.find("divSchemes");
    if (blk == std::string::npos) return "";
    const std::size_t open = text.find('{', blk);
    const std::size_t close = text.find('}', open == std::string::npos ? blk : open);
    if (open == std::string::npos) return "";
    // Prefer the explicit div(phi,U) entry; fall back to `default`.
    static const std::regex re(R"(div\(phi,U\)\s*([^;]*);)");
    std::smatch mm;
    std::string blkText = text.substr(open, (close == std::string::npos ? text.size() : close) - open);
    std::string entry;
    if (std::regex_search(blkText, mm, re)) entry = mm[1].str();
    else
    {
        static const std::regex rd(R"(default\s+([^;]*);)");
        if (std::regex_search(blkText, mm, rd)) entry = mm[1].str();
    }
    std::string last;
    std::regex tok(R"([^\s]+)");
    for (std::sregex_iterator it(entry.begin(), entry.end(), tok), e; it != e; ++it)
    {
        const std::string w = it->str();
        if (w == "Gauss" || w == "bounded" || w == "none") continue;
        if (std::regex_match(w, std::regex(R"([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)"))) continue;
        last = w;
        break;      // the FIRST such word is the scheme; anything after is its coefficient
    }
    return last;
}

// The gradient `linearUpwind` NAMES, resolved through gradSchemes -- and whether it is one we compute.
//
// `div(phi,U) bounded Gauss linearUpwind grad(U);` does not mean "use fvc::grad". linearUpwind's
// constructor reads the word after the scheme (linearUpwind.H, gradSchemeName_(schemeData)) and looks it
// up with mesh.gradScheme(name), which falls back to gradSchemes `default`. brae computes a plain Gauss
// linear gradient; a case naming cellLimited or leastSquares there would get a DIFFERENT correction,
// silently, and this correction does NOT vanish at convergence -- so the answer would simply be wrong.
// Returns the offending scheme word, or empty when the resolved scheme is Gauss linear.
std::string linearUpwindGradUnsupported(const std::string& caseDir)
{
    std::string text;
    try { text = readFileExpanded(caseDir + "/system/fvSchemes"); } catch (...) { return ""; }

    // 1. the name linearUpwind gives -- the word immediately after it in the div(phi,U) entry.
    std::string gradName = "default";
    {
        const std::size_t blk = text.find("divSchemes");
        if (blk != std::string::npos)
        {
            const std::size_t open = text.find('{', blk);
            const std::size_t close = text.find('}', open == std::string::npos ? blk : open);
            if (open != std::string::npos)
            {
                const std::string b =
                    text.substr(open, (close == std::string::npos ? text.size() : close) - open);
                static const std::regex re(R"(linearUpwind\s+([^\s;]+))");
                std::smatch mm;
                if (std::regex_search(b, mm, re)) gradName = mm[1].str();
            }
        }
    }

    // 2. that name's entry in gradSchemes, falling back to `default`.
    const std::size_t gblk = text.find("gradSchemes");
    if (gblk == std::string::npos) return "";          // no block -> OpenFOAM's default is Gauss linear
    const std::size_t open = text.find('{', gblk);
    if (open == std::string::npos) return "";
    const std::size_t close = text.find('}', open);
    const std::string b = text.substr(open, (close == std::string::npos ? text.size() : close) - open);

    std::string entry;
    {
        // The name contains parentheses (`grad(U)`), so every regex metacharacter in it is escaped rather
        // than pasted in raw -- `grad(U)` as a pattern would match the bare word `gradU`.
        std::string esc;
        for (char c : gradName)
        {
            if (std::strchr("().[]{}*+?^$|\\", c)) esc += '\\';
            esc += c;
        }
        std::smatch mm;
        if (std::regex_search(b, mm, std::regex(esc + R"(\s+([^;]*);)"))) entry = mm[1].str();
        else if (std::regex_search(b, mm, std::regex(R"(default\s+([^;]*);)"))) entry = mm[1].str();
    }
    if (entry.empty()) return "";

    // 3. accept only `Gauss linear`.
    static const std::regex tok(R"([^\s]+)");
    for (std::sregex_iterator it(entry.begin(), entry.end(), tok), e; it != e; ++it)
    {
        const std::string w = it->str();
        if (w == "Gauss" || w == "linear") continue;
        return w;
    }
    return "";
}

// The numeric coefficient a limited scheme carries: the `1` of `limitedLinear 1`. OpenFOAM reads it off
// the scheme stream, so it is the first number after the scheme word. It is NOT cosmetic: twoByk = 2/k
// scales the limiter, and `limitedLinear 0.2` is a materially different scheme from `limitedLinear 1`.
scalar divUSchemeCoeff(const std::string& caseDir, scalar def)
{
    std::string text;
    try { text = readFileExpanded(caseDir + "/system/fvSchemes"); } catch (...) { return def; }
    const std::size_t blk = text.find("divSchemes");
    if (blk == std::string::npos) return def;
    const std::size_t open = text.find('{', blk);
    if (open == std::string::npos) return def;
    const std::size_t close = text.find('}', open);
    const std::string b = text.substr(open, (close == std::string::npos ? text.size() : close) - open);
    static const std::regex re(R"(div\(phi,U\)\s*([^;]*);)");
    std::smatch mm;
    std::string entry;
    if (std::regex_search(b, mm, re)) entry = mm[1].str();
    else
    {
        static const std::regex rd(R"(default\s+([^;]*);)");
        if (std::regex_search(b, mm, rd)) entry = mm[1].str();
    }
    static const std::regex num(R"([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)");
    std::smatch nm;
    if (std::regex_search(entry, nm, num)) return std::atof(nm[0].str().c_str());
    return def;
}

bool divUBounded(const std::string& caseDir)
{
    std::string text;
    try { text = readFileExpanded(caseDir + "/system/fvSchemes"); } catch (...) { return false; }
    const std::size_t blk = text.find("divSchemes");
    if (blk == std::string::npos) return false;
    const std::size_t open = text.find('{', blk);
    if (open == std::string::npos) return false;
    const std::size_t close = text.find('}', open);
    const std::string b = text.substr(open, (close == std::string::npos ? text.size() : close) - open);
    static const std::regex re(R"(div\(phi,U\)\s*([^;]*);)");
    std::smatch mm;
    std::string entry;
    if (std::regex_search(b, mm, re)) entry = mm[1].str();
    else
    {
        static const std::regex rd(R"(default\s+([^;]*);)");
        if (std::regex_search(b, mm, rd)) entry = mm[1].str();
    }
    return entry.find("bounded") != std::string::npos;
}

// Does the LAPLACIAN carry the non-orthogonal correction?
//
// The laplacian's OWN scheme decides. `laplacianSchemes { default Gauss linear orthogonal; }` builds an
// orthogonal laplacian whatever `snGradSchemes` says -- snGradSchemes governs EXPLICIT fvc::snGrad, which
// in simpleFoam appears only in the SIMPLEC branch (already refused). An earlier version of this check
// blocked on either block and so refused pitzDailyTurb, a case whose laplacian is orthogonal and which
// the rebuilt path handles exactly.
//
// What `laplacianSchemes` asked for: whether the non-orthogonal correction is on, and the name of a
// scheme that is recognised but not ported (empty when there is none).
struct LaplacianScheme
{
    bool        corrected = true;   // OpenFOAM's default when the word is absent
    std::string unsupported;
};

// Both halves of `corrected` are now implemented on both paths (fvm.cuh; UEqn.cu; pEqn.cu), so this READS
// the scheme instead of refusing it. OpenFOAM's default when the word is absent IS corrected, so an absent
// block returns true.
//
// `limited <coeff>` is NOT the same scheme: limitedSnGrad caps the correction against the orthogonal part,
// and `limited 1` alone is equivalent to `corrected`. Anything else is reported through `unsupported` so
// the envelope can refuse it rather than quietly running the uncapped correction.
LaplacianScheme laplacianScheme(const std::string& caseDir)
{
    LaplacianScheme r;
    std::string text;
    try { text = readFileExpanded(caseDir + "/system/fvSchemes"); } catch (...) { r.corrected = false; return r; }
    const std::size_t blk = text.find("laplacianSchemes");
    if (blk == std::string::npos) return r;                 // absent -> OpenFOAM default is `corrected`
    const std::size_t open = text.find('{', blk);
    if (open == std::string::npos) return r;
    const std::size_t close = text.find('}', open);
    const std::string b = text.substr(open, (close == std::string::npos ? text.size() : close) - open);

    if (b.find("limited") != std::string::npos)
    {
        // `limited 1` is exactly `corrected`; any other coefficient is limitedSnGrad, which is not ported.
        const std::size_t k = b.find("limited");
        std::istringstream ls(b.substr(k + 7));
        scalar coeff = -1.0;
        if (!(ls >> coeff) || std::fabs(coeff - 1.0) > 1e-12)
            r.unsupported = "limited";
        return r;
    }
    if (b.find("uncorrected") != std::string::npos) { r.corrected = false; return r; }
    if (b.find("orthogonal") != std::string::npos && b.find("nonOrthogonal") == std::string::npos)
    { r.corrected = false; return r; }
    return r;   // `corrected` or unspecified
}

bool switchOn(const FoamDict& d, const std::string& key, bool def)
{
    const auto* v = d.find(key);
    if (!v || v->empty()) return def;
    const std::string& s = v->back();
    if (s == "true" || s == "on" || s == "yes" || s == "y" || s == "1") return true;
    if (s == "false" || s == "off" || s == "no" || s == "n" || s == "0") return false;
    return def;
}

} // namespace


EnvelopeReport simpleFoamV2Envelope(const std::string& caseDir)
{
    EnvelopeReport r;

    const FoamDict fvSolution = readDict(caseDir + "/system/fvSolution");
    const FoamDict turbProps  = readDict(caseDir + "/constant/turbulenceProperties");

    // --- things that change the equations and are not implemented ---------------------------
    if (fileExists(caseDir + "/constant/MRFProperties"))
        r.blockers.push_back("constant/MRFProperties is present. UEqn.H applies MRF via "
                             "correctBoundaryVelocity(U) and MRF.DDt(U) (UEqn.H:3,8) and pEqn.H via "
                             "makeRelative(phiHbyA) (pEqn.H:5); the rebuilt path implements none of it.");
    if (fileExists(caseDir + "/constant/fvOptions") || fileExists(caseDir + "/system/fvOptions"))
        r.blockers.push_back("fvOptions is present. UEqn.H applies it three times (UEqn.H:11,17,23) and "
                             "pEqn.H once (pEqn.H:49); the rebuilt path implements none of it.");

    if (const FoamDict* s = fvSolution.subDict("SIMPLE"))
    {
        // SIMPLEC is implemented (matrixH1 + fvc::snGrad); it is READ here, not refused. What is still
        // missing is constrainPressure, which pEqn.H calls with rAtU right after -- it only does anything
        // on a fixedFluxPressure patch, so that patch type is what gets refused, below.
        (void)s;
    }

    // --- this is the STEADY solver ------------------------------------------------------------
    {
        const std::string ddt = readDdtSchemeWord(caseDir + "/system/fvSchemes");
        if (!ddt.empty() && ddt != "steadyState")
            r.blockers.push_back("ddtSchemes.default is `" + ddt + "`, not steadyState. simpleFoam is the "
                                 "steady solver; a transient case belongs to pimpleFoam.");
    }

    // --- the convection scheme ----------------------------------------------------------------
    // The rebuilt UEqn implements OpenFOAM's `upwind` implicit weights. Running a case that asks for
    // limitedLinear/linearUpwind/LUST on those weights would silently solve a different discretisation --
    // which is exactly how brae's LUST implicit-weight defect stayed hidden.
    {
        // `bounded` FIRST: the scheme word after it is still `upwind`, so a guard that only looked at
        // the scheme word accepted `bounded Gauss upwind` and ran it without the Sp term. Measured on
        // pitzDaily: the existing solver's Ux residual fell 1 -> 0.022 over 20 iterations while the
        // rebuilt path plateaued at ~0.5. The term vanishes at convergence, so a converged comparison
        // cannot see it -- which is precisely why the guard has to.
        const LaplacianScheme lap = laplacianScheme(caseDir);
        if (!lap.unsupported.empty())
            r.blockers.push_back("laplacianSchemes asks for `" + lap.unsupported + "`, which caps the "
                                 "non-orthogonal correction against the orthogonal part (limitedSnGrad). "
                                 "Only the uncapped `corrected` is ported; running it would apply a larger "
                                 "correction than the case asked for.");

        const std::string sc = divUScheme(caseDir);
        // OpenFOAM registers 78 surfaceInterpolationSchemes (ofscan: impls surfaceInterpolationScheme).
        // These five are the ones ported and gated; anything else is refused by name.
        if (!sc.empty() && sc != "upwind" && sc != "linearUpwind" && sc != "limitedLinear"
            && sc != "limitedLinearV" && sc != "LUST")
            r.blockers.push_back("div(phi,U) asks for `" + sc + "`; the rebuilt UEqn implements `upwind`, "
                                 "`linearUpwind`, `limitedLinear`, `limitedLinearV` and `LUST`. Running it "
                                 "anyway would solve a different discretisation than the case specifies.");
        if (sc == "linearUpwind")
        {
            const std::string bad = linearUpwindGradUnsupported(caseDir);
            if (!bad.empty())
                r.blockers.push_back("div(phi,U) is `linearUpwind`, whose named gradient resolves to `" +
                                     bad + "` in gradSchemes; brae computes a plain Gauss linear gradient. "
                                     "This correction does not vanish at convergence, so running it would "
                                     "be wrong rather than merely slower to converge.");
        }
    }

    // --- turbulence ---------------------------------------------------------------------------
    {
        const std::string simType = turbProps.wordOr("simulationType", "laminar");
        if (simType == "RAS")
        {
            const FoamDict* ras = turbProps.subDict("RAS");
            const std::string model = ras ? ras->wordOr("RASModel", "") : "";
            if (model != "kEpsilon" && model != "kOmegaSST")
                r.blockers.push_back("RASModel is `" + model + "`; the rebuilt path wires kEpsilon and "
                                     "kOmegaSST only. OpenFOAM registers 26 incompressible turbulence "
                                     "models (ofscan: impls incompressible::turbulenceModel).");
        }
        else if (simType != "laminar")
        {
            r.blockers.push_back("simulationType is `" + simType + "`; the rebuilt path supports laminar "
                                 "and RAS/kEpsilon only.");
        }
    }

    // --- coupled patches ----------------------------------------------------------------------
    {
        PrimitiveMesh m;
        try { m.read(caseDir + "/constant/polyMesh"); }
        catch (const std::exception& e)
        {
            r.blockers.push_back(std::string("cannot read the mesh: ") + e.what());
            r.supported = r.blockers.empty();
            return r;
        }
        for (const auto& b : m.patches())
            if (isCoupledInterfaceType(b.type) || b.type == "processor")
                r.blockers.push_back("patch `" + b.name + "` is of coupled type `" + b.type +
                                     "`; the rebuilt components handle no coupled interfaces.");
    }

    // --- pressure BCs that pEqn.H reaches through constrainPressure ---------------------------
    // fixedFluxPressure is a fixedGradient patch whose gradient constrainPressure RESETS every outer
    // iteration to match the flux actually being imposed. brae's field builder maps it to zeroGradient,
    // which is the right answer only when that flux is zero. Left unchecked the case would run and
    // quietly impose a different boundary condition than it asked for.
    {
        const FoamDict cd0 = readDict(caseDir + "/system/controlDict");
        const std::string st0 = cd0.wordOr("startFrom", "startTime") == "latestTime"
                              ? std::string("0") : cd0.wordOr("startTime", "0");
        std::string ptext;
        try { ptext = readFileExpanded(caseDir + "/" + st0 + "/p"); } catch (...) { ptext.clear(); }
        if (ptext.find("fixedFluxPressure") != std::string::npos)
            r.blockers.push_back("a pressure patch is `fixedFluxPressure`; pEqn.H resets its gradient "
                                 "every iteration through constrainPressure(p, U, phiHbyA, rAtU, MRF) "
                                 "(pEqn.H:21), which is not ported. brae would substitute zeroGradient, "
                                 "which is only the same boundary condition when the imposed flux is 0.");
    }

    // --- substitutions that are supported but must be SAID ------------------------------------
    if (const FoamDict* solvers = fvSolution.subDict("solvers"))
        if (const FoamDict* ps = solvers->subDict("p"))
        {
            const std::string sel = ps->wordOr("solver", "");
            if (sel == "GAMG")
                r.notices.push_back("system/fvSolution asks for `GAMG` on p; brae runs an "
                                    "AMG-preconditioned PCG instead. Same operator, different Krylov "
                                    "method and iteration count.");
        }

    r.supported = r.blockers.empty();
    return r;
}


bool simpleFoamV2Selected()
{
    const char* e = std::getenv("BRAE_SIMPLEFOAM_V2");
    return e && e[0] == '1';
}


namespace { double g_hookSeconds = 0.0; double g_refreshSeconds = 0.0; }

int runSimpleFoamV2(const std::string& caseDir)
{
    const EnvelopeReport env = simpleFoamV2Envelope(caseDir);
    for (const std::string& n : env.notices)
        std::printf("NOTICE (simpleFoam v2): %s\n", n.c_str());
    if (!env.supported)
    {
        std::string msg =
            "BRAE_SIMPLEFOAM_V2=1 selected the rebuilt simpleFoam, but this case is outside what it "
            "implements:\n";
        for (const std::string& b : env.blockers) msg += "  - " + b + "\n";
        msg += "Refusing rather than solving a different problem. Unset BRAE_SIMPLEFOAM_V2 to run the "
               "existing solver.";
        throw std::runtime_error(msg);
    }

    // ---- case setup --------------------------------------------------------------------------
    PrimitiveMesh m;  m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;     g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();

    const FoamDict controlDict = readDict(caseDir + "/system/controlDict");
    const FoamDict fvSolution  = readDict(caseDir + "/system/fvSolution");
    const FoamDict transport   = readDict(caseDir + "/constant/transportProperties");

    const scalar nu = transport.scalarOr("nu", 1e-5);
    const FoamDict* simpleDict = fvSolution.subDict("SIMPLE");

    const std::string startTime = controlDict.wordOr("startFrom", "startTime") == "latestTime"
                                ? std::string("0") : controlDict.wordOr("startTime", "0");
    cpu::SimpleFields f = cpu::createFields(caseDir + "/" + startTime, simpleDict, m, g, fvp);

    cpu::SimpleControlDict cd = cpu::readSimpleControl(fvSolution);
    cpu::SimpleControl ctl(cd);

    scalar relaxU = 0.7, relaxP = 0.3;
    if (const FoamDict* rf = fvSolution.subDict("relaxationFactors"))
    {
        if (const FoamDict* eq = rf->subDict("equations")) relaxU = eq->scalarOr("U", relaxU);
        if (const FoamDict* fl = rf->subDict("fields"))    relaxP = fl->scalarOr("p", relaxP);
    }

    const label endTime = static_cast<label>(controlDict.scalarOr("endTime", 100));

    // ---- device state ------------------------------------------------------------------------
    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    const DeviceVectorBoundary dbU = buildDeviceVectorBoundary(f.U, fvp, g);
    const DeviceBoundary dbP = buildDeviceBoundary(f.p, fvp, g);

    SolverFields gf;
    {
        std::vector<scalar> ux(nC), uy(nC), uz(nC);
        for (label c = 0; c < nC; ++c)
        { ux[c] = f.U.internal[c].x; uy[c] = f.U.internal[c].y; uz[c] = f.U.internal[c].z; }
        gf.Ux.copyFrom(ux); gf.Uy.copyFrom(uy); gf.Uz.copyFrom(uz);
        gf.p.copyFrom(f.p.internal);
        gf.phiInt.copyFrom(f.phi.internal);
        std::vector<scalar> pb;
        for (const auto& v : f.phi.boundary) for (scalar x : v) pb.push_back(x);
        pb.resize(dm.nBndFaces, 0.0);
        gf.phiBnd.copyFrom(pb);
    }

    const FoamDict turbProps = readDict(caseDir + "/constant/turbulenceProperties");
    const bool ras = (turbProps.wordOr("simulationType", "laminar") == "RAS");

    std::vector<scalar> nuEffC(nC, nu);
    std::vector<std::vector<scalar>> nuEffB(fvp.size());
    for (std::size_t pi = 0; pi < fvp.size(); ++pi) nuEffB[pi].assign(fvp[pi].size, nu);
    const SurfaceScalarField nuFace = cpu::effectiveFaceViscosity(nuEffC, nuEffB, m, g, fvp);

    std::vector<scalar> nuBndFlat;
    for (const auto& v : nuEffB) for (scalar x : v) nuBndFlat.push_back(x);
    nuBndFlat.resize(dm.nBndFaces, nu);

    DeviceBuffer<scalar> dNuCell(nuEffC), dNuFace(nuFace.internal), dNuBnd(nuBndFlat);

    std::vector<label> takeU, adjustable;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            takeU.push_back(f.U.boundary[pi]->assignable() ? 0 : 1);
            adjustable.push_back(f.U.boundary[pi]->fixesValue() ? 0 : 1);
        }
    takeU.resize(dm.nBndFaces, 0);
    adjustable.resize(dm.nBndFaces, 0);
    DeviceBuffer<label> dTakeU(takeU), dAdjust(adjustable);

    // ---- turbulence: k-epsilon on the device, behind the driver's hook ------------------------
    //
    // The hook exists so the driver never names a model (that is what made the old solver a god object).
    // Everything the model needs is set up here and captured; the driver only knows that SOMETHING runs
    // at the end of the iteration, which is where simpleFoam.C:94 calls turbulence->correct().
    //
    // The hook also owns the nuEff REFRESH. The driver reads nuEff through pointers, so updating those
    // buffers here is what makes the coupling lagged in the right direction: iteration n+1's momentum
    // equation sees the nut this correct() just produced, and nothing else in the loop has to know.
    GeometricField<scalar> kF, epsF, nutF;
    DeviceBuffer<scalar> dK, dEps, dNut;
    DeviceWallData wall;
    DeviceBoundary dbK, dbEps, dbNut;
    DeviceBuffer<label> bndIsWall;
    DeviceBuffer<scalar> bndY;
    scalar relaxK = 0.7, relaxEps = 0.7;
    // k/epsilon linear-solver settings, read from fvSolution below. Declared here because the turbulence
    // hook is a lambda defined above that point and captures them by reference; it does not run until the
    // outer loop, by which time they are set.
    scalar tolKE = 1e-10, relTolKE = 0.0;
    bool   gsKE  = false;
    bool   sstModel = false;
    std::string secondField = "epsilon";
    KOmegaSSTCoeffs sstCoeffs;
    DeviceBuffer<scalar> dY;             // cell wall distance -- kOmegaSST's F1/F2 need it per CELL

    StepInput in;
    if (ras)
    {
        // kOmegaSST's second transport variable is omega, and brae holds it in the same slot epsilon
        // uses -- the fused device correct() for each model takes that slot and knows what it means.
        // `ras` here is the simulationType switch, not the dict -- the RAS sub-dictionary is re-fetched.
        const FoamDict* rasDict = turbProps.subDict("RAS");
        sstModel = rasDict && rasDict->wordOr("RASModel", "") == "kOmegaSST";
        secondField = sstModel ? "omega" : "epsilon";
        kF   = buildField<scalar>(readField<scalar>(caseDir + "/" + startTime + "/k"), fvp, nC);
        epsF = buildField<scalar>(readField<scalar>(caseDir + "/" + startTime + "/" + secondField), fvp, nC);
        nutF = buildField<scalar>(readField<scalar>(caseDir + "/" + startTime + "/nut"), fvp, nC);
        kF.evaluateBoundary(); epsF.evaluateBoundary(); nutF.evaluateBoundary();

        dK.copyFrom(kF.internal); dEps.copyFrom(epsF.internal); dNut.copyFrom(nutF.internal);
        dbK   = buildDeviceBoundary(kF,   fvp, g);
        dbEps = buildDeviceBoundary(epsF, fvp, g);
        dbNut = buildDeviceBoundary(nutF, fvp, g);

        // Wall geometry for the wall functions. wallU is the patch velocity; a static mesh with no
        // movingWallVelocity means the field's own boundary value is what the solver imposes.
        std::vector<std::vector<vector>> wallU(fvp.size());
        for (std::size_t pi = 0; pi < fvp.size(); ++pi) wallU[pi] = f.U.boundary[pi]->value();
        wall = buildDeviceWallData(m, g, fvp, wallU);

        if (sstModel)
        {
            // F1/F2 blend on the wall distance at EVERY cell (kOmegaSSTBase.C), not the near-wall face
            // distance the wall functions use. Two different quantities with the same symbol in OpenFOAM.
            dY.copyFrom(cellWallDist(m, g, fvp));
            readKOmegaSSTCoeffs(turbProps.subDict("RAS"), sstCoeffs);
        }

        {
            const std::vector<std::vector<scalar>> yW = nearWallDist(m, g, fvp);
            std::vector<label> isW; std::vector<scalar> yv;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                if (isCoupledInterfaceType(fvp[pi].type)) continue;
                const bool isWall = (fvp[pi].type == "wall");
                for (label i = 0; i < fvp[pi].size; ++i)
                {
                    isW.push_back(isWall ? 1 : 0);
                    yv.push_back(isWall ? yW[pi][i] : 0.0);
                }
            }
            bndIsWall.copyFrom(isW);
            bndY.copyFrom(yv);
        }

        if (const FoamDict* rf = fvSolution.subDict("relaxationFactors"))
            if (const FoamDict* eq = rf->subDict("equations"))
            {
                relaxK   = eq->scalarOr("k", relaxK);
                relaxEps = eq->scalarOr(secondField, relaxEps);   // `omega` under kOmegaSST
            }

        // Seed nuEff from the nut just read, so iteration 1 already uses it.
        {
            DeviceBuffer<scalar> nb;
            deviceBoundaryNut(dbNut, bndIsWall, bndY, dK, dNut, nu, nb);
            const std::vector<scalar> nutC = dNut.host(), nutB = nb.host();
            for (label c = 0; c < nC; ++c) nuEffC[c] = nu + nutC[c];
            std::size_t j = 0;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                if (isCoupledInterfaceType(fvp[pi].type)) continue;
                for (label i = 0; i < fvp[pi].size; ++i, ++j)
                    if (j < nutB.size()) nuEffB[pi][i] = nu + nutB[j];
            }
            const SurfaceScalarField nf2 = cpu::effectiveFaceViscosity(nuEffC, nuEffB, m, g, fvp);
            dNuCell.copyFrom(nuEffC);
            dNuFace.copyFrom(nf2.internal);
            std::vector<scalar> flat;
            for (const auto& v : nuEffB) for (scalar x : v) flat.push_back(x);
            flat.resize(dm.nBndFaces, nu);
            dNuBnd.copyFrom(flat);
        }

        in.correct = [&]()
        {
            // Phase timing, on demand: the wall-clock investigation needed to know how much of an outer
            // iteration is the turbulence hook (which round-trips nut to the host) versus the solve.
            const auto tHook0 = std::chrono::steady_clock::now();
            // Solver settings from the CASE, exactly as for U and p. This was hardcoded to tol=1e-10
            // with relTol 0 and BiCGStab, so k and epsilon were driven to near machine precision every
            // outer iteration while the case asks for 1e-05 / relTol 0.1 / symGaussSeidel. Measured: the
            // turbulence hook was 168 ms of a ~300 ms outer iteration on 440k cells -- 56% of the run --
            // and almost none of that was the nuEff host copy (6 ms); it was over-solving these two.
            if (sstModel)
                // Mirrors kOmegaSSTBase::correct(): GbyNu0 -> F1/F2/CDkOmega/S2 -> omega eqn (loose solve,
                // omega-wall setValues) -> bound -> k eqn -> bound -> correctNut (Bradshaw limiter).
                deviceKOmegaSSTCorrect(dm, wall, dbEps, dbK, dbU, gf.Ux, gf.Uy, gf.Uz,
                                       dK, dEps, dNut, dY, gf.phiInt, gf.phiBnd,
                                       nu, relaxEps, relaxK, tolKE,
                                       /*bounded*/false, /*boundedEps*/false,
                                       /*limitedK*/false, /*limitedOmega*/false, 2.0, 2.0,
                                       sstCoeffs, relTolKE, /*keCheckEvery*/1,
                                       /*linearUpwindK*/false, /*linearUpwindOmega*/false,
                                       /*nonOrth*/false, /*gradULimitK*/0.0, gsKE, gsKE);
            else
            deviceKEpsilonCorrect(dm, wall, dbEps, dbK, dbU, gf.Ux, gf.Uy, gf.Uz,
                                  dK, dEps, dNut, gf.phiInt, gf.phiBnd,
                                  nu, relaxEps, relaxK, tolKE,
                                  /*bounded*/false, /*boundedEps*/false,
                                  /*limitedK*/false, /*limitedEps*/false, 2.0, 2.0,
                                  KEpsilonCoeffs{}, relTolKE, /*keCheckEvery*/1,
                                  /*linearUpwindK*/false, /*linearUpwindEps*/false, /*nonOrth*/false,
                                  gsKE, gsKE);

            const auto tRefresh0 = std::chrono::steady_clock::now();
            // nuEff for the NEXT iteration: nu + nut, with the boundary value from the wall function --
            // NOT the owner cell's. That distinction is the defect that once made boundary viscosity
            // 2000x too small; deviceBoundaryNut is what applies the wall function per face.
            DeviceBuffer<scalar> nb;
            deviceBoundaryNut(dbNut, bndIsWall, bndY, dK, dNut, nu, nb);

            const std::vector<scalar> nutC = dNut.host(), nutB = nb.host();
            for (label c = 0; c < nC; ++c) nuEffC[c] = nu + nutC[c];
            std::size_t j = 0;
            for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            {
                if (isCoupledInterfaceType(fvp[pi].type)) continue;
                for (label i = 0; i < fvp[pi].size; ++i, ++j)
                    if (j < nutB.size()) nuEffB[pi][i] = nu + nutB[j];
            }
            const SurfaceScalarField nf2 = cpu::effectiveFaceViscosity(nuEffC, nuEffB, m, g, fvp);
            dNuCell.copyFrom(nuEffC);
            dNuFace.copyFrom(nf2.internal);
            std::vector<scalar> flat;
            for (const auto& v : nuEffB) for (scalar x : v) flat.push_back(x);
            flat.resize(dm.nBndFaces, nu);
            dNuBnd.copyFrom(flat);
            const auto tNow = std::chrono::steady_clock::now();
            g_hookSeconds    += std::chrono::duration<double>(tNow - tHook0).count();
            g_refreshSeconds += std::chrono::duration<double>(tNow - tRefresh0).count();
        };
    }

    in.nuEffCell = &dNuCell; in.nuEffFace = &dNuFace; in.nuEffBndFace = &dNuBnd;
    in.relaxU = relaxU; in.relaxP = relaxP;

    // ---- linear-solver controls, from the case ------------------------------------------------
    // These were HARDCODED at tolerance 1e-10 / relTol 0, which is four to five orders tighter than any
    // case asks for. SIMPLE deliberately solves each inner system LOOSELY -- pitzDaily says
    // `relTol 0.1` -- because the outer iteration is what converges, not the inner solve. Ignoring that
    // does not give a wrong answer, it gives the right answer for roughly ten times the linear-algebra
    // work: measured 614 fine-grid SpMV per SIMPLE iteration against the existing solver's 62.
    //
    // maxIter follows OpenFOAM's lduMatrix::solver default of 1000, and is READ, because an `maxIter 10`
    // in a case is a cap on the answer and not a performance hint.
    if (const FoamDict* solvers = fvSolution.subDict("solvers"))
    {
        // subDict resolves OpenFOAM regex keys, so `"(U|k|epsilon|omega|f|v2)"` is found by "U".
        if (const FoamDict* sp = solvers->subDict("p"))
        {
            in.tolP    = sp->scalarOr("tolerance", in.tolP);
            in.relTolP = sp->scalarOr("relTol", 0.0);
            in.maxIter = static_cast<int>(sp->scalarOr("maxIter", 1000));
        }
        // k and epsilon take their own entry, which in most tutorials is the same regex key as U's.
        if (const FoamDict* sk = solvers->subDict("k"))
        {
            tolKE    = sk->scalarOr("tolerance", tolKE);
            relTolKE = sk->scalarOr("relTol", 0.0);
            gsKE     = (sk->wordOr("solver", "") == "smoothSolver" &&
                        (sk->wordOr("smoother", "") == "symGaussSeidel" ||
                         sk->wordOr("smoother", "") == "GaussSeidel"));
            std::printf("  k/epsilon solves: tol=%.1e relTol=%.3g  solver=%s\n",
                        tolKE, relTolKE, gsKE ? "smoothSolver/symGaussSeidel" : "Jacobi-BiCGStab");
        }
        if (const FoamDict* su = solvers->subDict("U"))
        {
            in.tolU    = su->scalarOr("tolerance", in.tolU);
            in.relTolU = su->scalarOr("relTol", 0.0);
            // OpenFOAM's selection, exactly: `solver smoothSolver` + a GaussSeidel-family smoother.
            // Anything else (PBiCG[Stab], GAMG, ...) keeps Jacobi-BiCGStab, which is announced below as
            // the substitution it is.
            const std::string usolv = su->wordOr("solver", "");
            const std::string usm   = su->wordOr("smoother", "");
            in.uSymGaussSeidel = (usolv == "smoothSolver" &&
                                  (usm == "symGaussSeidel" || usm == "GaussSeidel"));
            if (!in.uSymGaussSeidel && !usolv.empty())
                std::printf("NOTICE (simpleFoam v2): system/fvSolution asks for `%s` on U; brae runs a "
                            "Jacobi-preconditioned BiCGStab instead. Same matrix, different Krylov "
                            "method and iteration count.\n", usolv.c_str());
        }
    }
    std::printf("  linear solves: p tol=%.1e relTol=%.3g   U tol=%.1e relTol=%.3g   maxIter=%d\n",
                in.tolP, in.relTolP, in.tolU, in.relTolU, in.maxIter);
    // Both are pure execution strategy -- same matrix, same stopping criterion -- so they are on by
    // default and env-overridable for A/B measurement rather than being case settings.
    // The AMG hierarchy is already reused across ITERATIONS (SolverWorkspace::amgBuilt); this reuses it
    // across RUNS. Opt-in, because it writes a file into the case.
    if (const char* e = std::getenv("BRAE_AMG_CACHE"))
        if (e[0] == '1') in.amgCacheDir = caseDir + "/constant/polyMesh";
    if (const char* e = std::getenv("BRAE_VCYCLE_GRAPH")) in.captureVcycle = (e[0] == '1');
    if (const char* e = std::getenv("BRAE_PCG_CHECK_EVERY")) in.pcgCheckEvery = std::max(1, std::atoi(e));
    std::printf("  pressure: V-cycle graph %s, residual read every %d PCG iteration(s)\n",
                in.captureVcycle ? "ON" : "off", in.pcgCheckEvery);
    std::printf("  U solver: %s\n",
                in.uSymGaussSeidel ? "smoothSolver/symGaussSeidel (as the case asks)"
                                   : "Jacobi-BiCGStab");
    in.momentumPredictor = cd.momentumPredictor;
    in.nNonOrthogonalCorrectors = cd.nNonOrthogonalCorrectors;
    in.pRefCell = f.pRefCell; in.pRefValue = f.pRefValue;
    in.takeUAtBoundary = &dTakeU; in.adjustable = &dAdjust;
    in.consistent = cd.consistent;
    if (in.consistent)
        std::printf("  SIMPLE/consistent: SIMPLEC, rAtU = 1/(1/rAU - UEqn.H1())\n");
    // `bounded Gauss <scheme>`: -fvm::Sp(fvc::div(phi), U), implemented on both paths and matched to
    // 2.9e-16 (tests/test_ueqn_cuda.cu), so it is READ rather than refused.
    {
        const std::string sc = divUScheme(caseDir);
        in.scheme = sc == "linearUpwind"   ? cpu::DivScheme::linearUpwind
                  : sc == "limitedLinear"  ? cpu::DivScheme::limitedLinear
                  : sc == "limitedLinearV" ? cpu::DivScheme::limitedLinearV
                  : sc == "LUST"           ? cpu::DivScheme::LUST
                                           : cpu::DivScheme::upwind;
        in.linearUpwind = (in.scheme == cpu::DivScheme::linearUpwind);
        in.schemeCoeff = divUSchemeCoeff(caseDir, 1.0);
        std::printf("  div(phi,U) scheme: %s", sc.empty() ? "upwind" : sc.c_str());
        if (in.scheme == cpu::DivScheme::limitedLinear || in.scheme == cpu::DivScheme::limitedLinearV)
            std::printf(" %g", in.schemeCoeff);
        std::printf("\n");
    }
    in.bounded = divUBounded(caseDir);
    if (in.bounded) std::printf("  div(phi,U) is bounded: applying -fvm::Sp(fvc::div(phi), U)\n");

    // The non-orthogonal correction, both halves: nonOrthDeltaCoeffs in the implicit coefficients and the
    // deferred correction in the source, in the momentum equation and the pressure equation alike. Matched
    // to the reference to 5e-16 (tests/test_ueqn_cuda.cu, tests/test_peqn_cuda.cu) and gated against real
    // OpenFOAM on a non-orthogonal mesh (tests/nonorth_vs_openfoam.sh), so it is READ rather than refused.
    in.correctedLaplacian = laplacianScheme(caseDir).corrected;
    std::printf("  laplacianSchemes: non-orthogonal correction %s\n",
                in.correctedLaplacian ? "ON (`corrected`)" : "OFF (`uncorrected`/`orthogonal`)");

    // ---- the SIMPLE loop ---------------------------------------------------------------------
    SolverWorkspace ws;
    std::map<std::string, scalar> residuals;
    label iter = 0;
    while (ctl.loop(iter + 1, endTime, residuals))
    {
        ++iter;
        residuals = simpleStep(gf, ws, dm, dbU, dbP, in);
        std::printf("Time = %d   U initial residual = %.6e   p initial residual = %.6e\n",
                    static_cast<int>(iter),
                    residuals.count("U") ? residuals.at("U") : 0.0,
                    residuals.count("p") ? residuals.at("p") : 0.0);
    }
    if (ctl.converged())
        std::printf("SIMPLE solution converged in %d iterations\n", static_cast<int>(iter));
    if (std::getenv("BRAE_PHASE_TIME"))
        std::printf("  [phase] turbulence hook %.3f s (%.1f ms/iter), of which the nuEff host "
                    "round-trip %.3f s (%.1f ms/iter), over %d iterations\n",
                    g_hookSeconds,    iter ? 1e3 * g_hookSeconds    / (double)iter : 0.0,
                    g_refreshSeconds, iter ? 1e3 * g_refreshSeconds / (double)iter : 0.0,
                    static_cast<int>(iter));

    // ---- write -------------------------------------------------------------------------------
    {
        const std::string outDir = caseDir + "/" + std::to_string(static_cast<int>(iter));
        std::filesystem::create_directories(outDir);
        const std::string src = caseDir + "/" + startTime;

        const std::vector<scalar> pOut = gf.p.host();
        const std::vector<scalar> ux = gf.Ux.host(), uy = gf.Uy.host(), uz = gf.Uz.host();
        std::vector<vector> UOut(nC);
        for (label c = 0; c < nC; ++c) UOut[c] = {ux[c], uy[c], uz[c]};

        writeVolField<scalar>(src + "/p", outDir + "/p", pOut, fvp);
        writeVolField<vector>(src + "/U", outDir + "/U", UOut, fvp);
        if (ras)
        {
            writeVolField<scalar>(src + "/k",       outDir + "/k",       dK.host(),   fvp);
            writeVolField<scalar>(src + "/" + secondField, outDir + "/" + secondField, dEps.host(), fvp);
            writeVolField<scalar>(src + "/nut",     outDir + "/nut",     dNut.host(), fvp);
        }
        std::printf("written %s/{U,p%s}\n", outDir.c_str(), ras ? ",k,epsilon,nut" : "");
    }
    return static_cast<int>(iter);
}

} // namespace gpu
} // namespace brae
