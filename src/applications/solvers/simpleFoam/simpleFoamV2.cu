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
#include "near_wall_dist.cuh"
#include "solver_dispatch.cuh"   // readDdtSchemeWord: the same steady/transient test the dispatcher uses

#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <regex>
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

// Does the case ask for the NON-ORTHOGONAL correction? OpenFOAM's `corrected` snGrad / `Gauss linear
// corrected` laplacian add the explicit over-relaxed correction for mesh non-orthogonality; only
// `orthogonal` and `uncorrected` switch it off.
//
// The rebuilt path's fvm::laplacian is ORTHOGONAL ONLY (fvm.cuh:4). On a non-orthogonal mesh omitting the
// correction leaves the pressure equation inconsistent with the flux, so continuity never closes and the
// residual plateaus instead of falling -- measured on pitzDaily, where the existing GPU solver's Ux
// residual falls 1 -> 0.234 over 20 iterations while the rebuilt path oscillates around 0.4. It is not a
// small error that accumulates; it is a term that is simply absent.
bool nonOrthogonalCorrection(const std::string& caseDir)
{
    std::string text;
    try { text = readFileExpanded(caseDir + "/system/fvSchemes"); } catch (...) { return false; }
    for (const char* blkName : {"laplacianSchemes", "snGradSchemes"})
    {
        const std::size_t blk = text.find(blkName);
        if (blk == std::string::npos) continue;
        const std::size_t open = text.find('{', blk);
        if (open == std::string::npos) continue;
        const std::size_t close = text.find('}', open);
        const std::string b = text.substr(open, (close == std::string::npos ? text.size() : close) - open);
        // OF's default when the word is absent is `corrected`; `orthogonal`/`uncorrected` disable it.
        if (b.find("orthogonal") != std::string::npos && b.find("nonOrthogonal") == std::string::npos)
            continue;                       // `orthogonal` -- correction off
        if (b.find("uncorrected") != std::string::npos) continue;
        if (b.find("corrected") != std::string::npos || b.find("limited") != std::string::npos)
            return true;
    }
    return false;
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
        if (switchOn(*s, "consistent", false))
            r.blockers.push_back("SIMPLE/consistent is set (SIMPLEC). pEqn.H:10-16 then replaces rAU with "
                                 "1/(1/rAU - UEqn.H1()) and corrects phiHbyA/HbyA with fvc::snGrad(p); "
                                 "neither H1() nor snGrad is ported.");
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
        if (divUBounded(caseDir))
            r.blockers.push_back("div(phi,U) is `bounded`, which adds -fvm::Sp(fvc::div(phi), U) to the "
                                 "momentum equation. The rebuilt UEqn does not implement it; the term is "
                                 "zero only at convergence, so omitting it changes the path there.");

        if (nonOrthogonalCorrection(caseDir))
            r.blockers.push_back("laplacianSchemes/snGradSchemes ask for the NON-ORTHOGONAL correction "
                                 "(`corrected`/`limited`); the rebuilt path's fvm::laplacian is orthogonal "
                                 "only. On a non-orthogonal mesh the term is not a refinement -- without it "
                                 "the pressure equation and the flux disagree and continuity never closes.");

        const std::string sc = divUScheme(caseDir);
        if (!sc.empty() && sc != "upwind")
            r.blockers.push_back("div(phi,U) asks for `" + sc + "`; the rebuilt UEqn implements `upwind` "
                                 "implicit weights only. Running it anyway would solve a different "
                                 "discretisation than the case specifies.");
    }

    // --- turbulence ---------------------------------------------------------------------------
    {
        const std::string simType = turbProps.wordOr("simulationType", "laminar");
        if (simType == "RAS")
        {
            const FoamDict* ras = turbProps.subDict("RAS");
            const std::string model = ras ? ras->wordOr("RASModel", "") : "";
            if (model != "kEpsilon")
                r.blockers.push_back("RASModel is `" + model + "`; the rebuilt path wires kEpsilon only.");
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

    StepInput in;
    if (ras)
    {
        kF   = buildField<scalar>(readField<scalar>(caseDir + "/" + startTime + "/k"), fvp, nC);
        epsF = buildField<scalar>(readField<scalar>(caseDir + "/" + startTime + "/epsilon"), fvp, nC);
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
                relaxEps = eq->scalarOr("epsilon", relaxEps);
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
            deviceKEpsilonCorrect(dm, wall, dbEps, dbK, dbU, gf.Ux, gf.Uy, gf.Uz,
                                  dK, dEps, dNut, gf.phiInt, gf.phiBnd,
                                  nu, relaxEps, relaxK, 1e-10);

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
        };
    }

    in.nuEffCell = &dNuCell; in.nuEffFace = &dNuFace; in.nuEffBndFace = &dNuBnd;
    in.relaxU = relaxU; in.relaxP = relaxP;
    in.momentumPredictor = cd.momentumPredictor;
    in.nNonOrthogonalCorrectors = cd.nNonOrthogonalCorrectors;
    in.pRefCell = f.pRefCell; in.pRefValue = f.pRefValue;
    in.takeUAtBoundary = &dTakeU; in.adjustable = &dAdjust;
    in.consistent = cd.consistent;

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
            writeVolField<scalar>(src + "/epsilon", outDir + "/epsilon", dEps.host(), fvp);
            writeVolField<scalar>(src + "/nut",     outDir + "/nut",     dNut.host(), fvp);
        }
        std::printf("written %s/{U,p%s}\n", outDir.c_str(), ras ? ",k,epsilon,nut" : "");
    }
    return static_cast<int>(iter);
}

} // namespace gpu
} // namespace brae
