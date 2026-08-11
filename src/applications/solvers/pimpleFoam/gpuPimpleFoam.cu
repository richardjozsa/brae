// cf gpuPimpleFoam -- transient incompressible PIMPLE solver, single-GPU device-resident. Reads a standard OpenFOAM
// case (controlDict / fvSolution / fvSchemes / transportProperties / turbulenceProperties + start fields) and marches
// the transient loop on the GPU via DeviceSimpleSolver::pimpleStep -- the SAME three composable phases as steady brae
// (momentum predictor / pressure-velocity / turbulence), with the implicit fvm::ddt(U) folded into the predictor.
// Writes standard OpenFOAM time directories.
//
// Scope: Euler/backward/CrankNicolson ddt; laminar, RAS (kEpsilon/realizableKE/kOmegaSST/kOmegaSSTLM/SpalartAllmaras),
// or DES/LES (SA-DDES/IDDES, kOmegaSST-DDES/IDDES, Smagorinsky). Reuses the steady driver's field I/O
// (foam_field_reader/writer) + dict parsing (foam_dict) + turbulence model setup. BOTH the momentum ddt AND the
// turbulence transport ddt (fvm::ddt(k/eps/omega/nuTilda)) are fully implicit + OF-exact -- the turbulence is transient
// URANS/DES, not quasi-steady (see device_simple_foam.cu:294). fvSchemes div/laplacian schemes are parsed; phi output +
// restart, CrankNicolson ddt0 restart, coded (fixedValue/mixed) BCs and forceCoeffs (Cd/Cl/Cm, sampled on the write
// cadence to postProcessing/forceCoeffs/) are wired. NOT yet: MRF, fvOptions,
// adjustTimeStep + maxCo (adaptive dt), a general functionObject framework (forceCoeffs above is hard-wired, NOT
// a framework -- probes/fieldAverage/sampling/surfaces are still absent and are silently ignored), distributed
// (multi-GPU) transient. The first three are REFUSED at start-up rather than silently skipped (see main).
// Kept a SEPARATE executable (brae_pimpleFoam) so it cannot regress the validated steady brae; `brae` hands
// over to it whenever a case's controlDict says `application pimpleFoam` (solvers/common/solver_dispatch.cuh).
#include "primitive_mesh.cuh"
#include "../common/read_surface_field.cuh"   // readSurfaceField / readPhiIfPresent (OF READ_IF_PRESENT)
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_field_writer.cuh"
#include "foam_dict.cuh"
#include "mrf_read.cuh"             // readMRFProperties: only to REFUSE an MRF case (not applied transient yet)
#include "linear_solver_setup.cuh"   // readLinearSolverControls: shared with gpuSimpleFoam/gpuRhoSimpleFoam
#include "dict_audit.cuh"
#include "scheme_parse.cuh"         // parseFvSchemesControls: shared fvSchemes div/laplacian scheme parse
#include "turbulence_setup.cuh"    // readTurbulenceModel + readTurbulenceFields (shared with brae)
#include "komega_sst_coeffs.cuh"    // readKOmegaSSTCoeffs
#include "fvc.cuh"
#include "forces.cuh"               // wallForces + forceCoeffs (shared with gpuSimpleFoam)
#include "device_simple_foam.cuh"
#include "coded_bc_setup.cuh"       // CodedBCSpec + parseCodedBCs + setupCodedBCs (shared with gpuSimpleFoam)
#include "brae_time.cuh"
#include "scalar_transport_fo.cuh"   // OF functionObjects::scalarTransport, on the device flux   // OF Time/functionObjectList lifecycle, owned centrally (not per solver)
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

using namespace brae;

namespace {

std::string timeName(scalar t)
{
    if (t == std::floor(t) && std::fabs((double)t) < 1e15)
        return std::to_string((long long)std::llround((double)t));
    char b[64];
    std::snprintf(b, sizeof b, "%g", (double)t);
    return std::string(b);
}

// fvSchemes holds multi-token scheme values + $-vars (not a plain key/value dict), so text-parse ddtSchemes.default.
DdtScheme parseDdtScheme(const std::string& fvSchemesPath, scalar& ocCoeff)
{
    ocCoeff = 1.0;                                    // CrankNicolson off-centring; default = pure CN (1)
    std::ifstream f(fvSchemesPath);
    if (!f) throw std::runtime_error("cannot read " + fvSchemesPath);
    const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const std::size_t blk = text.find("ddtSchemes");
    if (blk == std::string::npos)
        throw std::runtime_error("fvSchemes has no ddtSchemes block (transient pimpleFoam needs Euler|backward).");
    std::size_t d = text.find("default", blk);
    if (d == std::string::npos) throw std::runtime_error("fvSchemes ddtSchemes has no 'default' entry.");
    d += 7;
    while (d < text.size() && std::isspace((unsigned char)text[d])) ++d;
    std::string w;
    while (d < text.size() && !std::isspace((unsigned char)text[d]) && text[d] != ';') w += text[d++];
    if (w == "steadyState")
        throw std::runtime_error("fvSchemes ddtSchemes.default = steadyState -> use the steady solver 'brae'.");
    if (w == "backward")                return DdtScheme::backward;
    if (w == "Euler" || w == "bounded") return DdtScheme::Euler;
    if (w == "CrankNicolson")
    {
        // read the off-centring coefficient (OF "CrankNicolson <ocCoeff>", oc in [0,1]); absent -> pure CN (1).
        while (d < text.size() && std::isspace((unsigned char)text[d])) ++d;
        std::string oc;
        while (d < text.size() && !std::isspace((unsigned char)text[d]) && text[d] != ';') oc += text[d++];
        if (!oc.empty()) { try { ocCoeff = std::stod(oc); } catch (...) {} }
        return DdtScheme::CrankNicolson;
    }
    throw std::runtime_error("fvSchemes ddtSchemes.default '" + w + "' unsupported (Euler|backward|CrankNicolson|steadyState).");
}



// CodedBCSpec + parseCodedBCs + setupCodedBCs now live in coded_bc_setup.cuh (shared with gpuSimpleFoam).

void printUsage()
{
    std::printf(
"brae_pimpleFoam, the GPU-native transient incompressible solver (PIMPLE). Normally reached through `brae`,\n"
"which hands over any case whose controlDict says `application pimpleFoam`.\n\n"
"Usage:\n"
"  brae_pimpleFoam [-case <dir>]  march an OpenFOAM case in time (default: the current directory)\n"
"  brae_pimpleFoam <dir>          same, case directory as a positional argument\n"
"  brae_pimpleFoam --help         show this message\n\n"
"Needs ddtSchemes.default = Euler | backward | CrankNicolson and a PIMPLE dict in fvSolution. Writes standard\n"
"OpenFOAM time directories (U, p, phi, turbulence) that restart seamlessly with startFrom latestTime.\n\n"
"Docs: https://github.com/simd-ai/brae/blob/main/docs/solvers/pimplefoam.md\n");
}

}  // namespace

int main(int argc, char** argv)
try
{
    // Same argument surface as the steady driver, so `brae -case <dir>` forwards here verbatim on dispatch.
    std::string caseDir = ".";
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--help" || a == "-h")        { printUsage(); return 0; }
        else if (a == "-case" && i + 1 < argc) caseDir = argv[++i];
        else if (a[0] != '-')                  caseDir = a;
    }

    // ---- case dictionaries ----
    const FoamDict controlDict = readDict(caseDir + "/system/controlDict");
    const FoamDict fvSolution  = readDict(caseDir + "/system/fvSolution");
    const FoamDict transport   = readDict(caseDir + "/constant/transportProperties");
    scalar ddtOcCoeff = 1.0;
    const DdtScheme ddtScheme  = parseDdtScheme(caseDir + "/system/fvSchemes", ddtOcCoeff);

    scalar       startTime     = controlDict.scalarOr("startTime", 0.0);   // resolved below for startFrom latestTime (restart)
    const scalar deltaT        = controlDict.scalarOr("deltaT", 1e-3);
    const scalar endTime       = controlDict.scalarOr("endTime", 1.0);
    const std::string writeControl = controlDict.wordOr("writeControl", "timeStep");
    const scalar writeInterval = controlDict.scalarOr("writeInterval", 1e30);
    const int    purgeWrite    = std::max(0, controlDict.intOr("purgeWrite", 0));
    const int    precision     = controlDict.intOr("writePrecision", 16);
    if (deltaT <= 0)          throw std::runtime_error("controlDict deltaT must be > 0.");
    if (endTime <= startTime) throw std::runtime_error("controlDict endTime must be > startTime.");

    // Physics the STEADY driver applies and this one does not yet: refuse the case rather than march it with the
    // source terms quietly missing (an MRF rotation or an fvOptions source dropped is wrong physics that still
    // produces a plausible-looking field). Same rule the steady driver applies to unsupported fvOptions sources.
    if (readMRFProperties(caseDir + "/constant").active)
        throw std::runtime_error(
            "constant/MRFProperties has an active zone, and brae's transient solver does not apply MRF yet "
            "(the steady solver does). Marching this case would silently drop the rotation.");
    for (const std::string& fvo : {caseDir + "/system/fvOptions", caseDir + "/constant/fvOptions"})
        if (std::filesystem::exists(fvo))
            throw std::runtime_error(
                fvo + " is present, and brae's transient solver does not apply fvOptions yet (the steady solver "
                "does). Marching this case would silently drop those sources.");

    // adjustTimeStep needs the Courant-limited dt that brae does not compute yet; running the case at the fixed
    // deltaT instead is a different simulation, so say so rather than quietly ignore the switch.
    if (controlDict.wordOr("adjustTimeStep", "no") == "yes")
        throw std::runtime_error(
            "controlDict adjustTimeStep yes is not supported yet (no Courant-based dt). Set adjustTimeStep no and "
            "a fixed deltaT; keep it under your target maxCo.");

    // ---- PIMPLE + solver controls (fvSolution) ----
    const FoamDict* pimple  = fvSolution.subDict("PIMPLE");
    const int nOuter   = pimple ? std::max(1, pimple->intOr("nOuterCorrectors", 1)) : 1;
    const int nCorr    = pimple ? std::max(1, pimple->intOr("nCorrectors", 1)) : 1;
    const int nNonOrth = pimple ? pimple->intOr("nNonOrthogonalCorrectors", 0) : 0;
    const scalar nu = transport.scalarOr("nu", 1e-5);

    DeviceSimpleControls ctl;
    ctl.nu = nu;
    // relaxationFactors come from readRelaxationFactors BELOW (with the turbulence model known). Read here
    // it took epsilon/omega's factor from "k" -- `omega 0.4` was ignored -- skipped the legacy flat form,
    // and had no alpha<=0 guard, where OF's fvMatrix::relax skips relaxation for alpha <= 0.
    // tolerances / relTol / smoothSolver selection come from readLinearSolverControls BELOW, once the
    // turbulence model is known (it picks k+omega vs k+epsilon vs nuTilda). Reading them here read only
    // "k" -- a case with a tighter omega tolerance got the loose one -- and never read relTol or
    // smoothSolver at all, so every transient solve was absolute-tolerance BiCGStab whatever fvSolution said.
    ctl.nNonOrth = nNonOrth;
    ctl.caseDir = caseDir;                                          // AMG-hierarchy cache lives in <caseDir>/constant/polyMesh
    ctl.writeCache = std::getenv("BRAE_MESH_CACHE") != nullptr;     // BRAE_MESH_CACHE=1 -> build the AMG hierarchy ONCE, cache it, reload next run
    // (BRAE_AMG_SA=1 enables smoothed aggregation -- the mesh-independent fast path -- read inside buildAMG; when set at
    //  cache-build time the SA hierarchy is what gets cached, so later runs reload the SA hierarchy directly.)
    // fvSchemes div/laplacian/grad scheme flags (2nd-order div(phi,U) linearUpwind, non-orth correction, limitedLinear
    // turbulence scalars, ...). Same parse as steady brae -> the transient solve honours the case's schemes, not upwind.
    parseFvSchemesControls(caseDir, ctl);

    // ---- turbulence model (constant/turbulenceProperties). Absent -> laminar. ----
    std::string secondName = "epsilon";
    if (std::filesystem::exists(caseDir + "/constant/turbulenceProperties"))
    {
        const FoamDict turbProps = readDict(caseDir + "/constant/turbulenceProperties");
        const std::string simType = turbProps.wordOr("simulationType", "laminar");
        if (simType != "RAS" && simType != "laminar" && simType != "LES")
            throw std::runtime_error("pimpleFoam: unsupported simulationType '" + simType + "' (RAS, LES or laminar).");
        ctl.turbulent = (simType == "RAS" || simType == "LES");   // LES here == SA-DDES (a URANS-based hybrid); readTurbulenceModel sets ctl.des
        readTurbulenceModel(turbProps, ctl);
        secondName = ctl.sst ? "omega" : "epsilon";
    }

    // fvSolution -> ctl, through the SHARED reader. Must come AFTER the turbulence model is read: it
    // branches on ctl.turbulent/sa/sst to pick which scalar solver dicts to look at. "PIMPLE" is the
    // algorithm dict -- OF's solutionControl reads consistent/nNonOrthogonalCorrectors from the dict
    // named by algorithmName_, "PIMPLE" here (pimpleControl.H:135), not "SIMPLE".
    readLinearSolverControls(fvSolution, secondName, ctl, "PIMPLE");
    readRelaxationFactors(fvSolution, ctl);

    // ---- mesh + start fields ----
    PrimitiveMesh m; m.read(caseDir + "/constant/polyMesh");
    FvGeometry g; g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();
    // startFrom latestTime -> RESTART: resume from the newest numeric time dir that holds a U field (else startTime).
    std::string startStr = timeName(startTime);
    if (controlDict.wordOr("startFrom", "startTime") == "latestTime")
    {
        namespace fs = std::filesystem;
        std::error_code ec; scalar best = 0; std::string bestName; bool found = false;
        for (const auto& e : fs::directory_iterator(caseDir, ec))
        {
            if (!e.is_directory()) continue;
            const std::string nm = e.path().filename().string();
            char* end = nullptr; const scalar t = std::strtod(nm.c_str(), &end);
            if (end == nm.c_str() || *end != '\0') continue;             // not a pure number -> not a time dir
            if (!fs::exists(e.path() / "U")) continue;
            if (!found || t > best) { best = t; bestName = nm; found = true; }
        }
        if (found) { startTime = best; startStr = bestName;
            std::printf("gpuPimpleFoam: startFrom latestTime -> resuming from time '%s'\n", bestName.c_str()); }
    }
    const std::string fieldDir = caseDir + "/" + startStr;
    GeometricField<vector> U = buildField<vector>(readField<vector>(fieldDir + "/U"), fvp, nC); U.evaluateBoundary();
    GeometricField<scalar> p = buildField<scalar>(readField<scalar>(fieldDir + "/p"), fvp, nC); p.evaluateBoundary();
    // phi: on restart READ the previously-written conservative face flux (surfaceScalarField) so the resumed run
    // continues the EXACT flux state (the >=17-digit write makes the phi write->read round-trip bit-identical) -- no
    // first-step continuity transient. The overall restart is seamless to ~1e-10 (nut re-validated at startup, like OF),
    // not literally bit-for-bit. Fresh start / no phi -> recompute from U (fvc::flux).
    SurfaceScalarField phi;
    const std::string phiPath = fieldDir + "/phi";
    if (std::filesystem::exists(phiPath))
    {
        phi = readSurfaceField(phiPath, fvp, m.nInternalFaces());
        std::printf("gpuPimpleFoam: read phi from %s (restart flux)\n", phiPath.c_str());
    }
    else
        phi = fvc::flux(U, m, g, fvp);

    TurbulenceFields tf = readTurbulenceFields(fieldDir, fvp, nC, ctl, secondName, U);

    DeviceSimpleSolver solver(m, g, fvp, U, p, phi, ctl,
                              (ctl.turbulent && !ctl.les) ? &tf.k : nullptr, (ctl.turbulent && !ctl.sa && !ctl.les) ? &tf.eps : nullptr,
                              ctl.turbulent ? &tf.nut : nullptr, ctl.lm ? &tf.ReThetat : nullptr, ctl.lm ? &tf.gammaInt : nullptr);
    // OF re-evaluates the turbulent-inlet BCs every updateCoeffs; give the solver the per-face masks so it
    // refreshes them each iteration instead of freezing the set-up value.
    solver.setTurbulentInlets(tf.turbInletMasks.tiMask, tf.turbInletMasks.tiIntensity,
                              tf.turbInletMasks.mlMask, tf.turbInletMasks.mlLength);
    solver.setDdtScheme(ddtScheme, ddtOcCoeff);
    // CrankNicolson RESTART: if the previous run wrote the ddt0 fields (present in the restart time dir), reload them and
    // prime the time state so the FIRST resumed step is full CN using the exact stored old ddt -> seamless (like OF's
    // DDt0Field). Fresh start / no ddt0 file -> ddt0 stays 0 and the first step Euler-bootstraps, exactly as before.
    if (ddtScheme == DdtScheme::CrankNicolson && std::filesystem::exists(fieldDir + "/ddt0(U)")) {
        solver.setUddt0(buildField<vector>(readField<vector>(fieldDir + "/ddt0(U)"), fvp, nC).internal);
        if (ctl.turbulent && !ctl.les) {
            const std::string kn = ctl.sa ? "nuTilda" : "k";
            if (std::filesystem::exists(fieldDir + "/ddt0(" + kn + ")"))
                solver.setKddt0(buildField<scalar>(readField<scalar>(fieldDir + "/ddt0(" + kn + ")"), fvp, nC).internal);
            if (!ctl.sa && std::filesystem::exists(fieldDir + "/ddt0(" + secondName + ")"))
                solver.setE2ddt0(buildField<scalar>(readField<scalar>(fieldDir + "/ddt0(" + secondName + ")"), fvp, nC).internal);
            if (ctl.lm) {
                if (std::filesystem::exists(fieldDir + "/ddt0(ReThetat)")) solver.setReThetatddt0(buildField<scalar>(readField<scalar>(fieldDir + "/ddt0(ReThetat)"), fvp, nC).internal);
                if (std::filesystem::exists(fieldDir + "/ddt0(gammaInt)")) solver.setGammaIntddt0(buildField<scalar>(readField<scalar>(fieldDir + "/ddt0(gammaInt)"), fvp, nC).internal);
            }
        }
        solver.setCnRestart(deltaT);
        std::printf("gpuPimpleFoam: read CrankNicolson ddt0 from %s (seamless CN restart)\n", fieldDir.c_str());
    }
    // codedFixedValue / codedMixed on U / p / turbulence scalars -> NVRTC device kernel per coded patch (compiled once;
    // applied each step in the momentum predictor). Runs fully on the GPU -- no host-side per-step BC evaluation.
    setupCodedBCs(solver, fieldDir, fvp, ctl, secondName, "gpuPimpleFoam");

    std::printf("gpuPimpleFoam: deltaT=%g endTime=%g ddt=%s nOuterCorrectors=%d nCorrectors=%d nNonOrth=%d nu=%g "
                "turbulence=%s nCells=%d\n\n",
                (double)deltaT, (double)endTime, ddtScheme == DdtScheme::backward ? "backward" : ddtScheme == DdtScheme::CrankNicolson ? "CrankNicolson" : "Euler",
                nOuter, nCorr, nNonOrth, (double)nu,
                !ctl.turbulent ? "laminar" : ctl.les ? "Smagorinsky (LES)" : ctl.iddes ? (ctl.sst ? "kOmegaSSTIDDES" : "SpalartAllmarasIDDES") : ctl.sa ? "SpalartAllmaras" : ctl.sst ? (ctl.lm ? "kOmegaSSTLM" : "kOmegaSST") : "kEpsilon",
                nC);

    // ---- write cadence (Foam::Time: writeControl timeStep|runTime + writeInterval + purgeWrite FIFO) ----
    long writeTimeIndex = 0;
    std::deque<std::string> writtenTimes;
    auto isWriteTime = [&](long it, scalar tval) -> bool {
        if (writeControl == "timeStep")
            return writeInterval >= 1 && (it % (long)writeInterval) == 0;
        if (writeControl == "runTime" || writeControl == "adjustable" || writeControl == "adjustableRunTime") {
            const long wi = (long)(((tval - startTime) + 0.5 * deltaT) / writeInterval);
            if (wi > writeTimeIndex) { writeTimeIndex = wi; return true; }
            return false;
        }
        return false;
    };
    // Everything a functionObject can need already exists at this point in this driver, so the registry
    // is populated immediately. The lookups still happen on first execute() -- that indirection is what
    // lets the steady drivers build Time at start-up, and costs nothing here.
    ObjectRegistry timeRegistry;
    timeRegistry.store("mesh", &m);
    timeRegistry.store("geometry", &g);
    timeRegistry.store("patches", &fvp);
    timeRegistry.store("solver", &solver);
    std::vector<ScalarTransportFO*> scalarTransports;
    std::vector<std::pair<std::string, FunctionObjectList::Factory>> foTypes;
    foTypes.emplace_back(
        "scalarTransport",
        [&](const std::string& foName, const FoamDict& fd) -> std::unique_ptr<FunctionObject>
        {
            // Only OF's constant-D branch is implemented; nut-based and alphaD/alphaDt are refused by
            // name rather than quietly replaced by a constant (scalarTransport.C D()).
            if (!fd.found("D"))
            {
                noticeIgnored("functions/" + foName,
                              "scalarTransport without a constant `D` (nut-based or alphaD/alphaDt "
                              "diffusivity) is not implemented, so this tracer is NOT solved.");
                return nullptr;
            }
            const std::string fld = fd.wordOr("field", foName);
            // OF: schemesField defaults to the field name, and the scheme is looked up as
            // div(phi,<schemesField>). Absent under `default none` is fatal in OF, so it is here too --
            // reported and declined rather than run with a substituted discretisation.
            const std::string schemesField = fd.wordOr("schemesField", fld);
            FieldDivScheme scheme;
            try { scheme = parseFieldDivScheme(caseDir, schemesField); }
            catch (const std::exception& e)
            {
                noticeIgnored("functions/" + foName, std::string(e.what()) + " -- this tracer is NOT solved.");
                return nullptr;
            }
            auto fo = std::make_unique<ScalarTransportFO>(
                foName, fld, fieldDir + "/" + fld, timeRegistry,
                fd.scalarOr("D", 0.0), fd.scalarOr("relaxCoeff", 1.0), fd.scalarOr("tol", 1e-6),
                scheme);
            scalarTransports.push_back(fo.get());
            return fo;
        });
    Time time(controlDict, nullptr, 0, foTypes, {}, {"forceCoeffs"});

    auto writeTimeDir = [&](const std::string& tname) {
        const std::string outDir = caseDir + "/" + tname;
        std::filesystem::create_directories(outDir);
        // scalarTransport tracers: OF's transported field is AUTO_WRITE (scalarTransport.C
        // transportedField()), so it appears in every written time directory. AFTER the directory
        // exists, and pulled from the device only here -- on the write cadence, which is the reason OF
        // splits write() from execute().
        for (ScalarTransportFO* st : scalarTransports)
            if (st->ready())
                writeVolField(fieldDir + "/" + st->fieldName(), outDir + "/" + st->fieldName(),
                              st->hostField(), fvp, 12);
        std::error_code ec;
        if (std::filesystem::exists(fieldDir + "/include"))
            std::filesystem::copy(fieldDir + "/include", outDir + "/include",
                std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing, ec);
        writeVolField(fieldDir + "/U", outDir + "/U", solver.U(), fvp, precision, solver.UBoundary());
        writeVolField(fieldDir + "/p", outDir + "/p", solver.p(), fvp, precision, solver.pBoundary());
        // phi is the restart-critical conservative flux -> write it LOSSLESS (>=17 = double max_digits10) so its
        // write->read round-trip is bit-identical (16 sig figs loses the last bit) and a restart resumes the EXACT flux
        // with no continuity transient. The viz fields (U/p/turbulence) keep the user's writePrecision.
        writeSurfaceField(outDir + "/phi", solver.phiInternal(), solver.phiBoundary(), fvp, std::max(precision, 17));
        if (ctl.les) {   // pure LES Smagorinsky: only the algebraic sub-grid nut (no k/epsilon/omega/nuTilda field)
            writeVolField(fieldDir + "/nut",     outDir + "/nut",     solver.nut(), fvp, precision, solver.nutBoundary());
        } else if (ctl.sa) {
            writeVolField(fieldDir + "/nuTilda", outDir + "/nuTilda", solver.k(),   fvp, precision, solver.nuTildaBoundary());
            writeVolField(fieldDir + "/nut",     outDir + "/nut",     solver.nut(), fvp, precision, solver.nutBoundary());
        } else if (ctl.turbulent) {
            writeVolField(fieldDir + "/k",          outDir + "/k",          solver.k(),   fvp, precision, solver.kBoundary());
            writeVolField(fieldDir + "/" + secondName, outDir + "/" + secondName, solver.eps(), fvp, precision, solver.epsBoundary());
            writeVolField(fieldDir + "/nut",        outDir + "/nut",        solver.nut(), fvp, precision, solver.nutBoundary());
            if (ctl.lm) {
                writeVolField(fieldDir + "/ReThetat", outDir + "/ReThetat", solver.ReThetat(), fvp, precision);
                writeVolField(fieldDir + "/gammaInt", outDir + "/gammaInt", solver.gammaInt(), fvp, precision);
            }
        }
        if (solver.isCrankNicolson()) {
            // CrankNicolson stored old ddt (ddt0) per transported variable -> LOSSLESS (>=17) so a restart resumes the
            // EXACT recurrence (seamless CN restart, like OF's registered DDt0Field). No 0/ template -> writeVolFieldRaw.
            const int p17 = std::max(precision, 17);
            writeVolFieldRaw(outDir + "/ddt0(U)", "ddt0(U)", "[0 1 -2 0 0 0 0]", solver.Uddt0(), fvp, p17);
            if (ctl.turbulent && !ctl.les) {
                const std::string kn = ctl.sa ? "nuTilda" : "k";
                writeVolFieldRaw(outDir + "/ddt0(" + kn + ")", "ddt0(" + kn + ")", "[0 0 0 0 0 0 0]", solver.kddt0(), fvp, p17);
                if (!ctl.sa)
                    writeVolFieldRaw(outDir + "/ddt0(" + secondName + ")", "ddt0(" + secondName + ")", "[0 0 0 0 0 0 0]", solver.e2ddt0(), fvp, p17);
                if (ctl.lm) {
                    writeVolFieldRaw(outDir + "/ddt0(ReThetat)", "ddt0(ReThetat)", "[0 0 0 0 0 0 0]", solver.reThetatddt0(), fvp, p17);
                    writeVolFieldRaw(outDir + "/ddt0(gammaInt)", "ddt0(gammaInt)", "[0 0 0 0 0 0 0]", solver.gammaIntddt0(), fvp, p17);
                }
            }
        }
        std::printf("written %s\n", outDir.c_str());
        if (purgeWrite > 0) {
            if (writtenTimes.empty() || writtenTimes.back() != tname) writtenTimes.push_back(tname);
            while ((int)writtenTimes.size() > purgeWrite) {
                std::error_code pec;
                std::filesystem::remove_all(caseDir + "/" + writtenTimes.front(), pec);
                writtenTimes.pop_front();
            }
        }
    };

    // ---- forceCoeffs (Cd/Cl/Cm) sampling -------------------------------------------------------------------
    // The steady solver evaluates forces ONCE at convergence (gpuSimpleFoam.cu). A DES has no converged state:
    // the wake sheds, so Cd(t) oscillates and only its TIME AVERAGE is meaningful. So sample it through the run
    // and stream to postProcessing/forceCoeffs/<startTime>/coefficient.dat, matching OF's layout closely enough
    // that the same plotting scripts work.
    //
    // COST: wallForces runs fvc::gaussGrad over the whole mesh ON THE HOST and calls nearWallDist internally,
    // both mesh-wide and single-threaded. On 6M cells that is seconds per call, so sampling EVERY timestep would
    // dwarf the solve (16 000 calls). It is therefore sampled on the WRITE cadence by default -- 160 samples over
    // this run, ~7 per shedding cycle, enough for a stable mean if not a spectrum. BRAE_FORCE_INTERVAL overrides
    // it (in timesteps) when a finer Cd(t) trace is worth the cost.
    std::vector<std::string> forceWalls;
    for (const auto& q : fvp)
        if (q.type == "wall") forceWalls.push_back(q.name);
    const FoamDict* fcFuncs = controlDict.subDict("functions");
    // Account for EVERY controlDict.functions entry through the shared list. forceCoeffs is declared
    // APPLIED here (not approximated as in gpuSimpleFoam): this driver samples on the write cadence and
    // writes postProcessing/forceCoeffs/<time>/coefficient.dat, which is what OF's forceCoeffs does.
    // Anything else is reported as ignored instead of being silently passed over.
    // Time owns the functionObject lifecycle (OF Foam::Time). forceCoeffs is declared APPLIED here, not
    // approximated: this driver samples on the write cadence and writes
    // postProcessing/forceCoeffs/<time>/coefficient.dat, which is what OF's forceCoeffs does.

    const FoamDict* fcDict = nullptr;
    if (fcFuncs)
        for (const auto& s : fcFuncs->subs)
            if (s.second.wordOr("type", "") == "forceCoeffs") { fcDict = &s.second; break; }
    const long forceInterval = std::getenv("BRAE_FORCE_INTERVAL")
                             ? std::atol(std::getenv("BRAE_FORCE_INTERVAL")) : 0;   // 0 -> follow the write cadence
    std::ofstream fcOut;
    if (fcDict && ctl.turbulent && !forceWalls.empty()) {
        const std::string fdir = caseDir + "/postProcessing/forceCoeffs/" + timeName(startTime);
        std::error_code fec; std::filesystem::create_directories(fdir, fec);
        fcOut.open(fdir + "/coefficient.dat");
        fcOut << "# Time\tCd\tCl\tCm\tFx\tFy\tFz\n";
        fcOut.setf(std::ios::scientific); fcOut.precision(8);
        std::printf("  forceCoeffs: sampling Cd/Cl/Cm -> postProcessing/forceCoeffs/%s/coefficient.dat%s\n",
                    timeName(startTime).c_str(),
                    forceInterval > 0 ? (" every " + std::to_string(forceInterval) + " steps").c_str() : " on the write cadence");
    } else if (fcDict) {
        // Do not let a forceCoeffs block look like it is running when it is not.
        std::printf("  forceCoeffs: present in controlDict but NOT sampled (%s)\n",
                    !ctl.turbulent ? "laminar case" : "no wall patches");
    }

    // Evaluate forces from the CURRENT device state and append one row. Mirrors the steady solver's block
    // (gpuSimpleFoam.cu): pull U/p back, re-evaluate wall BCs, then wallForces on the requested patches.
    auto sampleForces = [&](scalar tval) {
        if (!fcOut.is_open()) return;
        const std::vector<vector> Ug = solver.U();
        const std::vector<scalar> pg = solver.p();
        for (label c = 0; c < nC; ++c) U.internal[c] = Ug[c];
        p.internal = pg;
        U.evaluateBoundary();
        p.evaluateBoundary();
        const scalar wCmu   = ctl.sst ? ctl.ksstCoeffs.betaStar : ctl.keCoeffs.Cmu;
        const scalar wKappa = ctl.sst ? ctl.ksstCoeffs.kappa    : ctl.keCoeffs.kappa;
        const scalar wE     = ctl.sst ? ctl.ksstCoeffs.E        : ctl.keCoeffs.E;
        const bool velNutWall = ctl.sa || ctl.nutWall != NutWall::Nutk;
        const std::vector<scalar> saNutWall = velNutWall ? solver.nutWall() : std::vector<scalar>();
        const std::vector<scalar>* nwb = velNutWall ? &saNutWall : nullptr;
        auto toV = [](const std::vector<scalar>& a, vector d){ return a.size() >= 3 ? vector{a[0],a[1],a[2]} : d; };
        const std::vector<std::string> fcP = fcDict->wordListOr("patches", forceWalls);
        const scalar rhoInf  = fcDict->scalarOr("rhoInf", 1.0), magUInf = fcDict->scalarOr("magUInf", 1.0);
        const scalar Aref    = fcDict->scalarOr("Aref", 1.0),   lRef    = fcDict->scalarOr("lRef", 1.0);
        const vector liftDir = toV(fcDict->scalarListOr("liftDir", {}), vector{0,1,0});
        const vector dragDir = toV(fcDict->scalarListOr("dragDir", {}), vector{1,0,0});
        const vector pitchAx = toV(fcDict->scalarListOr("pitchAxis", {}), vector{0,0,1});
        const vector CofR    = toV(fcDict->scalarListOr("CofR", {}), vector{0,0,0});
        const ForceResult F  = wallForces(U, p, solver.k(), ctl.nu, m, g, fvp, fcP, rhoInf, 0.0, CofR,
                                          wCmu, wKappa, wE, nwb);
        const ForceCoeffs cc = forceCoeffs(F, dragDir, liftDir, pitchAx, rhoInf, magUInf, Aref, lRef);
        const vector T = F.total();
        fcOut << tval << '\t' << cc.Cd << '\t' << cc.Cl << '\t' << cc.Cm << '\t'
              << T.x << '\t' << T.y << '\t' << T.z << '\n';
        fcOut.flush();                       // a long DES should not lose its Cd trace if the run is killed
        std::printf("  forceCoeffs: Cd=%.6e  Cl=%.6e  Cm=%.6e\n", cc.Cd, cc.Cl, cc.Cm);
    };

    // ---- transient time loop ----
    const scalar tEnd = endTime + 0.5 * deltaT;
    long timeIndex = 0;
    std::string lastWritten;
    // Time drives the functionObject lifecycle; the transient loop keeps its own time-valued
    // advancement, which is a genuinely different shape from the steady solvers' iteration index and is
    // not worth restructuring for this. loop() fires start()/execute() at OF's points.
    time.setSteps(static_cast<int>(std::lround((tEnd - (startTime + deltaT)) / deltaT)) + 1);
    for (scalar t = startTime + deltaT; t <= tEnd && time.loop(); t += deltaT) {
        ++timeIndex;
        solver.setTime(t);   // feed the current time to any codedFixedValue snippet's `t`
        const DeviceSimpleResidual r = solver.pimpleStep(deltaT, nOuter, nCorr);
        const std::string tn = timeName(t);
        std::printf("Time = %s\n  Ux %.3e  Uy %.3e  Uz %.3e  p %.3e  contLocal %.3e  contGlobal %.3e\n",
                    tn.c_str(), r.Ux, r.Uy, r.Uz, r.p, r.contLocal, r.contGlobal);
        if (!std::getenv("BRAE_ALLOW_NONFINITE")
            && !(std::isfinite(r.p) && std::isfinite(r.Ux) && std::isfinite(r.Uy) && std::isfinite(r.Uz)))
            throw std::runtime_error("solution diverged: non-finite residual at Time = " + tn + ". Reduce deltaT (CFL).");
        const bool isWrite = isWriteTime(timeIndex, t);
        if (isWrite) { writeTimeDir(tn); lastWritten = tn; time.write(); }
        // Sample Cd on the write cadence, or on BRAE_FORCE_INTERVAL when set (see the cost note above).
        if (forceInterval > 0 ? (timeIndex % forceInterval == 0) : isWrite) sampleForces(t);
    }
    time.end();   // OF Time.C:790-802: a final execute() so the last step is seen, then end()
    {
        const std::string tn = timeName(startTime + deltaT * (scalar)timeIndex);
        if (tn != lastWritten) { writeTimeDir(tn); sampleForces(startTime + deltaT * (scalar)timeIndex); }
    }
    return 0;
}
catch (const std::exception& e)
{
    std::fprintf(stderr, "gpuPimpleFoam error: %s\n", e.what());
    return 1;
}
