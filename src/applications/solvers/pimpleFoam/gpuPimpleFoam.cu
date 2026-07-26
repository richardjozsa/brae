// cf gpuPimpleFoam -- transient incompressible PIMPLE solver, single-GPU device-resident. Reads a standard OpenFOAM
// case (controlDict / fvSolution / fvSchemes / transportProperties / turbulenceProperties + start fields) and marches
// the transient loop on the GPU via DeviceSimpleSolver::pimpleStep -- the SAME three composable phases as steady brae
// (momentum predictor / pressure-velocity / turbulence), with the implicit fvm::ddt(U) folded into the predictor.
// Writes standard OpenFOAM time directories.
//
// v1 scope: Euler/backward ddt, laminar OR RAS (kEpsilon/realizableKE/kOmegaSST/kOmegaSSTLM/SpalartAllmaras). Reuses the
// steady driver's field I/O (foam_field_reader/writer) + dict parsing (foam_dict) + turbulence model setup. NOTE: the
// MOMENTUM ddt is fully implicit + OF-exact; the TURBULENCE transport currently runs quasi-steady per time step (no
// fvm::ddt(k/eps/omega) yet -- that ddt wiring through the scalar-transport scaffold is the next step). Follow-ups: the
// full fvSchemes div/laplacian-scheme parse (v1 uses brae defaults: upwind div), phi output + restart, fvOptions/MRF.
// Kept a SEPARATE executable (brae_pimpleFoam) so it cannot regress the validated steady brae.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_field_writer.cuh"
#include "foam_dict.cuh"
#include "scheme_parse.cuh"         // parseFvSchemesControls: shared fvSchemes div/laplacian scheme parse
#include "turbulence_setup.cuh"    // readTurbulenceModel + readTurbulenceFields (shared with brae)
#include "komega_sst_coeffs.cuh"    // readKOmegaSSTCoeffs
#include "fvc.cuh"
#include "device_simple_foam.cuh"
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
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
DdtScheme parseDdtScheme(const std::string& fvSchemesPath)
{
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
    throw std::runtime_error("fvSchemes ddtSchemes.default '" + w + "' unsupported (Euler|backward|steadyState).");
}

}  // namespace

int main(int argc, char** argv)
try
{
    const std::string caseDir = argc > 1 ? argv[1] : ".";

    // ---- case dictionaries ----
    const FoamDict controlDict = readDict(caseDir + "/system/controlDict");
    const FoamDict fvSolution  = readDict(caseDir + "/system/fvSolution");
    const FoamDict transport   = readDict(caseDir + "/constant/transportProperties");
    const DdtScheme ddtScheme  = parseDdtScheme(caseDir + "/system/fvSchemes");

    const scalar startTime     = controlDict.scalarOr("startTime", 0.0);
    const scalar deltaT        = controlDict.scalarOr("deltaT", 1e-3);
    const scalar endTime       = controlDict.scalarOr("endTime", 1.0);
    const std::string writeControl = controlDict.wordOr("writeControl", "timeStep");
    const scalar writeInterval = controlDict.scalarOr("writeInterval", 1e30);
    const int    purgeWrite    = std::max(0, controlDict.intOr("purgeWrite", 0));
    const int    precision     = controlDict.intOr("writePrecision", 16);
    if (deltaT <= 0)          throw std::runtime_error("controlDict deltaT must be > 0.");
    if (endTime <= startTime) throw std::runtime_error("controlDict endTime must be > startTime.");

    // ---- PIMPLE + solver controls (fvSolution) ----
    const FoamDict* pimple  = fvSolution.subDict("PIMPLE");
    const int nOuter   = pimple ? std::max(1, pimple->intOr("nOuterCorrectors", 1)) : 1;
    const int nCorr    = pimple ? std::max(1, pimple->intOr("nCorrectors", 1)) : 1;
    const int nNonOrth = pimple ? pimple->intOr("nNonOrthogonalCorrectors", 0) : 0;
    const FoamDict* rf   = fvSolution.subDict("relaxationFactors");
    const FoamDict* rfEq = rf ? rf->subDict("equations") : nullptr;
    const FoamDict* rfFl = rf ? rf->subDict("fields") : nullptr;
    const scalar relaxU  = rfEq ? rfEq->scalarOr("U", 1.0) : 1.0;   // transient default: no relaxation
    const scalar relaxP  = rfFl ? rfFl->scalarOr("p", 1.0) : 1.0;
    const scalar relaxK  = rfEq ? rfEq->scalarOr("k", 1.0) : 1.0;
    const FoamDict* solvers = fvSolution.subDict("solvers");
    auto solveTol = [&](const char* f, scalar dflt) {
        const FoamDict* s = solvers ? solvers->subDict(f) : nullptr;
        return s ? s->scalarOr("tolerance", dflt) : dflt;
    };
    const scalar nu = transport.scalarOr("nu", 1e-5);

    DeviceSimpleControls ctl;
    ctl.nu = nu; ctl.relaxU = relaxU; ctl.relaxP = relaxP; ctl.relaxK = relaxK; ctl.relaxEps = relaxK;
    ctl.tolU = solveTol("U", 1e-7); ctl.tolP = solveTol("p", 1e-7); ctl.tolKE = solveTol("k", 1e-8);
    ctl.nNonOrth = nNonOrth;
    // fvSchemes div/laplacian/grad scheme flags (2nd-order div(phi,U) linearUpwind, non-orth correction, limitedLinear
    // turbulence scalars, ...). Same parse as steady brae -> the transient solve honours the case's schemes, not upwind.
    parseFvSchemesControls(caseDir, ctl);

    // ---- turbulence model (constant/turbulenceProperties). Absent -> laminar. ----
    std::string secondName = "epsilon";
    if (std::filesystem::exists(caseDir + "/constant/turbulenceProperties"))
    {
        const FoamDict turbProps = readDict(caseDir + "/constant/turbulenceProperties");
        const std::string simType = turbProps.wordOr("simulationType", "laminar");
        if (simType != "RAS" && simType != "laminar")
            throw std::runtime_error("pimpleFoam: unsupported simulationType '" + simType + "' (RAS or laminar).");
        ctl.turbulent = (simType == "RAS");
        readTurbulenceModel(turbProps, ctl);
        secondName = ctl.sst ? "omega" : "epsilon";
    }

    // ---- mesh + start fields ----
    PrimitiveMesh m; m.read(caseDir + "/constant/polyMesh");
    FvGeometry g; g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();
    const std::string fieldDir = caseDir + "/" + timeName(startTime);
    GeometricField<vector> U = buildField<vector>(readField<vector>(fieldDir + "/U"), fvp, nC); U.evaluateBoundary();
    GeometricField<scalar> p = buildField<scalar>(readField<scalar>(fieldDir + "/p"), fvp, nC); p.evaluateBoundary();
    SurfaceScalarField phi = fvc::flux(U, m, g, fvp);

    TurbulenceFields tf = readTurbulenceFields(fieldDir, fvp, nC, ctl, secondName, U);

    DeviceSimpleSolver solver(m, g, fvp, U, p, phi, ctl,
                              ctl.turbulent ? &tf.k : nullptr, (ctl.turbulent && !ctl.sa) ? &tf.eps : nullptr,
                              ctl.turbulent ? &tf.nut : nullptr, ctl.lm ? &tf.ReThetat : nullptr, ctl.lm ? &tf.gammaInt : nullptr);
    solver.setDdtScheme(ddtScheme);

    std::printf("gpuPimpleFoam: deltaT=%g endTime=%g ddt=%s nOuterCorrectors=%d nCorrectors=%d nNonOrth=%d nu=%g "
                "turbulence=%s nCells=%d\n\n",
                (double)deltaT, (double)endTime, ddtScheme == DdtScheme::backward ? "backward" : "Euler",
                nOuter, nCorr, nNonOrth, (double)nu,
                !ctl.turbulent ? "laminar" : ctl.sa ? "SpalartAllmaras" : ctl.sst ? (ctl.lm ? "kOmegaSSTLM" : "kOmegaSST") : "kEpsilon",
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
    auto writeTimeDir = [&](const std::string& tname) {
        const std::string outDir = caseDir + "/" + tname;
        std::filesystem::create_directories(outDir);
        std::error_code ec;
        if (std::filesystem::exists(fieldDir + "/include"))
            std::filesystem::copy(fieldDir + "/include", outDir + "/include",
                std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing, ec);
        writeVolField(fieldDir + "/U", outDir + "/U", solver.U(), fvp, precision);
        writeVolField(fieldDir + "/p", outDir + "/p", solver.p(), fvp, precision);
        if (ctl.sa) {
            writeVolField(fieldDir + "/nuTilda", outDir + "/nuTilda", solver.k(),   fvp, precision);
            writeVolField(fieldDir + "/nut",     outDir + "/nut",     solver.nut(), fvp, precision);
        } else if (ctl.turbulent) {
            writeVolField(fieldDir + "/k",          outDir + "/k",          solver.k(),   fvp, precision);
            writeVolField(fieldDir + "/" + secondName, outDir + "/" + secondName, solver.eps(), fvp, precision);
            writeVolField(fieldDir + "/nut",        outDir + "/nut",        solver.nut(), fvp, precision);
            if (ctl.lm) {
                writeVolField(fieldDir + "/ReThetat", outDir + "/ReThetat", solver.ReThetat(), fvp, precision);
                writeVolField(fieldDir + "/gammaInt", outDir + "/gammaInt", solver.gammaInt(), fvp, precision);
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

    // ---- transient time loop ----
    const scalar tEnd = endTime + 0.5 * deltaT;
    long timeIndex = 0;
    std::string lastWritten;
    for (scalar t = startTime + deltaT; t <= tEnd; t += deltaT) {
        ++timeIndex;
        const DeviceSimpleResidual r = solver.pimpleStep(deltaT, nOuter, nCorr);
        const std::string tn = timeName(t);
        std::printf("Time = %s\n  Ux %.3e  Uy %.3e  Uz %.3e  p %.3e  contLocal %.3e  contGlobal %.3e\n",
                    tn.c_str(), r.Ux, r.Uy, r.Uz, r.p, r.contLocal, r.contGlobal);
        if (!std::getenv("BRAE_ALLOW_NONFINITE")
            && !(std::isfinite(r.p) && std::isfinite(r.Ux) && std::isfinite(r.Uy) && std::isfinite(r.Uz)))
            throw std::runtime_error("solution diverged: non-finite residual at Time = " + tn + ". Reduce deltaT (CFL).");
        if (isWriteTime(timeIndex, t)) { writeTimeDir(tn); lastWritten = tn; }
    }
    {
        const std::string tn = timeName(startTime + deltaT * (scalar)timeIndex);
        if (tn != lastWritten) writeTimeDir(tn);
    }
    return 0;
}
catch (const std::exception& e)
{
    std::fprintf(stderr, "gpuPimpleFoam error: %s\n", e.what());
    return 1;
}
