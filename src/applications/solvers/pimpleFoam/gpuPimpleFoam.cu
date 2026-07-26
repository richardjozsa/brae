// cf gpuPimpleFoam -- transient incompressible PIMPLE solver, single-GPU device-resident. Reads a standard OpenFOAM
// case (controlDict / fvSolution / fvSchemes / transportProperties + start fields) and marches the transient loop on the
// GPU via DeviceSimpleSolver::pimpleStep -- the SAME three composable phases as steady brae (momentum predictor /
// pressure-velocity / turbulence), with the implicit fvm::ddt(U) folded into the predictor. Writes standard time dirs.
//
// v1 scope: LAMINAR, Euler/backward ddt, the essential controlDict/fvSolution/PIMPLE controls. Deliberately reuses the
// steady driver's field I/O (foam_field_reader/writer) + dict parsing (foam_dict). Follow-ups (lift from gpuSimpleFoam):
// turbulence (ctl.turbulent + k/eps/omega fields), the full fvSchemes div/laplacian-scheme parse, fvOptions, MRF, phi
// output + restart. Kept a SEPARATE executable (brae_pimpleFoam) so it cannot regress the validated steady brae.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_field_writer.cuh"
#include "foam_dict.cuh"
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

// OF-style time-directory name: an integer for whole times (deltaT=1), else %g (matches Foam::Time::timeName).
std::string timeName(scalar t)
{
    if (t == std::floor(t) && std::fabs((double)t) < 1e15)
        return std::to_string((long long)std::llround((double)t));
    char b[64];
    std::snprintf(b, sizeof b, "%g", (double)t);
    return std::string(b);
}

// Parse fvSchemes ddtSchemes.default from raw text (fvSchemes holds multi-token scheme values + $-vars, so it is not a
// plain key/value dict -- the steady driver text-parses it too). Returns the word after "default" inside ddtSchemes{}.
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
    d += 7;                                                        // skip "default"
    while (d < text.size() && std::isspace((unsigned char)text[d])) ++d;
    std::string w;
    while (d < text.size() && !std::isspace((unsigned char)text[d]) && text[d] != ';') w += text[d++];
    if (w == "steadyState")
        throw std::runtime_error("fvSchemes ddtSchemes.default = steadyState -> use the steady solver 'brae'. "
                                 "pimpleFoam needs Euler or backward.");
    if (w == "backward")                          return DdtScheme::backward;
    if (w == "Euler" || w == "bounded")           return DdtScheme::Euler;   // "bounded Euler" -> Euler (bounding is a no-op here)
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
    if (deltaT <= 0)      throw std::runtime_error("controlDict deltaT must be > 0.");
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
    const FoamDict* solvers = fvSolution.subDict("solvers");
    auto solveTol = [&](const char* f, scalar dflt) {
        const FoamDict* s = solvers ? solvers->subDict(f) : nullptr;
        return s ? s->scalarOr("tolerance", dflt) : dflt;
    };
    const scalar tolU = solveTol("U", 1e-7), tolP = solveTol("p", 1e-7);
    const scalar nu = transport.scalarOr("nu", 1e-5);

    // ---- mesh + start fields ----
    PrimitiveMesh m; m.read(caseDir + "/constant/polyMesh");
    FvGeometry g; g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();
    const std::string fieldDir = caseDir + "/" + timeName(startTime);
    const FieldData<vector> Ufd = readField<vector>(fieldDir + "/U");
    const FieldData<scalar> pfd = readField<scalar>(fieldDir + "/p");
    GeometricField<vector> U = buildField<vector>(Ufd, fvp, nC); U.evaluateBoundary();
    GeometricField<scalar> p = buildField<scalar>(pfd, fvp, nC); p.evaluateBoundary();
    SurfaceScalarField phi = fvc::flux(U, m, g, fvp);

    DeviceSimpleControls ctl;
    ctl.nu = nu; ctl.relaxU = relaxU; ctl.relaxP = relaxP;
    ctl.tolU = tolU; ctl.tolP = tolP; ctl.nNonOrth = nNonOrth;
    ctl.turbulent = false;                                          // v1: laminar
    DeviceSimpleSolver solver(m, g, fvp, U, p, phi, ctl);
    solver.setDdtScheme(ddtScheme);

    std::printf("gpuPimpleFoam (laminar): deltaT=%g endTime=%g ddt=%s nOuterCorrectors=%d nCorrectors=%d "
                "nNonOrth=%d nCells=%d\n\n",
                (double)deltaT, (double)endTime, ddtScheme == DdtScheme::backward ? "backward" : "Euler",
                nOuter, nCorr, nNonOrth, nC);

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
        std::printf("written %s/{U,p}\n", outDir.c_str());
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
    const scalar tEnd = endTime + 0.5 * deltaT;                     // include endTime against FP drift
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
            throw std::runtime_error("solution diverged: non-finite residual at Time = " + tn
                + ". Reduce deltaT (CFL), or check the case. No field written past here.");
        if (isWriteTime(timeIndex, t)) { writeTimeDir(tn); lastWritten = tn; }
    }
    // Always write the final state (OF writes endTime even if writeInterval did not land on it).
    {
        scalar tLast = startTime + deltaT * (scalar)timeIndex;
        const std::string tn = timeName(tLast);
        if (tn != lastWritten) writeTimeDir(tn);
    }
    return 0;
}
catch (const std::exception& e)
{
    std::fprintf(stderr, "gpuPimpleFoam error: %s\n", e.what());
    return 1;
}
